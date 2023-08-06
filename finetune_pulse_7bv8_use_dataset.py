# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain Llama"""
import torch
import math
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import get_num_microbatches
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.hfdatasets_helper import mkdir_json_dataset
from megatron.model import LlamaModel, LlamaModelPipe
from megatron.training import pretrain_with_dataloader
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import os
import numpy as np
import subprocess
import datasets

from tqdm.auto import tqdm

from transformers import DataCollatorForTokenClassification
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import DataLoader

from torch import nn
import torch.nn.functional as F



def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building llama model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = LlamaModelPipe(parallel_output=True)
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=get_accelerator().current_device_name())).view(
                1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

        else:
            model = LlamaModel(
                parallel_output=True,
                add_pooler=False,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_ids', "labels"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['input_ids'].long()[:, :-1].contiguous()
    labels = data_b['labels'].long()[:, 1:].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, _ = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss
    )

    loss_mask = (labels >=0).long()
    labels = torch.clamp(labels, min=0, max=None)

    return tokens, labels, loss_mask, attention_mask



def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_ids', "labels"]
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack. # 限制长度
    tokens = data_b['input_ids'].long()[:, :-1].contiguous()
    labels = data_b['labels'].long()[:, 1:].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, _ = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss
    )
    
    loss_mask = (labels >= 0).long()
    labels = torch.clamp(labels, min=0, max=None)

    return (tokens, attention_mask), (labels, loss_mask)

def loss_func(loss_mask, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, attention_mask, labels=labels)
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_dataloader_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets for GPT ...')

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        tokenizer = get_tokenizer().tokenizer

        dig_features = datasets.Features(
            {
                'instruction': datasets.Value("string"),
                "digs": datasets.Sequence(feature={
                        "speaker": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        'choices': datasets.Sequence(datasets.Value("string")),
                    }
                ),
            }
        )

        # raw_datasets
        raw_train_datasets = mkdir_json_dataset(
            json_data_paths=[
                # 标准对话训练数据
                # chatgpt med_dig_datasets
                [
                    args.data_path[0] + "chatgpt_baiduzhidao_copy-train.jsonl.gz",
                ],

            ], 
            probabilities=[
                # 标准对话训练数据
                # chatgpt med_dig_datasets
                1.0,
            ],
            seed=args.seed,
            features=dig_features,
        )

        raw_train_datasets = raw_train_datasets.shuffle(seed=args.seed, buffer_size=100000)

        raw_eval_datasets = mkdir_json_dataset(
            json_data_paths=[
                # 标准对话训练数据
                # chatgpt med_dig_datasets
                [
                    args.data_path[0] + "chatgpt_baiduzhidao_copy-validation.jsonl.gz",
                ],
            ], 
            probabilities=[
                # 标准对话训练数据
                # chatgpt med_dig_datasets
                1.0,
            ],
            seed=args.seed,
            features=dig_features,
        )

        #  tokenizer deal
        train_max_len = args.seq_length + 1
        # pretrain_cut_step = train_max_len - 128
        pretrain_cut_step = train_max_len

        speaker_mapper = {
            "from user": "User: ",
            "to user": "Helper: ",
            "to note": "Record: ",
            "to terminal": "Command: ",
            "from terminal": "Terminal: ",
        }

        def train_map(batch):

            all_input_ids = []
            all_labels = []

            for prompt, dig in zip(batch['instruction'], batch['digs']):

                input_ids = []
                labels = []

                if len(dig['speaker']) == 0:
                    continue

                # 预训练数据不一样的逻辑
                if "Pretrain" in dig['speaker'][0].lower():

                    for text in dig['text']:
                        input_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
                        labels.extend(tokenizer(text, add_special_tokens=False).input_ids)
                    
                    #文档结束
                    input_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
                    labels.append(tokenizer.convert_tokens_to_ids("</s>"))

                else:
                    prompt_ids = tokenizer("Instructions: " + prompt, add_special_tokens=False).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]

                    input_ids += prompt_ids
                    labels += [-100] * len(prompt_ids)

                    for didx, (old_speaker, text) in enumerate(zip(dig['speaker'], dig['text'])):
                        
                        old_speaker = old_speaker.lower()
                        
                        speaker = None

                        for k,v in speaker_mapper.items():
                            if k in old_speaker:
                                speaker = v
                        
                        if speaker is None:
                            continue

                        # 合并对象
                        dig_ids = tokenizer(speaker + text, add_special_tokens=False).input_ids
                        input_ids += dig_ids
                        input_ids += [tokenizer.convert_tokens_to_ids("</s>")]

                        # 生成的
                        if speaker in {"Helper: ", "Record: ", "Command: "}:
                            labels += dig_ids
                            labels += [tokenizer.convert_tokens_to_ids("</s>")]
                        else:
                            labels += [-100] * len(dig_ids)
                            labels += [-100]

                    # input_ids = input_ids[:train_max_len]
                    # labels = labels[:train_max_len]

                # 事先裁剪过长的句子
                if len(input_ids) > train_max_len:
                    for i in range(0, len(input_ids), pretrain_cut_step):
                        cut_input_ids = input_ids[i: i+train_max_len]
                        cut_labels = labels[i: i+train_max_len]

                        # 删除一句回复都没有的情况
                        if len(cut_labels) >= 12 and np.any(np.array(cut_labels)[1:] >= 0):

                            all_input_ids.append(cut_input_ids)
                            all_labels.append(cut_labels)
                else:
                    all_input_ids.append(input_ids)
                    all_labels.append(labels)


            # 统一合并逻辑
            batch_input_ids = [[]]
            batch_labels = [[]]

            for input_ids, labels in zip(all_input_ids, all_labels):

                if len(batch_input_ids[-1]) + len(input_ids) > train_max_len:
                    batch_input_ids.append(input_ids)
                    batch_labels.append(labels)
                else:
                    batch_input_ids[-1].extend(input_ids)
                    batch_labels[-1].extend(labels)

            # 最后一个可能为空
            if len(batch_input_ids[-1]) == 0:
                batch_input_ids.pop(-1)
                batch_labels.pop(-1)

            return {
                "input_ids": batch_input_ids,
                "labels": batch_labels,
            }
        
        
        train_dataset = raw_train_datasets.map(
            train_map,
            batched=True,
            batch_size=65536,
            # num_proc=32,
            remove_columns=['digs', "instruction"],
            # desc="Running train_map",
        )

        eval_dataset = raw_eval_datasets.map(
            train_map,
            batched=True,
            batch_size=65536,
            # num_proc=32,
            remove_columns=['digs', "instruction"],
            # desc="Running train_map",
        )

        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding="max_length", max_length=(args.seq_length + 1))

        if mpu.get_data_parallel_world_size() > 1:
            train_dataset = IterableDatasetShard(
                train_dataset,
                batch_size=args.micro_batch_size,
                drop_last=True,
                num_processes=mpu.get_data_parallel_world_size(),
                process_index=mpu.get_data_parallel_rank(),
            )

            eval_dataset = IterableDatasetShard(
                eval_dataset,
                batch_size=args.micro_batch_size,
                drop_last=True,
                num_processes=mpu.get_data_parallel_world_size(),
                process_index=mpu.get_data_parallel_rank(),
            )

        # train_dataset.__len__ = lambda self: 999999999999999999999999
        # eval_dataset.__len__ = lambda self: 999999999999999999999999

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            collate_fn=data_collator,
            num_workers=0,
            # generator=torch.Generator().manual_seed(args.seed),
            pin_memory=True,
        )

        valid_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.micro_batch_size,
            collate_fn=data_collator,
            num_workers=0,
            # generator=torch.Generator().manual_seed(args.seed),
            pin_memory=True,
        )

        test_dataloader = None

        # Need to broadcast num_tokens and num_type_tokens.
        flags = get_accelerator().LongTensor([
            1,
            1, # eval_iters == 0 is equivalent to having no validation
            0, # eval_iters == 0 is equivalent to having no test
        ])
    else:
        flags = get_accelerator().LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Build iterators.
    # dl_type = args.dataloader_type

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    # skip
    if args.consumed_train_samples > 0 and train_data_iterator is not None:
        print("iteration > 0 skip train dataloader start")
        for _ in tqdm(range(args.consumed_train_samples)):
            next(train_data_iterator)
        
        print("iteration > 0 skip train dataloader finish")

    if args.consumed_valid_samples > 0 and valid_data_iterator is not None:
        print("iteration > 0 skip valid dataloader start")
        for _ in tqdm(range(args.consumed_valid_samples)):
            next(valid_data_iterator)

        print("iteration > 0 skip valid dataloader finish")

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0("> finished creating GPT datasets ...")
    return train_data_iterator, valid_data_iterator, test_data_iterator



def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')


if __name__ == "__main__":
    git_ds_info()
    pretrain_with_dataloader(train_valid_test_dataloader_provider, model_provider, forward_step,
             data_post_process=None)
