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

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import datasets
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
import numpy as np
from datasets.iterable_dataset import _BaseExamplesIterable, HasNextIterator, deepcopy, IterableDataset, Features, DatasetInfo
from typing import Optional, List, Any, Dict, Optional, Union, Tuple, Iterator
from datasets import load_dataset, IterableDataset, set_caching_enabled

class MultiSourcesExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterables,
        generator: np.random.Generator,
        probabilities: Optional[List[float]] = None
    ):
        self.ex_iterables = ex_iterables
        self.generator = deepcopy(generator)
        self.probabilities = probabilities
        

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        random_batch_size=1000,
        p: Optional[List[float]] = None,
    ) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        if p is None:
            while True:
                yield from (int(i) for i in rng.integers(0, num_sources, size=random_batch_size))
        else:
            while True:
                yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=p))

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(rng, len(self.ex_iterables), p=self.probabilities)


    def __iter__(self):
        iterators = [[HasNextIterator(ex) for ex in ex_iterable] for ex_iterable in self.ex_iterables]
        ex_idxs =  [0 for _ in self.ex_iterables]

        indices_iterator = self._give_indice_iterator()

        for i in indices_iterator:

            j = ex_idxs[i]

            try:  # let's pick one example from the iterator at index i
                yield next(iterators[i][j])
                # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                if not iterators[i][j].hasnext():
                    iterators[i][j] = HasNextIterator(self.ex_iterables[i][j])

            except StopIteration:
                iterators[i][j] = HasNextIterator(self.ex_iterables[i][j])
            
            ex_idxs[i] = (j + 1) % len(iterators[i])


    def shuffle_data_sources(self, generator: np.random.Generator) -> "MultiSourcesExamplesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [[ex.shuffle_data_sources(generator) for ex in ex_iterable] for ex_iterable in self.ex_iterables]
        return MultiSourcesExamplesIterable(
            ex_iterables, generator=generator, probabilities=self.probabilities
        )

    def shard_data_sources(self, shard_idx: int) -> "MultiSourcesExamplesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        raise NotImplementedError("Sharding a RandomlyCyclingMultiSourcesExamplesIterable is not implemented")

def mkdir_json_dataset(
    json_data_paths: List[str], 
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    features: Optional[Features] = None,
):
    generator = np.random.default_rng(seed)

    json_datasets = [
        [
            load_dataset(
                "json", 
                data_files=data_path, 
                streaming=True,
                split="train",
                features=features
            )
            for data_path in json_data_path
        ]
        for json_data_path in json_data_paths
    ]

    ex_iterables = [[d._ex_iterable for d in json_dataset] for json_dataset in json_datasets]

    ex_iterable = MultiSourcesExamplesIterable(
        ex_iterables, 
        generator=generator, 
        probabilities=probabilities
    )

    flatten_json_datasets = []
    for item in json_datasets:
        flatten_json_datasets.extend(item)

    info = DatasetInfo.from_merge([d.info for d in flatten_json_datasets])

    token_per_repo_id = {
        repo_id: token for dataset in flatten_json_datasets for repo_id, token in dataset._token_per_repo_id.items()
    }

    # Return new daset
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=None, token_per_repo_id=token_per_repo_id)



def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                        help='Pad the vocab size to this value.'
                        'This value must be greater than the initial size of the tokenizer'
                        ', needs to be divisible by TP size and `make-vocab-size-divisible-by`.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()
    
    tokenizer = build_tokenizer(args)
    _tokenizer = tokenizer.tokenizer

    # build output
    level = "document"
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    key = "text"
    output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                    key, level)
    output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                    key, level)
    builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                            impl=args.dataset_impl,
                                            vocab_size=tokenizer.vocab_size)


    print("Opening", args.input)

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
                args.input + "chatgpt_med_copy-train.jsonl.gz",
            ],
            # chatgpt 问诊_datasets
            [
                args.input + "chatgpt_qmed_copy-train.jsonl.gz",
            ],
            # chatgpt nb问诊_datasets
            [
                args.input + "chatgpt_nbmed_copy-train.jsonl.gz",
            ],
            # chatgpt tongyong
            [
                args.input + "chatgpt_copy-train.jsonl.gz",
            ],
            # chatgpt_sharegpt_copy
            [
                args.input + "chatgpt_sharegpt_copy-train.jsonl.gz",
            ],
            # chatgpt_med_dig_copy
            [
                args.input + "chatgpt_med_dig_copy-train.jsonl.gz",
            ],
            # chatgpt tongyong duihua
            [
                args.input + "chatgpt_dig_copy-train.jsonl.gz",
            ],
            # chatgpt tongyong half duihua
            [
                args.input + "chatgpt_half_dig-train.jsonl.gz",
            ],
            # chatgpt_plugins_copy
            [
                args.input + "chatgpt_plugins_copy-train.jsonl.gz",
            ],
            # chatgpt cn kbqa copy
            [
                args.input + "chatgpt_baiduzhidao_copy-train.jsonl.gz",
                args.input + "chatgpt_zhihu_copy-train.jsonl.gz",
            ],
            # chatgpt en kbqa copy
            [
                args.input + "chatgpt_en_qa_copy-train.jsonl.gz",
            ],
            # chatgpt_code_copy
            [
                args.input + "chatgpt_codesearchnet_copy-train.jsonl.gz",
                args.input + "chatgpt_lang2code_copy-train.jsonl.gz",
            ],
            # chatgpt en med copy
            [
                args.input + "chatgpt_en_med_copy-train.jsonl.gz",
            ],
            # chatgpt_en_instructions_copy
            [
                args.input + "chatgpt_en_instructions_copy-train.jsonl.gz",
            ],
            # chatgpt_cn_instructions_copy
            [
                args.input + "chatgpt_cn_instructions_copy-train.jsonl.gz",
            ],
            # chatgpt_xp3_copy
            [
                args.input + "chatgpt_xp3_copy-train.jsonl.gz",
            ],
            # chatgpt_thought_source_copy
            [
                args.input + "chatgpt_thought_source_copy-train.jsonl.gz",
            ],
            # chatgpt_role_play_copy
            [
                args.input + "chatgpt_role_play_copy-train.jsonl.gz",
            ],
            # chatgpt_small_task
            [
                args.input + "chatgpt_small_task_BMI计算_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_fan_归一化_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_new_apx210k_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_new_math23k_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_xiaoshuo_kuoxie_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_xiaoshuo_sammary_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_病例抽取_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_病例生成_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_知识问答生成_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_丁金如_归一化_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_模仿知乎_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_判断模型_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_前nbmed_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_四则运算_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_问题生成_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_药物分类_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_医学指令构造_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_英文报告翻译_copy-train.jsonl.gz",
                args.input + "chatgpt_small_task_自动随访_copy-train.jsonl.gz",

            ],
            # chatgpt_self_identity
            [
                args.input + "chatgpt_self_identity/chatgpt_baiduzhidao_copy-train.jsonl.gz",
                args.input + "chatgpt_self_identity/chatgpt_copy-train.jsonl.gz",
                args.input + "chatgpt_self_identity/chatgpt_med_copy-train.jsonl.gz",
                args.input + "chatgpt_self_identity/chatgpt_nbmed_copy-train.jsonl.gz",
                args.input + "chatgpt_self_identity/chatgpt_qmed_copy-train.jsonl.gz",
                args.input + "chatgpt_self_identity/chatgpt_sharegpt_copy-train.jsonl.gz",
            ],
            # gpt 4 copy
            [
                args.input + "chatgpt_gpt4_copy-train.jsonl.gz",
            ],
            
            # 自己标注对话（部分体检）
            [
                args.input + "our_med_qa-train.jsonl.gz",
            ],
            # not_med_datasets
            [
                args.input + "not_med-train.jsonl.gz",
            ], 

            # 预训练数据
            # 医学书+指南
            [
                args.input + "our_med_kb-train.jsonl.gz",
            ],

            # 指令微调数据
            # med_dig_datasets
            [
                args.input + "med_dig-train.jsonl.gz",
            ],
            #  问诊_datasets
            [
                args.input + "qmed_dig-train.jsonl.gz",
            ],
            # covid_19_datasets
            [
                args.input + 'covid_19_kb-train.jsonl.gz',
                args.input + 'covid_med_dig-train.jsonl.gz',
            ],
            # med_qa_datasets
            [
                args.input + 'MedQA-train.jsonl.gz',
                args.input + 'mlec-qa-train.jsonl.gz',
            ],
        ], 
        probabilities=[
            # 标准对话训练数据
            # chatgpt med_dig_datasets
            0.065,
            # chatgpt 问诊_datasets
            0.025,
            # chatgpt nb问诊_datasets
            0.065,
            # chatgpt tongyong
            0.075,
            # chatgpt_sharegpt_copy
            0.07,
            # chatgpt_med_dig_copy
            0.005,
            # chatgpt tongyong duihua
            0.035,
            # chatgpt tongyong half duihua
            0.015,
            # chatgpt_plugins_copy
            0.025,
            # chatgpt cn kbqa copy
            0.075,
            # chatgpt en kbqa copy
            0.075,
            # chatgpt_code_copy
            0.045,
            # chatgpt en med copy
            0.025,
            # chatgpt_en_instructions_copy
            0.05,
            # chatgpt_cn_instructions_copy
            0.05,
            # chatgpt_xp3_copy
            0.075,
            # chatgpt_thought_source_copy
            0.05,
            # chatgpt_role_play_copy
            0.005,
            # chatgpt_small_task
            0.045,
            # chatgpt_self_identity
            0.01,
            # gpt 4 copy
            0.05,
            
            # 自己标注对话（部分体检）
            0.005,
            # not_med_datasets
            0.005, 

            # 预训练数据
            # 医学书+指南
            0.025,

            # 指令微调数据
            # med_dig_datasets
            0.01,
            # 问诊_datasets
            0.01,
            # covid_19_datasets
            0.005,
            # med_qa_datasets
            0.005,
        ],
        seed=42,
        features=dig_features,
    )


    # self.prompt = "Instructions: You are Helper, a large language model trained by ShLab."

    speaker_mapper = {
        "from user": "User: ",
        "to user": "Helper: ",
        "to note": "Record: ",
        "to terminal": "Command: ",
        "from terminal": "Terminal: ",
    }
    def train_map(batch):

        all_input_ids = []

        for prompt, dig in zip(batch['instruction'], batch['digs']):

            input_ids = []

            if len(dig['speaker']) == 0:
                continue

            # 预训练数据不一样的逻辑
            if "Pretrain" in dig['speaker'][0].lower():

                for text in dig['text']:
                    input_ids.extend(_tokenizer(text, add_special_tokens=False).input_ids)
                
                #文档结束
                input_ids.append(_tokenizer.convert_tokens_to_ids("</s>"))

            else:
                prompt_ids = _tokenizer("Instructions: " + prompt, add_special_tokens=False).input_ids + [_tokenizer.convert_tokens_to_ids("</s>")]

                input_ids += prompt_ids

                for didx, (old_speaker, text) in enumerate(zip(dig['speaker'], dig['text'])):
                    
                    old_speaker = old_speaker.lower()
                    
                    speaker = None

                    for k,v in speaker_mapper.items():
                        if k in old_speaker:
                            speaker = v
                    
                    if speaker is None:
                        continue

                    # 合并对象
                    dig_ids = _tokenizer(speaker + text, add_special_tokens=False).input_ids
                    input_ids += dig_ids
                    input_ids += [_tokenizer.convert_tokens_to_ids("</s>")]

            all_input_ids.append(input_ids)

        return {
            "input_ids": all_input_ids,
        }

    train_dataset = raw_train_datasets.map(
        train_map,
        batched=True,
        batch_size=1000,
        # num_proc=128,
        remove_columns=['digs', "instruction"],
        # desc="Running train_map",
    )

    train_dataset = train_dataset.take(16 * 64 * 73728)

    startup_end = time.time()
    proc_start = time.time()
    print("Time to startup:", startup_end - startup_start)

    for i, doc in enumerate(train_dataset, start=1):

        builders[key].add_item(torch.IntTensor(doc['input_ids']))
        builders[key].end_document()

        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s).",
                  file=sys.stderr)
            
    builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main()
