
import argparse
import json
import os
import re

import torch

from transformers import LlamaConfig
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging


logging.set_verbosity_info()

WEIGHTS_TO_AVERAGE_ENDSWITH = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "rotary_emb.inv_freq",
]

WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "mlp.down_proj.weight",
    "attention.dense.weight",
]


def layer_name_mapping(key, file):
    """Convert Megatron-DeepSpeed TP/PP weights mapping in transformers PP only"""
    # Handle first and last layers
    layer_rename_map = {
        "word_embeddings.weight": "model.embed_tokens.weight",
        "weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    if key in layer_rename_map:
        return layer_rename_map[key]
    

    # Handle transformer blocks
    layer_number = int(re.match(r".*layer_(\d*).*", file)[1])
    layer_number -= 3
    return f"model.layers.{layer_number}." + key


def get_dtype_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def convert_llama_checkpoint_to_pytorch(
    llama_checkpoint_path, llama_config_file, pytorch_dump_folder_path, pretraining_tp
):
    # Construct model
    assert llama_config_file != ""

    config = LlamaConfig.from_json_file(llama_config_file)

    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    hidden_size = config.hidden_size

    base = 10000
    alpha = 1

    base = base * alpha ** (head_dim / (head_dim-2))

    file_names = os.listdir(llama_checkpoint_path)
    file_names = sorted(filter(lambda s: s.startswith("layer") and "model_00" in s, file_names))

    index_dict = {"weight_map": {}, "metadata": {}}
    total_size = 0

    missing_keys = None

    for j, file in enumerate(file_names):
        print("Processing file: {}".format(file))
        tensors = None

        for i in range(pretraining_tp):
            # load all TP files
            f_name = file.replace("model_00", f"model_0{i}")
            temp = torch.load(os.path.join(llama_checkpoint_path, f_name), map_location="cpu")

            # Rename keys in the transformers names
            keys = list(temp.keys())
            for key in keys:
                temp[layer_name_mapping(key, file)] = temp.pop(key)

            if tensors is None:
                tensors = temp
            else:
                for key in tensors.keys():
                    if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                        # We average (sum and then divide) some weights accross TP ranks (see https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/olruwase/sync_layer_norms/megatron/training.py#L425)
                        tensors[key] += temp[key]
                    else:
                        # Some weights are RowParallelLinear in Megatron-Deepspeed, others are ColumnParallel
                        cat_dim = 1 if any(text in key for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN) else 0
                        # We concatenate these weights accross TP ranks
                        tensors[key] = torch.cat([tensors[key], temp[key]], dim=cat_dim)

        # Divide by the number of TP the weights we want to average
        for key in tensors.keys():
            if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                tensors[key] = tensors[key] / pretraining_tp

        # attention change
        attention_wname = layer_name_mapping('attention.query_key_value.weight', file)
        if attention_wname in tensors:
            query_key_value = tensors.pop(attention_wname).view((num_heads, 3, head_dim, hidden_size))

            tensors[layer_name_mapping('self_attn.q_proj.weight', file)] = query_key_value[:,0,:,:].reshape((num_heads * head_dim, hidden_size))
            tensors[layer_name_mapping('self_attn.k_proj.weight', file)] = query_key_value[:,1,:,:].reshape((num_heads * head_dim, hidden_size))
            tensors[layer_name_mapping('self_attn.v_proj.weight', file)] = query_key_value[:,2,:,:].reshape((num_heads * head_dim, hidden_size))

            tensors[layer_name_mapping('self_attn.o_proj.weight', file)] = tensors.pop(layer_name_mapping('attention.dense.weight', file))
            
            tensors.pop(layer_name_mapping('attention.rotary_emb.inv_freq', file))
            tensors[layer_name_mapping('self_attn.rotary_emb.inv_freq', file)] = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


        torch.save(
            tensors,
            os.path.join(
                pytorch_dump_folder_path,
                "pytorch_model_{}-of-{}.bin".format(str(j + 1).zfill(5), str(len(file_names)).zfill(5)),
            ),
        )

        for key in tensors.keys():
            value = tensors[key]
            total_size += value.numel() * get_dtype_size(value.dtype)
            if key not in index_dict["weight_map"]:
                index_dict["weight_map"][key] = "pytorch_model_{}-of-{}.bin".format(
                    str(j + 1).zfill(5), str(len(file_names)).zfill(5)
                )

    # config = LlamaConfig()
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    index_dict["metadata"]["total_size"] = total_size
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())
    with open(os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME + ".index.json"), "w", encoding="utf-8") as f:
        json_config = json.dumps(index_dict, indent=2, sort_keys=True) + "\n"
        f.write(json_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--llama_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the Megatron-LM checkpoint path.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--llama_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pretraining_tp",
        default=4,
        type=int,
        help="Pretraining TP rank that has been used when training the model in Megatron-LM \n",
    )
    args = parser.parse_args()
    convert_llama_checkpoint_to_pytorch(
        args.llama_checkpoint_path,
        args.llama_config_file,
        args.pytorch_dump_folder_path,
        args.pretraining_tp,
    )