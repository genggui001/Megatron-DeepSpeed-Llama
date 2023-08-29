import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from transformers import LlamaForCausalLM
import torch

model = LlamaForCausalLM.from_pretrained(
    "/mnt/petrelfs/test/pretrain_weights/nlp/15760_hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = model.eval()

# %%
all_named_parameters = dict(model.named_parameters())

# %%
list(all_named_parameters.keys())

# %%
from collections import OrderedDict

base_path = "/mnt/petrelfs/test/pretrain_weights/nlp/15760_100b-megatron-states/"
save_path = base_path + "global_step0/"
# load_path = "/mnt/data/smart_health_02/test/pretrain_weights/nlp/chinese-llama-plus-7b-megatron-states/global_step0/"

pad_vocab_size = 65792
vocab_size = model.config.vocab_size

# mid
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // model.config.num_attention_heads
hidden_size = model.config.hidden_size
layer_count = model.config.num_hidden_layers

# %%
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/test/pretrain_weights/nlp/15760_hf",
    padding_side='right',
    pad_token='<pad>',
)

# %%
tokenizer.save_pretrained(base_path + "tokenizer_v9/")

# %%
model.config

# %%
# emb

with torch.no_grad():
    temp = OrderedDict()
    temp['word_embeddings.weight'] = torch.normal(0, 0.02, size=(pad_vocab_size, model.config.hidden_size), dtype=torch.bfloat16)
    temp['word_embeddings.weight'][:vocab_size, :] = all_named_parameters['model.embed_tokens.weight'].data
    torch.save(temp, save_path + "layer_01-model_00-model_states.pt")


# %%
from tqdm.auto import tqdm


with torch.no_grad():
    # layer_idx = 0
    for layer_idx in tqdm(range(3, 3+layer_count)):
        temp = OrderedDict()

        temp['input_layernorm.weight'] = all_named_parameters[f'model.layers.{layer_idx-3}.input_layernorm.weight'].data
        temp['post_attention_layernorm.weight'] = all_named_parameters[f'model.layers.{layer_idx-3}.post_attention_layernorm.weight'].data

        temp['mlp.gate_proj.weight'] = all_named_parameters[f'model.layers.{layer_idx-3}.mlp.gate_proj.weight'].data
        temp['mlp.up_proj.weight'] = all_named_parameters[f'model.layers.{layer_idx-3}.mlp.up_proj.weight'].data
        temp['mlp.down_proj.weight'] = all_named_parameters[f'model.layers.{layer_idx-3}.mlp.down_proj.weight'].data

        temp['attention.rotary_emb.inv_freq'] = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # 'attention.rotary_emb.inv_freq' 不变

        temp['attention.query_key_value.weight'] = torch.cat((
            all_named_parameters[f'model.layers.{layer_idx-3}.self_attn.q_proj.weight'].data.view((num_heads, 1, head_dim, hidden_size)),
            all_named_parameters[f'model.layers.{layer_idx-3}.self_attn.k_proj.weight'].data.view((num_heads, 1, head_dim, hidden_size)),
            all_named_parameters[f'model.layers.{layer_idx-3}.self_attn.v_proj.weight'].data.view((num_heads, 1, head_dim, hidden_size)),
        ), dim=1).view((num_heads * 3 * head_dim, hidden_size))
        temp['attention.dense.weight'] = all_named_parameters[f'model.layers.{layer_idx-3}.self_attn.o_proj.weight'].data

        torch.save(temp, save_path + "layer_%02d-model_00-model_states.pt" % layer_idx)

# %%
# model.norm.weight

with torch.no_grad():
    temp = OrderedDict()
    temp['weight'] = all_named_parameters['model.norm.weight'].data
    torch.save(temp, save_path + f"layer_{4+layer_count}-model_00-model_states.pt")

# %%
# lm_head.weight

with torch.no_grad():
    temp = OrderedDict()
    temp['lm_head.weight'] = torch.normal(0, 0.02, size=(pad_vocab_size, model.config.hidden_size), dtype=torch.bfloat16)
    temp['lm_head.weight'][:vocab_size, :] = all_named_parameters['lm_head.weight'].data
    torch.save(temp, save_path + f"layer_{5+layer_count}-model_00-model_states.pt")


# %%
# temp['lm_head.weight'].shape

# %%



