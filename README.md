## Megatron-DeepSpeed-Llama
Llama in Megatron-DeepSpeed

主要参考:

[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

[LydiaXiaohongLi-Megatron-DeepSpeed](https://github.com/LydiaXiaohongLi/Megatron-DeepSpeed)

支持 xformer 加速 FusedRMSNorm 加速

注意：代码开发中ing，并不是最终版，随时可能大改配置


### 1. 环境配置
```
conda env create -f torch1.13.yml

# 编译安装deepspeed定制版
git clone -b v0.8.0_genggui001 https://github.com/genggui001/DeepSpeed.git
DS_BUILD_OPS=1 pip install . --global-option="build_ext" --global-option="-j1" --no-cache -v --disable-pip-version-check

# 编译安装apex
pip install https://github.com/NVIDIA/apex/archive/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28.zip --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
```

### 2. 已经转化好格式的预训练权重

#### decapoda-research/llama
原始权重：[decapoda-research](https://huggingface.co/decapoda-research)

[decapoda-research-llama-7b-megatron-states](https://huggingface.co/genggui001/decapoda-research-llama-7b-megatron-states)

loss on commoncrawl (和论文基本一致)
```
validation loss at the end of training for val data | lm loss value: 1.795973E+00 | lm loss PPL: 6.025333E+00 |
```

[decapoda-research-llama-13b-megatron-states](https://huggingface.co/genggui001/decapoda-research-llama-13b-megatron-states)

loss on commoncrawl (和论文基本一致)
```
validation loss at the end of training for val data | lm loss value: 1.709547E+00 | lm loss PPL: 5.526456E+00 |
```

[decapoda-research-llama-30b-megatron-states](https://huggingface.co/genggui001/decapoda-research-llama-30b-megatron-states)

loss on commoncrawl (和论文基本一致)
```
validation loss at the end of training for val data | lm loss value: 1.579006E+00 | lm loss PPL: 4.850134E+00 |
```

[decapoda-research-llama-65b-megatron-states](https://huggingface.co/genggui001/decapoda-research-llama-65b-megatron-states)

loss on commoncrawl (和论文基本一致)
```
validation loss at the end of training for val data | lm loss value: 1.519600E+00 | lm loss PPL: 4.570395E+00 |
```

#### Chinese-LLaMA-Plus
原始权重：[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

[chinese-llama-plus-7b-megatron-states](https://huggingface.co/genggui001/chinese-llama-plus-7b-megatron-states)

loss on zh-cn_wikipedia_text (略高原因不明)
```
validation loss at the end of training for val data | lm loss value: 3.349298E+00 | lm loss PPL: 2.848272E+01 |
```

loss on commoncrawl (同样略高 英文遗忘的不少)
```
validation loss at the end of training for val data | lm loss value: 2.483371E+00 | lm loss PPL: 1.198159E+01 | 
```

### 3. 数据预处理
```
python -u tools/preprocess_data.py \
    --input /mnt/data/smart_health_02/xuekui/dataset/test/RedPajama-Data-1T/common_crawl/2019-30/en_head_0000.json.gz.dedup.classifier.jsonl \
    --json-keys text \
    --output-prefix /mnt/data/smart_health_02/xuekui/dataset/test/pretrain_data/en/common_crawl_2019_30_tmp \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --vocab-file /mnt/data/smart_health_02/xuekui/pretrain_weights/nlp/decapoda-research-llama-7b-megatron-states/tokenizer \
    --append-eod \
    --pad-vocab-size-to 32128 \
    --workers 32
```


### 4. Finetune


修改finetune_llama_en.sh中的

LOAD_CHECKPOINT_PATH

DATA_PATH

执行

```
bash finetune_llama_en.sh
```




