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

[decapoda-research/llama-7b](https://huggingface.co/genggui001/decapoda-research-llama-7b-megatron-states)

loss on commoncrawl (和论文基本一致)
```
iteration        1/   73728 | consumed samples:          128 | consumed tokens:       262144 | elapsed time per iteration (ms): 65067.1 | learning rate: 6.000E-05 | global batch size:   128 | lm loss: 1.831802E+00 | loss scale: 4096.0 | grad norm: 0.588 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 1.967 | TFLOPs: 28.52 |  
```

#### Chinese-LLaMA-Plus
原始权重：[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

[Chinese-LLaMA-Plus-7b](https://huggingface.co/genggui001/chinese-llama-plus-7b-megatron-states)

loss on zh-cn_wikipedia_text (略高原因不明)
```
iteration        1/   73728 | consumed samples:          128 | consumed tokens:       262144 | elapsed time per iteration (ms): 65787.8 | learning rate: 6.000E-05 | global batch size:   128 | lm loss: 3.341448E+00 | loss scale: 4096.0 | grad norm: 5.496 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 1.946 | TFLOPs: 28.42 |
```

loss on commoncrawl (同样略高 英文遗忘的不少)
```
iteration        1/   73728 | consumed samples:          128 | consumed tokens:       262144 | elapsed time per iteration (ms): 66938.9 | learning rate: 6.000E-05 | global batch size:   128 | lm loss: 2.483702E+00 | loss scale: 4096.0 | grad norm: 3.814 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 1.912 | TFLOPs: 27.94 |
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




