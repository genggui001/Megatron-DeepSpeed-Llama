#!/bin/bash

python -u tools/preprocess_data.py \
  --input /mnt/data/smart_health_02/xuekui/dataset/test/pretrain_data/zhs/lm_zh-cn_wikipedia.jsonl \
  --json-keys text \
  --tokenizer-type PretrainedFromHF \
  --append-eod \
  --pad-vocab-size-to 50176 \
  --vocab-file /mnt/data/smart_health_02/xuekui/code/Megatron-DeepSpeed-Llama/hchinese_llama_tokenizer \
  --output-prefix /mnt/data/smart_health_02/xuekui/dataset/pretrain_data/zhs/lm_zh-cn_wikipedia \
  --workers 16 \
  --dataset-impl mmap


