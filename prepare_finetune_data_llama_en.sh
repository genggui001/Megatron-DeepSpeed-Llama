#!/bin/bash

python -u tools/preprocess_data.py \
    --input /mnt/data/smart_health_02/xuekui/dataset/test/RedPajama-Data-1T/common_crawl/2019-30/en_head_0000.json.gz.dedup.classifier.jsonl \
    --json-keys text \
    --output-prefix /mnt/data/smart_health_02/xuekui/dataset/test/pretrain_data/en/common_crawl_2019_30_tmp \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --vocab-file /mnt/data/smart_health_02/xuekui/code/Megatron-DeepSpeed-Llama/llama_tokenizer \
    --append-eod \
    --pad-vocab-size-to 32128 \
    --workers 32


