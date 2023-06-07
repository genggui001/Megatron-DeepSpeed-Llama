#!/bin/bash

export NV_LIBNCCL_DEV_PACKAGE=
export NV_LIBNCCL_DEV_PACKAGE_VERSION=
export NV_LIBNCCL_DEV_PACKAGE_NAME=
export NV_LIBNCCL_PACKAGE=
export NV_LIBNCCL_PACKAGE_NAME=
export NV_LIBNCCL_PACKAGE_VERSION=

# 单节点 bug
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SLURM_NNODES=1
SLURM_PROCID=0
NNODES=1
GPUS_PER_NODE=8
SLURM_JOB_ID=23333
MASTER_ADDR=localhost
MASTER_PORT=46282

# 多节点 srun
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR: " $MASTER_ADDR
# export MASTER_PORT=$(expr 45000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_PORT: " $MASTER_PORT

# GPUS_PER_NODE=4
# NNODES=$SLURM_NNODES

# echo "NNODES: " $NNODES

#export RANK=$SLURM_PROCID
#echo "RANK: " $RANK
#export LOCAL_RANK=$SLURM_LOCALID
#echo "LOCAL_RANK: " $LOCAL_RANK
#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
#echo "WORLD_SIZE: " $WORLD_SIZE


# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error-genggui001.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

# defining the right environment variables
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="/root/miniconda3/envs/torch1.13/bin:$PATH"


START_DATE=$(date "+%Y_%m_%d_%H:%M:%S")

echo "START TIME: $START_DATE"

variant=main

LOAD_CHECKPOINT_PATH=/mnt/data/smart_health_02/xuekui/pretrain_weights/nlp/decapoda-research-llama-7b-megatron-states

DATA_OUTPUT_PATH=./model_dir/llama_7b_en
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/experiment
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

KILL_SWITCH_PATH=$REPO_PATH/kill-switch

DATA_PATH="/mnt/data/smart_health_02/xuekui/dataset/test/pretrain_data/en/common_crawl_2019_30_tmp_text_document"
TOKENIZER_NAME_OR_PATH=$LOAD_CHECKPOINT_PATH/tokenizer


TP_SIZE=2
PP_SIZE=4

MICRO_BATCH_SIZE=1  # was MBS=1 till GBS=784
GLOBAL_BATCH_SIZE=128  # 4.2M tokens. It is larger than the initial plan of 3.2M tokens to get higher throughput

NHIDDEN=4096
FFN_HIDDEN_SIZE=11008
NLAYERS=32
NHEADS=32
SEQ_LEN=2048

SP=12

ADAPTER_SIZE=0
SAVE_INTERVAL=256

TRAIN_SAMPLES=9_437_184  # 450B tokens
LR_DECAY_SAMPLES=8_437_120  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=64  # 375M tokens

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "
# for 20h 1190, for 100h 5990
#    --exit-duration-in-mins 1190
EXIT_OPTS=" \
    --exit-duration-in-mins 99999999 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --vocab-file $TOKENIZER_NAME_OR_PATH \
    --loss-scale $SP \
    --init-method-std 0.0048 \
    --fp16 \
    --seed 42 \
    --checkpoint-activations \
    --pad-vocab-size-to 32128 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

# TODO: decide on efficient eval-interval + eval-iters

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $SAVE_INTERVAL \
    --eval-iters 32 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1

config_json="./ds_config.$SLURM_JOB_ID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "checkpoint": {
    "load_universal": false
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": $SP
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    pretrain_llama.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $LOAD_CHECKPOINT_PATH \
    --finetune \
    --data-path $DATA_PATH \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

#     --skip-train-iteration-range 22528-45056 \
# export CMD=" \
#     finetune_dig_gpt.py \
#     --tensor-model-parallel-size $TP_SIZE \
#     --pipeline-model-parallel-size $PP_SIZE \
#     $GPT_ARGS \
#     $OUTPUT_ARGS \
#     --save $CHECKPOINT_PATH \
#     --load $CHECKPOINT_PATH \
#     --pretrained-checkpoint $LOAD_CHECKPOINT_PATH \
#     --data-path $DATA_PATH \
#     --distributed-backend nccl \
#      $DEEPSPEED_ARGS \
#     "
# export CMD=" \
#     finetune_dig_gpt.py \
#     --tensor-model-parallel-size $TP_SIZE \
#     --pipeline-model-parallel-size $PP_SIZE \
#     $GPT_ARGS \
#     $OUTPUT_ARGS \
#     --save $CHECKPOINT_PATH \
#     --load $CHECKPOINT_PATH \
#     --data-path $DATA_PATH \
#     --distributed-backend nccl \
#      $DEEPSPEED_ARGS \
#     "

# echo "$LAUNCHER --node_rank $SLURM_PROCID $CMD" 

echo "$LAUNCHER --node_rank $SLURM_PROCID $CMD" > $LOGS_PATH/main_${SLURM_PROCID}.log

echo "-----------------------------------------------" > $LOGS_PATH/main_${SLURM_PROCID}.log

bash -c "$LAUNCHER --node_rank $SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_${SLURM_PROCID}.log


