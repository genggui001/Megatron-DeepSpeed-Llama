#!/bin/bash

export NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=
export NV_LIBNCCL_DEV_PACKAGE_VERSION=libnccl-dev=
export NV_LIBNCCL_DEV_PACKAGE_NAME=
export NV_LIBNCCL_PACKAGE=
export NV_LIBNCCL_PACKAGE_NAME=
export NV_LIBNCCL_PACKAGE_VERSION=


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SLURM_NNODES=1
SLURM_PROCID=0
NNODES=1
GPUS_PER_NODE=4
SLURM_JOB_ID=23333
MASTER_ADDR=localhost
MASTER_PORT=46282

# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR: " $MASTER_ADDR
# export MASTER_PORT=$(expr 45000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_PORT: " $MASTER_PORT

# GPUS_PER_NODE=8
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
# export PATH="/mnt/petrelfs/share/gcc/gcc-7.5.0/bin:/mnt/petrelfs/share/cuda-11.3/bin:$PATH"

START_DATE=$(date "+%Y_%m_%d_%H:%M:%S")

echo "START TIME: $START_DATE"

variant=main

LOAD_CHECKPOINT_PATH=/mnt/petrelfs/xuekui/code/BLOOM-Megatron-DeepSpeed-175b-all-finetune/pretrain_weights/bloomz-mt

DATA_OUTPUT_PATH=/mnt/petrelfs/xuekui/code/BLOOM-Megatron-DeepSpeed-175b-all-finetune/model_dir/bloomz-mt
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/experiment
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

KILL_SWITCH_PATH=$REPO_PATH/kill-switch

DATA_PATH="/mnt/petrelfs/xuekui/dataset/nlp/chatgpt_sft_copy/all/"
TOKENIZER_NAME_OR_PATH="./bigscience_bloom_tokenizer"


TP_SIZE=2
PP_SIZE=24

MICRO_BATCH_SIZE=1  # was MBS=1 till GBS=784
GLOBAL_BATCH_SIZE=256  # 4.2M tokens. It is larger than the initial plan of 3.2M tokens to get higher throughput

NHIDDEN=14336
NLAYERS=70
NHEADS=112
SEQ_LEN=2048

ADAPTER_SIZE=0
SAVE_INTERVAL=256

TRAIN_SAMPLES=18_874_368  # 450B tokens
LR_DECAY_SAMPLES=18_874_368  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=0  # 375M tokens


OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6.03e-6 \
    --min-lr 6.03e-7 \
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
    --pp-partition-method 'type:transformer|embedding' \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --adapter-size $ADAPTER_SIZE \
    --not-optimize-emb \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --init-method-std 0.0048 \
    --embed-layernorm \
    --bf16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --kill-switch-path $KILL_SWITCH_PATH \
    --pad-vocab-size-to 250880 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

# TODO: decide on efficient eval-interval + eval-iters

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-total-limit 2 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $SAVE_INTERVAL \
    --eval-iters 32 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent

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
  "bf16": {
    "enabled": true
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
    finetune_dig_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load /mnt/petrelfs/xuekui/code/BLOOM-Megatron-DeepSpeed-175b-all-finetune/model_dir/bloomz-mt/checkpoints/main_no_opt/ \
    --data-path $DATA_PATH \
    --skip-train-iteration-range 30208-37760 \
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

echo "$LAUNCHER --node_rank $SLURM_PROCID $CMD" > $LOGS_PATH/main_${SLURM_PROCID}_log.txt

echo "-----------------------------------------------" > $LOGS_PATH/main_${SLURM_PROCID}_log.txt

bash -c "$LAUNCHER --node_rank $SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_${SLURM_PROCID}_log.txt


