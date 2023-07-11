#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#SLURM_NNODES=1
#SLURM_PROCID=0
#NNODES=1
#GPUS_PER_NODE=4
#SLURM_JOB_ID=23333
#MASTER_ADDR=localhost
#MASTER_PORT=46282

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR: " $MASTER_ADDR
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 2))
export MASTER_PORT=10059

echo "MASTER_PORT: " $MASTER_PORT

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

echo "NNODES: " $NNODES

#export RANK=$SLURM_PROCID
#echo "RANK: " $RANK
#export LOCAL_RANK=$SLURM_LOCALID
#echo "LOCAL_RANK: " $LOCAL_RANK
#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
#echo "WORLD_SIZE: " $WORLD_SIZE

# # do not remove or the training will hang and nodes will be lost w/o this workaround
# export CUDA_LAUNCH_BLOCKING=1

# # hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error-genggui001.json

# # force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1



# defining the right environment variables
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="/mnt/petrelfs/share/gcc/gcc-7.5.0/bin:$PATH"

START_DATE=$(date "+%Y_%m_%d_%H:%M:%S")

echo "START TIME: $START_DATE"

variant=main

LOAD_CHECKPOINT_PATH=/mnt/petrelfs/xuekui/pretrain_weights/nlp/new_100b-megatron-states

DATA_OUTPUT_PATH=/mnt/petrelfs/xuekui/code/Megatron-DeepSpeed-Llama/model_dir/new_100b
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/experiment
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

KILL_SWITCH_PATH=$REPO_PATH/kill-switch

DATA_PATH="/mnt/petrelfs/xuekui/dataset/nlp/chatgpt_sft_copy_new/all/"
TOKENIZER_NAME_OR_PATH=$LOAD_CHECKPOINT_PATH/tokenizer


TP_SIZE=4
PP_SIZE=10

MICRO_BATCH_SIZE=1  # was MBS=1 till GBS=784
GLOBAL_BATCH_SIZE=32  # 4.2M tokens. It is larger than the initial plan of 3.2M tokens to get higher throughput

NHIDDEN=10240
FFN_HIDDEN_SIZE=27392
NLAYERS=82
NHEADS=80
SEQ_LEN=8192

SP=12

ADAPTER_SIZE=0
SAVE_INTERVAL=512

# TRAIN_SAMPLES=4_718_592  # 73728 * 64
# LR_DECAY_SAMPLES=4_718_592  # 73728 * 64
TRAIN_SAMPLES=2_359_296  # 73728 * 32
LR_DECAY_SAMPLES=2_359_296  # 73728 * 32
LR_WARMUP_SAMPLES=0  # 


OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 9e-6 \
    --min-lr 9e-7 \
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
    --pad-vocab-size-to 65792 \
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
    train_llama_use_dataset.py \
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

echo "$LAUNCHER --node_rank $SLURM_PROCID $CMD" > $LOGS_PATH/main_${SLURM_PROCID}_log.txt

echo "-----------------------------------------------" > $LOGS_PATH/main_${SLURM_PROCID}_log.txt

bash -c "$LAUNCHER --node_rank $SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_${SLURM_PROCID}_log.txt


