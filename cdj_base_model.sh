#!/bin/bash
export WANDB_API_KEY=5117b0efc6fd7faf972b98be176874ead86ccd43

# DEBUG=true
dsw=true  

TIME=$(date +%Y%m%d%H%M%S)
SCRIPT_DIR=/workspace/model/prts
MODEL_NAME=24L2048H
METHOD=scratch
OUTPUT_DIR="${SCRIPT_DIR}/${METHOD}/${MODEL_NAME}_llama2"
DATASET_BASE=/mnt/nas_v2/common/public/dataset

if [[ -z ${DEBUG} ]]; then
    debug_option="--debug=False"
    GPUS_PER_NODE=8    
else
    debug_option="--debug=True"
    export CUDA_VISIBLE_DEVICES=2,3
    GPUS_PER_NODE=2
fi

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi
if [ "$dsw" = true ]; then
  GPUS_PER_NODE=${TQ_GPU_NUM}
  NNODES=$WORLD_SIZE
else
  MASTER_ADDR=localhost
  MASTER_PORT=$(shuf -i 6000-6100 -n 1)

  export CUDA_VISIBLE_DEVICES=4,5,6,7
  # export CUDA_VISIBLE_DEVICES=6,7
  GPUS_PER_NODE=4
  NNODES=1
  RANK=0
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun ${DISTRIBUTED_ARGS} pretrain/run_pretrain.py \
    --num_nodes=${NNODES} \
    --model_name=${MODEL_NAME} \
    --name=${MODEL_NAME}_llama2 \
    --method=${METHOD} \
    --out_dir=${OUTPUT_DIR}\
    --train_data_dir=${DATASET_BASE}/SlimPajama-627B/llama2-train \
    --val_data_dir=${DATASET_BASE}/SlimPajama-627B/llama2-validation \
    --devices=${GPUS_PER_NODE} \
    --global_batch_size=2048 \
    --learning_rate=3e-4 \
    --min_lr=3e-5 \
    --micro_batch_size=8 \
    --max_step=300000 \
    --warmup_steps=3000 \
    --log_step_interval=1 \
    --eval_iters=1000 \
    --save_step_interval=500 \
    --eval_step_interval=5000 \
    --weight_decay=1e-1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=True \
    ${debug_option} 2>&1 |tee ${OUTPUT_DIR}/training-log-${RANK}-${TIME}.txt