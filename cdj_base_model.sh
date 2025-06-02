#!/bin/bash
export WANDB_API_KEY=5117b0efc6fd7faf972b98be176874ead86ccd43
source /workspace/model/prts/venv_prts/bin/activate
which python

#DEBUG=true
#DLC=true

TIME=$(date +%Y%m%d%H%M%S)
SCRIPT_DIR=/workspace/model/prts

#METHOD=scratch
#MODEL_NAME=6L2048H_llama3
#MODEL_NAME=tiny_LLaMA_400M_like1.1B
#MODEL_NAME=tiny_LLaMA_400M_like1.1B_width
#MODEL_NAME=tiny_LLaMA_1.1B


METHOD=stacking
MODEL_NAME=tiny_LLaMA_410M_1.1B_10B_interp3 # only for log
CONFIG="${SCRIPT_DIR}/prts_configs/stacking_tinyllama1.1b_interpolation3.json"


METHOD=b2b
# # MODEL_NAME=6L2048H_6L4096H # only for log
# # CONFIG="${SCRIPT_DIR}/prts_configs/b2b_6L1024H.json"
MODEL_NAME=tiny_LLaMA_400M_1.1B_width
CONFIG="${SCRIPT_DIR}/prts_configs/b2b_tinyllama1.1b_width.json"
# MODEL_NAME=tiny_LLaMA_400M_1.1B_depth
# CONFIG="${SCRIPT_DIR}/prts_configs/b2b_tinyllama1.1b_depth.json"

if [ "${METHOD}" == "scratch" ]; then
    resume_option=""
else
    resume_option="--resume_id=80000"
fi

OUTPUT_DIR="${SCRIPT_DIR}/${METHOD}/${MODEL_NAME}"
DATASET_BASE=/mnt/nas_v2/common/public/dataset


if [ -z ${DLC} ]; then
  MASTER_ADDR=localhost
  MASTER_PORT=$(shuf -i 6000-6100 -n 1)

  GPUS_PER_NODE=8
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   GPUS_PER_NODE=4
#   export CUDA_VISIBLE_DEVICES=4,5,6,7
  # GPUS_PER_NODE=2  
  # export CUDA_VISIBLE_DEVICES=6,7

  NNODES=1
  RANK=0  
else
  GPUS_PER_NODE=${TQ_GPU_NUM}
  NNODES=$WORLD_SIZE
fi

if [[ -z ${DEBUG} ]]; then
    debug_option="--debug=False"
else
    debug_option="--debug=True"
    export CUDA_VISIBLE_DEVICES=2,3
    GPUS_PER_NODE=2
    OUTPUT_DIR="${SCRIPT_DIR}/${METHOD}/${MODEL_NAME}_debug"
fi

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

if [[ "${MODEL_NAME,,}" == *llama3* ]]; then
    dataset_option="
    --train_data_dir=${DATASET_BASE}/SlimPajama-627B/llama3-train \
    --val_data_dir=${DATASET_BASE}/SlimPajama-627B/llama3-validation 
    "
elif [[ "${MODEL_NAME,,}" == *pythia* ]]; then
    dataset_option="
    --train_data_dir=${DATASET_BASE}/SlimPajama-627B/pythia-train \
    --val_data_dir=${DATASET_BASE}/SlimPajama-627B/pythia-validation 
    "
else
    dataset_option="
    --train_data_dir=${DATASET_BASE}/SlimPajama-627B/llama2-train \
    --val_data_dir=${DATASET_BASE}/SlimPajama-627B/llama2-validation 
    "
fi

if [[ -z ${CONFIG} ]]; then
    config_option=""
else
    config_option="--config_path=${CONFIG}"
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
    #step from 0
    # --checkpoint_path=/workspace/model/prts/scratch/6L2048H_llama2/iter-064000-ckpt.pth \
    #step from resume_id
    # --resume_ckpt=/workspace/model/prts/scratch/6L2048H_llama2/iter-064000-ckpt.pth \
    # --resume_id=64000 \
torchrun ${DISTRIBUTED_ARGS} pretrain/run_pretrain.py \
    --num_nodes=${NNODES} \
    --model_name=${MODEL_NAME} \
    --name=${METHOD}+${MODEL_NAME} \
    --method=${METHOD} \
    ${config_option} \
    --out_dir=${OUTPUT_DIR}\
    ${dataset_option} \
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
    --eval_step_interval=50000 \
    --weight_decay=1e-1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=True \
    ${resume_option} \
    ${debug_option} 2>&1 |tee ${OUTPUT_DIR}/training-log-${RANK}-${TIME}.txt