#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job prts
#SBATCH --partition=Your partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=28
#SBATCH --time=7-00:00:00
#SBATCH --output=prts.out
#SBATCH --error=prts.err
#SBATCH --exclusive

cd /workspace/model/prts

TIME=$(date +%Y-%m-%d-%H-%M-%S)
SCRIPT_DIR=/workspace/model/prts
MODEL_NAME=24L2048H
METHOD=stacking
CONFIG="./prts_configs/stacking_6L_24L.json"
export WANDB_API_KEY=5117b0efc6fd7faf972b98be176874ead86ccd43

# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -i 6000-6100 -n 1)
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
    # --val_data_dir=/mnt/nas_v2/common/public/dataset/SlimPajama-627B/slimpajama-validation \

torchrun ${DISTRIBUTED_ARGS} pretrain/run_pretrain.py \
    --num_nodes=${NNODES} \
    --model_name=${MODEL_NAME} \
    --name=${MODEL_NAME}_stacking \
    --method=${METHOD} \
    --config_path=${CONFIG} \
    --out_dir=${SCRIPT_DIR}/${METHOD}/${TIME} \
    --train_data_dir=/mnt/nas_v2/common/public/dataset/SlimPajama-627B/slimpajama \
    --devices=${GPUS_PER_NODE} \
    --global_batch_size=2048 \
    --learning_rate=3e-4 \
    --min_lr=3e-5 \
    --micro_batch_size=8 \
    --max_step=300000 \
    --warmup_steps=3000 \
    --log_step_interval=1 \
    --eval_iters=10000 \
    --save_step_interval=5000 \
    --eval_step_interval=5000 \
    --weight_decay=1e-1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=True \
    --resume_id=64000
