#!/bin/bash
#SBATCH --job-name=xlstm-wm
#SBATCH --output=output_%j.out
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "--------------------------"
echo "--- GPU Information ---"
GPU_ID=${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}
GPU_NAME=$(nvidia-smi --id=$GPU_ID --query-gpu=gpu_name --format=csv,noheader)
MEM_TOTAL=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.total --format=csv,noheader)
MEM_USED=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader)
echo "GPU Name: $GPU_NAME"
echo "Total Memory: $MEM_TOTAL"
echo "Memory Used: $MEM_USED"
echo "--------------------------"

echo "Job started on $(date)"

RUN_DIR="output/run_$SLURM_JOB_ID"
mkdir -p $RUN_DIR

python train.py --run_dir $RUN_DIR $@

echo "Job finished on $(date)"
