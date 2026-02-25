#!/bin/bash
#SBATCH --job-name=xlstm-wm
#SBATCH --output=output_%j.out
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

echo "--------------------------"
echo "--- GPU Information ---"
GPU_NAME=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=gpu_name --format=csv,noheader)
MEM_TOTAL=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total --format=csv,noheader)
MEM_USED=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader)
echo "GPU Name: $GPU_NAME"
echo "Total Memory: $MEM_TOTAL"
echo "Memory Used (before script): $MEM_USED"
echo "--------------------------"

echo "Job started on $(date)"

echo "--------------------------"
echo "--- GPU Information ---"
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)
MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
echo "GPU Name: $GPU_NAME"
echo "Total Memory: $MEM_TOTAL"
echo "Memory Used (before script): $MEM_USED"
echo "--------------------------"

RUN_DIR="output/run_$SLURM_JOB_ID"
mkdir -p $RUN_DIR

python train.py --run_dir $RUN_DIR $@

echo "Job finished on $(date)"
