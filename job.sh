#!/bin/bash
#SBATCH --job-name=xlstm-wm
#SBATCH --output=slurm_logs/%j.out
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1

RUN_DIR="output/run_$SLURM_JOB_ID"
mkdir -p $RUN_DIR

python train.py --run_dir $RUN_DIR $@
