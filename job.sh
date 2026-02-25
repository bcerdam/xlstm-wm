#!/bin/bash
#SBATCH --job-name=xlstm-wm
#SBATCH --output=slurm_logs/%j.out
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1

RUN_DIR="output/run_$SLURM_JOB_ID"
mkdir -p $RUN_DIR

python train.py --run_dir $RUN_DIR --train_wm.wm_batch_size 32 --train_wm.world_model_learning_rate 0.0005
