#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-warp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=bert_all_2110
#SBATCH --output=bert_all_2110_%j.log
#SBATCH --exclude=gpunode003
source ~/.bashrc
conda activate slm
unset PYTHONPATH
export HF_HOME=./hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd .
python train_bert_kfold_all.py
