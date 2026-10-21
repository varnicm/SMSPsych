#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-warp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=lora_qwen9b
#SBATCH --output=lora_qwen9b_%j.log
source ~/.bashrc
conda activate slm
unset PYTHONPATH
export HF_HOME=./hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd .
python train_lora_qwen9b.py
