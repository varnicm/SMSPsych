#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-warp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=lora_llama
#SBATCH --output=lora_llama_%j.log
#SBATCH --exclude=gpunode003
source ~/.bashrc
conda activate slm
unset PYTHONPATH
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd .
python train_lora_llama.py
