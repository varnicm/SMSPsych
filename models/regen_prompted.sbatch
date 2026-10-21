#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-warp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=regen_prompted
#SBATCH --output=regen_prompted_%j.log
#SBATCH --exclude=gpunode003
source ~/.bashrc
conda activate slm
unset PYTHONPATH
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd .
for m in "google/gemma-4-E4B-it" "Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-9B" "meta-llama/Llama-3.1-8B-Instruct" "microsoft/Phi-3-mini-4k-instruct"; do
  echo "=== $m ==="
  sed -i 's|^MODEL_ID = .*|MODEL_ID = "'"$m"'"|' run_inference_poster.py
  python run_inference_poster.py
done
