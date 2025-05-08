#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=ocr_benchmark
#SBATCH --output=logs/ocr_benchmark_%j.out

# load UV Python & your venv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12
uv venv --python 3.12

uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio

# run experiment
cd src
python experiment.py \
  --architecture EmnistCNN_32_128_256 \
  --device cuda \
  --cpu-workers 8 \
  --subsample-size 50000 \
  --k-folds 5 \
  --epochs 50 \
  --batch-size 256 \
  --lr 0.0001 \
  --weight-decay 0.0001
