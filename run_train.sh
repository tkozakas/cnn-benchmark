#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=ocr_benchmark
#SBATCH --output=logs/ocr_benchmark_%j.out

# load UV Python & your venv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio

# run training
cd src
python train.py \
  --architecture EmnistCNN_32_128_256 \
  --emnist-type balanced \
  --device cuda \
  --cpu-workers 8 \
  --k-folds 5 \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 5