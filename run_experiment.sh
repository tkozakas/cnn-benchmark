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
source $HOME/.local/bin/env

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio

# run experiment
cd src
python experiment.py \
  --architecture EmnistCNN_32_128_256 \
  --device cuda \
  --cpu-workers 6 \
  --subsample-size 10000 \
  --k-folds 3 \
  --epochs 20 \
  --batch-size 256 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --patience 20