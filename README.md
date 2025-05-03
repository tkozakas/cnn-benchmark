# OCR Model Benchmark

## Project Goals

This project aims to benchmark various OCR models to evaluate their performance in terms of
accuracy, edit distance, and runtime.

## Datasets
EMNIST

## The experiments
1. Optimizer accuracy comparison (Adam is better -> use Adam)
2. Learning rate scheduler comparison (CosineAnnealingLR is better -> use CosineAnnealingLR)
3. Regularization comparison (L2 is better -> use L2)
4. Batch Size & Learning‐Rate Grid (Find sweet spot for batch size and learning rate)
5. Architecture comparison (Emnist with 2 layers or 3 or RNN or RCNN (existing))


## [uv](https://github.com/astral-sh/uv) based python wrapper environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

# Install dependencies
```bash
    uv pip install -r requirements.txt
```

# Setup
## Seting up ROCm on Fedora 41 (AMD GPU)
! Note: If you have an NVIDIA GPU, you should use CUDA instead of ROCm.
```bash
sudo usermod -a -G video $LOGNAME
sudo dnf install rocminfo rocm-opencl rocm-clinfo
echo “export HSA_OVERRIDE_GFX_VERSION=10.3.0” >> ~/.bashrc
source ~/.bashrc # or zshrc

# Install PyTorch with ROCm support (in a virtual environment)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

# Usage
Note: The following commands are examples and may need to be adjusted based on your specific setup and requirements.

## Train and evaluate model

Usage:
```bash
python train.py \
  --architecture EmnistCNN_32_128_256 \
  --emnist-type balanced \
  --device cuda \
  --cpu-workers 6 \
  --k-folds 5 \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 5
```

## Experiment on finding the best hyperparameters
```bash
python experiment.py \
  --architecture EmnistCNN_32_128_256 \
  --device cuda \
  --cpu-workers 6 \
  --subsample-size 10000 \
  --k-folds 3 \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.0005 \
  --weight-decay 0.0001 \
  --patience 5
```

## Train on VU supercomputer
1. Connect to the supercomputer
```bash
ssh mifvu_username@uosis.mif.vu.lt
ssh hpc

srun \
  --partition=gpu \
  --gres=gpu:1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=16G \
  --time=02:00:00 \
  --pty bash
```