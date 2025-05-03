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
    
    uv python install 3.9
    uv venv --python 3.9
    source .venv/bin/activate
    uv pip install -r requirements.txt
```

# Setup
## Install dependencies
```bash
    pip install -r requirements.txt
```
## Seting up ROCm on Fedora 41 (AMD GPU)
! Note: If you have an NVIDIA GPU, you should use CUDA instead of ROCm.
```bash
    sudo usermod -a -G video $LOGNAME
    sudo dnf install rocminfo rocm-opencl rocm-clinfo
    echo “export HSA_OVERRIDE_GFX_VERSION=10.3.0” >> ~/.bashrc
    source ~/.bashrc # or zshrc
    
    # Install PyTorch with ROCm support (in a virtual environment)
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```


# Usage
```bash
    # Train models
    python train.py
    # Evaluate models
    python evaluate.py
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
      
    git clone https://github.com/tkozakas/ocr-models.git
```