# OCR Model Benchmark

## Project Goals

This project aims to benchmark various OCR models to evaluate their performance in terms of
accuracy, edit distance, and runtime.

## Datasets
EMNIST

## Plan and Methodology
TODO

## Usage
```bash
# Activate virtual environment
conda activate ocr-benchmark
# Train models
python train.py
# Evaluate models
python evaluate.py
```

# Seting up ROCm on Fedora 41 (AMD GPU)
! Note: If you have an NVIDIA GPU, you should use CUDA instead of ROCm.
```bash
sudo usermod -a -G video $LOGNAME
sudo dnf install rocminfo rocm-opencl rocm-clinfo
echo “export HSA_OVERRIDE_GFX_VERSION=10.3.0” >> ~/.bashrc
source ~/.bashrc # or zshrc

# Install PyTorch with ROCm support (in a virtual environment)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```