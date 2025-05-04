# CNN Architecture Benchmark

## Project Goals

This project aims to benchmark various CNN architectures to evaluate their performance in terms of
accuracy, using various metrics, and training time. The goal is to identify the best-performing model for text
recognition trained on EMNIST datasets. The project will also explore the impact of different hyperparameters, such as
learning rate, batch size, and architecture, on the model's performance.

## The experiments
1. Optimizer accuracy comparison (Adam is better -> use Adam)
2. Learning rate scheduler comparison (CosineAnnealingLR is better -> use CosineAnnealingLR)
3. Regularization comparison (L2 is better -> use L2)
4. Batch Size & Learning‐Rate Grid (Find sweet spot for batch size and learning rate)
5. Architecture comparison (Emnist with 2 layers or 3 or RNN or RCNN (existing))

---

## Environment

- **GPU driver / backend**: AMD ROCm 6.2 (but the same code should work on NVIDIA/CUDA)
- **Device**: AMD Radeon RX 6600 XT
- **OS**: Fedora
- **Python**: 3.12

## 6600 XT Workaround

If you encounter the error it’s because ROCm 6.2 doesn’t recognize the 6600 XT’s `gfx1032` architecture by default. You
can work around this by forcing the HSA version override:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```  

# Installation
### Install [uv](https://github.com/astral-sh/uv) python wrapper environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```
### Install required packages
```bash
    uv pip install -r requirements.txt
```
### For AMD
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```
### For NVIDIA
```bash
uv pip install torch torchvision torchaudio
```

# Usage
Note: The following commands are examples and may need to be adjusted based on your specific setup and requirements.

### Train and evaluate model
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

### Experiment on finding the best hyperparameters
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
### Predict image with trained model
```bash
python predict.py \
  path/to/best_model.pth \
  path/to/input_image.png \
  --device cuda \
  --architecture EmnistCNN_32_128_256 \
  --emnist-type balanced
```

# Train on VU supercomputer
1. Generate an SSH key locally (if you haven’t already)
```bash
ssh-keygen
```
2. [Upload](https://mif.vu.lt/ldap/sshkey.php) your public key to MIF’s LDAP
3. Connect to the supercomputer
```bash
ssh -i ~/.ssh/id_ed25519 mifvu_username@hpc.mif.vu.lt
```
4. Clone the repository and run the scripts
```bash
git clone https://github.com/tkozakas/cnn-benchmark && cd cnn-benchmark
chmod +x run_train.sh run_experiment.sh follow_logs.sh

# Run the training script
./follow_logs.sh $(sbatch --parsable run_train.sh)
# Run the experiment script
./follow_logs.sh $(sbatch --parsable run_experiment.sh)
```
## Copy Results Back to Your Local Machine
You can omit these if you don’t need the outputs locally.
1. Copy the test_data directory from the supercomputer to your local machine
```bash
scp -i ~/.ssh/id_ed25519 -r \
  mifvu_username@hpc.mif.vu.lt:/scratch/lustre/home/mifvu_username/cnn-benchmark/test_data \
  ~/Documents/cnn-benchmark/test_data/
```
2. Copy trained model from the supercomputer to your local machine
```bash
scp -i ~/.ssh/id_ed25519 -r \
  mifvu_username@hpc.mif.vu.lt:/scratch/lustre/home/mifvu_username/cnn-benchmark/trained \
  ~/Documents/cnn-benchmark/trained/
```