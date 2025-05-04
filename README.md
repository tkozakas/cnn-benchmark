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

# Install dependencies
### Install [uv](https://github.com/astral-sh/uv) python wrapper environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```
### For AMD GPUs
```bash
    uv pip install -r requirements-amd.txt
    # uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```
### Other
```bash
uv pip install -r requirements.txt
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
git clone https://github.com/tkozakas/ocr-models && cd ocr-models
chmod +x run_train.sh run_experiment.sh follow_logs.sh

# Run the training script
./follow_logs.sh $(sbatch --parsable run_train.sh)
# Run the experiment script
./follow_logs.sh $(sbatch --parsable run_experiment.sh)
```

### *Optional: Copy the test_data directory from the supercomputer to your local machine
```bash
scp -i ~/.ssh/id_ed25519 -r \
  mifvu_username@hpc.mif.vu.lt:/scratch/lustre/home/mifvu_username/ocr-models/test_data \
  ~/Documents/ocr-models/test_data/
```
### *Optional: Copy trained model from the supercomputer to your local machine
```bash
scp -i ~/.ssh/id_ed25519 -r \
  mifvu_username@hpc.mif.vu.lt:/scratch/lustre/home/mifvu_username/ocr-models/trained \
  ~/Documents/ocr-models/trained/
```