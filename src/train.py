__doc__ = r"""
Usage:
    train.py [--architecture=ARCH]
             [--emnist-type=TYPE]
             [--device=DEVICE]
             [--cpu-workers=NUM]
             [--subsample-size=S]
             [--k-folds=K]
             [--epochs=N]
             [--batch-size=B]
             [--lr=LR]
             [--weight-decay=WD]
             [--patience=P]

Options:
    -h --help               Show this help message.
    --architecture=ARCH     Model architecture [default: EmnistCNN_32_128_256].
    --emnist-type=TYPE      EMNIST type (letters, digits, balanced) [default: balanced].
    --device=DEVICE         Device to use (cpu or cuda) [default: cpu].
    --cpu-workers=NUM       Number of CPU workers for data loading [default: 4].
    --subsample-size=S      Subsample size for training set [default: None].
    --k-folds=K             Number of CV folds [default: 5].
    --epochs=N              Max epochs per fold [default: 20].
    --batch-size=B          Training batch size [default: 128].
    --lr=LR                 Learning rate [default: 0.001].
    --weight-decay=WD       Weight decay (L2) [default: 0.0001].
    --patience=P            Early-stop patience [default: 5].
"""
import re
import subprocess
import time
import warnings

import psutil
import torch
from docopt import docopt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from config import train_config, model_config
from model import get_model, save_model, load_model
from utility import get_transforms
from utility import parse_args, get_subsample
from visualise import (
    plot_confusion_matrix
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*hipBLASLt.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")
cudnn.benchmark = True

transform = get_transforms()

def get_gpu_usage_percent():
    """Return GPU utilization percent for NVIDIA or AMD/ROCm."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        pynvml.nvmlShutdown()
        return float(util)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            m = re.search(r"GPU use\s*\(%\)\s*:\s*(\d+)", line)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return 0.0

def evaluate_and_metrics(model, loader, criterion, device):
    """Run evaluation and compute loss, accuracy and confusion‐based metrics."""
    model.eval()
    running_loss = correct = total = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    loss = running_loss / total
    acc = correct / total
    cm = confusion_matrix(y_true, y_pred)
    tp = int(cm.diagonal().sum())
    fp = int(cm.sum(axis=0).sum() - tp)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return loss, acc, tp, fp, prec, rec, f1


def get_data_loaders(dataset, train_idx, test_idx,
                     batch_size, num_workers):
    """Split dataset into train/val/test and return DataLoaders."""
    train_subset = Subset(dataset, train_idx)
    test_subset  = Subset(dataset, test_idx)
    val_size = int(len(train_subset) * train_config["val_split"])
    train_size = len(train_subset) - val_size
    train_data, val_data = torch.utils.data.random_split(
        train_subset, [train_size, val_size]
    )
    loader_args = dict(
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        num_workers=num_workers,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader   = DataLoader(val_data,   shuffle=False, **loader_args)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch, returning loss, acc, peak CPU/GPU usage."""
    model.train()
    running_loss = correct = total = 0
    max_cpu = max_gpu = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # resource sampling
        cpu = psutil.cpu_percent(interval=None)
        gpu = get_gpu_usage_percent()
        max_cpu = max(max_cpu, cpu)
        max_gpu = max(max_gpu, gpu)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total, max_cpu, max_gpu


def init_model_optimizer_scheduler(model_fn, learning_rate,
                                   weight_decay,
                                   optimizer_fn, scheduler_fn,
                                   device):
    """Instantiate model, optimizer, and scheduler."""
    model = model_fn().to(device)
    optimizer = optimizer_fn(model.parameters()) if optimizer_fn else optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = scheduler_fn(optimizer) if scheduler_fn else None
    return model, optimizer, scheduler


def train(architecture, dataset, model_fn,
          k_folds, epochs,
          batch_size, learning_rate,
          weight_decay,
          optimizer_fn=None,
          scheduler_fn=None,
          criterion=None,
          early_stopping_patience=None,
          device='cuda',
          cpu_workers=4,
          random_state=42):
    device = torch.device(device)
    criterion = criterion or nn.CrossEntropyLoss()
    kfold = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=random_state
    )
    all_results = []
    total_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset), start=1):
        # Number of samples in this fold’s training split
        num_train_samples = len(train_idx)

        # DataLoaders
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset, train_idx, test_idx,
            batch_size, cpu_workers
        )

        # Initialize model, optimizer, scheduler
        model, optimizer, scheduler = init_model_optimizer_scheduler(
            model_fn, learning_rate, weight_decay,
            optimizer_fn, scheduler_fn, device
        )

        # Count trainable params
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        best_val_loss = float('inf')
        patience_cnt = 0

        # History containers
        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss':   [], 'val_accuracy':   [],
            'f1_score': [], 'precision': [], 'recall': [],
            'lr': [], 'epoch_time': [],
            'cpu_usage': [], 'gpu_usage': [],
            'samples_per_sec': []
        }

        # --- Per-epoch training loop ---
        for epoch in range(1, epochs+1):
            t0 = time.time()
            tloss, tacc, cpu_peak, gpu_peak = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            epoch_duration = time.time() - t0

            vloss, vacc, tp, fp, prec, rec, f1 = evaluate_and_metrics(
                model, val_loader, criterion, device
            )

            # Throughput = samples / second
            throughput = num_train_samples / epoch_duration

            # Record epoch metrics
            history['train_loss'].append(tloss)
            history['train_accuracy'].append(tacc)
            history['val_loss'].append(vloss)
            history['val_accuracy'].append(vacc)
            history['f1_score'].append(f1)
            history['precision'].append(prec)
            history['recall'].append(rec)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_duration)
            history['samples_per_sec'].append(throughput)
            history['cpu_usage'].append(cpu_peak)
            history['gpu_usage'].append(gpu_peak)

            # Step scheduler and early stop
            if scheduler:
                scheduler.step()
            if early_stopping_patience is not None:
                if vloss < best_val_loss:
                    best_val_loss, patience_cnt = vloss, 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch} (patience: {patience_cnt})")
                        break

            print(
                f"Fold {fold} | Epoch {epoch}/{epochs} | "
                f"Train Loss: {tloss:.3f} | Train Acc: {tacc:.3f} | "
                f"Val Loss: {vloss:.3f} | Val Acc: {vacc:.3f} | "
                f"F1: {f1:.3f} | CPU: {cpu_peak:.1f}% | GPU: {gpu_peak:.1f}% | "
                f"Time: {epoch_duration:.2f}s"
            )

        # --- End of epochs: evaluate on test set ---
        tloss, tacc, tp, fp, prec, rec, f1 = evaluate_and_metrics(
            model, test_loader, criterion, device
        )
        print(f"Test | Loss: {tloss:.3f} | Acc: {tacc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        # Measure inference latency per sample
        model.eval()
        # Warm-up (esp. for GPU)
        dummy_input = next(iter(test_loader))[0].to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        # Timed full pass
        start_inf = time.time()
        total_samples = 0
        with torch.no_grad():
            for xb, _ in test_loader:
                bs = xb.size(0)
                total_samples += bs
                _ = model(xb.to(device))
        inf_elapsed = time.time() - start_inf
        inference_latency = inf_elapsed / total_samples

        # Collect final test + fold metadata
        history.update({
            'test_loss': tloss,
            'test_accuracy': tacc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1_score': f1,
        })

        save_model(model, f"{architecture}_fold{fold}.pth")
        all_results.append({
            'fold': fold,
            'param_count': param_count,
            'inference_latency': inference_latency,
            **history
        })

    total_time = time.time() - total_start
    print(f"Total training time across all folds: {total_time:.2f}s")
    return all_results

def main():
    print("Starting training...")
    args = docopt(__doc__)
    ARCHITECTURE, B, CPU_WORKERS, DEVICE, EMNIST_TYPE, K, LR, N, PAT, SUBSAMPLE_SIZE, WD = parse_args(args)

    full = datasets.EMNIST(
        root="../data",
        split=EMNIST_TYPE,
        train=True,
        download=True,
        transform=transform
    )
    ds = get_subsample(full, SUBSAMPLE_SIZE)

    # train + CV
    all_results, _ = train(
        device=DEVICE,
        architecture=ARCHITECTURE,
        dataset=ds,
        model_fn=lambda: get_model(ARCHITECTURE),
        k_folds=K,
        epochs=N,
        batch_size=B,
        learning_rate=LR,
        weight_decay=WD,
        early_stopping_patience=PAT,
        cpu_workers=CPU_WORKERS,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    )

    # final confusion + test metrics
    test_loader = DataLoader(
        ds,
        batch_size=B,
        num_workers=CPU_WORKERS,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    model = get_model(ARCHITECTURE).to(DEVICE)
    model = load_model(
        model, f"{ARCHITECTURE}_fold{K}.pth"
    )
    plot_confusion_matrix(
        model, test_loader, DEVICE,
        classes=list(range(model_config[ARCHITECTURE]["num_classes"]))
    )
    test_loss, test_acc, tp, fp, precision, sensitivity, f1_score = evaluate_and_metrics(
        model, test_loader, nn.CrossEntropyLoss(), DEVICE
    )
    print("\nTest results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()
