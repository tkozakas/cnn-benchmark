"""
Usage:
    train.py [--architecture=ARCH]
             [--k-folds=K] [--epochs=N] [--batch-size=B]
             [--lr=LR] [--weight-decay=WD] [--patience=P]

Options:
    -h --help               Show this help message.
    --architecture=ARCH     Model architecture to use
                            [default: EmnistCNN_16_64_128].
    --k-folds=K             Number of CV folds         [default: {k_folds}].
    --epochs=N              Max epochs per fold        [default: {epochs}].
    --batch-size=B          Training batch size        [default: {train_batch_size}].
    --lr=LR                 Learning rate              [default: {learning_rate}].
    --weight-decay=WD       Weight decay (L2)          [default: {weight_decay}].
    --patience=P            EarlyStop patience         [default: {early_stopping_patience}].
""".format(**__import__('config', fromlist=['train_config']).train_config)

import re
import subprocess
import time
import warnings

import numpy as np
import psutil
import torch
from docopt import docopt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import train_config, model_config
from src.model import get_model, save_model, load_model
from src.visualise import (
    plot_aggregated_learning_curves,
    plot_confusion_matrix
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*hipBLASLt.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
cudnn.benchmark = True

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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
          batch_size, learning_rate=None,
          weight_decay=0.0,
          random_state=42,
          optimizer_fn=None,
          scheduler_fn=None,
          criterion=None,
          early_stopping_patience=None):
    """Run k-fold CV, returning per-fold histories and aggregated averages."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion or nn.CrossEntropyLoss()
    kfold = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=random_state
    )
    all_results = []
    total_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset), start=1):
        fold_start = time.time()
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset, train_idx, test_idx,
            batch_size, train_config['cpu_workers']
        )
        model, optimizer, scheduler = init_model_optimizer_scheduler(
            model_fn, learning_rate, weight_decay,
            optimizer_fn, scheduler_fn, device
        )
        best_val_loss = float('inf')
        patience_cnt = 0
        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss':   [], 'val_accuracy':   [],
            'cpu_usage':  [], 'gpu_usage':      []
        }

        for epoch in range(1, epochs+1):
            t0 = time.time()
            tloss, tacc, cpu_peak, gpu_peak = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            vloss, vacc, tp, fp, prec, rec, f1 = evaluate_and_metrics(
                model, val_loader, criterion, device
            )
            history['train_loss'].append(tloss)
            history['train_accuracy'].append(tacc)
            history['val_loss'].append(vloss)
            history['val_accuracy'].append(vacc)
            history['cpu_usage'].append(cpu_peak)
            history['gpu_usage'].append(gpu_peak)

            if scheduler:
                scheduler.step()
            if early_stopping_patience is not None:
                if vloss < best_val_loss:
                    best_val_loss, patience_cnt = vloss, 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= early_stopping_patience:
                        print(
                            f"Early stopping at epoch {epoch}"
                            f" (patience: {patience_cnt})"
                        )
                        break

            print(
                f"Fold {fold} | Epoch {epoch}/{epochs} | "
                f"Train Loss: {tloss:.3f} | Train Acc: {tacc:.3f} | "
                f"Validation Loss: {vloss:.3f} | Validation Acc: {vacc:.3f} | "
                f"Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | "
                f"CPU: {cpu_peak:.1f}% | GPU: {gpu_peak:.1f}% | "
                f"Time: {time.time() - t0:.2f}s"
            )

        tloss, tacc, tp, fp, prec, rec, f1 = evaluate_and_metrics(
            model, test_loader, criterion, device
        )
        save_model(model, f"{architecture}_fold{fold}_best.pth")

        all_results.append({
            'fold': fold,
            **history,
            'test_loss': tloss,
            'test_accuracy': tacc,
            'TP': tp, 'FP': fp,
            'precision': prec, 'sensitivity': rec,
            'f1_score': f1,
            'elapsed_time': time.time() - fold_start,
            'epochs': len(history['train_loss'])
        })

    total_time = time.time() - total_start
    avg = lambda k: np.mean([np.mean(r[k]) for r in all_results])
    avg_results = {
        'avg_train_loss':  avg('train_loss'),
        'avg_train_acc':   avg('train_accuracy'),
        'avg_val_loss':    avg('val_loss'),
        'avg_val_acc':     avg('val_accuracy'),
        'avg_test_loss':   avg('test_loss'),
        'avg_test_acc':    avg('test_accuracy'),
        'avg_precision':   avg('precision'),
        'avg_sensitivity': avg('sensitivity'),
        'avg_f1_score':    avg('f1_score'),
        'avg_epochs': avg('epochs'),
        'avg_cpu_usage': np.mean([np.mean(r['cpu_usage']) for r in all_results]),
        'avg_gpu_usage': np.mean([np.mean(r['gpu_usage']) for r in all_results]),
        'total_time': total_time
    }

    return all_results, avg_results

def main(architecture):
    args = docopt(__doc__)
    # override or use defaults
    K = int(args['--k-folds'] or train_config['k_folds'])
    N = int(args['--epochs'] or train_config['epochs'])
    B = int(args['--batch-size'] or train_config['train_batch_size'])
    LR = float(args['--lr'] or train_config['learning_rate'])
    WD = float(args['--weight-decay'] or train_config['weight_decay'])
    PAT = int(args['--patience'] or train_config['early_stopping_patience'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {architecture}")
    print(f"Using K={K}, epochs={N}, batch_size={B}, lr={LR}, wd={WD}, patience={PAT}")

    # load dataset
    full = datasets.EMNIST(
        root="../data",
        split=train_config['emnist_type'],
        train=True,
        download=True,
        transform=transform
    )
    if train_config.get('subsample_size'):
        full, _ = torch.utils.data.random_split(
            full,
            [train_config['subsample_size'], len(full) - train_config['subsample_size']]
        )

    # train + CV
    all_results, avg_results = train(
        architecture=architecture,
        dataset=full,
        model_fn=lambda: get_model(architecture),
        k_folds=K,
        epochs=N,
        batch_size=B,
        learning_rate=LR,
        weight_decay=WD,
        early_stopping_patience=PAT,
        optimizer_fn=optim.Adam,
        scheduler_fn=None
    )

    print("\nPlotting results...")
    plot_aggregated_learning_curves(
        all_results, "Accuracy", "train_accuracy", "val_accuracy"
    )
    plot_aggregated_learning_curves(
        all_results, "Loss", "train_loss", "val_loss"
    )

    # final confusion + test metrics
    test_loader = DataLoader(
        full,
        batch_size=B,
        num_workers=train_config['cpu_workers'],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    model = get_model(architecture).to(device)
    model = load_model(
        model, f"{architecture}_fold{K}_best.pth"
    )
    plot_confusion_matrix(
        model, test_loader, device,
        classes=list(range(model_config[architecture]["num_classes"]))
    )
    test_loss, test_acc, tp, fp, precision, sensitivity, f1_score = evaluate_and_metrics(
        model, test_loader, nn.CrossEntropyLoss(), device
    )
    print("\nTest results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args['--architecture'])
