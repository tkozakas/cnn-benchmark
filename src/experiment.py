__doc__ = r"""
Usage:
    experiment.py [--architecture=ARCH]
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
    --emnist-type=TYPE      EMNIST type (letters, digits, balanced) [default: balanced].
    --device=DEVICE         Device to use for training (cpu or cuda) [default: cuda].
    --cpu-workers=NUM       Number of CPU workers for data loading [default: 6].
    --architecture=ARCH     Model architecture [default: EmnistCNN_32_128_256].
    --subsample-size=S      Subsample size for training set [default: 1000].
    --k-folds=K             Number of CV folds              [default: 3].
    --epochs=N              Max epochs per fold             [default: 10].
    --batch-size=B          Training batch size             [default: 128].
    --lr=LR                 Learning rate                   [default: 0.001].
    --weight-decay=WD       Weight decay (L2)               [default: 0.0001].
    --patience=P            Early-stop patience             [default: 5].
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from docopt import docopt
from torchvision import datasets

from model import get_model
from utility import parse_args, get_subsample
from train import train, transform
from visualise import plot_metrics, plot_test_accuracy, plot_time

os.makedirs('../test_data', exist_ok=True)
warnings.filterwarnings("ignore", message=".*GoogleNet.*", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")

def run_experiment(name, architecture, dataset, **kwargs):
    """Run one experimental configuration and collect metrics."""
    folds_data, avg_results = train(
        architecture=architecture,
        dataset=dataset,
        model_fn=lambda arch=architecture: get_model(arch),
        **kwargs
    )

    # per-epoch averages for plotting
    per_fold = {
        'train_loss':   [f['train_loss'] for f in folds_data],
        'val_loss':     [f['val_loss']   for f in folds_data],
        'train_acc':    [f['train_accuracy'] for f in folds_data],
        'val_acc':      [f['val_accuracy']   for f in folds_data],
    }
    avg_curve = {
        k: [sum(vals) / len(vals) for vals in zip(*v)]
        for k, v in per_fold.items()
    }

    return {
        'name':                name,
        'batch_size':          kwargs['batch_size'],
        'learning_rate':       kwargs['learning_rate'],
        'avg_train_loss':      float(np.mean(avg_curve['train_loss'])),
        'avg_val_loss':        float(np.mean(avg_curve['val_loss'])),
        'avg_train_accuracy':  float(np.mean(avg_curve['train_acc'])),
        'avg_val_accuracy':    float(np.mean(avg_curve['val_acc'])),
        'test_accuracy':       avg_results['avg_test_acc'],
        'time':                avg_results.get('avg_time', avg_results.get('elapsed_time')),
        # curves for plotting
        'train_loss_curve':    avg_curve['train_loss'],
        'val_loss_curve':      avg_curve['val_loss'],
        'train_acc_curve':     avg_curve['train_acc'],
        'val_acc_curve':       avg_curve['val_acc'],
        'cpu_usage':           avg_results.get('avg_cpu_usage', []),
        'gpu_usage':           avg_results.get('avg_gpu_usage', []),
    }


def save_test_data(data, filename):
    """Save the summary table to CSV."""
    pd.DataFrame(data).to_csv(
        filename, index=False,
        columns=[
            'name', 'batch_size', 'learning_rate',
            'avg_train_loss', 'avg_val_loss',
            'avg_train_accuracy', 'avg_val_accuracy',
            'test_accuracy', 'time'
        ]
    )


def main():
    args = docopt(__doc__)
    ARCHITECTURE, B, CPU_WORKERS, DEVICE, EMNIST_TYPE, K, LR, N, PAT, SUBSAMPLE_SIZE, WD = parse_args(args)

    full = datasets.EMNIST(
        root="../data",
        split=EMNIST_TYPE,
        train=True, download=True,
        transform=transform
    )
    ds = get_subsample(full, SUBSAMPLE_SIZE)

    # 1) Optimizer Comparison
    optim_map = {
        'Adam':    lambda p: torch.optim.Adam(p, lr=LR, weight_decay=WD),
        'SGD':     lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9, weight_decay=WD),
        'RMSprop': lambda p: torch.optim.RMSprop(p, lr=LR, weight_decay=WD)
    }
    runs = [
        run_experiment(
            name, ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=B,
            learning_rate=LR, optimizer_fn=opt_fn,
            weight_decay=WD,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for name, opt_fn in optim_map.items()
    ]
    save_test_data(runs, '../test_data/optimizer_comparison.csv')
    plot_metrics(runs, 'Optimizer Comparison')
    plot_test_accuracy(runs, 'Optimizer: Test Accuracy Comparison')
    best_opt = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_opt_fn = optim_map[best_opt]
    print(f"Best optimizer: {best_opt}")

    # 2) Scheduler Comparison
    sched_map = {
        'None':            {},
        'StepLR':          {'scheduler_fn': lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=10, gamma=0.1)},
        'CosineAnnealing': {'scheduler_fn': lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=N)},
        'OneCycle':        {'scheduler_fn': lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=LR, total_steps=N)}
    }
    runs = [
        run_experiment(
            name, ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=B,
            learning_rate=LR, optimizer_fn=best_opt_fn,
            weight_decay=WD,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE,
            **params
        )
        for name, params in sched_map.items()
    ]
    save_test_data(runs, '../test_data/scheduler_comparison.csv')
    plot_metrics(runs, 'Scheduler Comparison')
    plot_test_accuracy(runs, 'Scheduler: Test Accuracy Comparison')
    best_sched = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_sched_fn = sched_map[best_sched].get('scheduler_fn')
    print(f"Best scheduler: {best_sched}")

    # 3) Regularization Comparison
    reg_map = {'No WD': 0.0, 'WD=1e-5': 1e-5, 'WD=1e-4': 1e-4, 'WD=1e-3': 1e-3}
    runs = [
        run_experiment(
            name, ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=B,
            learning_rate=LR, optimizer_fn=best_opt_fn,
            scheduler_fn=best_sched_fn,
            weight_decay=wd,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for name, wd in reg_map.items()
    ]
    save_test_data(runs, '../test_data/regularization_comparison.csv')
    plot_metrics(runs, 'Regularization Comparison')
    plot_test_accuracy(runs, 'Regularization: Test Accuracy Comparison')
    best_reg = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_wd  = reg_map[best_reg]
    print(f"Best weight decay: {best_reg}")

    # 4) Batch Size Comparison
    batch_sizes = [64, 128, 256, 512, 1024]
    runs_bs = [
        run_experiment(
            f"BS={b}", ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=b,
            learning_rate=LR, optimizer_fn=best_opt_fn,
            scheduler_fn=best_sched_fn,
            weight_decay=best_wd,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for b in batch_sizes
    ]
    save_test_data(runs_bs, '../test_data/batch_size_comparison.csv')
    plot_metrics(runs_bs, 'Batch Size Comparison')
    plot_time(runs_bs, 'Batch Size: Training Time')
    plot_test_accuracy(runs_bs, 'Batch Size: Test Accuracy Comparison')
    best_bs = max(runs_bs, key=lambda r: r['test_accuracy'])['batch_size']
    print(f"Best batch size: {best_bs}")

    # 5) Learning-Rate Grid at Best Batch Size
    lr_grid = [1e-3, 1e-4, 1e-5]
    runs_lr = [
        run_experiment(
            f"BS={best_bs}, LR={l}", ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=best_bs,
            learning_rate=l, optimizer_fn=best_opt_fn,
            scheduler_fn=best_sched_fn,
            weight_decay=best_wd,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for l in lr_grid
    ]
    save_test_data(runs_lr, '../test_data/lr_grid_comparison.csv')
    plot_metrics(runs_lr, f'LR Grid @ BS={best_bs}')
    plot_test_accuracy(runs_lr, f'LR Grid @ BS={best_bs}: Test Accuracy Comparison')
    best_lr = max(runs_lr, key=lambda r: r['test_accuracy'])['learning_rate']
    print(f"Best learning rate: {best_lr}")

    # 6) Architecture Comparison
    archs = [
        'EmnistCNN_16_64_128', 'EmnistCNN_32_128_256',
        'EmnistCNN_8_32_64',  'EmnistCNN_16_64',
        'EmnistCNN_32_128',   'GoogleNet', 'ResNet18'
    ]
    runs = [
        run_experiment(
            arch, arch, ds,
            k_folds=K, epochs=N, batch_size=best_bs,
            learning_rate=best_lr, optimizer_fn=best_opt_fn,
            scheduler_fn=best_sched_fn,
            weight_decay=best_wd,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for arch in archs
    ]
    save_test_data(runs, '../test_data/architecture_comparison.csv')
    plot_metrics(runs, 'Architecture Comparison')
    plot_time(runs, 'Architecture: Training Time')
    plot_test_accuracy(runs, 'Architecture: Test Accuracy Comparison')


if __name__ == "__main__":
    main()
