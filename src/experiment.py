"""
Usage:
    experiment.py [--architecture=ARCH]

Options:
    --architecture=ARCH   Specify the architecture to use [default: EmnistCNN_16_64_128].
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from docopt import docopt
from torchvision import datasets

from src.config import test_config
from src.model import get_model
from src.train import k_fold_cross_validation, transform

os.makedirs('../test_data', exist_ok=True)
warnings.filterwarnings("ignore", message=".*GoogleNet.*", category=UserWarning)

def get_subsample(full_dataset):
    if test_config.get("subsample_size"):
        subsample_size = test_config["subsample_size"]
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [subsample_size, len(full_dataset) - subsample_size]
        )
    return full_dataset

def plot_metrics(runs, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax00, ax01, ax10, ax11 = axes.flatten()
    for run in runs:
        epochs = range(1, len(run['train_loss_curve']) + 1)
        ax00.plot(epochs, run['train_loss_curve'], label=run['name'])
        ax01.plot(epochs, run['val_loss_curve'],   label=run['name'])
        ax10.plot(epochs, run['train_acc_curve'],  label=run['name'])
        ax11.plot(epochs, run['val_acc_curve'],    label=run['name'])
    ax00.set_title('Train Loss')
    ax01.set_title('Validation Loss')
    ax10.set_title('Train Accuracy')
    ax11.set_title('Validation Accuracy')
    ax00.set_ylabel('Loss')
    ax01.set_ylabel('Loss')
    ax10.set_ylabel('Accuracy')
    ax11.set_ylabel('Accuracy')
    for ax in [ax00, ax01, ax10, ax11]:
        ax.set_xlabel('Epoch')
        ax.grid(True)
        ax.legend(fontsize='small')
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_time(runs, title):
    names = [r['name'] for r in runs]
    times = [r['time'] for r in runs]
    plt.figure(figsize=(8, 4))
    plt.bar(names, times)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Configuration')
    plt.ylabel('Average Training Time (s)')
    plt.tight_layout()
    plt.show()


def plot_test_accuracy(runs, title):
    names = [r['name'] for r in runs]
    accs  = [r['test_accuracy'] for r in runs]
    plt.figure(figsize=(8, 4))
    plt.bar(names, accs)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Configuration')
    plt.ylabel('Average Test Accuracy')
    plt.tight_layout()
    plt.show()


def run_experiment(name, architecture, dataset, **kwargs):
    folds_data, avg_results = k_fold_cross_validation(
        architecture=architecture,
        dataset=dataset,
        model_fn=lambda arch=architecture: get_model(arch),
        **kwargs
    )
    # per-epoch averages (for plotting only)
    per_fold_train_loss = [f['train_loss'] for f in folds_data]
    per_fold_val_loss   = [f['val_loss']   for f in folds_data]
    per_fold_train_acc  = [f['train_accuracy'] for f in folds_data]
    per_fold_val_acc    = [f['val_accuracy']   for f in folds_data]

    avg_train_loss_curve = [sum(vals)/len(vals) for vals in zip(*per_fold_train_loss)]
    avg_val_loss_curve   = [sum(vals)/len(vals) for vals in zip(*per_fold_val_loss)]
    avg_train_acc_curve  = [sum(vals)/len(vals) for vals in zip(*per_fold_train_acc)]
    avg_val_acc_curve    = [sum(vals)/len(vals) for vals in zip(*per_fold_val_acc)]

    # scalar summaries
    avg_train_loss    = float(np.mean(avg_train_loss_curve))
    avg_val_loss      = float(np.mean(avg_val_loss_curve))
    avg_train_accuracy= float(np.mean(avg_train_acc_curve))
    avg_val_accuracy  = float(np.mean(avg_val_acc_curve))
    test_accuracy     = avg_results['avg_test_acc']
    time_taken        = avg_results.get('avg_time', avg_results.get('elapsed_time'))

    return {
        'name': name,
        'batch_size':      kwargs.get('batch_size'),
        'learning_rate':   kwargs.get('learning_rate'),
        'avg_train_loss':    avg_train_loss,
        'avg_val_loss':      avg_val_loss,
        'avg_train_accuracy':avg_train_accuracy,
        'avg_val_accuracy':  avg_val_accuracy,
        'test_accuracy':     test_accuracy,
        'time':              time_taken,
        # keep these for plotting:
        'train_loss_curve':  avg_train_loss_curve,
        'val_loss_curve':    avg_val_loss_curve,
        'train_acc_curve':   avg_train_acc_curve,
        'val_acc_curve':     avg_val_acc_curve,
        'cpu_usage':         avg_results.get('avg_cpu_usage', []),
        'gpu_usage':         avg_results.get('avg_gpu_usage', []),
    }

def save_test_data(data, filename):
    pd.DataFrame(data).to_csv(filename, index=False,
                              columns=[
                                  'name', 'batch_size', 'learning_rate',
                                  'avg_train_loss', 'avg_val_loss',
                                  'avg_train_accuracy', 'avg_val_accuracy',
                                  'test_accuracy', 'time'
                              ])

def main(architecture):
    print("Loading EMNIST dataset...")
    full = datasets.EMNIST(root="../data", split=test_config["emnist_type"],
                           train=True, download=True, transform=transform)
    ds = get_subsample(full)

    lr     = test_config['learning_rate']
    epochs = test_config['epochs']
    folds  = test_config['k_folds']
    bs     = test_config['train_batch_size']
    early_stopping_patience = test_config['early_stopping_patience']

    # 1) Optimizer Comparison
    optim_map = {
        'Adam':    lambda p: torch.optim.Adam(p, lr=lr),
        'SGD':     lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
        'RMSprop': lambda p: torch.optim.RMSprop(p, lr=lr)
    }
    runs = [run_experiment(name, architecture, ds,
                           k_folds=folds, epochs=epochs,
                           batch_size=bs, learning_rate=lr,
                           optimizer_fn=opt_fn,
                           early_stopping_patience=early_stopping_patience)
            for name, opt_fn in optim_map.items()]
    save_test_data(runs, '../test_data/optimizer_comparison.csv')
    plot_metrics(runs, 'Optimizer Comparison')
    plot_test_accuracy(runs, 'Optimizer: Test Accuracy Comparison')
    best_opt = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_optimizer_fn = optim_map[best_opt]
    print(f"Best optimizer: {best_opt}")

    # 2) Scheduler Comparison
    sched_map = {
        'None':            {},
        'StepLR':          {'scheduler_fn': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)},
        'CosineAnnealing': {'scheduler_fn': lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)},
        'OneCycle':        {'scheduler_fn': lambda opt: torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs)}
    }
    runs = [run_experiment(name, architecture, ds,
                           k_folds=folds, epochs=epochs,
                           batch_size=bs, learning_rate=lr,
                           optimizer_fn=best_optimizer_fn,
                           early_stopping_patience=early_stopping_patience,
                           **params)
            for name, params in sched_map.items()]
    save_test_data(runs, '../test_data/scheduler_comparison.csv')
    plot_metrics(runs, 'Scheduler Comparison')
    plot_test_accuracy(runs, 'Scheduler: Test Accuracy Comparison')
    best_sched = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_scheduler_fn = sched_map[best_sched].get('scheduler_fn')
    print(f"Best scheduler: {best_sched}")

    # 3) Regularization Comparison
    reg_map = {'No WD': 0.0, 'WD=1e-5': 1e-5, 'WD=1e-4': 1e-4, 'WD=1e-3': 1e-3}
    runs = [run_experiment(name, architecture, ds,
                           k_folds=folds, epochs=epochs,
                           batch_size=bs, learning_rate=lr,
                           optimizer_fn=best_optimizer_fn,
                           scheduler_fn=best_scheduler_fn,
                           weight_decay=wd,
                           early_stopping_patience=early_stopping_patience)
            for name, wd in reg_map.items()]
    save_test_data(runs, '../test_data/regularization_comparison.csv')
    plot_metrics(runs, 'Regularization Comparison')
    plot_test_accuracy(runs, 'Regularization: Test Accuracy Comparison')
    best_reg = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_weight_decay = reg_map[best_reg]
    print(f"Best weight decay: {best_reg}")

    # 4) Batch Size Comparison
    batch_sizes = [64, 128, 256, 512, 1024]
    runs_bs = [
        run_experiment(
            f"BS={b}", architecture, ds,
            k_folds=folds, epochs=epochs,
            batch_size=b, learning_rate=lr,
            optimizer_fn=best_optimizer_fn,
            scheduler_fn=best_scheduler_fn,
            weight_decay=best_weight_decay,
            early_stopping_patience=early_stopping_patience
        )
        for b in batch_sizes
    ]
    save_test_data(runs_bs, '../test_data/batch_size_comparison.csv')
    plot_metrics(runs_bs, 'Batch Size Comparison')
    plot_test_accuracy(runs_bs, 'Batch Size: Test Accuracy Comparison')
    plot_time(runs_bs, 'Batch Size: Training Time')
    best_bs = max(runs_bs, key=lambda r: r['test_accuracy'])['batch_size']
    print(f"Best batch size: {best_bs}")

    # 5) Learning‐Rate Grid at Best Batch Size
    lr_grid = [1e-3, 1e-4, 1e-5]
    runs_lr = [
        run_experiment(
            f"BS={best_bs}, LR={l}", architecture, ds,
            k_folds=folds, epochs=epochs,
            batch_size=best_bs, learning_rate=l,
            optimizer_fn=best_optimizer_fn,
            scheduler_fn=best_scheduler_fn,
            weight_decay=best_weight_decay,
            early_stopping_patience=early_stopping_patience
        )
        for l in lr_grid
    ]
    save_test_data(runs_lr, '../test_data/lr_grid_comparison.csv')
    plot_metrics(runs_lr, f'LR Grid @ BS={best_bs}')
    plot_test_accuracy(runs_lr, f'LR Grid @ BS={best_bs}: Test Accuracy Comparison')
    best_lr = max(runs_lr, key=lambda r: r['test_accuracy'])['learning_rate']
    print(f"Best learning rate at BS={best_bs}: LR={best_lr}")

    print(f"Best config: {best_opt}, {best_sched}, {best_reg}, BS={best_bs}, LR={best_lr}")
    # 6) Architecture Comparison
    archs = [
        'EmnistCNN_16_64_128', 'EmnistCNN_32_128_256',
        'EmnistCNN_8_32_64', 'EmnistCNN_16_64',
        'EmnistCNN_32_128', 'GoogleNet', 'ResNet18'
    ]
    runs = [run_experiment(arch, arch, ds,
                           k_folds=folds, epochs=epochs,
                           batch_size=best_bs,
                           learning_rate=best_lr,
                           optimizer_fn=best_optimizer_fn,
                           scheduler_fn=best_scheduler_fn,
                           weight_decay=best_weight_decay,
                           early_stopping_patience=early_stopping_patience)
            for arch in archs]
    save_test_data(runs, '../test_data/architecture_comparison.csv')
    plot_metrics(runs, 'Architecture Comparison')
    plot_time(runs, 'Architecture: Training Time')
    plot_test_accuracy(runs, 'Architecture: Test Accuracy Comparison')

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args['--architecture'])
