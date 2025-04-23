"""
Usage:
    experiment.py [--architecture=ARCH]

Options:
    --architecture=ARCH   Specify the architecture to use [default: EmnistCNN_16_64_128].
"""

import matplotlib.pyplot as plt
import torch
from docopt import docopt
from torchvision import datasets

from src.config import train_config
from src.model import get_model
from src.train import k_fold_cross_validation, transform


def get_subsample(full_dataset):
    if train_config.get("subsample_size"):
        subsample_size = train_config["subsample_size"]
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [subsample_size, len(full_dataset) - subsample_size]
        )
    return full_dataset


def plot_metrics(runs, title):
    """
    Plot a 2x2 grid of loss & accuracy curves for given runs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax00, ax01, ax10, ax11 = axes.flatten()
    for run in runs:
        epochs = range(1, len(run['train_loss']) + 1)
        ax00.plot(epochs, run['train_loss'], label=run['name'])
        ax01.plot(epochs, run['val_loss'],   label=run['name'])
        ax10.plot(epochs, run['train_accuracy'], label=run['name'])
        ax11.plot(epochs, run['val_accuracy'],   label=run['name'])
    ax00.set_title('Train Loss'); ax01.set_title('Val Loss')
    ax10.set_title('Train Acc');  ax11.set_title('Val Acc')
    for ax in [ax00, ax01, ax10, ax11]:
        ax.set_xlabel('Epoch'); ax.grid(True); ax.legend(fontsize='small')
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
    plt.ylabel('Avg Training Time (s)')
    plt.tight_layout()
    plt.show()


def plot_test_accuracy(runs, title):
    names = [r['name'] for r in runs]
    accs  = [r['test_accuracy'] for r in runs]
    plt.figure(figsize=(8,4))
    plt.bar(names, accs)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Configuration')
    plt.ylabel('Avg Test Accuracy')
    plt.tight_layout()
    plt.show()


def run_experiment(name, architecture, dataset, **kwargs):
    """
    Run k-fold CV and return averaged metric curves, train time,
    plus test accuracy, batch_size and learning_rate if provided.
    """
    folds_data, avg_results = k_fold_cross_validation(
        architecture=architecture,
        dataset=dataset,
        model_fn=lambda arch=architecture: get_model(arch),
        **kwargs
    )
    # aggregate per-epoch metrics
    per_fold_train_loss = [f['train_loss'] for f in folds_data]
    per_fold_val_loss   = [f['val_loss']   for f in folds_data]
    per_fold_train_acc  = [f['train_accuracy'] for f in folds_data]
    per_fold_val_acc    = [f['val_accuracy']   for f in folds_data]
    avg_train_loss = [sum(vals)/len(vals) for vals in zip(*per_fold_train_loss)]
    avg_val_loss   = [sum(vals)/len(vals) for vals in zip(*per_fold_val_loss)]
    avg_train_acc  = [sum(vals)/len(vals) for vals in zip(*per_fold_train_acc)]
    avg_val_acc    = [sum(vals)/len(vals) for vals in zip(*per_fold_val_acc)]
    return {
        'name': name,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_accuracy': avg_train_acc,
        'val_accuracy': avg_val_acc,
        'test_accuracy': avg_results['avg_test_acc'],
        'time': avg_results.get('avg_time', avg_results.get('elapsed_time')),
        'batch_size': kwargs.get('batch_size'),
        'learning_rate': kwargs.get('learning_rate')
    }


def main(architecture):
    print("Loading EMNIST dataset...")
    full = datasets.EMNIST(root="../data", split=train_config["emnist_type"],
                           train=True, download=True, transform=transform)
    ds = get_subsample(full)

    # baseline hyperparams
    lr     = train_config['learning_rate']
    epochs = train_config['epochs']
    folds  = train_config['k_folds']
    bs     = train_config['train_batch_size']
    early_stopping_patience = train_config['early_stopping_patience']

    # 1) Optimizer Comparison
    optim_map = {
        'Adam':    lambda p: torch.optim.Adam(p, lr=lr),
        'SGD':     lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
        'RMSprop': lambda p: torch.optim.RMSprop(p, lr=lr)
    }
    runs = [run_experiment(name, architecture, ds,
                           k_folds=folds, epochs=epochs,
                           batch_size=bs, learning_rate=lr,
                           optimizer_fn=opt_fn)
            for name, opt_fn in optim_map.items()]
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
    plot_metrics(runs, 'Scheduler Comparison')
    plot_test_accuracy(runs, 'Scheduler: Test Accuracy Comparison')
    best_sched = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_scheduler_fn = sched_map[best_sched].get('scheduler_fn')
    print(f"Best scheduler: {best_sched}")

    # 3) Regularization Comparison
    reg_map = {'No WD': 0.0, 'WD=1e-4': 1e-4, 'WD=1e-3': 1e-3}
    runs = [run_experiment(name, architecture, ds,
                           k_folds=folds, epochs=epochs,
                           batch_size=bs, learning_rate=lr,
                           optimizer_fn=best_optimizer_fn,
                           scheduler_fn=best_scheduler_fn,
                           weight_decay=wd,
                           early_stopping_patience=early_stopping_patience)
            for name, wd in reg_map.items()]
    plot_metrics(runs, 'Regularization Comparison')
    plot_test_accuracy(runs, 'Regularization: Test Accuracy Comparison')
    best_reg = max(runs, key=lambda r: r['test_accuracy'])['name']
    best_weight_decay = reg_map[best_reg]
    print(f"Best weight decay: {best_reg}")

    # 4) Batch Size & LR Grid
    grid = [(32, 1e-3), (32, 1e-4), (64, 1e-3), (64, 1e-4)]
    runs = [run_experiment(f"BS={bs2}, LR={lr2}", architecture, ds,
                           k_folds=folds, epochs=epochs,
                           batch_size=bs2, learning_rate=lr2,
                           optimizer_fn=best_optimizer_fn,
                           scheduler_fn=best_scheduler_fn,
                           weight_decay=best_weight_decay,
                           early_stopping_patience=early_stopping_patience)
            for bs2, lr2 in grid]
    plot_metrics(runs, 'Batch Size & LR Grid')
    plot_test_accuracy(runs, 'Batch/LR: Test Accuracy Comparison')
    best_run = max(runs, key=lambda r: r['test_accuracy'])
    best_bs = best_run['batch_size']
    best_lr = best_run['learning_rate']
    print(f"Best BS/LR: BS={best_bs}, LR={best_lr}")

    # 5) Architecture Comparison
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
    plot_metrics(runs, 'Architecture Comparison')
    plot_time(runs, 'Architecture: Training Time')
    plot_test_accuracy(runs, 'Architecture: Test Accuracy Comparison')

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args['--architecture'])