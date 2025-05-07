import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

BASE_DIR = "../test_data/plot"

def _save_plot(fig, category, name, suffix):
    path = os.path.join(BASE_DIR, category)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{name}_{suffix}.png"))
    plt.close(fig)

def _epochs(r):
    return range(1, len(r['train_loss_curve']) + 1)

def _plot_line(runs, key, title, xlabel, ylabel, category, name, suffix):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in runs:
        ax.plot(_epochs(r), r[key], label=r['name'])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(fontsize='small')
    _save_plot(fig, category, name, suffix)

def _plot_bar(labels, values, title, xlabel, ylabel, category, name, suffix):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    _save_plot(fig, category, name, suffix)

def plot_learning_rate_comparison(runs, title):
    name = title.replace(' ', '_')
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for r in runs:
        ax3.plot(_epochs(r), r['train_accuracy_curve'], label=r['name'])
    ax3.set_title('Mokymo tikslumas per epochas')
    ax3.set_xlabel('Epochos')
    ax3.set_ylabel('Tikslumas')
    ax3.grid(True)
    ax3.legend(fontsize='small')
    _save_plot(fig3, "learning_rate", name, "train_acc")


def plot_optimizer_comparison(runs, title, loss_threshold=0.3):
    name = title.replace(' ', '_')
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'optimizer', name, 'train_loss')
    _plot_line(runs, 'val_loss_curve', 'Validavimo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'optimizer', name, 'val_loss')
    _plot_line(runs, 'val_accuracy_curve', 'Validavimo tikslumas per epochas',
               'Epochos', 'Tikslumas', 'optimizer', name, 'val_acc')

def plot_scheduler_comparison(runs, title):
    name = title.replace(' ', '_')
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'scheduler', name, 'train_loss')
    _plot_line(runs, 'val_loss_curve', 'Validavimo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'scheduler', name, 'val_loss')
    _plot_line(runs, 'val_accuracy_curve', 'Validavimo tikslumas per epochas',
               'Epochos', 'Tikslumas', 'scheduler', name, 'val_acc')
    _plot_line(runs, 'lr_curve', 'Mokymosi greitis per epochas',
               'Epochos', 'Greičio koeficientas', 'scheduler', name, 'lr')

def plot_regularization_comparison(runs, title):
    name = title.replace(' ', '_')
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'regularization', name, 'train_loss')
    _plot_line(runs, 'val_loss_curve', 'Validavimo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'regularization', name, 'val_loss')
    _plot_line(runs, 'val_accuracy_curve', 'Validavimo tikslumas per epochas',
               'Epochos', 'Tikslumas', 'regularization', name, 'val_acc')
    labels = [r['name'] for r in runs]
    values = [
        sum(r['train_accuracy_curve'][i] - r['val_accuracy_curve'][i]
            for i in range(len(r['val_accuracy_curve'])))
        for r in runs
    ]
    _plot_bar(labels, values, 'Bendroji spraga mokymas–validavimas',
              'Regularizacija', 'Spraga', 'regularization', name, 'gap')

def plot_batch_size_comparison(runs, title, acc_target=0.9):
    name = title.replace(' ', '_')
    labels = [str(r['batch_size']) for r in runs]
    _plot_bar(labels, [r['training_time'] for r in runs],
              'Visas mokymo laikas (s)', 'Batšo dydis', 'Sekundės',
              'batch_size', name, 'time')
    _plot_bar(labels, [r['avg_samples_per_sec'] for r in runs],
              'Pralaidumas (pavyzdžių/s)', 'Batšo dydis', 'Pavyzdžiai/s',
              'batch_size', name, 'throughput')
    _plot_bar(labels, [r['avg_gpu_usage'] for r in runs],
              'Vidutinis GPU naudojimas (%)', 'Batšo dydis', 'Procentai',
              'batch_size', name, 'gpu')

def plot_architecture_comparison(runs, title, acc_target=0.9):
    name = title.replace(' ', '_')
    labels = [r['name'] for r in runs]
    _plot_line(runs, 'f1_score_curve', 'F1 rodiklis per epochas',
               'Epochos', 'F1 rodiklis', 'architecture', name, 'f1_score')
    fig, ax = plt.subplots(figsize=(8,6))
    params = [r['param_count'] for r in runs]
    accs = [r['test_accuracy'] for r in runs]
    ax.scatter(params, accs)
    for n,x,y in zip(labels, params, accs):
        ax.text(x,y,n,fontsize=8,ha='right')
    ax.set_title('Parametrų skaičius vs tikslumas')
    ax.set_xlabel('Parametrų skaičius')
    ax.set_ylabel('Tikslumas')
    ax.grid(True)
    _save_plot(fig, 'architecture', name, 'params')
    _plot_bar(labels, [r['inference_latency'] for r in runs],
              'Inferencijos vėlinimas (ms)', 'Architektūra', 'Milisekundės',
              'architecture', name, 'latency')
    values = [
        sum(r['epoch_time_curve'][: next((i for i,v in enumerate(r['val_accuracy_curve'],1) if v>=acc_target),
                                        len(r['val_accuracy_curve']))])
        for r in runs
    ]
    _plot_bar(labels, values, f'Laikas iki tikslumo ≥ {int(acc_target*100)}%',
              'Architektūra', 'Sekundės', 'architecture', name, 'time_to_acc')

def plot_test_accuracy(runs, title):
    name = title.replace(' ', '_')
    _plot_bar([r['name'] for r in runs], [r['test_accuracy'] for r in runs],
              'Testavimo tikslumas', 'Konfigūracija', 'Tikslumas',
              'test_accuracy', name, 'accuracy')

def plot_architecture_by_fold(folds_data, title):
    name = title.replace(' ', '_')
    _plot_line(folds_data, 'f1_score', 'F1 rodiklis per epochas',
                'Epochos', 'F1 rodiklis', 'architecture', name, 'f1_score')
    for key, t, y in [
        ('train_accuracy', 'Mokymo tikslumas per epochas', 'Tikslumas'),
        ('val_accuracy',   'Validavimo tikslumas per epochas', 'Tikslumas'),
        ('f1_score',       'Validavimo F1 per epochas',       'F1 rodiklis'),
    ]:
        fig, ax = plt.subplots(figsize=(8,6))
        epochs = range(1, len(folds_data[0][key])+1)
        for idx, f in enumerate(folds_data,1):
            ax.plot(epochs, f[key], label=f'Fold {idx}')
        ax.set_title(t)
        ax.set_xlabel('Epochos')
        ax.set_ylabel(y)
        ax.grid(True)
        ax.legend(fontsize='small')
        _save_plot(fig, 'architecture', name, key)
    plot_confusion_matrix(folds_data[-1]['model'], folds_data[-1]['test_loader'],
                          folds_data[-1]['device'], folds_data[-1]['classes'])

def plot_confusion_matrix(model, loader, device, classes):
    name = "confusion_matrix"
    fig, ax = plt.subplots(figsize=(8, 8))
    preds, labels = [], []
    model.eval(); model.to(device)
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = model(imgs).argmax(dim=1).cpu().numpy()
            preds.extend(out); labels.extend(lbls.numpy())
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Prognozuojama klasė')
    ax.set_ylabel('Tikroji klasė')
    ax.set_title('Sumaišties matrica')
    _save_plot(fig, "confusion", name, "cm")
