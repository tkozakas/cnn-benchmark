import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

os.makedirs("../test_data/plot", exist_ok=True)


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

    outpath = f"../test_data/plot/{title.replace(' ', '_')}_metrics.png"
    fig.savefig(outpath)
    plt.close(fig)


def plot_time(runs, title):
    names = [r['name'] for r in runs]
    times = [r['time'] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, times)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Configuration')
    plt.ylabel('Average Training Time (s)')
    plt.tight_layout()

    outpath = f"../test_data/plot/{title.replace(' ', '_')}_time.png"
    fig.savefig(outpath)
    plt.close(fig)


def plot_test_accuracy(runs, title):
    names = [r['name'] for r in runs]
    accs  = [r['test_accuracy'] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, accs)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Configuration')
    plt.ylabel('Average Test Accuracy')
    plt.tight_layout()

    outpath = f"../test_data/plot/{title.replace(' ', '_')}_accuracy.png"
    fig.savefig(outpath)
    plt.close(fig)


def plot_aggregated_learning_curves(all_results, metric_label, train_key, val_key):
    max_epochs = max(len(result[train_key]) for result in all_results)

    def pad_metric(metric, max_length):
        return metric + [metric[-1]] * (max_length - len(metric))

    train_values = np.array([pad_metric(result[train_key], max_epochs) for result in all_results])
    val_values   = np.array([pad_metric(result[val_key],   max_epochs) for result in all_results])

    train_mean = train_values.mean(axis=0)
    train_std  = train_values.std(axis=0)
    val_mean   = val_values.mean(axis=0)
    val_std    = val_values.std(axis=0)

    epochs = range(1, max_epochs + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_mean, label=f"Train {metric_label}")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(epochs, val_mean,   label=f"Validation {metric_label}")
    ax.fill_between(epochs, val_mean - val_std,   val_mean + val_std,   alpha=0.2)
    ax.set_title(f"Average {metric_label} Across Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()

    outpath = f"../test_data/plot/aggregated_{metric_label.replace(' ', '_')}.png"
    fig.savefig(outpath)
    plt.close(fig)


def plot_confusion_matrix(model, loader, device, classes):
    all_preds = []
    all_labels = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 7}, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    outpath = "../test_data/plot/confusion_matrix.png"
    fig.savefig(outpath)
    plt.close(fig)
