import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

os.makedirs("../test_data/plot", exist_ok=True)


def plot_optimizer_comparison(runs, title, loss_threshold=0.2):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_train_loss, ax_val_loss, ax_val_acc, ax_speed = axes.flatten()

    for run in runs:
        epochs = range(1, len(run['train_loss_curve']) + 1)
        ax_train_loss.plot(epochs, run['train_loss_curve'], label=f"{run['name']} Train Loss")
        ax_val_loss.plot(epochs, run['val_loss_curve'], label=f"{run['name']} Val Loss")
        ax_val_acc.plot(epochs, run['val_accuracy_curve'], label=f"{run['name']} Val Acc")

    # Configure loss vs epoch
    ax_train_loss.set_title('Training Loss vs Epoch')
    ax_train_loss.set_xlabel('Epoch')
    ax_train_loss.set_ylabel('Loss')
    ax_train_loss.grid(True)
    ax_train_loss.legend(fontsize='small')

    # Configure validation loss vs epoch
    ax_val_loss.set_title('Validation Loss vs Epoch')
    ax_val_loss.set_xlabel('Epoch')
    ax_val_loss.set_ylabel('Loss')
    ax_val_loss.grid(True)
    ax_val_loss.legend(fontsize='small')

    # Configure validation accuracy vs epoch
    ax_val_acc.set_title('Validation Accuracy vs Epoch')
    ax_val_acc.set_xlabel('Epoch')
    ax_val_acc.set_ylabel('Accuracy')
    ax_val_acc.grid(True)
    ax_val_acc.legend(fontsize='small')

    # Compute epochs to threshold for each optimizer run
    names = []
    epochs_to_thresh = []
    for run in runs:
        try:
            idx = next(i for i, v in enumerate(run['val_loss_curve'], start=1) if v <= loss_threshold)
        except StopIteration:
            idx = len(run['val_loss_curve'])
        names.append(run['name'])
        epochs_to_thresh.append(idx)

    # Bar chart for speed to threshold
    ax_speed.bar(names, epochs_to_thresh)
    ax_speed.set_title(f'Epochs to reach Val Loss ≤ {loss_threshold}')
    ax_speed.set_xlabel('Optimizer')
    ax_speed.set_ylabel('Epochs')
    ax_speed.grid(axis='y', linestyle='--', alpha=0.7)

    # Main title and layout
    fig.suptitle(f"{title} — Optimizer Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and show
    outpath = f"../test_data/plot/{title.replace(' ', '_')}_optimizer_comparison.png"
    fig.savefig(outpath)
    plt.show()


def plot_scheduler_comparison(runs, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_train_loss, ax_val_loss, ax_val_acc, ax_lr = axes.flatten()

    for run in runs:
        epochs = range(1, len(run['train_loss_curve']) + 1)
        ax_train_loss.plot(epochs, run['train_loss_curve'], label=run['name'])
        ax_val_loss.plot(epochs, run['val_loss_curve'], label=run['name'])
        ax_val_acc.plot(epochs, run['val_accuracy_curve'], label=run['name'])
        ax_lr.plot(epochs, run['lr_curve'], label=run['name'])

    ax_train_loss.set_title('Training Loss vs Epoch')
    ax_train_loss.set_xlabel('Epoch')
    ax_train_loss.set_ylabel('Loss')
    ax_train_loss.grid(True)
    ax_train_loss.legend(fontsize='small')

    ax_val_loss.set_title('Validation Loss vs Epoch')
    ax_val_loss.set_xlabel('Epoch')
    ax_val_loss.set_ylabel('Loss')
    ax_val_loss.grid(True)
    ax_val_loss.legend(fontsize='small')

    ax_val_acc.set_title('Validation Accuracy vs Epoch')
    ax_val_acc.set_xlabel('Epoch')
    ax_val_acc.set_ylabel('Accuracy')
    ax_val_acc.grid(True)
    ax_val_acc.legend(fontsize='small')

    ax_lr.set_title('Learning Rate vs Epoch')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.grid(True)
    ax_lr.legend(fontsize='small')

    fig.suptitle(f"{title} — Scheduler Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = f"../test_data/plot/{title.replace(' ', '_')}_scheduler_comparison.png"
    fig.savefig(outpath)
    plt.show()


def plot_regularization_comparison(runs, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_train_loss, ax_val_loss, ax_val_acc, ax_gap = axes.flatten()

    for run in runs:
        epochs = range(1, len(run['train_loss_curve']) + 1)
        ax_train_loss.plot(epochs, run['train_loss_curve'], label=run['name'])
        ax_val_loss.plot(epochs, run['val_loss_curve'], label=run['name'])
        ax_val_acc.plot(epochs, run['val_accuracy_curve'], label=run['name'])
        gap = [v - t for t, v in zip(run['train_accuracy_curve'], run['val_accuracy_curve'])]
        ax_gap.plot(epochs, gap, label=run['name'])

    ax_train_loss.set_title('Training Loss vs Epoch')
    ax_train_loss.set_xlabel('Epoch')
    ax_train_loss.set_ylabel('Loss')
    ax_train_loss.grid(True)
    ax_train_loss.legend(fontsize='small')

    ax_val_loss.set_title('Validation Loss vs Epoch')
    ax_val_loss.set_xlabel('Epoch')
    ax_val_loss.set_ylabel('Loss')
    ax_val_loss.grid(True)
    ax_val_loss.legend(fontsize='small')

    ax_val_acc.set_title('Validation Accuracy vs Epoch')
    ax_val_acc.set_xlabel('Epoch')
    ax_val_acc.set_ylabel('Accuracy')
    ax_val_acc.grid(True)
    ax_val_acc.legend(fontsize='small')

    ax_gap.set_title('Generalization Gap (Val Acc - Train Acc)')
    ax_gap.set_xlabel('Epoch')
    ax_gap.set_ylabel('Accuracy Gap')
    ax_gap.grid(True)
    ax_gap.legend(fontsize='small')

    fig.suptitle(f"{title} — Regularization Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = f"../test_data/plot/{title.replace(' ', '_')}_regularization_comparison.png"
    fig.savefig(outpath)
    plt.show()

def plot_batch_size_comparison(runs, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_time, ax_throughput, ax_gpu, ax_acc = axes.flatten()

    names = [str(run['batch_size']) for run in runs]
    times = [run['training_time'] for run in runs]
    throughputs = [run['avg_samples_per_sec'] for run in runs]
    gpu_usages = [run['avg_gpu_usage'] for run in runs]
    final_accs = [run['test_accuracy'] for run in runs]

    ax_time.bar(names, times)
    ax_time.set_title('Total Training Time (s)')
    ax_time.set_xlabel('Batch Size')
    ax_time.set_ylabel('Seconds')
    ax_time.grid(axis='y', linestyle='--', alpha=0.7)

    ax_throughput.bar(names, throughputs)
    ax_throughput.set_title('Throughput (samples/sec)')
    ax_throughput.set_xlabel('Batch Size')
    ax_throughput.set_ylabel('Samples/sec')
    ax_throughput.grid(axis='y', linestyle='--', alpha=0.7)

    ax_gpu.bar(names, gpu_usages)
    ax_gpu.set_title('Average GPU Utilization (%)')
    ax_gpu.set_xlabel('Batch Size')
    ax_gpu.set_ylabel('Utilization')
    ax_gpu.grid(axis='y', linestyle='--', alpha=0.7)

    ax_acc.bar(names, final_accs)
    ax_acc.set_title('Final Test Accuracy')
    ax_acc.set_xlabel('Batch Size')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(f"{title} — Batch Size Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = f"../test_data/plot/{title.replace(' ', '_')}_batch_size_comparison.png"
    fig.savefig(outpath)
    plt.show()


def plot_learning_rate_comparison(runs, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_train_loss, ax_val_loss, ax_val_acc, ax_lr = axes.flatten()

    for run in runs:
        epochs = range(1, len(run['train_loss_curve']) + 1)
        ax_train_loss.plot(epochs, run['train_loss_curve'], label=run['name'])
        ax_val_loss.plot(epochs, run['val_loss_curve'],   label=run['name'])
        ax_val_acc.plot(epochs, run['val_accuracy_curve'], label=run['name'])
        ax_lr.plot(epochs, run['lr_curve'],                  label=run['name'])

    ax_train_loss.set_title('Training Loss vs Epoch')
    ax_train_loss.set_xlabel('Epoch')
    ax_train_loss.set_ylabel('Loss')
    ax_train_loss.grid(True)
    ax_train_loss.legend(fontsize='small')

    ax_val_loss.set_title('Validation Loss vs Epoch')
    ax_val_loss.set_xlabel('Epoch')
    ax_val_loss.set_ylabel('Loss')
    ax_val_loss.grid(True)
    ax_val_loss.legend(fontsize='small')

    ax_val_acc.set_title('Validation Accuracy vs Epoch')
    ax_val_acc.set_xlabel('Epoch')
    ax_val_acc.set_ylabel('Accuracy')
    ax_val_acc.grid(True)
    ax_val_acc.legend(fontsize='small')

    ax_lr.set_title('Learning Rate vs Epoch')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.grid(True)
    ax_lr.legend(fontsize='small')

    fig.suptitle(f"{title} — Learning Rate Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = f"../test_data/plot/{title.replace(' ', '_')}_lr_comparison.png"
    fig.savefig(outpath)
    plt.show()

# Architecture comparison visualization
def plot_architecture_comparison(runs, title, acc_threshold=None):
    """
    Visualize final architecture performance:
    1) Test Accuracy (bar chart)
    2) Params vs Test Accuracy (scatter plot)
    3) Inference Latency (bar chart)
    4) Training Time to reach acc_threshold (bar chart)
    If acc_threshold is None, uses total training_time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_acc, ax_params, ax_latency, ax_time = axes.flatten()

    names = [run['name'] for run in runs]
    accuracies = [run['test_accuracy'] for run in runs]
    params = [run.get('param_count') for run in runs]
    latencies = [run.get('inference_latency') for run in runs]
    # compute time to threshold or total training time
    times = []
    for run in runs:
        if acc_threshold is not None and 'val_accuracy_curve' in run:
            # find first epoch reaching threshold
            try:
                idx = next(i for i, v in enumerate(run['val_accuracy_curve'], start=1) if v >= acc_threshold)
                # sum epoch_time_curve up to idx
                t = sum(run.get('epoch_time_curve', [])[:idx])
            except StopIteration:
                t = sum(run.get('epoch_time_curve', []))
        else:
            t = run.get('training_time')
        times.append(t)

    # 1) Test Accuracy
    ax_acc.bar(names, accuracies)
    ax_acc.set_title('Final Test Accuracy')
    ax_acc.set_xlabel('Architecture')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax_acc.get_xticklabels(), rotation=45, ha='right')

    # 2) Params vs Accuracy scatter
    ax_params.scatter(params, accuracies)
    for name, x, y in zip(names, params, accuracies):
        ax_params.text(x, y, name, fontsize=8, ha='right')
    ax_params.set_title('Params vs Test Accuracy')
    ax_params.set_xlabel('Parameter Count')
    ax_params.set_ylabel('Accuracy')
    ax_params.grid(True)

    # 3) Inference Latency
    ax_latency.bar(names, latencies)
    ax_latency.set_title('Inference Latency (ms)')
    ax_latency.set_xlabel('Architecture')
    ax_latency.set_ylabel('Latency (ms)')
    ax_latency.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax_latency.get_xticklabels(), rotation=45, ha='right')

    # 4) Training Time to X% or total
    xlabel = f"Time to reach {acc_threshold*100:.1f}% Acc" if acc_threshold else 'Total Training Time (s)'
    ax_time.bar(names, times)
    ax_time.set_title(xlabel)
    ax_time.set_xlabel('Architecture')
    ax_time.set_ylabel('Seconds')
    ax_time.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax_time.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(f"{title} — Architecture Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = f"../test_data/plot/{title.replace(' ', '_')}_architecture_comparison.png"
    fig.savefig(outpath)
    plt.show()

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
    plt.show()


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
    plt.show()


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
    plt.show()
