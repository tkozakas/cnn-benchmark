import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def plot_aggregated_learning_curves(all_results, metric_label, train_key, val_key):
    max_epochs = max(len(result[train_key]) for result in all_results)

    def pad_metric(metric, max_length):
        return metric + [metric[-1]] * (max_length - len(metric))

    train_values = np.array([pad_metric(result[train_key], max_epochs) for result in all_results])
    val_values = np.array([pad_metric(result[val_key], max_epochs) for result in all_results])

    train_mean = train_values.mean(axis=0)
    train_std = train_values.std(axis=0)
    val_mean = val_values.mean(axis=0)
    val_std = val_values.std(axis=0)

    epochs = range(1, max_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mean, label=f"Train {metric_label}", color='blue')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)

    plt.plot(epochs, val_mean, label=f"Validation {metric_label}", color='orange')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)

    plt.title(f"Average Learning Progress of {metric_label} Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(metric_label)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 7})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_bar_chart(ax, x_values, heights, title, xlabel, ylabel, bar_width=0.5):
    ax.bar(x_values, heights, color='skyblue', width=bar_width, align='center')

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, rotation=45, ha='right')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def plot_model_diffs(results, dataset_name):
    plt.figure(figsize=(10, 6))

    for architecture, data in results.items():
        val_accuracy = data["val_accuracy"]
        epochs_range = range(1, len(val_accuracy) + 1)
        plt.plot(epochs_range, val_accuracy, label=architecture)

    plt.title(f"Validation Accuracy vs. Epochs for Different Architectures on {dataset_name} Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend(title="Architecture")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
