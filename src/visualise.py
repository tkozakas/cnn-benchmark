import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def show_performance_curve(
        results,
        train_key,
        val_key,
        metric_label=None,
        intersection_tol=1e-2
):
    train_values = results[train_key]
    val_values = results[val_key]

    intersection_indices = np.argwhere(
        np.isclose(train_values, val_values, atol=intersection_tol)
    ).flatten()

    # Plot the curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=f"Train {metric_label}")
    plt.plot(val_values, label=f"Validation {metric_label}")

    if len(intersection_indices) > 0:
        intersection_idx = intersection_indices[0]
        intersection_value = train_values[intersection_idx]

        plt.axvline(
            x=intersection_idx, color='r', linestyle='--',
            label='Intersection'
        )
        plt.annotate(
            f'Value: {intersection_value:.4f}',
            xy=(intersection_idx, intersection_value),
            xytext=(intersection_idx + 0.5, intersection_value),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='green'),
            color='green'
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(model, loader, device, classes):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_metric_vs_epochs(ax, epochs_range, metric_data, label, title, xlabel, ylabel):
    ax.plot(epochs_range, metric_data, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)


def plot_bar_chart(ax, x_values, heights, title, xlabel, ylabel, bar_width=0.5):
    ax.bar(x_values, heights, color='skyblue', width=bar_width, align='center')

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, rotation=45, ha='right')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def plot_results(results):
    show_performance_curve(results, train_key="train_accuracy", val_key="val_accuracy", metric_label="Accuracy")
    show_performance_curve(results, train_key="train_loss", val_key="val_loss", metric_label="Loss")


def print_test_results(results, title="test"):
    headers = ["Parameter", "Test Loss", "Test Accuracy", "Time", "Epochs"]

    row_format = (
        "{:<15} "  # Parameter
        "{:<10} "  # Test Loss
        "{:<10} "  # Test Accuracy
        "{:<10} "  # Test Time
        "{:<10}"   # Epochs
    )

    print(row_format.format(*headers))
    print("-" * 60)

    table_data = []

    for param, config_results in results.items():
        row = [
            param,
            f"{config_results['test_loss']:.4f}",
            f"{config_results['test_accuracy']:.4f}",
            f"{config_results['elapsed_time']:.2f}",
            f"{config_results['epoch_count']}"
        ]
        table_data.append(row)
        print(row_format.format(*row))

    save_test_results_to_csv(f"{title}_test_results.csv", headers, table_data)

def save_test_results_to_csv(file_name, headers, table_data):
    directory = os.path.join(os.path.dirname(os.getcwd()), "test_data")
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(table_data)

def print_results_table(results):
    num_epochs = len(results["train_loss"])

    headers = [
        "Epoch",
        "Train Loss",
        "Train Acc",
        "Val Loss",
        "Val Acc",
    ]

    row_format = (
        "{:<5}  "  # Epoch
        "{:<10} "  # Train Loss
        "{:<10} "  # Train Acc
        "{:<10} "  # Val Loss
        "{:<10} "  # Val Acc
    )

    print(row_format.format(*headers))
    print("-" * 60)

    for epoch_idx in range(num_epochs):
        print(
            row_format.format(
                epoch_idx + 1,
                f"{results['train_loss'][epoch_idx]:.4f}",
                f"{results['train_accuracy'][epoch_idx]:.4f}"
                f"{results['val_loss'][epoch_idx]:.4f}",
                f"{results['val_accuracy'][epoch_idx]:.4f}"
            )
        )


def save_results_to_csv(file_name, results, additional_fields=None):
    directory = os.path.join(os.path.dirname(os.getcwd()), "test_data")
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)

    headers = [
        "Epoch",
        "Train Loss",
        "Train Accuracy"
        "Validation Loss",
        "Validation Accuracy"
    ]

    if additional_fields:
        headers += list(additional_fields.keys())

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for epoch in range(len(results["train_loss"])):
            row = [
                epoch + 1,
                f"{results['train_loss'][epoch]:.4f}",
                f"{results['train_accuracy'][epoch]:.4f}",
                f"{results['val_loss'][epoch]:.4f}",
                f"{results['val_accuracy'][epoch]:.4f}",
            ]
            if additional_fields:
                row += [additional_fields[key] for key in additional_fields]
            writer.writerow(row)
