import matplotlib.pyplot as plt
import numpy as np


def show_performance_curve(
        results,
        train_key,
        val_key,
        metric_label=None,
        intersection_tol=1e-2
):
    train_values = results[train_key]
    val_values = results[val_key]

    # Find intersection index (if any) using np.isclose
    intersection_indices = np.argwhere(
        np.isclose(train_values, val_values, atol=intersection_tol)
    ).flatten()

    # Plot the curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=f"Train {metric_label}")
    plt.plot(val_values, label=f"Validation {metric_label}")

    # If an intersection is found, mark the first one
    if len(intersection_indices) > 0:
        intersection_idx = intersection_indices[0]
        intersection_value = train_values[intersection_idx]

        plt.axvline(
            x=intersection_idx, color='r', linestyle='--',
            label='Intersection'
        )
        plt.annotate(
            f'Optimal Value: {intersection_value:.4f}',
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


def plot_results(results):
    show_performance_curve(results, train_key="epoch_accuracy", val_key="val_accuracy", metric_label="Accuracy")
    show_performance_curve(results, train_key="epoch_loss", val_key="val_loss", metric_label="Loss")
    show_performance_curve(results, train_key="epoch_precision", val_key="val_precision", metric_label="Precision")


def print_results_table(results):
    num_epochs = len(results["epoch_loss"])

    headers = [
        "Epoch",
        "Train Loss",
        "Train Acc",
        "Val Loss",
        "Val Acc"
    ]

    row_format = (
        "{:<5}  "  # Epoch
        "{:<10} "  # Train Loss
        "{:<10} "  # Train Acc
        "{:<10} "  # Val Loss
        "{:<10}"  # Val Acc
    )

    print(row_format.format(*headers))
    print("-" * 45)

    for epoch_idx in range(num_epochs):
        print(
            row_format.format(
                epoch_idx + 1,
                f"{results['epoch_loss'][epoch_idx]:.4f}",
                f"{results['epoch_accuracy'][epoch_idx]:.4f}",
                f"{results['val_loss'][epoch_idx]:.4f}",
                f"{results['val_accuracy'][epoch_idx]:.4f}"
            )
        )
