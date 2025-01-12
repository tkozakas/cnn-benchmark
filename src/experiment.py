"""
Usage:
    experiment.py [--architecture=ARCH]

Options:
    --architecture=ARCH   Specify the architecture to use [default: EmnistCNN_16_64_128].
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from docopt import docopt
from torchvision import datasets

from src.config import train_config
from src.model import get_model
from src.train import k_fold_cross_validation, transform
from src.visualise import plot_bar_chart


def train_configuration_test(architecture, dataset, epochs, k_folds, batch_sizes, learning_rates):
    configurations = [
        {"learning_rate": lr, "train_batch_size": bs}
        for lr in learning_rates
        for bs in batch_sizes
    ]

    results = []
    train_times = []

    all_val_accuracies = {}
    all_val_losses = {}

    for config in configurations:
        print(f"\n--- Testing Configuration: LR={config['learning_rate']}, Batch Size={config['train_batch_size']} ---")
        all_results, avg_results = k_fold_cross_validation(
            architecture=architecture,
            dataset=dataset,
            model_fn=lambda: get_model(architecture),
            k_folds=k_folds,
            epochs=epochs,
            learning_rate=config["learning_rate"],
            batch_size=config["train_batch_size"],
            random_state=42,
        )

        config_key = f"LR={config['learning_rate']}, BS={config['train_batch_size']}"
        avg_results["Configuration"] = config_key
        results.append(avg_results)

        train_times.append(avg_results["elapsed_time"])

        val_acc = avg_results["val_accuracy"]
        val_loss = avg_results["val_loss"]

        all_val_accuracies[config_key] = val_acc
        all_val_losses[config_key] = val_loss

    plt.figure(figsize=(10, 6))
    for config_key, val_acc in all_val_accuracies.items():
        actual_epochs = len(val_acc)
        epochs_range = range(1, actual_epochs + 1)
        plt.plot(epochs_range, val_acc, label=config_key)
    plt.title("Validation Accuracy Across Configurations")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Configurations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for config_key, val_loss in all_val_losses.items():
        actual_epochs = len(val_loss)
        epochs_range = range(1, actual_epochs + 1)
        plt.plot(epochs_range, val_loss, label=config_key)
    plt.title("Validation Loss Across Configurations")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Configurations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    config_labels = [f"LR={config['learning_rate']}, BS={config['train_batch_size']}" for config in configurations]
    plt.figure(figsize=(10, 6))
    plot_bar_chart(
        plt.gca(),
        x_values=config_labels,
        heights=train_times,
        title="Training Time for Configurations",
        xlabel="Configuration (LR, BS)",
        ylabel="Training Time (seconds)"
    )
    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(results)
    results_file = "../test_data/all_configuration_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    return results


def model_config_test(datasets, epochs, k_folds, batch_size, learning_rate):
    model_architectures = ["EmnistCNN_16_64_128", "EmnistCNN_32_128_256", "EmnistCNN_8_32_64"]
    results = []

    for dataset_name, dataset in datasets.items():
        print(f"\n--- Dataset: {dataset_name} ---")

        all_val_accuracies = {}
        all_val_losses = {}
        dataset_train_times = []

        for architecture in model_architectures:
            print(f"\n--- Testing Model: {architecture} on {dataset_name} ---")
            all_results, avg_results = k_fold_cross_validation(
                architecture=architecture,
                dataset=dataset,
                model_fn=lambda: get_model(architecture),
                k_folds=k_folds,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_state=42,
            )

            avg_results["Architecture"] = architecture
            avg_results["Dataset"] = dataset_name
            results.append(avg_results)

            dataset_train_times.append(avg_results["elapsed_time"])

            val_acc = avg_results["val_accuracy"]
            val_loss = avg_results["val_loss"]

            all_val_accuracies[architecture] = val_acc
            all_val_losses[architecture] = val_loss

        plt.figure(figsize=(10, 6))
        for architecture, val_acc in all_val_accuracies.items():
            actual_epochs = len(val_acc)
            epochs_range = range(1, actual_epochs + 1)
            plt.plot(epochs_range, val_acc, label=architecture)
        plt.title(f"Validation Accuracy Across Models on {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Models")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        for architecture, val_loss in all_val_losses.items():
            actual_epochs = len(val_loss)
            epochs_range = range(1, actual_epochs + 1)
            plt.plot(epochs_range, val_loss, label=architecture)
        plt.title(f"Validation Loss Across Models on {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Models")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plot_bar_chart(
            plt.gca(),
            x_values=model_architectures,
            heights=dataset_train_times,
            title=f"Training Time for Models on {dataset_name}",
            xlabel="Model Architecture",
            ylabel="Training Time (seconds)"
        )
        plt.tight_layout()
        plt.show()

    results_df = pd.DataFrame(results)
    results_file = "../test_data/all_model_config_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    return results


def main(architecture):
    test_epochs = 3

    print("\n--- Testing different configurations for training ---")
    train_configuration_test(
        architecture=architecture,
        dataset=get_subsample(datasets.EMNIST(
            root="../data",
            split=train_config["emnist_type"],
            train=True,
            download=True,
            transform=transform
        )),
        epochs=test_epochs,
        k_folds=train_config["k_folds"],
        batch_sizes=[32, 64, 128],
        learning_rates=[0.1, 0.01, 0.001, 0.0001]
    )

    print("\n--- Testing different model configurations ---")
    model_config_test(
        datasets={
            "letters": get_subsample(datasets.EMNIST(
                root="../data",
                split="letters",
                train=True,
                download=True,
                transform=transform
            )),
            "digits": get_subsample(datasets.EMNIST(
                root="../data",
                split="digits",
                train=True,
                download=True,
                transform=transform
            )),
            "balanced": get_subsample(datasets.EMNIST(
                root="../data",
                split="balanced",
                train=True,
                download=True,
                transform=transform
            ))
        },
        epochs=test_epochs,
        k_folds=train_config["k_folds"],
        batch_size=train_config["train_batch_size"],
        learning_rate=train_config["learning_rate"]
    )


def get_subsample(full_dataset):
    if train_config["subsample_size"]:
        print(f"Subsampling dataset to {train_config['subsample_size']} samples...")
        subsample_size = train_config["subsample_size"]
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [subsample_size, len(full_dataset) - subsample_size]
        )
    return full_dataset

if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
