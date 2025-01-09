"""
Usage:
    experiment.py [--architecture=ARCH]

Options:
    --architecture=ARCH   Specify the architecture to use [default: EmnistCNN].
"""

import time

import matplotlib.pyplot as plt
import torch
from docopt import docopt
from torch import optim, nn

from src.config import train_config
from src.model import get_model
from src.train import train, load_emnist_data
from src.visualise import plot_results, print_results_table, save_results_to_csv, plot_bar_chart, plot_metric_vs_epochs, print_test_results


def epoch_test(architecture, device, loaders, criterion, epochs):
    model = get_model(architecture).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    results = train(model, loaders, criterion, optimizer, device, epochs)

    save_results_to_csv(f"epoch_test_results.csv", results)

    plot_results(results=results)
    print_results_table(results)


def learning_rate_test(architecture, device, loaders, criterion, epochs):
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    results = {}
    time_results = {}

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        model = get_model(architecture).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        start_time = time.time()
        lr_results = train(model, loaders, criterion, optimizer, device, epochs)
        elapsed_time = time.time() - start_time

        results[lr] = lr_results
        time_results[lr] = elapsed_time

        save_results_to_csv(f"learning_rate_{lr}_results.csv", lr_results, {"Learning Rate": lr})

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for lr in learning_rates:
        epoch_accuracy = results[lr]['val_accuracy']
        epoch_loss = results[lr]['val_loss']
        epochs_range = range(1, len(epoch_accuracy) + 1)

        plot_metric_vs_epochs(
            axs[0, 0],
            epochs_range,
            epoch_accuracy,
            label=f"LR={lr}",
            title="Validation Accuracy vs. Epochs for Learning Rates",
            xlabel="Epochs",
            ylabel="Validation Accuracy"
        )

        plot_metric_vs_epochs(
            axs[0, 1],
            epochs_range,
            epoch_loss,
            label=f"LR={lr}",
            title="Validation Loss vs. Epochs for Learning Rates",
            xlabel="Epochs",
            ylabel="Validation Loss"
        )

    plot_bar_chart(
        axs[1, 0],
        [str(lr) for lr in learning_rates],
        time_results.values(),
        title="Training Speed for Different Learning Rates",
        xlabel="Learning Rate",
        ylabel="Training Time (seconds)"
    )

    print_test_results(results)

    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

def configuration_test(architecture, device, criterion, epochs):
    configurations = [
        {"learning_rate": 0.0005, "train_batch_size": 64},
        {"learning_rate": 0.001, "train_batch_size": 64},
        {"learning_rate": 0.01, "train_batch_size": 64},
        {"learning_rate": 0.0005, "train_batch_size": 256},
        {"learning_rate": 0.001, "train_batch_size": 256},
        {"learning_rate": 0.01, "train_batch_size": 256}
    ]

    results = {}
    time_results = {}

    for config in configurations:
        print(f"\nTesting configuration: {config}")
        model = get_model(architecture).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        loaders = load_emnist_data(
            train_config["emnist_type"],
            config["train_batch_size"],
            train_config["subsample_size"],
            train_config["cpu_workers"]
        )

        start_time = time.time()
        config_results = train(model, loaders, criterion, optimizer, device, epochs)
        elapsed_time = time.time() - start_time

        results[str(config)] = config_results
        time_results[str(config)] = elapsed_time

        save_results_to_csv(f"config_{config}_results.csv", config_results, {"Configuration": str(config)})

    config_labels = [f"LR={config['learning_rate']}, BS={config['train_batch_size']}" for config in configurations]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for config, label in zip(configurations, config_labels):
        config_key = str(config)
        epoch_accuracy = results[config_key]['val_accuracy']
        epoch_loss = results[config_key]['val_loss']
        epochs_range = range(1, len(epoch_accuracy) + 1)

        plot_metric_vs_epochs(
            axs[0, 0],
            epochs_range,
            epoch_accuracy,
            label=label,
            title="Validation Accuracy vs. Epochs for Configurations",
            xlabel="Epochs",
            ylabel="Validation Accuracy"
        )

        plot_metric_vs_epochs(
            axs[0, 1],
            epochs_range,
            epoch_loss,
            label=label,
            title="Validation Loss vs. Epochs for Configurations",
            xlabel="Epochs",
            ylabel="Validation Loss"
        )

    plot_bar_chart(
        axs[1, 0],
        config_labels,
        time_results.values(),
        title="Training Speed for Different Configurations",
        xlabel="Configuration (LR, BS)",
        ylabel="Training Time (seconds)"
    )

    print_test_results(results)

    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def batch_size_test(architecture, device, criterion, epochs):
    batch_sizes = [16, 32, 64, 128, 256]
    results = {}
    speed_results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        loaders = load_emnist_data(
            train_config["emnist_type"],
            batch_size,
            train_config["subsample_size"],
            train_config["cpu_workers"]
        )

        model = get_model(architecture).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])

        start_time = time.time()
        batch_results = train(model, loaders, criterion, optimizer, device, epochs)
        elapsed_time = time.time() - start_time

        results[batch_size] = batch_results
        speed_results[batch_size] = elapsed_time

        save_results_to_csv(f"batch_size_{batch_size}_results.csv", batch_results, {"Batch Size": batch_size})

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for batch_size in batch_sizes:
        epoch_accuracy = results[batch_size]['val_accuracy']
        epoch_loss = results[batch_size]['val_loss']
        epochs_range = range(1, len(epoch_accuracy) + 1)

        plot_metric_vs_epochs(
            axs[0, 0],
            epochs_range,
            epoch_accuracy,
            label=f"Batch={batch_size}",
            title="Validation Accuracy vs. Epochs for Batch Sizes",
            xlabel="Epochs",
            ylabel="Validation Accuracy"
        )

        plot_metric_vs_epochs(
            axs[0, 1],
            epochs_range,
            epoch_loss,
            label=f"Batch={batch_size}",
            title="Validation Loss vs. Epochs for Batch Sizes",
            xlabel="Epochs",
            ylabel="Validation Loss"
        )

    print(speed_results)
    plot_bar_chart(
        axs[1, 0],
        [str(batch_size) for batch_size in batch_sizes],
        speed_results.values(),
        title="Training Speed for Different Batch Sizes",
        xlabel="Batch Size",
        ylabel="Training Time (seconds)"
    )

    print_test_results(results)

    # Hide unused subplot
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def configure_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using {torch.cuda.device_count()} GPU(s)")

    loaders = load_emnist_data(
        train_config["emnist_type"],
        train_config["train_batch_size"],
        train_config["subsample_size"],
        train_config["cpu_workers"]
    )
    criterion = nn.CrossEntropyLoss()
    return device, loaders, criterion


def main(architecture):
    test_epochs = 2
    device, loaders, criterion = configure_test()
    print(f"\n--- Testing {architecture} model ---")
    epoch_test(architecture, device, loaders, criterion, test_epochs)

    print("\n--- Testing different hyperparameters ---")
    learning_rate_test(architecture, device, loaders, criterion, test_epochs)

    print("\n--- Testing different batch sizes ---")
    batch_size_test(architecture, device, criterion, test_epochs)

    print("\n--- Testing different configurations for training ---")
    configuration_test(architecture, device, criterion, test_epochs)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
