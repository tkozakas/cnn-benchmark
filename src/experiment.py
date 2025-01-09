"""
Usage:
    experiment.py [--architecture=ARCH]

Options:
    --architecture=ARCH   Specify the architecture to use [default: EmnistCNN_16_64_128].
"""

import matplotlib.pyplot as plt
import torch
from docopt import docopt
from torch import optim, nn

from src.config import train_config
from src.model import get_model
from src.train import train, load_emnist_data
from src.visualise import plot_results, print_results_table, save_results_to_csv, plot_bar_chart, plot_metric_vs_epochs, print_test_results, \
    plot_model_diffs


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

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        model = get_model(architecture).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        lr_results = train(model, loaders, criterion, optimizer, device, epochs)

        results[lr] = lr_results
        save_results_to_csv(f"learning_rate_{lr}_results.csv", lr_results, {"Learning Rate": lr})

    batch_times = [results[lr]['elapsed_time'] for lr in learning_rates]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for lr in learning_rates:
        train_accuracy = results[lr]['val_accuracy']
        train_loss = results[lr]['val_loss']
        epochs_range = range(1, len(train_accuracy) + 1)

        plot_metric_vs_epochs(
            axs[0, 0],
            epochs_range,
            train_accuracy,
            label=f"LR={lr}",
            title="Validation Accuracy vs. Epochs for Learning Rates",
            xlabel="Epochs",
            ylabel="Validation Accuracy"
        )

        plot_metric_vs_epochs(
            axs[0, 1],
            epochs_range,
            train_loss,
            label=f"LR={lr}",
            title="Validation Loss vs. Epochs for Learning Rates",
            xlabel="Epochs",
            ylabel="Validation Loss"
        )

    plot_bar_chart(
        axs[1, 0],
        [str(lr) for lr in learning_rates],
        batch_times,
        title="Training Speed for Different Learning Rates",
        xlabel="Learning Rate",
        ylabel="Training Time (seconds)"
    )

    print_test_results(results, title="learning_rate")

    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def configuration_test(architecture, device, criterion, epochs):
    configurations = [
        {"learning_rate": 0.001, "train_batch_size": 16},
        {"learning_rate": 0.01, "train_batch_size": 16},
        {"learning_rate": 0.001, "train_batch_size": 64},
        {"learning_rate": 0.01, "train_batch_size": 64},
        {"learning_rate": 0.001, "train_batch_size": 128},
        {"learning_rate": 0.01, "train_batch_size": 128},
    ]

    results = {}

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
        config_results = train(model, loaders, criterion, optimizer, device, epochs)

        config_key = str(config)
        results[config_key] = config_results

        save_results_to_csv(f"config_{config_key}_results.csv", config_results, {"Configuration": config_key})

    config_labels = [f"LR={config['learning_rate']}, BS={config['train_batch_size']}" for config in configurations]
    batch_times = [results[str(config)]['elapsed_time'] for config in configurations]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for config, label in zip(configurations, config_labels):
        config_key = str(config)
        train_accuracy = results[config_key]['val_accuracy']
        train_loss = results[config_key]['val_loss']
        epochs_range = range(1, len(train_accuracy) + 1)

        plot_metric_vs_epochs(
            axs[0, 0],
            epochs_range,
            train_accuracy,
            label=label,
            title="Validation Accuracy vs. Epochs for Configurations",
            xlabel="Epochs",
            ylabel="Validation Accuracy"
        )

        plot_metric_vs_epochs(
            axs[0, 1],
            epochs_range,
            train_loss,
            label=label,
            title="Validation Loss vs. Epochs for Configurations",
            xlabel="Epochs",
            ylabel="Validation Loss"
        )

    # Plot bar chart for training times
    plot_bar_chart(
        axs[1, 0],
        config_labels,
        batch_times,
        title="Training Speed for Different Configurations",
        xlabel="Configuration (LR, BS)",
        ylabel="Training Time (seconds)"
    )

    print_test_results(results, title="configuration")

    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def batch_size_test(architecture, device, criterion, epochs):
    batch_sizes = [16, 32, 64, 128, 256]
    results = {}

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
        batch_results = train(model, loaders, criterion, optimizer, device, epochs)

        results[batch_size] = batch_results
        save_results_to_csv(f"batch_size_{batch_size}_results.csv", batch_results, {"Batch Size": batch_size})

    batch_times = [results[batch_size]['elapsed_time'] for batch_size in batch_sizes]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for batch_size in batch_sizes:
        train_accuracy = results[batch_size]['val_accuracy']
        train_loss = results[batch_size]['val_loss']
        epochs_range = range(1, len(train_accuracy) + 1)

        plot_metric_vs_epochs(
            axs[0, 0],
            epochs_range,
            train_accuracy,
            label=f"Batch={batch_size}",
            title="Validation Accuracy vs. Epochs for Batch Sizes",
            xlabel="Epochs",
            ylabel="Validation Accuracy"
        )

        plot_metric_vs_epochs(
            axs[0, 1],
            epochs_range,
            train_loss,
            label=f"Batch={batch_size}",
            title="Validation Loss vs. Epochs for Batch Sizes",
            xlabel="Epochs",
            ylabel="Validation Loss"
        )

    plot_bar_chart(
        axs[1, 0],
        batch_sizes,
        batch_times,
        title="Training Speed for Different Batch Sizes",
        xlabel="Batch Size",
        ylabel="Training Time (seconds)",
        bar_width=6
    )

    print_test_results(results, title="batch_size")

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


def model_config_test(device, loaders, criterion, epochs):
    architectures = ["EmnistCNN_16_64_128", "EmnistCNN_32_128_256", "EmnistCNN_8_32_64"]
    results = {}

    for architecture in architectures:
        print(f"\nTesting model: {architecture}")
        model = get_model(architecture).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
        model_results = train(model, loaders, criterion, optimizer, device, epochs)

        results[architecture] = model_results
        save_results_to_csv(f"model_{architecture}_results.csv", model_results, {"Model": architecture})

    print_test_results(results, title="model")
    plot_model_diffs(results)

def main(architecture):
    test_epochs = 10
    device, loaders, criterion = configure_test()
    # print(f"\n--- Testing {architecture} model ---")
    # epoch_test(architecture, device, loaders, criterion, test_epochs)

    # print("\n--- Testing different learning rates ---")
    # learning_rate_test(architecture, device, loaders, criterion, test_epochs)

    # print("\n--- Testing different batch sizes ---")
    # batch_size_test(architecture, device, criterion, test_epochs)

    # print("\n--- Testing different configurations for training ---")
    # configuration_test(architecture, device, criterion, test_epochs)

    print("\n--- Testing different model configurations ---")
    model_config_test(device, loaders, criterion, test_epochs)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
