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
from src.visualise import plot_results, print_results_table, save_results_to_csv


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

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        epoch_accuracy = results[lr]['epoch_accuracy']
        epochs_range = range(1, len(epoch_accuracy) + 1)
        plt.plot(epochs_range, epoch_accuracy, label=f'LR={lr}')
    plt.title("Validation Accuracy vs. Epochs for Different Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    bar_width = 0.5 / len(learning_rates)
    plt.bar(
        [str(lr) for lr in learning_rates],
        time_results.values(),
        color='skyblue',
        width=bar_width
    )
    plt.xticks(rotation=45, ha='right')
    plt.title("Training Time for Different Learning Rates")
    plt.xlabel("Learning Rate")
    plt.ylabel("Training Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()



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

    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        epoch_accuracy = results[batch_size]['epoch_accuracy']
        epochs_range = range(1, len(epoch_accuracy) + 1)
        plt.plot(epochs_range, epoch_accuracy, label=f'Batch={batch_size}')
    plt.title("Validation Accuracy vs. Epochs for Different Batch Sizes")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    bar_width = 5
    plt.bar(
        list(speed_results.keys()),
        speed_results.values(),
        color='skyblue',
        width=bar_width,
        align='center'
    )
    plt.xticks(list(speed_results.keys()), rotation=45, ha='right')
    plt.title("Training Speed for Different Batch Sizes")
    plt.xlabel("Batch Size")
    plt.ylabel("Training Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
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

    plt.figure(figsize=(12, 6))
    for config in configurations:
        config_key = str(config)
        epoch_accuracy = results[config_key]['epoch_accuracy']
        epochs_range = range(1, len(epoch_accuracy) + 1)
        plt.plot(epochs_range, epoch_accuracy, label=f"LR={config['learning_rate']}, BS={config['train_batch_size']}")
    plt.title("Validation Accuracy vs. Epochs for Different Configurations")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    bar_width = 0.5 / len(configurations)
    plt.bar(
        list(time_results.keys()),
        time_results.values(),
        color='skyblue',
        width=bar_width
    )
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Training Time for Different Configurations")
    plt.xlabel("Configuration (LR, BS, E)")
    plt.ylabel("Training Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main(architecture):
    test_epochs = 100
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
