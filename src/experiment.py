import time

import matplotlib.pyplot as plt
import torch
from docopt import docopt
from torch import optim, nn

from src.config import train_config
from src.model import get_model
from src.train import train, load_emnist_data
from src.visualise import plot_results, print_results_table


def epoch_test(architecture, device, loaders, criterion, epochs):
    model = get_model(architecture).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    results = train(model, loaders, criterion, optimizer, device, epochs)
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

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        plt.plot(epochs_range, results[lr]['epoch_accuracy'], label=f'LR={lr}')

    plt.title("Validation Accuracy vs. Epochs for Different Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
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

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        plt.plot(epochs_range, results[batch_size]['epoch_accuracy'], label=f'Batch={batch_size}')

    plt.title("Validation Accuracy vs. Epochs for Different Batch Sizes")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def training_speed_test(architecture, device, criterion, epochs):
    batch_sizes = [16, 32, 64, 128, 256]
    speed_results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting training speed with batch size: {batch_size}")

        loaders = load_emnist_data(
            train_config["emnist_type"],
            batch_size,
            train_config["subsample_size"],
            train_config["cpu_workers"]
        )

        model = get_model(architecture).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])

        start_time = time.time()
        train(model, loaders, criterion, optimizer, device, epochs)
        elapsed_time = time.time() - start_time

        print(f"Batch size: {batch_size}, Time taken: {elapsed_time:.2f} seconds")
        speed_results[batch_size] = elapsed_time

    plt.figure(figsize=(10, 6))
    plt.bar(speed_results.keys(), list(speed_results.values()), color='skyblue')
    plt.title("Training Speed for Different Batch Sizes")
    plt.xlabel("Batch Size")
    plt.ylabel("Training Time (seconds)")
    plt.grid(axis='y')
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
    test_epochs = 50
    device, loaders, criterion = configure_test()
    print(f"\n--- Testing {architecture} model ---")
    epoch_test(architecture, device, loaders, criterion, test_epochs)

    print("\n--- Testing different hyperparameters ---")
    learning_rate_test(architecture, device, loaders, criterion, test_epochs)

    print("\n--- Testing different batch sizes ---")
    batch_size_test(architecture, device, criterion, test_epochs)

    print("\n--- Testing training speed ---")
    training_speed_test(architecture, device, criterion, test_epochs)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
