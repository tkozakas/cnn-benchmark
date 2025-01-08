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

    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    for lr in learning_rates:
        epoch_accuracy = results[lr]['epoch_accuracy']
        epochs_range = range(1, len(epoch_accuracy) + 1)
        axs[0].plot(epochs_range, epoch_accuracy, label=f'LR={lr}')
    axs[0].set_title("Validation Accuracy vs. Epochs for Different Learning Rates")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Validation Accuracy")
    axs[0].legend()
    axs[0].grid(True)

    for lr in learning_rates:
        val_loss = results[lr]['val_loss']
        epochs_range = range(1, len(val_loss) + 1)
        axs[1].plot(epochs_range, val_loss, label=f'LR={lr}')
    axs[1].set_title("Validation Loss vs. Epochs for Different Learning Rates")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Validation Loss")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].bar(
        time_results.keys(),
        time_results.values(),
        color='skyblue',
        width=0.05
    )
    axs[2].set_xticks(learning_rates)
    axs[2].set_title("Training Time for Different Learning Rates")
    axs[2].set_xlabel("Learning Rate")
    axs[2].set_ylabel("Training Time (seconds)")
    axs[2].grid(axis='y')

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

        # Measure training time
        start_time = time.time()
        batch_results = train(model, loaders, criterion, optimizer, device, epochs)
        elapsed_time = time.time() - start_time

        results[batch_size] = batch_results
        speed_results[batch_size] = elapsed_time

    # Create subplots for validation accuracy and training speed
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot validation accuracy
    for batch_size in batch_sizes:
        epoch_accuracy = results[batch_size]['epoch_accuracy']
        epochs_range = range(1, len(epoch_accuracy) + 1)
        axs[0].plot(epochs_range, epoch_accuracy, label=f'Batch={batch_size}')
    axs[0].set_title("Validation Accuracy vs. Epochs for Different Batch Sizes")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Validation Accuracy")
    axs[0].legend()
    axs[0].grid(True)

    # Plot training speed
    axs[1].bar(
        speed_results.keys(),
        speed_results.values(),
        color='skyblue',
        width=10  # Adjust width for better visualization
    )
    axs[1].set_xticks(batch_sizes)  # Ensure x-axis matches actual batch sizes
    axs[1].set_title("Training Speed for Different Batch Sizes")
    axs[1].set_xlabel("Batch Size")
    axs[1].set_ylabel("Training Time (seconds)")
    axs[1].grid(axis='y')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

def configure_test():
    """
    Configures device, data loaders, and loss criterion.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using {torch.cuda.device_count()} GPU(s)")

    # Load data and prepare common resources
    loaders = load_emnist_data(
        train_config["emnist_type"],
        train_config["train_batch_size"],
        train_config["subsample_size"],
        train_config["cpu_workers"]
    )
    criterion = nn.CrossEntropyLoss()
    return device, loaders, criterion


def main(architecture):
    test_epochs = 30
    device, loaders, criterion = configure_test()
    print(f"\n--- Testing {architecture} model ---")
    epoch_test(architecture, device, loaders, criterion, test_epochs)

    print("\n--- Testing different hyperparameters ---")
    learning_rate_test(architecture, device, loaders, criterion, test_epochs)

    print("\n--- Testing different batch sizes ---")
    batch_size_test(architecture, device, criterion, test_epochs)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
