import matplotlib.pyplot as plt
import torch
from torch import optim, nn

from src.config import train_config
from src.model import get_model
from src.train import train, load_emnist_data


def plot_results(epochs, accuracies, losses, title="Training Progress", ylabel="Metric"):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, label="Accuracy")
    plt.plot(epochs, losses, label="Loss", color="orange")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def epoch_test(model, device, loaders, criterion, optimizer, epochs):
    results = train(model, device, loaders, criterion, optimizer, epochs)
    plot_results(
        range(1, epochs + 1),
        results['epoch_accuracy'],
        results['epoch_loss'],
        title="Epoch Test: Accuracy and Loss vs. Epochs"
    )


def learning_rate_test(model, device, loaders, criterion, optimizer, epochs):
    learning_rates = [0.001, 0.01, 0.1, 0.0001]
    results = {}

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        lr_results = train(model, device, loaders, criterion, optimizer, epochs)
        results[lr] = lr_results

    epochs = range(1, epochs + 1)
    for lr in learning_rates:
        plt.plot(epochs, results[lr]['epoch_accuracy'], label=f'LR={lr}')

    plt.title("Validation Accuracy vs. Epochs for Different Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def configure_test(architecture):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using {torch.cuda.device_count()} GPUs")
    # Load data and model
    loaders = load_emnist_data(
        train_config["emnist_type"],
        train_config["train_batch_size"],
        train_config["subsample_size"],
        train_config["cpu_workers"]
    )
    model = get_model(architecture).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    return criterion, device, loaders, model, optimizer


def main(architecture):
    test_epoch = 500

    criterion, device, loaders, model, optimizer = configure_test(architecture)
    epoch_test(model, device, loaders, criterion, optimizer, test_epoch)
    learning_rate_test(model, device, loaders, criterion, optimizer, test_epoch)


if __name__ == "__main__":
    main("EmnistCNN")
