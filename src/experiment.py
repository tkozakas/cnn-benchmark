import matplotlib.pyplot as plt
import torch

from src.config import train_config, model_config
from src.model import get_model
from src.train import get_emnist_loaders
from src.train import train_model


def train_and_evaluate_model(device, loaders, epochs, learning_rate=None):
    model = get_model("EmnistCNN", model_config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate or train_config['learning_rate'])

    results = train_model(
        model,
        loaders["train"],
        loaders["test"],
        loaders["validation"],
        criterion,
        optimizer,
        device,
        train_config['show_interval'],
        train_config['valid_interval'],
        train_config['save_interval'],
        epochs=epochs
    )
    return model, results


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


def epoch_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_emnist_loaders(
        train_batch_size=train_config['train_batch_size'],
        eval_batch_size=train_config['eval_batch_size'],
        cpu_workers=train_config['cpu_workers'],
        val_split=0.2
    )

    _, results = train_and_evaluate_model(device, loaders, epochs=train_config['epochs'])
    plot_results(
        range(1, train_config['epochs'] + 1),
        results['epoch_accuracy'],
        results['epoch_loss'],
        title="Epoch Test: Accuracy and Loss vs. Epochs"
    )

def learning_rate_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [0.001, 0.01, 0.1, 0.0001]
    results = {}

    loaders = get_emnist_loaders(
        train_batch_size=train_config['train_batch_size'],
        eval_batch_size=train_config['eval_batch_size'],
        cpu_workers=train_config['cpu_workers'],
        val_split=0.2
    )

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        _, lr_results = train_and_evaluate_model(device, loaders, epochs=train_config['epochs'], learning_rate=lr)
        results[lr] = lr_results

    epochs = range(1, train_config['epochs'] + 1)
    for lr in learning_rates:
        plt.plot(epochs, results[lr]['epoch_accuracy'], label=f'LR={lr}')

    plt.title("Validation Accuracy vs. Epochs for Different Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# Accuracy Growth Test
def accuracy_growth_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specific_epochs = [10, 20, 40, 80, 160, 320]
    results = []

    loaders = get_emnist_loaders(
        train_batch_size=train_config['train_batch_size'],
        eval_batch_size=train_config['eval_batch_size'],
        cpu_workers=train_config['cpu_workers'],
        val_split=0.2
    )

    model = get_model("EmnistCNN", model_config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    for target_epoch in specific_epochs:
        train_model(
            model,
            loaders["train"],
            loaders["test"],
            loaders["validation"],
            criterion,
            optimizer,
            device,
            train_config['show_interval'],
            train_config['valid_interval'],
            train_config['save_interval'],
            epochs=target_epoch
        )

        # Evaluate model
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in loaders["validation"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {target_epoch}: Accuracy = {accuracy * 100:.2f}%")
        results.append((target_epoch, accuracy * 100))

    # Plot Accuracy Growth
    epochs, accuracies = zip(*results)
    plot_results(epochs, accuracies, [0] * len(accuracies), title="Accuracy Growth over Specific Epochs", ylabel="Validation Accuracy (%)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using {torch.cuda.device_count()} GPUs")

    # Execute Tests
    epoch_test()
    accuracy_growth_test()
    learning_rate_test()


if __name__ == "__main__":
    main()
