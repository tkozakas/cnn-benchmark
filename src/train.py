"""
Usage:
    train.py [--architecture=ARCH]

Options:
    -h --help                     Show this help message.
    --architecture=ARCH           Model architecture to use (e.g., EmnistCNN, CRNN) [default: EmnistCNN].
"""
import torch
from docopt import docopt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassPrecision
from torchvision import datasets, transforms

from src.config import train_config
from src.model import save_model, get_model
from src.visualise import plot_results

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

NUM_CLASSES = 47

def train(model, loaders, criterion, optimizer, device, epochs, scheduler=None):
    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    test_loader = loaders["test"]

    results = {
        "epoch_loss": [],
        "epoch_accuracy": [],
        "epoch_precision": [],  # Added
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": []  # Added
    }

    # Initialize Precision metric for training
    train_precision = MulticlassPrecision(num_classes=NUM_CLASSES).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Reset Precision metric at the start of each epoch
        train_precision.reset()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += images.size(0)

            # Update Precision metric with current batch
            train_precision.update(predictions, labels)

        # Compute average training loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Compute Precision for the epoch
        epoch_precision = train_precision.compute().item()

        results["epoch_loss"].append(epoch_loss)
        results["epoch_accuracy"].append(epoch_acc)
        results["epoch_precision"].append(epoch_precision)  # Added

        # Validation phase
        val_loss, val_acc, val_precision = evaluate(model, val_loader, criterion, device)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_acc)
        results["val_precision"].append(val_precision)  # Added

        if scheduler:
            scheduler.step()

        # Print epoch results
        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.4f} | "
              f"Train Precision: {epoch_precision:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} | "
              f"Val Precision: {val_precision:.4f}")

        # Save model at intervals
        if epoch % train_config["save_interval"] == 0:
            save_model(model, f"{model.__class__.__name__}_epoch_{epoch}.pt")

    # Evaluate on the test set (optional: you can add test_precision similarly)
    test_loss, test_acc, test_precision = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test Precision: {test_precision:.4f}")

    return results


def evaluate(model, loader, criterion, device):
    """Evaluate model on a given loader. Returns loss, accuracy, and precision."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Initialize Precision metric for evaluation
    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES).to(device)
    precision_metric.reset()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += images.size(0)

            # Update Precision metric with current batch
            precision_metric.update(predictions, labels)

    loss = running_loss / total
    accuracy = correct / total
    precision = precision_metric.compute().item()

    return loss, accuracy, precision


def load_emnist_data(emnist_type, batch_size, subsample_size, cpu_workers, val_split=0.2):
    print("Loading EMNIST dataset...")
    full_train_dataset = datasets.EMNIST(root="../data", split=emnist_type, train=True, download=True, transform=transform)

    if subsample_size:
        full_train_dataset, _ = random_split(full_train_dataset, [subsample_size, len(full_train_dataset) - subsample_size])

    train_size = int(len(full_train_dataset) * (1 - val_split))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = datasets.EMNIST(root="../data", split=emnist_type, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_workers, pin_memory=True)

    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }

def main(architecture):
    criterion, device, loaders, model, optimizer, scheduler = configure_training(architecture)
    results = train(model, loaders, criterion, optimizer, device, train_config["epochs"], scheduler)

    plot_results(
        results=results
    )

def configure_training(architecture):
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
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    return criterion, device, loaders, model, optimizer, scheduler


if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
