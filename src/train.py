"""
Usage:
    train.py [--architecture=ARCH]

Options:
    -h --help                     Show this help message.
    --architecture=ARCH           Model architecture to use (e.g., EmnistCNN, CRNN) [default: EmnistCNN].
"""
import time

import torch
from docopt import docopt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassPrecision
from torchvision import datasets, transforms

from src.config import train_config
from src.model import save_model, get_model
from src.visualise import plot_results, plot_confusion_matrix

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

NUM_CLASSES = 47


def train(model, loaders, criterion, optimizer, device, epochs, scheduler=None, early_stopping_patience=10):

    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    test_loader = loaders["test"]

    results = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "test_accuracy": [],
        "test_loss": [],
        "epoch_count": epochs,
        "elapsed_time": 0,
    }

    total_start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    training_elapsed_time = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_start_time = time.time()

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

        # End timing the training phase
        train_end_time = time.time()
        training_elapsed_time += train_end_time - train_start_time

        # Compute average training loss and accuracy
        train_loss = running_loss / total
        epoch_acc = correct / total

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(epoch_acc)

        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        results["learning_rate"].append(current_lr)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                save_model(model, f"{model.__class__.__name__}_best.pth")
                total_elapsed_time = time.time() - total_start_time
                results["elapsed_time"] = total_elapsed_time
                results["epoch_count"] = epoch
                print(f"Early stopping triggered after epoch {epoch}.")
                break

        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Accuracy: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if epoch % train_config["save_interval"] == 0:
            save_model(model, f"{model.__class__.__name__}_epoch_{epoch}.pth")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    results["test_loss"] = test_loss
    results["test_accuracy"] = test_acc

    total_elapsed_time = time.time() - total_start_time
    results["elapsed_time"] = total_elapsed_time

    print(f"Total Elapsed Time: {total_elapsed_time:.2f} seconds")

    return results

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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

            precision_metric.update(predictions, labels)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


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

    plot_results(results=results)
    plot_confusion_matrix(
        model=model,
        loader=loaders["test"],
        device=device,
        classes=list(range(NUM_CLASSES))
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
