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
from torchvision import datasets, transforms

from src.config import train_config
from src.model import save_model, get_model


def train(model, loaders, criterion, optimizer, device, epochs):
    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    test_loader = loaders["test"]

    results = {"epoch_loss": [], "epoch_accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        results["epoch_loss"].append(epoch_loss)
        results["epoch_accuracy"].append(epoch_acc)
        print(f"Epoch [{epoch}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        if epoch % train_config["save_interval"] == 0:
            save_model(model, f"{model.__class__.__name__}_epoch_{epoch}.pth")

    return results


def load_emnist_data(emnist_type, batch_size, subsample_size, cpu_workers, val_split=0.2):
    print("Loading EMNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
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
    criterion, device, loaders, model, optimizer = configure_training(architecture)
    train(model, loaders, criterion, optimizer, device, train_config["epochs"])


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
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    return criterion, device, loaders, model, optimizer


if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
