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
from src.model import get_model, save_model


def train(model, loader, criterion, optimizer, device, epochs):
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
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
        print(f"Epoch [{epoch}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        if epoch % train_config["save_interval"] == 0:
            save_model(model, f"{model.__class__.__name__}_epoch_{epoch}.pth")


def load_emnist_data(emnist_type, batch_size, subsample_size, cpu_workers):
    print("Loading EMNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.EMNIST(root="../data", split=emnist_type, train=True, download=True, transform=transform)
    dataset, _ = random_split(dataset, [subsample_size, len(dataset) - subsample_size])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_workers, pin_memory=True)
    return loader


def main(architecture):
    # Load configurations
    emnist_type = train_config["emnist_type"]
    subsample_size = train_config["subsample_size"]
    epochs = train_config["epochs"]
    train_batch_size = train_config["train_batch_size"]
    learning_rate = train_config["learning_rate"]
    cpu_workers = train_config["cpu_workers"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using {torch.cuda.device_count()} GPUs")

    # Load data and model
    loader = load_emnist_data(emnist_type, train_batch_size, subsample_size, cpu_workers)
    model = get_model(architecture).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    train(model, loader, criterion, optimizer, device, epochs)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
