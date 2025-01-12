"""
Usage:
    train.py [--architecture=ARCH]

Options:
    -h --help                     Show this help message.
    --architecture=ARCH           Model architecture to use (e.g., EmnistCNN_16_64_128, CRNN) [default: EmnistCNN_16_64_128].
"""
import time

import numpy as np
import torch
from docopt import docopt
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import train_config, model_config
from src.model import save_model, get_model
from src.visualise import plot_confusion_matrix, plot_aggregated_learning_curves

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def k_fold_cross_validation(architecture, dataset, model_fn, k_folds, epochs, batch_size, learning_rate, random_state=42):
    global model, test_loader
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    all_results = []

    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        val_size = int(len(train_dataset) * train_config["val_split"])
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        loaders = {"train": train_loader, "validation": val_loader, "test": test_loader}
        model = model_fn().to("cuda" if torch.cuda.is_available() else "cpu")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        results = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_accuracy": [],
            "test_loss": [],
            "epoch_count": epochs,
            "elapsed_time": 0,
        }
        start_time = time.time()

        results.update(
            train(
                model=model,
                architecture=architecture,
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                device="cuda" if torch.cuda.is_available() else "cpu",
                epochs=epochs
            )
        )
        results["elapsed_time"] = time.time() - start_time
        all_results.append(results)

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, "cuda" if torch.cuda.is_available() else "cpu")
        results["test_loss"] = test_loss
        results["test_accuracy"] = test_accuracy

        print(f"Test Loss for Fold {fold + 1}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    avg_results = {
        "train_loss": np.mean([np.mean(r["train_loss"]) for r in all_results]),
        "train_accuracy": np.mean([np.mean(r["train_accuracy"]) for r in all_results]),
        "val_loss": np.mean([np.mean(r["val_loss"]) for r in all_results]),
        "val_accuracy": np.mean([np.mean(r["val_accuracy"]) for r in all_results]),
        "test_loss": np.mean([r["test_loss"] for r in all_results]),
        "test_accuracy": np.mean([r["test_accuracy"] for r in all_results]),
        "elapsed_time": np.mean([r["elapsed_time"] for r in all_results]),
        "epoch_count": np.mean([r["epoch_count"] for r in all_results]),
    }

    print("\nFinal Aggregated Results:")
    print(f"Train Loss: {avg_results['train_loss']:.4f}")
    print(f"Train Accuracy: {avg_results['train_accuracy']:.4f}")
    print(f"Validation Loss: {avg_results['val_loss']:.4f}")
    print(f"Validation Accuracy: {avg_results['val_accuracy']:.4f}")
    print(f"Test Loss: {avg_results['test_loss']:.4f}")
    print(f"Test Accuracy: {avg_results['test_accuracy']:.4f}")
    print(f"Average Epochs per Fold: {avg_results['epoch_count']:.2f}")
    print(f"Average Elapsed Time per Fold: {avg_results['elapsed_time']:.2f} seconds")

    return all_results, avg_results

def train(model, architecture, loaders, criterion, optimizer, device, epochs, scheduler=None, early_stopping_patience=10):
    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    test_loader = loaders["test"]
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "test_loss": [],
        "epoch_count": epochs,
        "elapsed_time": 0,
    }
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_accuracy)

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                save_model(model, f"{architecture}_best.pth")
                results["elapsed_time"] = time.time() - results["elapsed_time"]
                results["epoch_count"] = epoch
                print(f"Early stopping triggered at epoch {epoch}.")
                break
        if scheduler:
            scheduler.step()

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    results["test_loss"] = test_loss
    results["test_accuracy"] = test_accuracy
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    save_model(model, f"{architecture}_latest.pth")
    return results

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return running_loss / total, correct / total


def main(architecture):
    def model_fn():
        return get_model(architecture)

    print("Loading EMNIST dataset...")
    full_dataset = datasets.EMNIST(
        root="../data",
        split="balanced",
        train=True,
        download=True,
        transform=transform
    )

    if train_config["subsample_size"]:
        print(f"Subsampling dataset to {train_config['subsample_size']} samples...")
        subsample_size = train_config["subsample_size"]
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [subsample_size, len(full_dataset) - subsample_size]
        )

    all_results, avg_results = k_fold_cross_validation(
        architecture=architecture,
        dataset=full_dataset,
        model_fn=model_fn,
        k_folds=train_config["k_folds"],
        epochs=train_config["epochs"],
        batch_size=train_config["train_batch_size"],
        learning_rate=train_config["learning_rate"]
    )

    print("\nPlotting Aggregated Learning Curves...")
    plot_aggregated_learning_curves(all_results, "Accuracy", "train_accuracy", "val_accuracy")
    plot_aggregated_learning_curves(all_results, "Loss", "train_loss", "val_loss")

    plot_confusion_matrix(
        model=model,
        loader=test_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        classes=list(range(model_config[architecture]["num_classes"]))
    )

if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
