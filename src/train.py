"""
Usage:
    train.py [--architecture=ARCH]

Options:
    -h --help                     Show this help message.
    --architecture=ARCH           Model architecture to use (e.g., EmnistCNN, CRNN) [default: EmnistCNN].
"""

import idx2numpy
import torch
from docopt import docopt
from torch.utils.data import TensorDataset, DataLoader

from src.config import train_config, model_config, path_config
from src.evaluate import evaluate_simple_cnn
from src.model import get_model, save_model


def train_model(
        model, train_loader, test_loader, validation_loader, criterion, optimizer, device,
        show_interval, valid_interval, save_interval, epochs=10, scheduler=None):
    """
    Train the model on the training set.

    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for the training set.
    - validation_loader: DataLoader for the validation set.
    - criterion: The loss function.
    - optimizer: The optimizer used for training.
    - device: The device to run the computations on.
    - show_interval: Interval for showing loss during training.
    - valid_interval: Interval for performing evaluation.
    - save_interval: Interval for saving the model.
    - epochs: The number of epochs to train the model.
    """
    results = {
        "epoch_loss": [],
        "epoch_accuracy": []
    }

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            loss = train_batch(model, data, optimizer, criterion, device)
            if scheduler:
                scheduler.step()
            running_loss += loss

            if i % show_interval == show_interval - 1:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / show_interval:.4f}")
                running_loss = 0.0

        if validation_loader and (epoch + 1) % valid_interval == 0:
            print(f"Evaluating at epoch {epoch + 1}...")
            eval_results = evaluate_simple_cnn(model, validation_loader, criterion, device=device)
            results["epoch_loss"].append(eval_results["loss"])
            results["epoch_accuracy"].append(eval_results["accuracy"])
            print(f"Validation Loss: {eval_results['loss']:.4f}, "
                  f"Accuracy: {eval_results['accuracy']:.4f}")

        if (epoch + 1) % save_interval == 0:
            save_model(model, f"{model.__class__.__name__}_epoch_{epoch + 1}.pth")

    print("Finished Training")
    save_model(model, f"../trained/{model.__class__.__name__}_final.pth")
    eval_results = evaluate_simple_cnn(model, test_loader, criterion, device=device)
    print(f"Test Loss: {eval_results['loss']:.4f}, "
          f"Accuracy: {eval_results['accuracy']:.4f}")
    return results


def train_batch(model, data, optimizer, criterion, device):
    model.to(device)
    model.train()

    images, labels = [d.to(device) for d in data]
    outputs = model.forward(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def get_emnist_loaders(train_batch_size, eval_batch_size, cpu_workers, val_split=0.2):
    print("Loading EMNIST dataset from provided paths...")

    # Load data using idx2numpy
    train_images = idx2numpy.convert_from_file(path_config["train_images"])
    train_labels = idx2numpy.convert_from_file(path_config["train_labels"])
    test_images = idx2numpy.convert_from_file(path_config["test_images"])
    test_labels = idx2numpy.convert_from_file(path_config["test_labels"])

    # Normalize and reshape images
    train_images = torch.tensor((train_images / 255.0 - 0.5).reshape(len(train_images), 1, 28, 28)).float()
    train_labels = torch.tensor(train_labels.astype('int64'))
    test_images = torch.tensor((test_images / 255.0 - 0.5).reshape(len(test_images), 1, 28, 28)).float()
    test_labels = torch.tensor(test_labels.astype('int64'))

    # Split training data into training and validation sets
    num_train = len(train_labels)
    num_val = int(num_train * val_split)
    num_actual_train = num_train - num_val

    valid_images = train_images[num_actual_train:]
    valid_labels = train_labels[num_actual_train:]
    train_images = train_images[:num_actual_train]
    train_labels = train_labels[:num_actual_train]

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images, train_labels)
    valid_dataset = TensorDataset(valid_images, valid_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=cpu_workers)
    validation_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=cpu_workers)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=cpu_workers)

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader
    }


def main(architecture):
    epochs = train_config["epochs"]
    train_batch_size = train_config["train_batch_size"]
    eval_batch_size = train_config["eval_batch_size"]
    learning_rate = train_config["learning_rate"]
    show_interval = train_config["show_interval"]
    valid_interval = train_config["valid_interval"]
    save_interval = train_config["save_interval"]
    cpu_workers = train_config["cpu_workers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using {torch.cuda.device_count()} GPUs")

    loaders = get_emnist_loaders(
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        cpu_workers=cpu_workers,
        val_split=0.2
    )
    train_loader = loaders["train"]
    validation_loader = loaders["validation"]
    test_loader = loaders["test"]

    # Instantiate model
    model = get_model(architecture, model_config)
    model.to(device)
    print(f"Using architecture: {architecture}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion.to(device)

    assert save_interval % valid_interval == 0

    # Train the model
    train_model(
        model,
        train_loader,
        test_loader,
        validation_loader,
        criterion,
        optimizer,
        device,
        show_interval,
        valid_interval,
        save_interval,
        epochs=epochs,
        scheduler=scheduler
    )


if __name__ == "__main__":
    arguments = docopt(__doc__)
    architecture = arguments['--architecture']
    main(architecture)
