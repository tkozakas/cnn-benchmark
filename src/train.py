import argparse

import torch
from torch.utils.data import DataLoader

from src.config import train_config, model_config
from src.dataset import EmnistDataset, load_datasets
from src.evaluate import evaluate_simple_cnn
from src.model import get_model, save_model


def train_batch(model, data, optimizer, criterion, device):
    """
    Train the model on a single batch of data.

    Parameters:
    - model: The model to train.
    - data: A tuple containing input images, labels, and target lengths.
    - optimizer: The optimizer used for training.
    - criterion: The loss function.
    - device: The device to run computations on.

    Returns:
    - loss: The loss value for the batch.
    """
    model.train()  # Ensure the model is in training mode

    # Unpack the data
    images, labels, target_lengths = [d.to(device) for d in data]

    # Forward pass
    outputs = model(images)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    return loss.item()  # Return the loss as a scalar



def train_model(model, train_loader, criterion, optimizer, device,
                show_interval, valid_interval, save_interval, epochs=10):
    """
    Train the model on the training set.
    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for the training set.
    - criterion: The loss function.
    - optimizer: The optimizer used for training.
    - device: The device to run the computations on.
    - show_interval: Interval for showing loss during training.
    - valid_interval: Interval for performing evaluation.
    - save_interval: Interval for saving the model.
    - epochs: The number of epochs to train the model.
    """
    model.to(device)  # Move model to device
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            loss = train_batch(model, data, optimizer, criterion, device)
            running_loss += loss

            if i % show_interval == show_interval - 1:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / show_interval:.4f}")
                running_loss = 0.0

        if (epoch + 1) % valid_interval == 0:
            print(f"Evaluating at epoch {epoch + 1}")
            eval_results = evaluate_simple_cnn(model, train_loader, criterion, device=device)
            print(f"Validation Loss: {eval_results['loss']:.4f}, "
                  f"Accuracy: {eval_results['accuracy']:.4f}")

        if (epoch + 1) % save_interval == 0:
            save_model(model, f"model_epoch_{epoch + 1}.pth")

    print("Finished Training")
    save_model(model, "model_final.pth")


def main(params):
    # Load train config
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

    # Load data
    datasets = load_datasets()
    train_data = datasets["balanced"]["train"]
    test_data = datasets["balanced"]["test"]
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    train_dataset = EmnistDataset(train_images, train_labels)
    test_dataset = EmnistDataset(test_images, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers
    )

    # Instantiate model
    model = get_model(params.architecture, model_config)
    model.to(device)
    print(f"Using architecture: {params.architecture}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion.to(device)

    assert save_interval % valid_interval == 0

    # Train the model
    train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        show_interval,
        valid_interval,
        save_interval,
        epochs=epochs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the EMNIST dataset")
    parser.add_argument(
        "--architecture",
        type=str,
        default="SimpleCNN",
        help="Model architecture to use (e.g., SimpleCNN, CRNN). Default is SimpleCNN."
    )
    args = parser.parse_args()
    main(args)
