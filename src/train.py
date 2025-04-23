"""
Usage:
    train.py [--architecture=ARCH]

Options:
    -h --help                     Show this help message.
    --architecture=ARCH           Model architecture to use (e.g., EmnistCNN_16_64_128, CRNN) [default: EmnistCNN_16_64_128].
"""

from docopt import docopt
from torchvision import datasets, transforms

from src.config import model_config
from src.model import get_model
from src.visualise import plot_confusion_matrix, plot_aggregated_learning_curves

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


import time
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from src.config import train_config
from src.model import save_model

def k_fold_cross_validation(architecture,
                            dataset,
                            model_fn,
                            k_folds,
                            epochs,
                            batch_size,
                            learning_rate=None,
                            weight_decay: float = 0.0,
                            random_state: int = 42,
                            optimizer_fn=None,
                            scheduler_fn=None,
                            criterion=None,
                            early_stopping_patience: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold  = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    all_results = []

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset), start=1):
        print(f"\nFold {fold}/{k_folds}")
        # split train/val/test
        train_subset = Subset(dataset, train_idx)
        test_subset  = Subset(dataset, test_idx)

        val_size   = int(len(train_subset) * train_config["val_split"])
        train_size = len(train_subset) - val_size
        train_data, val_data = torch.utils.data.random_split(
            train_subset, [train_size, val_size]
        )

        train_loader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True,  pin_memory=True)
        val_loader   = DataLoader(val_data,   batch_size=batch_size,
                                  shuffle=False, pin_memory=True)
        test_loader  = DataLoader(test_subset, batch_size=batch_size,
                                  shuffle=False, pin_memory=True)

        # model / optimizer / scheduler
        model = model_fn().to(device)
        if optimizer_fn is None:
            optimizer = optim.Adam(model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay)
        else:
            optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None

        best_val_loss = float('inf')
        patience_cnt = 0

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   []
        }
        start_time = time.time()

        for epoch in range(1, epochs+1):
            # training
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc  = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # validation
            model.eval()
            val_running, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_running += loss.item() * imgs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_running / val_total
            val_acc  = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            # early stopping
            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= early_stopping_patience:
                        print(f"→ Early stopping at epoch {epoch}")
                        break

            if scheduler:
                scheduler.step()

        elapsed = time.time() - start_time

        # test evaluation
        model.eval()
        test_running, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_running += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_loss = test_running / test_total
        test_acc  = test_correct / test_total
        print(f"Fold {fold} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        all_results.append({
            'fold': fold,
            'train_loss':    history['train_loss'],
            'train_accuracy':history['train_acc'],
            'val_loss':      history['val_loss'],
            'val_accuracy':  history['val_acc'],
            'test_loss':     test_loss,
            'test_accuracy': test_acc,
            'elapsed_time':  elapsed,
            'epochs':        len(history['train_loss'])
        })

        save_model(model, f"{architecture}_fold{fold}_best.pth")

    # aggregate
    avg_results = {
        'avg_train_loss':  np.mean([np.mean(r['train_loss'])     for r in all_results]),
        'avg_train_acc':   np.mean([np.mean(r['train_accuracy']) for r in all_results]),
        'avg_val_loss':    np.mean([np.mean(r['val_loss'])       for r in all_results]),
        'avg_val_acc':     np.mean([np.mean(r['val_accuracy'])   for r in all_results]),
        'avg_test_loss':   np.mean([r['test_loss']               for r in all_results]),
        'avg_test_acc':    np.mean([r['test_accuracy']           for r in all_results]),
        'avg_time':        np.mean([r['elapsed_time']            for r in all_results]),
    }

    print("\nCross-validation complete")
    print(f"Average Val Acc:  {avg_results['avg_val_acc']:.4f}")
    print(f"Average Test Acc: {avg_results['avg_test_acc']:.4f}")

    return all_results, avg_results



def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
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

    if train_config.get("subsample_size"):
        print(f"Subsampling to {train_config['subsample_size']} samples...")
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [train_config['subsample_size'], len(full_dataset) - train_config['subsample_size']]
        )

    all_results, avg_results = k_fold_cross_validation(
        architecture=architecture,
        dataset=full_dataset,
        model_fn=model_fn,
        k_folds=train_config["k_folds"],
        epochs=train_config["epochs"],
        batch_size=train_config["train_batch_size"],
        early_stopping_patience=train_config["early_stopping_patience"],
    )

    print("\nPlotting results...")
    plot_aggregated_learning_curves(all_results, "Accuracy", "train_accuracy", "val_accuracy")
    plot_aggregated_learning_curves(all_results, "Loss",     "train_loss",     "val_loss")

    test_loader = DataLoader(
        full_dataset,
        batch_size=train_config["train_batch_size"],
        shuffle=False,
        pin_memory=True
    )
    last_model = model_fn().to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_path = f"{architecture}_fold{train_config['k_folds']}_best.pth"
    last_model.load_state_dict(torch.load(load_path))
    plot_confusion_matrix(
        model=last_model,
        loader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        classes=list(range(model_config[architecture]["num_classes"]))
    )

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args['--architecture'])
