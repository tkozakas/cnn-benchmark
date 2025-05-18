import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import utility
import visualise
from src.model import get_model, load_model


def main():
    EMNIST_TYPE = "balanced"
    ARCHITECTURE = "EmnistCNN_32_128_256"
    SUBSAMPLE_SIZE = None
    CPU_WORKERS = 6
    DEVICE = "cuda"
    K = 5
    B = 64

    full = datasets.EMNIST(
        root="../data",
        split=EMNIST_TYPE,
        train=True,
        download=True,
        transform=utility.get_transforms()
    )
    ds = utility.get_subsample(full, SUBSAMPLE_SIZE)

    test_loader = DataLoader(
        ds,
        batch_size=B,
        num_workers=CPU_WORKERS,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    model = get_model(ARCHITECTURE, num_classes=utility.get_emnist_class_num(EMNIST_TYPE)).to(DEVICE)
    model = load_model(
        model, f"{ARCHITECTURE}_fold{K}.pth"
    )
    visualise.plot_confusion_matrix(
        model, test_loader, DEVICE,
        classes=list(range(utility.get_emnist_class_num(EMNIST_TYPE) + 1)),
    )


if __name__ == "__main__":
    main()
