import os
import struct

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import dataset_config


class EmnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if image.ndim == 2:  # If shape is (height, width)
            image = np.expand_dims(image, axis=0)  # Add channel dimension (1, height, width)

        if self.transform:
            image = self.transform(image)

        # Determine target length
        if isinstance(label, (list, np.ndarray)):  # Sequence label
            target_length = len(label)
        else:  # Single-label classification
            target_length = 1

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(target_length, dtype=torch.long)
        )

def preprocess_images(images):
    if images.ndim == 3:
        images = np.expand_dims(images, axis=1)
    elif images.ndim == 4 and images.shape[-1] == 1:
        images = np.transpose(images, (0, 3, 1, 2))
    return images

def load_ubyte_file(file_path, is_image=True):
    absolute_root_path = os.path.join(os.path.dirname(__file__), "..", file_path)
    with open(absolute_root_path, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if is_image:
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num_items, rows, cols, 1) / 255.0
        else:
            data = np.fromfile(f, dtype=np.uint8)
    return data


def load_dataset(images_path, labels_path):
    images = load_ubyte_file(images_path, is_image=True)
    labels = load_ubyte_file(labels_path, is_image=False)
    return images, labels


def load_datasets():
    datasets = {}
    emnist_config = dataset_config["datasets"]["emnist"]["datasets"]
    for dataset_name, paths in emnist_config.items():
        print(f"Loading dataset: {dataset_name}")

        train_images_path = paths["train_images"]
        train_labels_path = paths["train_labels"]
        test_images_path = paths["test_images"]
        test_labels_path = paths["test_labels"]

        train_images, train_labels = load_dataset(train_images_path, train_labels_path)
        test_images, test_labels = load_dataset(test_images_path, test_labels_path)

        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)

        datasets[dataset_name] = {"train": (train_images, train_labels), "test": (test_images, test_labels)}

        print(f"Dataset {dataset_name} loaded: "
              f"{len(train_images)} train samples, {len(test_images)} test samples")

    return datasets