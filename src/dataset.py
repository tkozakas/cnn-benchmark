import numpy as np
import torch
from torch.utils.data import Dataset
import struct

from src.config import dataset_config

class EmnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = preprocess_images(images)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def preprocess_images(images):
    return np.transpose(images, (0, 3, 1, 2)) if images.ndim == 4 else images

def load_ubyte_file(file_path, is_image=True):
    with open(file_path, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if is_image:
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num_items, rows, cols, 1) / 255.0
        else:
            data = np.fromfile(f, dtype=np.uint8)
    return data

def _load_ubyte_file(file_path, is_image=True):
    load_ubyte_file(file_path, is_image)

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

        datasets[dataset_name] = {
            "train": EmnistDataset(train_images, train_labels),
            "test": EmnistDataset(test_images, test_labels)
        }

        print(f"Dataset {dataset_name} loaded: "
              f"{len(train_images)} train samples, {len(test_images)} test samples")

    return datasets

