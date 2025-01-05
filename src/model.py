from pathlib import Path

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_model(architecture, model_config):
    if architecture == "SimpleCNN":
        return SimpleCNN(**model_config["SimpleCNN"])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def save_model(model, path):
    project_root = Path(__file__).parent.parent.resolve()
    trained_dir = project_root / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)
    full_path = trained_dir / Path(path).name
    torch.save({
        "architecture": "SimpleCNN",
        "state_dict": model.state_dict()
    }, str(full_path))
    print(f"Model saved to {full_path}")
