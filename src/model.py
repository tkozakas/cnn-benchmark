from pathlib import Path

import torch
import torch.nn as nn


class EmnistCNN(nn.Module):
    def __init__(self, fmaps1, fmaps2, dense, dropout):
        super(EmnistCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=fmaps1, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=fmaps1, out_channels=fmaps2, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fcon1 = nn.Sequential(nn.Linear(49 * fmaps2, dense), nn.LeakyReLU())
        self.fcon2 = nn.Linear(dense, 10)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x

    def init_weights(self):
        if isinstance(self, nn.Conv2d):
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

def get_model(architecture, model_config):
    if architecture == "EmnistCNN":
        return EmnistCNN(**model_config["EmnistCNN"])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def save_model(model, path):
    project_root = Path(__file__).parent.parent.resolve()
    trained_dir = project_root / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)
    full_path = trained_dir / Path(path).name
    torch.save({
        "architecture": "EMNISTCNN",
        "state_dict": model.state_dict()
    }, str(full_path))
    print(f"Model saved to {full_path}")
