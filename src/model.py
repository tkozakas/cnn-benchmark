from pathlib import Path

import torch
import torch.nn as nn

from src.config import model_config


class EmnistCNN(nn.Module):
    def __init__(self, fmaps1=32, fmaps2=64, fmaps3=None, dense=128, dropout=0.5,
                 input_size=28, num_classes=10, activation=nn.LeakyReLU):
        super(EmnistCNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, fmaps1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(fmaps1),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(fmaps1, fmaps2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(fmaps2),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Optional Third Convolutional Block
        self.use_third_conv = fmaps3 is not None
        if self.use_third_conv:
            self.conv3 = nn.Sequential(
                nn.Conv2d(fmaps2, fmaps3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(fmaps3),
                activation(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.flatten_size = (input_size // 8) ** 2 * fmaps3
        else:
            self.flatten_size = (input_size // 4) ** 2 * fmaps2

        # Fully Connected Layers
        self.fcon1 = nn.Sequential(
            nn.Linear(self.flatten_size, dense),
            activation(),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fcon2 = nn.Linear(dense, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_third_conv:
            x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x

def get_model(architecture):
    if architecture == 'GoogleNet':
        from torchvision.models import googlenet
        model = googlenet(num_classes=model_config[architecture]['num_classes'], aux_logits=False)
        return model
    elif architecture == 'ResNet18':
        from torchvision.models import resnet18
        model = resnet18(num_classes=model_config[architecture]['num_classes'])
        return model
    return EmnistCNN(**model_config[architecture])

def save_model(model, path):
    project_root = Path(__file__).parent.parent.resolve()
    trained_dir = project_root / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)
    full_path = trained_dir / Path(path).name
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")

def load_model(model, path):
    project_root = Path(__file__).parent.parent.resolve()
    trained_dir = project_root / "trained"
    full_path = trained_dir / Path(path).name
    model.load_state_dict(torch.load(full_path))
    print(f"Model loaded from {full_path}")
    return model