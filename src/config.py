from torch import nn

train_config = {
    'val_split':     0.2,
}

# "num_classes": # letters = 26, digits = 10, balanced = 47
model_config = {
    "EmnistCNN_16_64_128": {
        "fmaps1": 16,
        "fmaps2": 64,
        "fmaps3": 128,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "activation": nn.ReLU
    },
    "EmnistCNN_32_128_256": {  # Configuration with increased feature maps
        "fmaps1": 32,
        "fmaps2": 128,
        "fmaps3": 256,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "activation": nn.ReLU
    },
    "EmnistCNN_8_32_64": {  # Configuration with reduced feature maps
        "fmaps1": 8,
        "fmaps2": 32,
        "fmaps3": 64,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "activation": nn.ReLU
    },
    # 2 layers
    "EmnistCNN_16_64": {
        "fmaps1": 16,
        "fmaps2": 64,
        "fmaps3": None,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "activation": nn.ReLU
    },
    "EmnistCNN_32_128": {
        "fmaps1": 32,
        "fmaps2": 128,
        "fmaps3": None,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "activation": nn.ReLU
    },
    # Existing architectures
    "GoogleNet": {
    },
    "ResNet18" : {
    },
}
