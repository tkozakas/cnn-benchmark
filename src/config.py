test_config = {
    'emnist_type': 'balanced',
    'subsample_size': 10000,
    'epochs': 5,
    'val_split': 0.2,
    'k_folds': 5,
    'valid_interval': 1,
    'show_interval': 1,

    'train_batch_size': 512,
    'eval_batch_size': 128,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'cpu_workers': 6,

    'early_stopping_patience': 10
}

train_config = {
    'emnist_type':     'balanced',
    'subsample_size': None,
    'epochs': 100,
    'val_split':       0.2,
    'k_folds':         5,
    'valid_interval':  1,
    'show_interval':   1,

    'train_batch_size': 512,
    'eval_batch_size': 128,
    'learning_rate':   5e-4,
    'weight_decay':    1e-4,
    'cpu_workers': 6,

    'early_stopping_patience': 10
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
        "num_classes": 47
    },
    "EmnistCNN_32_128_256": {  # Configuration with increased feature maps
        "fmaps1": 32,
        "fmaps2": 128,
        "fmaps3": 256,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "num_classes": 47
    },
    "EmnistCNN_8_32_64": {  # Configuration with reduced feature maps
        "fmaps1": 8,
        "fmaps2": 32,
        "fmaps3": 64,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "num_classes": 47
    },
    # 2 layers
    "EmnistCNN_16_64": {
        "fmaps1": 16,
        "fmaps2": 64,
        "fmaps3": None,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "num_classes": 47
    },
    "EmnistCNN_32_128": {
        "fmaps1": 32,
        "fmaps2": 128,
        "fmaps3": None,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "num_classes": 47
    },
    # Existing architectures
    "GoogleNet": {
        "num_classes": 47
    },
    "ResNet18": {
        "num_classes": 47
    }
}
