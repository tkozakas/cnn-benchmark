train_config = {
    'emnist_type': 'digits', # ! Need to change number of classes in model_config if changing this
    'subsample_size': None,
    'epochs': 50,
    'valid_interval': 10,
    'save_interval': 20,
    'show_interval': 10,
    'train_batch_size': 64,
    'eval_batch_size': 128,
    'learning_rate': 0.001,
    'cpu_workers': 6
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
    }
}
