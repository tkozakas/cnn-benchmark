train_config = {
    'emnist_type': 'balanced', # ! Need to change number of classes in model_config if changing this
    'subsample_size': 10000,
    'epochs': 10,
    'valid_interval': 10,
    'save_interval': 20,
    'show_interval': 10,
    'train_batch_size': 64,
    'eval_batch_size': 128,
    'learning_rate': 0.0005,
    'cpu_workers': 6
}

# "num_classes": # letters = 26, digits = 10, balanced = 47
model_config = {
    "EmnistCNN": {
        "fmaps1": 16,
        "fmaps2": 64,
        "fmaps3": 128,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "num_classes": 47
    },
    "Resnet18": {
        "num_classes": 10
    },
    "Resnet18-pretrained": {
        "num_classes": 10
    }
}