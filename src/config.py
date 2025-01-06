train_config = {
    'emnist_type': 'letters', # ! Need to change number of classes in model_config if changing this
    'subsample_size': 1000,
    'epochs': 250,
    'valid_interval': 5,
    'save_interval': 250,
    'show_interval': 50,
    'train_batch_size': 128,
    'eval_batch_size': 256,
    'learning_rate': 0.001,
    'cpu_workers': 6
}

model_config = {
    "EmnistCNN": {
        "fmaps1": 32,
        "fmaps2": 128,
        "fmaps3": 256,
        "dense": 128,
        "dropout": 0.0,
        "input_size": 28,
        "num_classes": 26 # letters = 26, digits = 10, balanced = 47
    }
}