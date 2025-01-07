train_config = {
    'emnist_type': 'digits', # ! Need to change number of classes in model_config if changing this
    'subsample_size': 20000,
    'epochs': 150,
    'valid_interval': 2,
    'save_interval': 150,
    'show_interval': 10,
    'train_batch_size': 64,
    'eval_batch_size': 128,
    'learning_rate': 0.001,
    'cpu_workers': 6
}

model_config = {
    "EmnistCNN": {
        "fmaps1": 16,
        "fmaps2": 64,
        "fmaps3": 128,
        "dense": 256,
        "dropout": 0.3,
        "input_size": 28,
        "num_classes": 10 # letters = 26, digits = 10, balanced = 47
    }
}