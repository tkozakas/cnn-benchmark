train_config = {
    'epochs': 50,
    'valid_interval': 5,  # (IN EPOCHS)
    'save_interval': 10,  # (IN EPOCHS)
    'show_interval': 50,  # (IN BATCHES)
    'train_batch_size': 32,  # (INCREASE IF GPU MEMORY ALLOWS)
    'eval_batch_size': 128,  # (INCREASE IF GPU MEMORY ALLOWS)
    'learning_rate': 0.0001,  # (ADJUST IF NEEDED)
    'cpu_workers': 6
}


model_config = {
    "EmnistCNN": {
        "fmaps1": 64,
        "fmaps2": 128,
        "dense": 256,
        "dropout": 0.3
    }
}

path_config = {
    "train_images": "../data/emnist_source_files/emnist-balanced-train-images-idx3-ubyte",
    "train_labels": "../data/emnist_source_files/emnist-balanced-train-labels-idx1-ubyte",
    "test_images": "../data/emnist_source_files/emnist-balanced-test-images-idx3-ubyte",
    "test_labels": "../data/emnist_source_files/emnist-balanced-test-labels-idx1-ubyte"
}
