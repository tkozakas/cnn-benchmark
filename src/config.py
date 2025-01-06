train_config = {
    'epochs': 60,  # Number of epochs
    'show_interval': 50,  # Interval for showing progress
    'valid_interval': 20,  # Interval for validation
    'save_interval': 20,  # Interval for saving model
    'train_batch_size': 32,  # Batch size for training
    'eval_batch_size': 128,  # Batch size for evaluation
    'learning_rate': 0.0005,  # Learning rate
    'cpu_workers': 6  # Number of CPU workers
}

model_config = {
    "EmnistCNN": {
        "fmaps1": 32,  # Number of feature maps in the first convolutional layer
        "fmaps2": 64,  # Number of feature maps in the second convolutional layer
        "dense": 128,  # Number of units in the dense (fully connected) layer
        "dropout": 0.5  # Dropout rate
    }
}

path_config = {
    "train_images": "../data/emnist_source_files/emnist-balanced-train-images-idx3-ubyte",
    "train_labels": "../data/emnist_source_files/emnist-balanced-train-labels-idx1-ubyte",
    "test_images": "../data/emnist_source_files/emnist-balanced-test-images-idx3-ubyte",
    "test_labels": "../data/emnist_source_files/emnist-balanced-test-labels-idx1-ubyte"
}
