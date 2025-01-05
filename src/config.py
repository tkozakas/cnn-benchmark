dataset_config = {
    "datasets": {
        "emnist": {
            "datasets": {
                "balanced": {
                    "train_images": "data/emnist_source_files/emnist-balanced-train-images-idx3-ubyte",
                    "train_labels": "data/emnist_source_files/emnist-balanced-train-labels-idx1-ubyte",
                    "test_images": "data/emnist_source_files/emnist-balanced-test-images-idx3-ubyte",
                    "test_labels": "data/emnist_source_files/emnist-balanced-test-labels-idx1-ubyte"
                }
            }
        }
    }
}

train_config = {
    'epochs': 100,
    'show_interval': 5,
    'valid_interval': 20,
    'save_interval': 20,
    'train_batch_size': 32,
    'eval_batch_size': 512,
    'learning_rate': 0.0005,
    'cpu_workers': 4
}


model_config = {
    "SimpleCNN": {
        "num_classes": 47
    }
}