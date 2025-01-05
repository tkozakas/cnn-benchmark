dataset_config = {
    "datasets": {
        "emnist": {
            "datasets": {
                "balanced": {
                    "train_images": "data/emnist_source_files/emnist-balanced-train-images-idx3-ubyte",
                    "train_labels": "data/emnist_source_files/emnist-balanced-train-labels-idx1-ubyte",
                    "test_images": "data/emnist_source_files/emnist-balanced-test-images-idx3-ubyte",
                    "test_labels": "data//emnist_source_files/emnist-balanced-test-labels-idx1-ubyte"
                }
            }
        }
    }
}

train_config = {
    "model": {
        "achitecture": "SimpleCNN",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_classes": 47,
        "channels": 1
    },
    "device": "cuda"
}