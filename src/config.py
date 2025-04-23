train_config = {
    # Which EMNIST split to use: ‘balanced’ gives you all 47 classes.
    'emnist_type':     'balanced',

    # Use the full dataset (no forced subsampling)
    'subsample_size':  None,

    # Train up to 100 epochs, but rely on early stopping (patience=10)
    'epochs':          100,

    # Use 10 % of each fold as a held-out validation set
    'val_split':       0.1,

    # 5-fold cross-validation is a good compromise between stability and speed
    'k_folds':         5,

    # Evaluate on the validation set every epoch (interval=1)
    'valid_interval':  1,

    # Save the “best so far” model every time val-loss improves
    # and also checkpoint every 10 epochs in case you want to roll back
    'save_interval':   10,

    # Print training progress every epoch
    'show_interval':   1,

    # A slightly larger batch lets you fully utilize a typical GPU
    'train_batch_size':128,
    'eval_batch_size': 256,

    # Start with a modest LR and decay it with a scheduler (e.g. OneCycleLR or CosineAnnealing)
    'learning_rate':   5e-4,

    # Add a bit of L2 weight decay to help generalization
    'weight_decay':    1e-4,

    # Number of CPU data‐loading workers
    'cpu_workers':     8,

    # Early‐stopping patience (stop if val‐loss doesn’t improve in this many epochs)
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
