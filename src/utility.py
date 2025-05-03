import torch


def get_subsample(full_dataset, subsample_size):
    """Optionally subsample the dataset."""
    if subsample_size:
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [subsample_size, len(full_dataset) - subsample_size]
        )
    return full_dataset


def parse_args():
    args = docopt(__doc__)
    # override or use defaults
    ARCHITECTURE = args['--architecture']
    K = int(args['--k-folds'])
    N = int(args['--epochs'])
    B = int(args['--batch-size'])
    LR = float(args['--lr'])
    WD = float(args['--weight-decay'])
    PAT = int(args['--patience'])
    CPU_WORKERS = int(args['--cpu-workers'])
    DEVICE = args['--device']
    EMNIST_TYPE = args['--emnist-type']
    SUBSAMPLE_SIZE = int(args['--subsample-size']) if args['--subsample-size'] else None
    print(f"Device: {DEVICE}")
    print(f"Architecture: {ARCHITECTURE}")
    return ARCHITECTURE, B, CPU_WORKERS, DEVICE, EMNIST_TYPE, K, LR, N, PAT, SUBSAMPLE_SIZE, WD
