import torch

def get_subsample(full_dataset, subsample_size):
    """Optionally subsample the dataset."""
    if subsample_size:
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [subsample_size, len(full_dataset) - subsample_size]
        )
    return full_dataset


def parse_args(args):
    """Parse command line arguments"""
    ARCHITECTURE = args['--architecture']
    K            = int(args['--k-folds'])
    N            = int(args['--epochs'])
    B            = int(args['--batch-size'])
    LR           = float(args['--lr'])
    WD           = float(args['--weight-decay'])
    PAT          = int(args['--patience'])
    CPU_WORKERS  = int(args['--cpu-workers'])
    DEVICE       = args['--device']
    EMNIST_TYPE  = args['--emnist-type']

    # Safely handle the “None” default
    raw_ss = args['--subsample-size']
    if raw_ss is None or raw_ss.lower() == 'none' or raw_ss == '':
        SUBSAMPLE_SIZE = None
    else:
        SUBSAMPLE_SIZE = int(raw_ss)

    print(f"Device:       {DEVICE}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"EMNIST split: {EMNIST_TYPE}")
    print(f"Subsample:    {SUBSAMPLE_SIZE!r}")

    print(f"Config → "
          f"EMNIST split: {EMNIST_TYPE}, "
          f"Subsample size: {SUBSAMPLE_SIZE}, "
          f"Using K={K} folds, "
          f"N={N} epochs, "
          f"B={B} batch size, "
          f"LR={LR} learning rate, "
          f"WD={WD} weight decay, "
          f"PAT={PAT} patience, "
          f"CPU_WORKERS={CPU_WORKERS} workers")

    return (
        ARCHITECTURE,
        B,
        CPU_WORKERS,
        DEVICE,
        EMNIST_TYPE,
        K,
        LR,
        N,
        PAT,
        SUBSAMPLE_SIZE,
        WD
    )
