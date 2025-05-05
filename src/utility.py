import torch
from torchvision import transforms

def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

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
    CPU_WORKERS  = int(args['--cpu-workers'])
    DEVICE       = args['--device']
    EMNIST_TYPE  = args['--emnist-type']

    PAT = None if args['--patience'] is None or args['--patience'].lower() == 'none' else int(args['--patience'])
    SUBSAMPLE_SIZE = None if args['--subsample-size'] is None or args['--subsample-size'].lower() == 'none' else int(
        args['--subsample-size'])

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
