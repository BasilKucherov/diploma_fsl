"""Dataset loaders for self-supervised learning.

Provides CIFAR-10 and STL-10 datasets with contrastive learning augmentations.
"""

from pathlib import Path

from torchvision import datasets

from src.methods.simclr.augmentations import ContrastiveLearningViewGenerator


def get_cifar10_ssl(data_root, split="train", transform=None, n_views=2, download=True):
    """Get CIFAR-10 dataset for self-supervised learning.

    Args:
        data_root: Root directory for dataset
        split: 'train' or 'test' (default: 'train')
        transform: Transform to apply (if None, must be wrapped with ViewGenerator)
        n_views: Number of augmented views per image (default: 2)
        download: Whether to download dataset if not present (default: True)

    Returns:
        torch.utils.data.Dataset: CIFAR-10 dataset with contrastive views
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    is_train = split == "train"

    if transform is not None:
        transform = ContrastiveLearningViewGenerator(transform, n_views=n_views)

    dataset = datasets.CIFAR10(
        root=str(data_root), train=is_train, transform=transform, download=download
    )

    return dataset


def get_stl10_ssl(data_root, split="train+unlabeled", transform=None, n_views=2, download=True):
    """Get STL-10 dataset for self-supervised learning.

    STL-10 has a large unlabeled split that is commonly used for SSL pre-training.

    Args:
        data_root: Root directory for dataset
        split: 'train', 'test', 'unlabeled', or 'train+unlabeled' (default: 'train+unlabeled')
        transform: Transform to apply (if None, must be wrapped with ViewGenerator)
        n_views: Number of augmented views per image (default: 2)
        download: Whether to download dataset if not present (default: True)

    Returns:
        torch.utils.data.Dataset: STL-10 dataset with contrastive views
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if transform is not None:
        transform = ContrastiveLearningViewGenerator(transform, n_views=n_views)

    dataset = datasets.STL10(
        root=str(data_root), split=split, transform=transform, download=download
    )

    return dataset


def get_dataset_image_size(dataset_name):
    """Get the default image size for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        int: Image size (height and width)
    """
    sizes = {
        "cifar10": 32,
        "stl10": 96,
    }
    if dataset_name not in sizes:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(sizes.keys())}")
    return sizes[dataset_name]
