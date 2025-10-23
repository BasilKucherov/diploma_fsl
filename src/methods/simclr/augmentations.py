"""Data augmentations for SimCLR.

Adapted from SimCLR PyTorch implementation
Copyright (c) 2020 Thalles Silva
Repository: https://github.com/sthalles/SimCLR
Commit: 1848fc934ad844ae630e6c452300433fe99acfd9
MIT License

Optimized version: Uses native torchvision transforms instead of custom implementations
to avoid PIL↔Tensor conversion overhead.
"""

from torchvision import transforms


class ContrastiveLearningViewGenerator:
    """Generate multiple augmented views of a single image for contrastive learning."""

    def __init__(self, base_transform, n_views=2):
        """Initialize view generator.

        Args:
            base_transform: Transform to apply for each view
            n_views: Number of views to generate (default: 2)
        """
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        """Generate multiple views of input image.

        Args:
            x: PIL Image

        Returns:
            List of n_views augmented versions of the image
        """
        return [self.base_transform(x) for _ in range(self.n_views)]


def get_simclr_augmentations(size, s=1.0):
    """Get SimCLR data augmentation pipeline (optimized version).

    This implements the augmentation strategy from the SimCLR paper:
    - Random resized crop
    - Random horizontal flip
    - Color jitter (with probability 0.8)
    - Random grayscale (with probability 0.2)
    - Gaussian blur (using native torchvision transform)
    - Convert to tensor

    Optimization: Replaced custom GaussianBlur with torchvision.transforms.GaussianBlur
    to eliminate expensive PIL↔Tensor conversion overhead.

    Args:
        size: Target image size (height and width)
        s: Strength of color distortion (default: 1.0)

    Returns:
        torchvision.transforms.Compose: Composed augmentation transforms
    """
    # Calculate kernel size (must be odd)
    kernel_size = int(0.1 * size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # Native GaussianBlur: much faster, no PIL↔Tensor conversions
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ]
    )
    return data_transforms
