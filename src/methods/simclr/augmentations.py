"""Data augmentations for SimCLR.

Adapted from SimCLR PyTorch implementation
Copyright (c) 2020 Thalles Silva
Repository: https://github.com/sthalles/SimCLR
Commit: 1848fc934ad844ae630e6c452300433fe99acfd9
MIT License
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class GaussianBlur:
    """Apply Gaussian blur to a single image on CPU."""

    def __init__(self, kernel_size):
        """Initialize Gaussian blur transform.

        Args:
            kernel_size: Size of the Gaussian kernel
        """
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(
            3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3
        )
        self.blur_v = nn.Conv2d(
            3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3
        )
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        """Apply Gaussian blur to image.

        Args:
            img: PIL Image

        Returns:
            PIL Image with Gaussian blur applied
        """
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


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
    """Get SimCLR data augmentation pipeline.

    This implements the augmentation strategy from the SimCLR paper:
    - Random resized crop
    - Random horizontal flip
    - Color jitter (with probability 0.8)
    - Random grayscale (with probability 0.2)
    - Gaussian blur
    - Convert to tensor

    Args:
        size: Target image size (height and width)
        s: Strength of color distortion (default: 1.0)

    Returns:
        torchvision.transforms.Compose: Composed augmentation transforms
    """
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
        ]
    )
    return data_transforms
