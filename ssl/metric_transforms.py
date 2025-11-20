import random
from typing import Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import ImageFilter, Image

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def make_metric_transforms(
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
):
    """
    Returns T_metric_strong and T_metric_weak as composed transforms.
    """
    
    normalize = transforms.Normalize(mean=mean, std=std)

    # T_metric_strong
    strong_transforms = [
        transforms.RandomResizedCrop(84, scale=(0.5, 1.0), ratio=(4.0/5.0, 5.0/4.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            GaussianBlur(sigma=[0.1, 2.0])
        ], p=0.3),
        transforms.ToTensor(),
        normalize
    ]

    # T_metric_weak
    weak_transforms = [
        transforms.Resize(96),
        transforms.CenterCrop(84),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ]

    return transforms.Compose(strong_transforms), transforms.Compose(weak_transforms)

class MetricTransform:
    def __init__(self):
        self.transform_strong, self.transform_weak = make_metric_transforms()

    def __call__(self, x):
        # We need:
        # 1. Two strong views for alignment/invariance
        # 2. One weak view for uniformity/variance/covariance
        
        x1_strong = self.transform_strong(x)
        x2_strong = self.transform_strong(x)
        x_weak = self.transform_weak(x)
        
        return x1_strong, x2_strong, x_weak

