"""Encoder backbone architectures.

Provides ResNet-18 and ResNet-50 encoders without the final classification layer.
These backbones output feature vectors that can be used for SSL training or FSL evaluation.
"""

import torch.nn as nn
from torchvision import models


def get_resnet18(pretrained=False):
    """Get ResNet-18 backbone without final FC layer.

    Args:
        pretrained: Whether to load pretrained weights (default: False for SSL)

    Returns:
        nn.Module: ResNet-18 encoder that outputs 512-dimensional feature vectors
    """
    resnet = models.resnet18(pretrained=pretrained)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone


def get_resnet50(pretrained=False):
    """Get ResNet-50 backbone without final FC layer.

    Args:
        pretrained: Whether to load pretrained weights (default: False for SSL)

    Returns:
        nn.Module: ResNet-50 encoder that outputs 2048-dimensional feature vectors
    """
    resnet = models.resnet50(pretrained=pretrained)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone


def get_backbone_output_dim(backbone_name):
    """Get the output dimension for a given backbone.

    Args:
        backbone_name: Name of the backbone ('resnet18' or 'resnet50')

    Returns:
        int: Output feature dimension
    """
    dims = {
        "resnet18": 512,
        "resnet50": 2048,
    }
    if backbone_name not in dims:
        raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(dims.keys())}")
    return dims[backbone_name]
