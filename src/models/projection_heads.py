"""Projection heads for self-supervised learning.

MLP projection heads that transform backbone features into the representation space
where contrastive loss is computed.
"""

import torch.nn as nn


class MLPProjectionHead(nn.Module):
    """MLP projection head for SimCLR.

    Architecture: Linear -> ReLU -> Linear
    This matches the projection head from the SimCLR paper.

    Args:
        input_dim: Dimension of input features from backbone
        hidden_dim: Dimension of hidden layer (default: same as input_dim)
        output_dim: Dimension of output projection (default: 128)
    """

    def __init__(self, input_dim, hidden_dim=None, output_dim=128):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Project input features.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Projected features of shape (batch_size, output_dim)
        """
        return self.projection(x)
