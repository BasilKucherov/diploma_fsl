"""SimCLR model combining backbone and projection head."""

import torch.nn as nn


class SimCLRModel(nn.Module):
    """SimCLR model that combines a backbone encoder and projection head.

    During SSL training, the full model (backbone + projection head) is used.
    For FSL evaluation, only the backbone is used (via get_backbone() method).

    Args:
        backbone: Encoder backbone (e.g., ResNet)
        projection_head: MLP projection head
    """

    def __init__(self, backbone, projection_head):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x):
        """Forward pass through backbone and projection head.

        Args:
            x: Input images of shape (batch_size, 3, H, W)

        Returns:
            Projected features of shape (batch_size, projection_dim)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        projections = self.projection_head(features)
        return projections

    def get_backbone(self):
        """Get the backbone encoder without projection head.

        This is used for FSL evaluation where we only need feature extraction.

        Returns:
            nn.Module: The backbone encoder
        """
        return self.backbone

    def get_features(self, x):
        """Get backbone features without projection.

        Args:
            x: Input images of shape (batch_size, 3, H, W)

        Returns:
            Features from backbone of shape (batch_size, feature_dim)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return features
