"""Checkpoint saving and loading utilities."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, config, save_path):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        config: Configuration dict/object
        save_path: Path to save checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }

    torch.save(checkpoint, str(save_path))
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model=None, optimizer=None, device="cpu"):
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to map checkpoint to

    Returns:
        dict: Checkpoint dictionary containing epoch, config, and state dicts
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model state loaded from {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Optimizer state loaded from {checkpoint_path}")

    return checkpoint
