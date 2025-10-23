"""Training entry point for SSL methods.

Example usage:
    python scripts/train.py method=simclr backbone=resnet18 dataset=cifar10
    python scripts/train.py method=simclr backbone=resnet50 dataset=stl10 method.epochs=100
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.ssl_datasets import get_cifar10_ssl, get_stl10_ssl
from src.methods.simclr.augmentations import get_simclr_augmentations
from src.methods.simclr.model import SimCLRModel
from src.methods.simclr.trainer import SimCLRTrainer
from src.models.backbones import get_resnet18, get_resnet50
from src.models.projection_heads import MLPProjectionHead
from src.utils.logging_utils import setup_logging


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_backbone(config):
    """Get backbone model based on config."""
    backbone_name = config.backbone.architecture.lower()
    if backbone_name == "resnet18":
        return get_resnet18(pretrained=config.backbone.pretrained)
    elif backbone_name == "resnet50":
        return get_resnet50(pretrained=config.backbone.pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def get_dataset(config):
    dataset_name = config.dataset.name.lower()

    transform = get_simclr_augmentations(
        size=config.dataset.image_size, s=config.method.augmentation_strength
    )

    if dataset_name == "cifar10":
        dataset = get_cifar10_ssl(
            data_root=config.dataset.data_root,
            split=config.dataset.split,
            transform=transform,
            n_views=config.method.n_views,
            download=config.dataset.download,
        )
    elif dataset_name == "stl10":
        dataset = get_stl10_ssl(
            data_root=config.dataset.data_root,
            split=config.dataset.split,
            transform=transform,
            n_views=config.method.n_views,
            download=config.dataset.download,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    # Use Hydra's runtime output directory (timestamped per run)
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir=output_dir, log_file="train.log")

    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("=" * 80)
    logger.info(f"\n{OmegaConf.to_yaml(config)}")
    logger.info("=" * 80)

    set_seed(config.seed)
    logger.info(f"Random seed set to: {config.seed}")

    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")

    logger.info(f"Loading dataset: {config.dataset.name}")
    train_dataset = get_dataset(config)
    logger.info(f"Dataset size: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.method.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Number of batches: {len(train_loader)}")

    logger.info(f"Building model: {config.backbone.name}")
    backbone = get_backbone(config)
    projection_head = MLPProjectionHead(
        input_dim=config.backbone.feature_dim,
        hidden_dim=config.backbone.feature_dim,
        output_dim=config.method.projection_dim,
    )
    model = SimCLRModel(backbone=backbone, projection_head=projection_head)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.method.lr, weight_decay=config.method.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    logger.info("Initializing SimCLR trainer")
    trainer = SimCLRTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=str(output_dir),
        logger=logger,
        temperature=config.method.temperature,
        fp16_precision=config.fp16_precision,
        log_every_n_steps=config.method.log_every_n_steps,
        warmup_epochs=config.method.warmup_epochs,
    )

    logger.info("=" * 80)
    logger.info(f"Starting training for {config.method.epochs} epochs")
    logger.info("=" * 80)
    trainer.train(
        train_loader=train_loader,
        epochs=config.method.epochs,
        config=OmegaConf.to_container(config),
    )

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Run directory: {output_dir}")
    logger.info(f"Checkpoints: {output_dir / 'checkpoints'}")
    logger.info(f"TensorBoard: {output_dir / 'tensorboard'}")
    logger.info(f"Logs: {output_dir / 'train.log'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
