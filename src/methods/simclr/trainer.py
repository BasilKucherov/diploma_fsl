"""SimCLR training logic.

Adapted from SimCLR PyTorch implementation
Copyright (c) 2020 Thalles Silva
Repository: https://github.com/sthalles/SimCLR
Commit: 1848fc934ad844ae630e6c452300433fe99acfd9
MIT License
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import accuracy


class SimCLRTrainer:
    """Trainer for SimCLR self-supervised learning.

    Args:
        model: SimCLRModel (backbone + projection head)
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to train on (cuda or cpu)
        log_dir: Directory for logs and checkpoints
        logger: Logger instance (optional, will create default if not provided)
        temperature: Temperature parameter for NT-Xent loss
        fp16_precision: Whether to use mixed precision training
        log_every_n_steps: Log metrics every N steps
        warmup_epochs: Number of warmup epochs before using scheduler
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        log_dir,
        logger=None,
        temperature=0.07,
        fp16_precision=True,
        log_every_n_steps=100,
        warmup_epochs=10,
        profiler_config=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.temperature = temperature
        self.fp16_precision = fp16_precision
        self.log_every_n_steps = log_every_n_steps
        self.warmup_epochs = warmup_epochs
        self.profiler_config = profiler_config or {}

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Cross entropy loss for NT-Xent
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features, batch_size):
        """Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        This is the contrastive loss used in SimCLR.

        Args:
            features: Projected features from both views, shape (2*batch_size, projection_dim)
            batch_size: Original batch size (before creating views)

        Returns:
            tuple: (logits, labels) for cross entropy loss
        """
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, epochs, config):
        """Train the SimCLR model.

        Args:
            train_loader: DataLoader with contrastive learning views
            epochs: Number of epochs to train
            config: Configuration dict for checkpointing
        """
        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        self.logger.info(f"Start SimCLR training for {epochs} epochs.")
        self.logger.info(f"Training on device: {self.device}")

        # Setup profiler if enabled
        profiler_enabled = self.profiler_config.get("enabled", False)
        prof = None

        if profiler_enabled:
            self.logger.info("=" * 80)
            self.logger.info("PyTorch Profiler ENABLED")
            self.logger.info(f"Profiler config: {self.profiler_config}")
            self.logger.info("=" * 80)

            activities = [ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)

            profiler_schedule = schedule(
                wait=self.profiler_config.get("wait", 1),
                warmup=self.profiler_config.get("warmup", 1),
                active=self.profiler_config.get("active", 3),
                repeat=self.profiler_config.get("repeat", 2),
            )

            prof = profile(
                activities=activities,
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self.log_dir / "profiler")
                ),
                record_shapes=self.profiler_config.get("record_shapes", True),
                profile_memory=self.profiler_config.get("profile_memory", True),
                with_stack=self.profiler_config.get("with_stack", False),
            )
            prof.start()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc_top1 = 0.0
            epoch_acc_top5 = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for images, _ in pbar:
                # images is a list of 2 views: [view1, view2]
                # Concatenate views: (batch_size, 3, H, W) * 2 -> (2*batch_size, 3, H, W)
                images = torch.cat(images, dim=0)
                images = images.to(self.device)
                batch_size = len(images) // 2

                with autocast(enabled=self.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features, batch_size)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

                if n_iter % self.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar("loss", loss.item(), global_step=n_iter)
                    self.writer.add_scalar("acc/top1", top1[0].item(), global_step=n_iter)
                    self.writer.add_scalar("acc/top5", top5[0].item(), global_step=n_iter)
                    self.writer.add_scalar(
                        "learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        global_step=n_iter,
                    )
                    epoch_acc_top1 = top1[0].item()
                    epoch_acc_top5 = top5[0].item()

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "top1": f"{top1[0].item():.2f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )

                # Step profiler after each batch
                if prof is not None:
                    prof.step()

                n_iter += 1

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            avg_loss = epoch_loss / num_batches
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {avg_loss:.4f}, "
                f"Top1: {epoch_acc_top1:.2f}%, "
                f"Top5: {epoch_acc_top5:.2f}%, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            checkpoint_dir = self.log_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pth"

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                config=config,
                save_path=str(checkpoint_path),
            )

            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Stop profiler and export results
        if prof is not None:
            prof.stop()

            self.logger.info("=" * 80)
            self.logger.info("Profiler Results")
            self.logger.info("=" * 80)

            # Export table summary
            summary_path = self.log_dir / "profiler_summary.txt"
            with open(summary_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("PyTorch Profiler Summary\n")
                f.write("=" * 80 + "\n\n")

                f.write("Top 20 operations by CPU time:\n")
                f.write("-" * 80 + "\n")
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                f.write("\n\n")

                if self.device.type == "cuda":
                    f.write("Top 20 operations by CUDA time:\n")
                    f.write("-" * 80 + "\n")
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                    f.write("\n\n")

            self.logger.info(f"Profiler summary saved to: {summary_path}")
            self.logger.info(f"TensorBoard traces saved to: {self.log_dir / 'profiler'}")
            self.logger.info("View traces: tensorboard --logdir profiler/")
            self.logger.info("=" * 80)

        self.logger.info("Training completed!")
        self.writer.close()
