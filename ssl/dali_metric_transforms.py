import os
from pathlib import Path
from typing import Union

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from solo.data.dali_dataloader import (
    RandomColorJitter,
    RandomGaussianBlur,
    RandomGrayScaleConversion,
)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MetricPipelineBuilder:
    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int,
        device: str,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed + device_id
        self.device = device
        self.data_path = Path(data_path)

        # Find files and labels
        labels = sorted(Path(entry.name) for entry in os.scandir(self.data_path) if entry.is_dir())
        data = [
            (self.data_path / label / file, label_idx)
            for label_idx, label in enumerate(labels)
            for file in sorted(os.listdir(self.data_path / label))
        ]
        files, labels = map(list, zip(*data))

        self.files = files
        self.labels = labels

        self.reader = ops.readers.File(
            files=self.files,
            labels=self.labels,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle_after_epoch=False,  # Deterministic order if possible, or irrelevant for metrics
        )

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )

        # Define augmentation operations corresponding to MetricTransform
        # T_metric_strong
        # 1. RandomResizedCrop(84, scale=(0.5, 1.0), ratio=(4/5, 5/4))
        self.rrc_strong = ops.RandomResizedCrop(
            device=self.device,
            size=84,
            random_area=(0.5, 1.0),
            random_aspect_ratio=(4.0 / 5.0, 5.0 / 4.0),
            interp_type=types.INTERP_CUBIC,
        )
        # 2. RandomHorizontalFlip(p=0.5) - handled by CoinFlip + C crop
        self.coin_flip_strong = ops.random.CoinFlip(probability=0.5)

        # 3. ColorJitter(0.4, 0.4, 0.4, 0.1) p=0.8
        self.color_jitter_strong = RandomColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, prob=0.8, device=self.device
        )

        # 4. RandomGrayscale(p=0.1)
        self.grayscale_strong = RandomGrayScaleConversion(prob=0.1, device=self.device)

        # 5. GaussianBlur(p=0.3)
        self.blur_strong = RandomGaussianBlur(
            prob=0.3, window_size=23, device=self.device
        )  # window size approx for 84x84?

        # Normalization for strong
        self.cmn_strong = ops.CropMirrorNormalize(
            device=self.device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[v * 255 for v in IMAGENET_DEFAULT_MEAN],
            std=[v * 255 for v in IMAGENET_DEFAULT_STD],
        )

        # T_metric_weak
        # 1. Resize(96)
        self.resize_weak = ops.Resize(
            device=self.device,
            resize_shorter=96,
            interp_type=types.INTERP_CUBIC,
        )
        # 2. CenterCrop(84)
        # 3. RandomHorizontalFlip(p=0.5)
        self.coin_flip_weak = ops.random.CoinFlip(probability=0.5)

        # Normalization for weak (includes Crop and Flip)
        self.cmn_weak = ops.CropMirrorNormalize(
            device=self.device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(84, 84),
            mean=[v * 255 for v in IMAGENET_DEFAULT_MEAN],
            std=[v * 255 for v in IMAGENET_DEFAULT_STD],
        )

    @pipeline_def
    def pipeline(self):
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # --- Strong View 1 ---
        # RRC
        img_s1 = self.rrc_strong(images)
        # Color Jitter
        img_s1 = self.color_jitter_strong(img_s1)
        # Grayscale
        img_s1 = self.grayscale_strong(img_s1)
        # Blur
        img_s1 = self.blur_strong(img_s1)
        # Normalize + Flip
        img_s1 = self.cmn_strong(img_s1, mirror=self.coin_flip_strong())

        # --- Strong View 2 ---
        # RRC
        img_s2 = self.rrc_strong(images)
        # Color Jitter
        img_s2 = self.color_jitter_strong(img_s2)
        # Grayscale
        img_s2 = self.grayscale_strong(img_s2)
        # Blur
        img_s2 = self.blur_strong(img_s2)
        # Normalize + Flip
        img_s2 = self.cmn_strong(img_s2, mirror=self.coin_flip_strong())

        # --- Weak View ---
        # Resize
        img_w = self.resize_weak(images)
        # CenterCrop + Normalize + Flip
        img_w = self.cmn_weak(img_w, mirror=self.coin_flip_weak())

        return img_s1, img_s2, img_w


def build_dali_metric_loader(data_path, batch_size, num_workers=4, device="gpu", seed=12):
    if device == "cpu":
        dali_device = "cpu"
    else:
        dali_device = "gpu"

    pipeline_builder = MetricPipelineBuilder(
        data_path=data_path,
        batch_size=batch_size,
        device=dali_device,
        num_threads=num_workers,
        seed=seed,
    )

    pipeline = pipeline_builder.pipeline(
        batch_size=batch_size, num_threads=num_workers, device_id=0, seed=seed
    )
    pipeline.build()

    epoch_size = pipeline.epoch_size("Reader")

    class DALIMetricIterator(DALIGenericIterator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __len__(self):
            return epoch_size // batch_size

    loader = DALIMetricIterator(
        pipelines=[pipeline],
        output_map=["x1", "x2", "x_weak"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )

    return loader, epoch_size
