import argparse
import json
import re
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLO_PATH = REPO_ROOT / "ssl" / "solo-learn"
if str(SOLO_PATH) not in sys.path:
    sys.path.append(str(SOLO_PATH))

from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_datasets,
)
from solo.methods import METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract backbone/projector features from SSL checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to args/config JSON. Defaults to <checkpoint_dir>/args.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root to miniImageNet split directories containing train/val/test.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to extract features from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store extracted feature chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the dataloader (number of images, not crops).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=["backbone", "projector", "both"],
        default=["both"],
        help="Feature spaces to save. 'both' is shorthand for backbone+projector.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of images per saved chunk.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset order when extracting (default: False).",
    )
    return parser.parse_args()


def resolve_spaces(spaces: list[str]) -> list[str]:
    if "both" in spaces:
        return ["backbone", "projector"]
    return spaces


def load_cfg(args: argparse.Namespace) -> DictConfig:
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = args.checkpoint.parent / "args.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Could not locate config JSON at {cfg_path}")
    cfg = OmegaConf.create(json.loads(cfg_path.read_text()))
    # Ensure we don't rely on DALI when extracting locally.
    cfg.data.format = "image_folder"
    cfg.data.train_path = str(args.data_root / "train")
    cfg.data.val_path = str(args.data_root / "val")
    cfg.dali = OmegaConf.create({"device": "cpu", "encode_indexes_into_labels": False})
    cfg.num_nodes = 1
    cfg.devices = [0]
    cfg.accelerator = "cpu"
    cfg.precision = "32"
    return cfg


def instantiate_model(cfg: DictConfig, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    method_name = cfg.method
    if method_name not in METHODS:
        raise ValueError(f"Unknown method '{method_name}'. Available: {list(METHODS.keys())}")

    model = METHODS[method_name](cfg)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[extract_features] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[extract_features] Unexpected keys when loading checkpoint: {unexpected}")
    model.eval()
    model.to(device)
    return model


def build_transform(cfg: DictConfig) -> FullTransformPipeline:
    pipelines = []
    for aug_cfg in cfg.augmentations:
        pipelines.append(
            NCropAugmentation(
                build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
            )
        )
    return FullTransformPipeline(pipelines)


def build_loader(
    cfg: DictConfig,
    transform: FullTransformPipeline,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    if split == "train":
        data_path = cfg.data.train_path
    elif split == "val":
        data_path = cfg.data.val_path
    else:
        data_path = str(Path(cfg.data.train_path).parents[0] / "test")

    dataset = prepare_datasets(
        dataset=cfg.data.dataset,
        transform=transform,
        train_data_path=data_path,
        data_format="image_folder",
        no_labels=cfg.data.get("no_labels", False),
        data_fraction=cfg.data.get("fraction", -1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 2:
        x = torch.flatten(x, start_dim=1)
    return x


def forward_backbone(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = model.backbone(x)
    feats = _normalize_tensor(feats)
    return feats


def forward_projector(model: torch.nn.Module, feats: torch.Tensor) -> torch.Tensor | None:
    projector = getattr(model, "projector", None)
    if projector is None:
        return None
    proj = projector(feats)
    proj = _normalize_tensor(proj)
    return proj


def parse_epoch(checkpoint: Path) -> int | None:
    match = re.search(r"ep=(\d+)", checkpoint.name)
    if match:
        return int(match.group(1))
    return None


def save_chunk(
    output_dir: Path,
    method: str,
    split: str,
    epoch: int | None,
    chunk_id: int,
    data: dict[str, torch.Tensor],
) -> None:
    num_views = None
    if data.get("backbone") is not None:
        num_views = data["backbone"].shape[1]
    elif data.get("projector") is not None:
        num_views = data["projector"].shape[1]
    else:
        num_views = 0

    meta = {
        "method": method,
        "split": split,
        "epoch": epoch,
        "chunk": chunk_id,
        "num_samples": int(data["indexes"].shape[0]),
        "num_views": int(num_views),
    }
    payload = {"meta": meta, "data": data}
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{method}_split={split}"
    if epoch is not None:
        filename += f"_epoch={epoch:03d}"
    filename += f"_chunk={chunk_id:04d}.pt"
    torch.save(payload, output_dir / filename)


def flush_chunks(
    accum: dict[str, torch.Tensor | None],
    output_dir: Path,
    method: str,
    split: str,
    epoch: int | None,
    chunk_size: int,
    chunk_counter: int,
) -> int:
    total = accum["indexes"].shape[0] if accum.get("indexes") is not None else 0
    while total >= chunk_size and accum.get("indexes") is not None:
        slc = slice(0, chunk_size)
        chunk_data = {
            "indexes": accum["indexes"][slc].clone(),
            "labels": accum["labels"][slc].clone() if accum.get("labels") is not None else None,
        }
        if "backbone" in accum and accum["backbone"] is not None:
            chunk_data["backbone"] = accum["backbone"][slc].clone()
        if "projector" in accum and accum["projector"] is not None:
            chunk_data["projector"] = accum["projector"][slc].clone()
        save_chunk(output_dir, method, split, epoch, chunk_counter, chunk_data)
        chunk_counter += 1

        for key in list(accum.keys()):
            tensor = accum[key]
            if tensor is not None:
                accum[key] = tensor[chunk_size:].clone() if tensor.shape[0] > chunk_size else None
        total = accum["indexes"].shape[0] if accum.get("indexes") is not None else 0
    return chunk_counter


def run_extraction(args: argparse.Namespace) -> None:
    spaces = resolve_spaces(args.spaces)
    device = torch.device(args.device)
    cfg = load_cfg(args)
    epoch = parse_epoch(args.checkpoint)

    model = instantiate_model(cfg, args.checkpoint, device)
    transform = build_transform(cfg)
    loader = build_loader(
        cfg,
        transform,
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    accum: dict[str, torch.Tensor | None] = {
        "indexes": None,
        "labels": None,
    }
    if "backbone" in spaces:
        accum["backbone"] = None
    if "projector" in spaces:
        accum["projector"] = None
    chunk_counter = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting [{cfg.method}] {args.split}"):
            indexes, views, labels = batch
            # views is a list of tensors (num_crops) each [B, C, H, W]
            if not isinstance(views, list):
                views = [views]

            batch_backbone: list[torch.Tensor] = []
            batch_projector: list[torch.Tensor] = []

            for view in views:
                view = view.to(device, non_blocking=True)
                feats_backbone = forward_backbone(model, view)
                feats_backbone = feats_backbone.to("cpu", non_blocking=True).float()
                batch_backbone.append(feats_backbone)

            if "projector" in spaces:
                for feats in batch_backbone:
                    proj = forward_projector(model, feats.to(device))
                    if proj is None:
                        raise RuntimeError(
                            "Requested projector features, but the model lacks a projector."
                        )
                    proj = proj.to("cpu", non_blocking=True).float()
                    batch_projector.append(proj)

            stacked_backbone = torch.stack(batch_backbone, dim=1) if "backbone" in spaces else None
            stacked_projector = torch.stack(batch_projector, dim=1) if batch_projector else None

            batch_indexes = indexes.to("cpu", non_blocking=True)
            batch_labels = labels.to("cpu", non_blocking=True) if labels is not None else None

            batch_payload = {"indexes": batch_indexes, "labels": batch_labels}
            if "backbone" in spaces:
                batch_payload["backbone"] = stacked_backbone
            if "projector" in spaces:
                batch_payload["projector"] = stacked_projector

            for key, value in batch_payload.items():
                if key not in accum or value is None:
                    continue
                if accum[key] is None:
                    accum[key] = value
                else:
                    accum[key] = torch.cat([accum[key], value], dim=0)

            total_samples += batch_indexes.shape[0]
            chunk_counter = flush_chunks(
                accum,
                args.output_dir,
                cfg.method,
                args.split,
                epoch,
                args.chunk_size,
                chunk_counter,
            )

    # Flush remaining samples
    if accum["indexes"] is not None and accum["indexes"].shape[0] > 0:
        chunk_data = {
            "indexes": accum["indexes"],
            "labels": accum["labels"],
        }
        if "backbone" in spaces:
            chunk_data["backbone"] = accum["backbone"]
        if "projector" in spaces and "projector" in accum:
            chunk_data["projector"] = accum["projector"]
        save_chunk(args.output_dir, cfg.method, args.split, epoch, chunk_counter, chunk_data)
        chunk_counter += 1

    print(
        f"Extraction complete: {total_samples} samples processed, {chunk_counter} chunk(s) written to {args.output_dir}."
    )


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
