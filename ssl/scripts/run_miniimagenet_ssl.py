#!/usr/bin/env python3
"""
Helper launcher for solo-learn SSL runs on miniImageNet.

Builds the appropriate Hydra command pointing to the configs under
`ssl/configs/pretrain/miniimagenet/` and optionally executes it.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOLO_ROOT = REPO_ROOT / "ssl" / "solo-learn"
CONFIG_ROOT = REPO_ROOT / "ssl" / "configs" / "pretrain" / "miniimagenet"

SUPPORTED_METHODS = {
    "simclr",
    "mocov2plus",
    "swav",
    "byol",
    "vicreg",
}


def parse_devices(spec: str) -> list[int]:
    devices = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            devices.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid device id '{token}'") from exc
    if not devices:
        raise argparse.ArgumentTypeError("At least one GPU id must be provided")
    return devices


def build_command(
    method: str,
    devices: list[int],
    batch_size: int,
    fast_dev_run: bool,
    disable_multicrop: bool,
    data_backend: str,
    dali_device: str,
    dali_encode_indexes: bool,
    extra_overrides: list[str],
) -> list[str]:
    devices = list(devices)
    main_py = SOLO_ROOT / "main_pretrain.py"

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method '{method}'. Choose from {sorted(SUPPORTED_METHODS)}.")

    command: list[str] = [
        sys.executable,
        str(main_py),
        "--config-path",
        str(CONFIG_ROOT),
        "--config-name",
        method,
    ]

    hydra_overrides = [
        f"devices=[{','.join(str(d) for d in devices)}]",
        "accelerator=gpu",
    ]

    if len(devices) > 1:
        hydra_overrides.append("strategy=ddp")
    else:
        hydra_overrides.append("strategy=auto")

    hydra_overrides.append(f"optimizer.batch_size={batch_size}")

    if method == "swav" and disable_multicrop:
        hydra_overrides.append("augmentations=symmetric_96")

    if data_backend == "dali":
        hydra_overrides.append("base=dali")
        hydra_overrides.append(f"dali.device={dali_device}")
        if dali_encode_indexes:
            hydra_overrides.append("dali.encode_indexes_into_labels=true")
    elif data_backend != "image_folder":
        raise ValueError(f"Unsupported data backend '{data_backend}'")

    if fast_dev_run:
        hydra_overrides.append("fast_dev_run=true")

    hydra_overrides.extend(extra_overrides)

    command.extend(hydra_overrides)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="solo-learn miniImageNet launcher")
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(SUPPORTED_METHODS),
        help="SSL method to run.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        type=parse_devices,
        help="Comma separated GPU ids (e.g. '0,1,2,3').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Per-device batch size (global batch = batch_size * num_gpus).",
    )
    parser.add_argument(
        "--no-multicrop",
        action="store_true",
        help="Disable SwAV multi-crop (default config uses 2x96 + 6x48 crops). Only for --method swav.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Append fast_dev_run=true for smoke tests.",
    )
    parser.add_argument(
        "--data-backend",
        choices=("image_folder", "dali"),
        default="image_folder",
        help="Select the data loading backend. Use 'dali' after installing NVIDIA DALI.",
    )
    parser.add_argument(
        "--dali-device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Device type for NVIDIA DALI operators (only used when --data-backend=dali).",
    )
    parser.add_argument(
        "--dali-encode-indexes",
        action="store_true",
        help="Enable encode_indexes_into_labels in the DALI pipeline (indices stored in labels).",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help="Additional Hydra overrides, separated by spaces (e.g. 'max_epochs=10 wandb.enabled=true').",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the command after printing it. By default only prints.",
    )

    args = parser.parse_args()

    if args.method != "swav" and args.no_multicrop:
        parser.error("--no-multicrop is only supported for --method swav")

    extra_overrides = shlex.split(args.extra) if args.extra else []
    command = build_command(
        method=args.method,
        devices=args.gpus,
        batch_size=args.batch_size,
        fast_dev_run=args.fast_dev_run,
        disable_multicrop=args.no_multicrop,
        data_backend=args.data_backend,
        dali_device=args.dali_device,
        dali_encode_indexes=args.dali_encode_indexes,
        extra_overrides=extra_overrides,
    )

    printable = " ".join(shlex.quote(part) for part in command)
    print(printable)

    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
