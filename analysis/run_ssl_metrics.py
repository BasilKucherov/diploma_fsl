import argparse
import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from analysis.extract_features import run_extraction
from analysis.metrics import FeatureSet, compute_metric_suite, load_feature_chunks, trustworthiness


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SSL feature extraction and metric suite.")
    parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint (.ckpt).")
    parser.add_argument(
        "--config", type=Path, help="Optional args.json path (defaults to alongside checkpoint)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root directory with miniImageNet splits (train/val/test). Required for extraction.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory where extracted features are (or will be) stored.",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=["backbone", "projector", "both"],
        default=["both"],
        help="Feature spaces to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for extraction.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for extraction.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of samples per feature chunk file when extracting.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip feature extraction even if checkpoint is provided.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force feature extraction even if chunks already exist.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Path to write metrics summary JSON. Defaults to <features_dir>/metrics.json.",
    )
    parser.add_argument(
        "--uniformity-samples",
        type=int,
        default=20000,
        help="Number of pair samples for uniformity metric.",
    )
    parser.add_argument(
        "--id-subsample",
        type=int,
        default=5000,
        help="Subsample size for Two-NN intrinsic dimension (set <=0 to disable subsampling).",
    )
    return parser.parse_args()


def resolve_spaces(spaces: List[str]) -> List[str]:
    if "both" in spaces:
        return ["backbone", "projector"]
    return spaces


def ensure_features(
    checkpoint: Path,
    config: Path,
    data_root: Path,
    split: str,
    features_dir: Path,
    spaces: List[str],
    batch_size: int,
    num_workers: int,
    device: str,
    chunk_size: int,
    force: bool,
) -> None:
    if not checkpoint:
        raise ValueError("Checkpoint path is required for extraction.")

    features_exist = any(features_dir.glob("*.pt"))
    if features_exist and not force:
        print(f"[run_ssl_metrics] Reusing existing features in {features_dir}.")
        return

    print(f"[run_ssl_metrics] Extracting features into {features_dir} ...")
    extract_namespace = SimpleNamespace(
        checkpoint=checkpoint,
        config=config,
        data_root=data_root,
        split=split,
        output_dir=features_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        spaces=spaces,
        chunk_size=chunk_size,
        shuffle=False,
    )
    run_extraction(extract_namespace)


def main() -> None:
    args = parse_cli()
    spaces = resolve_spaces(args.spaces)

    if not args.skip_extraction:
        if not args.checkpoint or not args.data_root:
            raise ValueError("--checkpoint and --data-root are required unless --skip-extraction is set.")
        ensure_features(
            checkpoint=args.checkpoint,
            config=args.config,
            data_root=args.data_root,
            split=args.split,
            features_dir=args.features_dir,
            spaces=spaces,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            chunk_size=args.chunk_size,
            force=args.force_extract,
        )

    metrics: Dict[str, Dict[str, float]] = {}
    for space in spaces:
        print(f"[run_ssl_metrics] Computing metrics for space '{space}' ...")
        metrics[space] = compute_metric_suite(
            args.features_dir,
            space=space,
            sample_uniformity=args.uniformity_samples,
            id_subsample=args.id_subsample if args.id_subsample > 0 else None,
        )

    if set(spaces) == {"backbone", "projector"}:
        backbone_set: FeatureSet = load_feature_chunks(args.features_dir, "backbone").l2_normalize()
        projector_set: FeatureSet = load_feature_chunks(args.features_dir, "projector").l2_normalize()
        # Align on indexes
        if not np.array_equal(backbone_set.indexes, projector_set.indexes):
            raise RuntimeError("Backbone and projector feature indexes do not align.")
        metrics["cross_space"] = {
            "trustworthiness": trustworthiness(backbone_set.view(0), projector_set.view(0))
        }

    metrics_path = args.metrics_output or (args.features_dir / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[run_ssl_metrics] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

