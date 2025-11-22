import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Add cdfsl to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CDFSL_DIR = os.path.join(CURRENT_DIR, "cdfsl")
sys.path.insert(0, CDFSL_DIR)

# Patch configs before importing datasets
import configs


def natural_sort_key(s):
    """Sort strings containing numbers naturally."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Bulk Few-Shot Learning Evaluation")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--output_file", type=str, default="fsl_results.json", help="Output JSON file"
    )
    parser.add_argument(
        "--datasets_dir", type=str, required=True, help="Root directory for datasets"
    )
    parser.add_argument("--every_n", type=int, default=1, help="Process every Nth checkpoint")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=600)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_backbone_state(model, ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()

    layer_map = {
        "backbone.conv1": "trunk.0",
        "backbone.bn1": "trunk.1",
        "backbone.layer1.0": "trunk.4",
        "backbone.layer2.0": "trunk.5",
        "backbone.layer3.0": "trunk.6",
        "backbone.layer4.0": "trunk.7",
    }

    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue

        mapped = False
        for prefix, target in layer_map.items():
            if k.startswith(prefix):
                suffix = k[len(prefix) :]
                if "downsample.0" in suffix:
                    suffix = suffix.replace("downsample.0", "shortcut")
                elif "downsample.1" in suffix:
                    suffix = suffix.replace("downsample.1", "BNshortcut")
                elif "conv1" in suffix:
                    suffix = suffix.replace("conv1", "C1")
                elif "bn1" in suffix:
                    suffix = suffix.replace("bn1", "BN1")
                elif "conv2" in suffix:
                    suffix = suffix.replace("conv2", "C2")
                elif "bn2" in suffix:
                    suffix = suffix.replace("bn2", "BN2")

                new_key = target + suffix
                new_state_dict[new_key] = v
                mapped = True
                break

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()


def get_datamgr(dataset_name, data_path, n_way, n_shot, n_query, n_episodes):
    if dataset_name == "CropDisease":
        configs.CropDisease_path = data_path
        from datasets.CropDisease_few_shot import SetDataManager
    elif dataset_name == "EuroSAT":
        configs.EuroSAT_path = data_path
        from datasets.EuroSAT_few_shot import SetDataManager
    elif dataset_name == "ISIC":
        configs.ISIC_path = data_path
        from datasets.ISIC_few_shot import SetDataManager
    elif dataset_name == "ChestX":
        configs.ChestX_path = data_path
        from datasets.Chest_few_shot import SetDataManager

    datamgr = SetDataManager(
        image_size=224, n_way=n_way, n_support=n_shot, n_query=n_query, n_eposide=n_episodes
    )
    return datamgr


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def evaluate(model, dataloader, n_way, n_shot, n_query, device, batch_size=8):
    acc_all = []

    # Manual batching of episodes
    batch_x = []
    batch_y = []  # Not used but kept for structure

    with torch.no_grad():
        iterator = iter(dataloader)
        done = False

        while not done:
            # Collect batch
            batch_x = []
            try:
                for _ in range(batch_size):
                    x, y = next(iterator)
                    batch_x.append(x)
            except StopIteration:
                done = True

            if not batch_x:
                break

            # Stack batch: [B, n_way, n_s+n_q, C, H, W]
            x_stack = torch.stack(batch_x)

            # ... same logic as before ...
            x = x_stack

            if x.dim() == 5:
                x = x.unsqueeze(0)

            B = x.size(0)
            p = n_shot + n_query
            n_samples = B * n_way * p

            x = x.to(device)
            # Flatten for backbone: [B*n_way*p, C, H, W]
            x_flat = x.view(n_samples, *x.shape[3:])

            feats = model(x_flat)
            # Reshape: [B, n_way, p, d]
            feats = feats.view(B, n_way, p, -1)

            # Split support and query
            z_support = feats[:, :, :n_shot]  # [B, n_way, n_shot, d]
            z_query = feats[:, :, n_shot:]  # [B, n_way, n_query, d]

            # Prototypes: [B, n_way, d]
            z_proto = z_support.mean(2)

            # Query flat for distance: [B, n_way*n_query, d]
            z_query_flat = z_query.contiguous().view(B, n_way * n_query, -1)

            # Compute distances for each batch element
            for b in range(B):
                # Per episode
                zp = z_proto[b]  # [n_way, d]
                zq = z_query_flat[b]  # [n_query_total, d]

                dists = euclidean_dist(zq, zp)  # [n_query_total, n_way]
                scores = -dists

                y_query = torch.arange(n_way).repeat_interleave(n_query).to(device)

                topk_scores, topk_labels = scores.topk(1, 1, True, True)
                top1_correct = topk_labels.eq(y_query.unsqueeze(1)).sum().item()

                acc = top1_correct / (n_way * n_query) * 100
                acc_all.append(acc)

    acc_all = np.array(acc_all)
    mean = np.mean(acc_all)
    std = np.std(acc_all)
    conf_interval = 1.96 * std / np.sqrt(len(acc_all))

    return mean, conf_interval, acc_all


def main():
    args = parse_args()

    # Setup Datasets
    datasets_map = {
        "CropDisease": os.path.join(args.datasets_dir, "PlantDisease"),
        "EuroSAT": os.path.join(args.datasets_dir, "EuroSAT/2750"),
        "ISIC": os.path.join(args.datasets_dir, "ISIC"),
        "ChestX": os.path.join(args.datasets_dir, "ChestXrays"),
    }

    # Initialize Datasets once (with num_workers=0 for safety/local)
    dataloaders = {}
    print("Initializing datasets...")
    for name, path in datasets_map.items():
        print(f"  Loading {name} from {path}...")
        try:
            dm = get_datamgr(name, path, args.n_way, args.n_shot, args.n_query, args.n_episodes)
            # Apply the same logic as evaluate_custom.py (files are already patched to num_workers=0)
            dataloaders[name] = dm.get_data_loader(aug=False)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # Initialize Model
    from backbone import ResNet10

    model = ResNet10(flatten=True)
    model.to(args.device)

    # Find Checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    ckpts.sort(key=lambda x: natural_sort_key(x.name))

    # Filter checkpoints
    ckpts = ckpts[:: args.every_n]

    print(f"Found {len(ckpts)} checkpoints to evaluate.")

    results = {}

    # Resume functionality
    if os.path.exists(args.output_file):
        print(f"Loading existing results from {args.output_file}")
        try:
            with open(args.output_file, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not decode existing results file. Starting fresh.")

    # Add metadata to results if not present
    if "metadata" not in results:
        results["metadata"] = {
            "n_way": args.n_way,
            "n_shot": args.n_shot,
            "n_query": args.n_query,
            "n_episodes": args.n_episodes,
            "every_n": args.every_n,
        }

    for i, ckpt in enumerate(ckpts):
        # Extract epoch
        match = re.search(r"ep=(\d+)", ckpt.name)
        epoch_key = f"epoch_{match.group(1)}" if match else f"ckpt_{i}"

        # Check if epoch already exists and has all datasets
        if epoch_key in results:
            # Check if all datasets are present
            all_present = True
            for ds_name in dataloaders.keys():
                if ds_name not in results[epoch_key]:
                    all_present = False
                    break

            if all_present:
                print(f"[{i+1}/{len(ckpts)}] Skipping {ckpt.name} (already processed)")
                continue

        print(f"\n[{i+1}/{len(ckpts)}] Processing {ckpt.name}")

        # Load weights
        try:
            load_backbone_state(model, ckpt, args.device)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        # Evaluate on all datasets
        if epoch_key not in results:
            results[epoch_key] = {}

        epoch_results = results[epoch_key]

        for ds_name, loader in dataloaders.items():
            if ds_name in epoch_results:
                continue  # Skip dataset if already done for this epoch

            print(f"  Evaluating on {ds_name}...", end="", flush=True)
            try:
                # Run evaluation
                mean, conf, raw_acc = evaluate(
                    model, loader, args.n_way, args.n_shot, args.n_query, args.device
                )
                epoch_results[ds_name] = {
                    "acc": float(mean),
                    "conf": float(conf),
                    "raw_acc": raw_acc.tolist(),  # Store raw accuracy
                }
                print(f" Acc: {mean:.2f}%")
            except Exception as e:
                print(f" Error: {e}")
                import traceback

                traceback.print_exc()

        results[epoch_key] = epoch_results

        # Periodic Save (every checkpoint to be safe with large jobs)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4, sort_keys=True)

    # Final Save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
