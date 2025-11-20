#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to sys.path to allow imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root))
# Add ssl directory to path to avoid conflict with standard library 'ssl' module
sys.path.append(str(project_root / "ssl"))

from custom_metrics import (
    alignment,
    uniformity,
    vicreg_covariance,
    vicreg_invariance,
    vicreg_variance,
)
from metric_transforms import MetricTransform

from fsl.cdfsl.backbone import ResNet10


def adapt_state_dict(state_dict):
    """
    Adapts a solo-learn (torchvision) ResNet10 state_dict to fsl/cdfsl/backbone.py ResNet10.
    """
    new_state_dict = {}

    # Mapping for the initial part
    # torchvision: conv1 -> fsl: trunk.0
    # torchvision: bn1 -> fsl: trunk.1

    # Mapping for blocks
    # ResNet10 has 4 layers, each with 1 block.
    # torchvision: layer1.0 -> fsl: trunk.4
    # torchvision: layer2.0 -> fsl: trunk.5
    # torchvision: layer3.0 -> fsl: trunk.6
    # torchvision: layer4.0 -> fsl: trunk.7

    layer_to_trunk = {
        "layer1.0": "trunk.4",
        "layer2.0": "trunk.5",
        "layer3.0": "trunk.6",
        "layer4.0": "trunk.7",
    }

    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue

        # Remove prefix
        name = k.replace("backbone.", "")

        new_name = None

        if name.startswith("conv1."):
            new_name = name.replace("conv1.", "trunk.0.")
        elif name.startswith("bn1."):
            new_name = name.replace("bn1.", "trunk.1.")
        elif any(name.startswith(l) for l in layer_to_trunk):
            # Handle block layers
            for layer_prefix, trunk_prefix in layer_to_trunk.items():
                if name.startswith(layer_prefix):
                    inner_name = name.replace(layer_prefix + ".", "")

                    # Map BasicBlock (tv) to SimpleBlock (fsl)
                    if inner_name.startswith("conv1."):
                        inner_name = inner_name.replace("conv1.", "C1.")
                    elif inner_name.startswith("bn1."):
                        inner_name = inner_name.replace("bn1.", "BN1.")
                    elif inner_name.startswith("conv2."):
                        inner_name = inner_name.replace("conv2.", "C2.")
                    elif inner_name.startswith("bn2."):
                        inner_name = inner_name.replace("bn2.", "BN2.")
                    elif inner_name.startswith("downsample.0."):
                        inner_name = inner_name.replace("downsample.0.", "shortcut.")
                    elif inner_name.startswith("downsample.1."):
                        inner_name = inner_name.replace("downsample.1.", "BNshortcut.")

                    new_name = f"{trunk_prefix}.{inner_name}"
                    break

        if new_name:
            new_state_dict[new_name] = v
        else:
            # Ignore fc or other layers not in fsl ResNet
            pass

    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Calculate SSL metrics on miniImageNet")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset (miniImageNet)"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    # Initialize model
    model = ResNet10(flatten=True)

    # Adapt and load state dict
    state_dict = adapt_state_dict(ckpt["state_dict"])

    # Check for missing keys
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")

    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()

    print(f"Loading data from {args.data_path}")
    # Dataset
    try:
        from dali_metric_transforms import build_dali_metric_loader
        USE_DALI = True
    except ImportError:
        print("DALI loader not found or DALI not installed. Falling back to torchvision.")
        USE_DALI = False

    if USE_DALI:
        dataloader, epoch_size = build_dali_metric_loader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=4,
            device=args.device,
        )
        # DALI output is a bit different
    else:
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(args.data_path, transform=MetricTransform())
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False
        )

    # Collect embeddings
    z1_list = []
    z2_list = []
    z_weak_list = []

    print("Computing embeddings...", flush=True)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding batches"):
            if USE_DALI:
                # batch is a list of dictionaries or tuples from DALIGenericIterator
                # We configured output_map=["x1", "x2", "x_weak"]
                # DALIGenericIterator returns a list of batch objects (one per GPU if using multiple, but we use 1)
                # Each element is a dict: {'x1': tensor, 'x2': tensor, 'x_weak': tensor}
                
                # batch[0] is the dict for the first pipeline (we only have 1)
                data_dict = batch[0]
                x1 = data_dict["x1"]
                x2 = data_dict["x2"]
                x_weak = data_dict["x_weak"]
                
                # DALI tensors might need to be moved/converted if they aren't already torch tensors?
                # DALIGenericIterator returns torch tensors by default if using pytorch plugin.
            else:
                (x1, x2, x_weak), _ = batch

            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            x_weak = x_weak.to(args.device)

            # Forward pass
            # ResNet10(flatten=True) returns (batch, 512)
            out1 = model(x1)
            out2 = model(x2)
            out_weak = model(x_weak)

            z1_list.append(out1.cpu())
            z2_list.append(out2.cpu())
            z_weak_list.append(out_weak.cpu())

    z1 = torch.cat(z1_list, dim=0)
    z2 = torch.cat(z2_list, dim=0)
    z_weak = torch.cat(z_weak_list, dim=0)

    print(f"Collected embeddings: {z1.shape}")

    # Move to device for metric calculation (or keep on CPU if memory is tight)
    # miniImageNet is small enough for GPU usually, but let's be safe and use chunks if needed.
    # Actually, standard metrics are fast enough on CPU for this size, or we can put back to GPU.
    device = args.device

    z1 = z1.to(device)
    z2 = z2.to(device)
    z_weak = z_weak.to(device)

    # Normalize for Alignment / Uniformity
    z1_norm = torch.nn.functional.normalize(z1, p=2, dim=1)
    z2_norm = torch.nn.functional.normalize(z2, p=2, dim=1)
    z_weak_norm = torch.nn.functional.normalize(z_weak, p=2, dim=1)

    print("Calculating metrics...")

    # Alignment
    align_val = alignment(z1_norm, z2_norm).item()
    print(f"Alignment: {align_val:.4f}")

    # Uniformity
    # For uniformity, we use a subset if N is too large, but 38k^2 is ~1.4e9, which is big.
    # The formula sums over all pairs. O(N^2).
    # 38,000^2 interactions.
    # We can approximate it by subsampling.
    MAX_SAMPLES_UNIF = 4096
    if z_weak_norm.size(0) > MAX_SAMPLES_UNIF:
        indices = torch.randperm(z_weak_norm.size(0))[:MAX_SAMPLES_UNIF]
        z_weak_subset = z_weak_norm[indices]
    else:
        z_weak_subset = z_weak_norm

    unif_val = uniformity(z_weak_subset).item()
    print(f"Uniformity: {unif_val:.4f}")

    # VICReg
    # Invariance (MSE on unnormalized)
    inv_val = vicreg_invariance(z1, z2).item()
    print(f"VICReg Invariance: {inv_val:.4f}")

    # Variance (on weak view)
    var_val = vicreg_variance(z_weak).item()
    print(f"VICReg Variance: {var_val:.4f}")

    # Covariance (on weak view)
    cov_val = vicreg_covariance(z_weak).item()
    print(f"VICReg Covariance: {cov_val:.4f}")

    results = {
        "alignment": align_val,
        "uniformity": unif_val,
        "vicreg_invariance": inv_val,
        "vicreg_variance": var_val,
        "vicreg_covariance": cov_val,
    }

    print("\nSummary:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
