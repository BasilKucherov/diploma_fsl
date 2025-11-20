import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train SSL methods on miniImageNet")
    parser.add_argument(
        "method", choices=["vicreg", "simclr", "byol", "swav"], help="SSL method to train"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode (CPU, small batch, few epochs)"
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help="Comma-separated list of GPU IDs (e.g. '0,1')"
    )

    args = parser.parse_args()

    # Get absolute path to main_pretrain.py
    current_dir = Path(__file__).parent.absolute()
    script_path = current_dir / "main_pretrain.py"

    if not script_path.exists():
        print(f"Error: Could not find {script_path}")
        sys.exit(1)

    # Base command
    cmd = [
        sys.executable,
        str(script_path),
        "--config-path",
        "scripts/pretrain/miniimagenet",
        "--config-name",
        f"{args.method}.yaml",
    ]

    if args.debug:
        print("Running in DEBUG mode (CPU, small dataset)")
        cmd.extend(
            [
                "accelerator=cpu",
                "devices=1",
                "optimizer.batch_size=4",
                "data.num_workers=0",
                "max_epochs=2",
                "checkpoint.frequency=1",
                "wandb.enabled=False",
                "sync_batchnorm=False",
            ]
        )

        if sys.platform == "darwin":
            print("MacOS detected, switching to image_folder format (DALI not supported)")
            cmd.append("data.format=image_folder")
            cmd.append("+data.fraction=0.002")
        else:
            cmd.append("data.fraction=0.002")
            cmd.append("dali.device=cpu")

    else:
        print(f"Running in TRAIN mode on GPUs: {args.gpus}")
        # Convert "0,1" to "[0,1]"
        device_list = f"[{args.gpus}]"
        cmd.extend(
            [
                f"devices={device_list}",
                "dali.device=gpu",
            ]
        )

    print(f"Executing command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    try:
        subprocess.check_call(cmd, cwd=current_dir, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
