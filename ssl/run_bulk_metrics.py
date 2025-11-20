import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def natural_sort_key(s):
    """Sort strings containing numbers naturally."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


def main():
    parser = argparse.ArgumentParser(
        description="Run metrics calculation on all checkpoints in a directory"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset (miniImageNet)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics_results.json",
        help="Output file for collected metrics",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--script_path",
        type=str,
        default="ssl/calc_metrics.py",
        help="Path to calc_metrics.py script",
    )

    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"Error: Directory {ckpt_dir} does not exist")
        sys.exit(1)

    # Find all .ckpt files
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    # Sort naturally to handle ep=1, ep=2, ..., ep=10 correctly
    ckpts.sort(key=lambda x: natural_sort_key(x.name))

    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        sys.exit(0)

    print(f"Found {len(ckpts)} checkpoints.")

    all_results = {}

    for i, ckpt in enumerate(ckpts):
        print(f"\n[{i+1}/{len(ckpts)}] Processing {ckpt.name}...")

        # Extract epoch number for cleaner key
        match = re.search(r"ep=(\d+)", ckpt.name)
        epoch = int(match.group(1)) if match else i

        # Set environment variable for unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            sys.executable,
            "-u",  # Force unbuffered binary stdout/stderr
            args.script_path,
            "--ckpt_path",
            str(ckpt),
            "--data_path",
            args.data_path,
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
        ]

        try:
            # Run the script and capture output. Using Popen to stream stdout.
            # universal_newlines=True is same as text=True
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout so we capture tqdm
                text=True,
                bufsize=1,  # Line buffered
                env=env,
            )

            output_lines = []
            # Print output in real-time, but indented
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.rstrip()
                    # Forward tqdm/status updates
                    if "Encoding batches" in line or "%" in line:
                        print(f"\r    {line}", end="")
                    else:
                        print(f"    {line}")
                    output_lines.append(line)

            # Get remaining output
            stdout_rest, stderr = process.communicate()
            if stdout_rest:
                for line in stdout_rest.splitlines():
                    print(f"    {line}")
                    output_lines.append(line)

            if process.returncode != 0:
                print(f"\n  -> Error running script (Exit code {process.returncode})")
                # stderr is merged to stdout, so it's already in output_lines
                continue

            output = "\n".join(output_lines)

            # Parse the output to find metrics
            # Expecting format:
            # Alignment: 0.1234
            # Uniformity: -1.2345
            # ...
            metrics = {}
            for line in output.splitlines():
                if ": " in line:
                    key, val = line.split(": ", 1)
                    key = key.strip().lower().replace(" ", "_")
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        continue

            # Check if we found valid metrics
            if "alignment" in metrics:
                all_results[f"epoch_{epoch}"] = metrics
                print(f"  -> Alignment: {metrics.get('alignment', 'N/A')}")
                print(f"  -> Uniformity: {metrics.get('uniformity', 'N/A')}")
            else:
                print("  -> Warning: Could not parse metrics from output")
                print(output)  # Print output for debugging

        except subprocess.CalledProcessError as e:
            print(f"  -> Error running script: {e}")
            print(e.stderr)

    # Save results
    output_path = ckpt_dir / args.output_file
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4, sort_keys=True)

    print(f"\nSaved all metrics to {output_path}")


if __name__ == "__main__":
    main()
