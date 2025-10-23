"""Benchmark augmentation pipeline to identify bottlenecks.

This script measures the actual time spent in augmentations (which profiler can't see).
"""

import sys
import time
from pathlib import Path

from torchvision import datasets

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.methods.simclr.augmentations import get_simclr_augmentations


def benchmark_augmentations(num_samples=100):
    """Benchmark augmentation pipeline speed."""
    print("=" * 80)
    print("AUGMENTATION BENCHMARK")
    print("=" * 80)

    # Load CIFAR-10
    dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=None)

    # Get augmentation pipeline
    transform = get_simclr_augmentations(size=32, s=1.0)

    print(f"\nTesting {num_samples} images...")
    print("Image size: 32x32")
    print("\nTransform pipeline:")
    print(transform)
    print("\n" + "-" * 80)

    # Warmup
    for i in range(10):
        img, _ = dataset[i]
        _ = transform(img)

    # Benchmark
    times = []
    for i in range(num_samples):
        img, _ = dataset[i]

        start = time.perf_counter()
        augmented = transform(img)
        end = time.perf_counter()

        times.append(end - start)

    # Statistics
    times = [t * 1000 for t in times]  # Convert to ms
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\nResults (per image):")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")

    # Calculate throughput
    throughput_per_worker = 1000 / avg_time
    print(f"\nThroughput per worker: {throughput_per_worker:.1f} images/sec")

    # Simulate multi-worker scenario
    print("\n" + "-" * 80)
    print("MULTI-WORKER SIMULATION")
    print("-" * 80)

    for num_workers in [4, 8, 16, 24]:
        total_throughput = throughput_per_worker * num_workers
        images_per_batch = 256  # batch size
        batches_per_sec = total_throughput / images_per_batch
        ms_per_batch = 1000 / batches_per_sec if batches_per_sec > 0 else float("inf")

        print(f"\nWorkers: {num_workers}")
        print(f"  Total throughput: {total_throughput:.1f} images/sec")
        print(f"  Time per batch (bs=256): {ms_per_batch:.1f}ms")

        # For batch_size=512
        images_per_batch_512 = 512
        batches_per_sec_512 = total_throughput / images_per_batch_512
        ms_per_batch_512 = 1000 / batches_per_sec_512 if batches_per_sec_512 > 0 else float("inf")
        print(f"  Time per batch (bs=512): {ms_per_batch_512:.1f}ms")

    # GPU target
    print("\n" + "-" * 80)
    print("GPU TARGET ANALYSIS")
    print("-" * 80)

    # Assume GPU can process at 10ms/batch (conservative for A100)
    gpu_time_per_batch = 10  # ms
    gpu_images_per_sec = (512 / gpu_time_per_batch) * 1000

    print(f"\nAssuming GPU processes 512-batch in {gpu_time_per_batch}ms:")
    print(f"  GPU demand: {gpu_images_per_sec:.0f} images/sec")

    required_workers = gpu_images_per_sec / throughput_per_worker
    print(f"  Required workers to saturate GPU: {required_workers:.1f}")

    if required_workers > 24:
        print("\n⚠️  WARNING: Even with 24 workers, CPU can't keep up!")
        print("     Augmentations are TOO SLOW for this GPU.")
        print("     Consider:")
        print("       - GPU-based augmentations (Kornia)")
        print("       - Simpler augmentation pipeline")
        print("       - Larger batch size")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_augmentations(num_samples=100)
