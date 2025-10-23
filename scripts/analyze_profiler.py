"""Analyze PyTorch profiler trace files and generate summary statistics.

Usage:
    python scripts/analyze_profiler.py <path_to_trace.json>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_trace(trace_path):
    """Analyze a PyTorch profiler trace file."""
    print(f"Loading trace file: {trace_path}")

    with open(trace_path) as f:
        data = json.load(f)

    # Parse trace events
    events = data.get("traceEvents", [])

    # Categorize by operation name
    cpu_times = defaultdict(list)
    cuda_times = defaultdict(list)

    for event in events:
        if event.get("ph") == "X":  # Complete events (duration events)
            name = event.get("name", "")
            dur = event.get("dur", 0)  # Duration in microseconds
            cat = event.get("cat", "")

            # Skip very short events
            if dur < 1:
                continue

            # Categorize by thread
            if "cpu" in cat.lower() or event.get("pid") == 0:
                cpu_times[name].append(dur)
            elif "cuda" in cat.lower() or "gpu" in cat.lower():
                cuda_times[name].append(dur)

    # Calculate statistics
    def calc_stats(times_dict):
        stats = []
        for name, durations in times_dict.items():
            total = sum(durations)
            count = len(durations)
            avg = total / count if count > 0 else 0
            stats.append(
                {
                    "name": name,
                    "total_ms": total / 1000,  # Convert to ms
                    "count": count,
                    "avg_ms": avg / 1000,
                    "percent": 0,  # Will calculate later
                }
            )

        # Calculate percentages
        total_time = sum(s["total_ms"] for s in stats)
        for s in stats:
            s["percent"] = (s["total_ms"] / total_time * 100) if total_time > 0 else 0

        # Sort by total time
        stats.sort(key=lambda x: x["total_ms"], reverse=True)
        return stats

    cpu_stats = calc_stats(cpu_times)
    cuda_stats = calc_stats(cuda_times)

    # Print results
    print("\n" + "=" * 100)
    print("PROFILER ANALYSIS SUMMARY")
    print("=" * 100)

    print("\n" + "-" * 100)
    print("TOP 30 CPU OPERATIONS (by total time)")
    print("-" * 100)
    print(f"{'Operation':<60} {'Total (ms)':>12} {'Count':>8} {'Avg (ms)':>10} {'%':>6}")
    print("-" * 100)

    for i, stat in enumerate(cpu_stats[:30], 1):
        name = stat["name"][:58]
        print(
            f"{name:<60} {stat['total_ms']:>12.2f} {stat['count']:>8} {stat['avg_ms']:>10.3f} {stat['percent']:>6.2f}"
        )

    if cuda_stats:
        print("\n" + "-" * 100)
        print("TOP 30 CUDA OPERATIONS (by total time)")
        print("-" * 100)
        print(f"{'Operation':<60} {'Total (ms)':>12} {'Count':>8} {'Avg (ms)':>10} {'%':>6}")
        print("-" * 100)

        for i, stat in enumerate(cuda_stats[:30], 1):
            name = stat["name"][:58]
            print(
                f"{name:<60} {stat['total_ms']:>12.2f} {stat['count']:>8} {stat['avg_ms']:>10.3f} {stat['percent']:>6.2f}"
            )

    # Identify bottlenecks
    print("\n" + "=" * 100)
    print("BOTTLENECK ANALYSIS")
    print("=" * 100)

    # Look for common bottlenecks
    bottlenecks = []

    for stat in cpu_stats[:10]:
        name_lower = stat["name"].lower()
        if any(
            keyword in name_lower
            for keyword in ["dataloader", "augment", "transform", "blur", "crop", "flip"]
        ):
            bottlenecks.append(
                f"  - {stat['name']}: {stat['total_ms']:.1f}ms ({stat['percent']:.1f}%)"
            )

    if bottlenecks:
        print("\nData Loading / Augmentation Bottlenecks:")
        for b in bottlenecks:
            print(b)

    # Check GPU utilization
    total_cpu_time = sum(s["total_ms"] for s in cpu_stats)
    total_cuda_time = sum(s["total_ms"] for s in cuda_stats)

    print(f"\nTotal CPU time: {total_cpu_time:.1f}ms")
    print(f"Total CUDA time: {total_cuda_time:.1f}ms")

    if total_cuda_time > 0:
        ratio = total_cpu_time / total_cuda_time
        if ratio > 2:
            print("\n⚠️  WARNING: CPU time is significantly higher than GPU time!")
            print("   This suggests the GPU is starved and waiting for data.")
            print("   Solutions:")
            print("   - Increase num_workers in DataLoader")
            print("   - Increase prefetch_factor")
            print("   - Optimize CPU augmentations")
            print("   - Move augmentations to GPU")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_profiler.py <path_to_trace.json>")
        sys.exit(1)

    trace_path = Path(sys.argv[1])
    if not trace_path.exists():
        print(f"Error: File not found: {trace_path}")
        sys.exit(1)

    analyze_trace(trace_path)
