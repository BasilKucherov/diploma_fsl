from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors


@dataclass
class FeatureSet:
    """Container for stacked multi-view features."""

    indexes: np.ndarray
    labels: Optional[np.ndarray]
    features: np.ndarray  # shape (N, V, D)

    def l2_normalize(self) -> "FeatureSet":
        feats = self.features
        norms = np.linalg.norm(feats, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = feats / norms
        return FeatureSet(self.indexes.copy(), None if self.labels is None else self.labels.copy(), normalized)

    def view(self, idx: int) -> np.ndarray:
        return self.features[:, idx, :]

    def paired_views(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.features.shape[1] < 2:
            raise ValueError("Expected at least 2 views to compute paired metrics.")
        return self.view(0), self.view(1)


def load_feature_chunks(directory: Path, space: str) -> FeatureSet:
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No feature chunks found under {directory}")

    indexes: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    feats: List[torch.Tensor] = []

    for file in files:
        payload = torch.load(file, map_location="cpu")
        data: Dict[str, torch.Tensor] = payload["data"]
        if space not in data or data[space] is None:
            continue

        indexes.append(data["indexes"].long())
        if data.get("labels") is not None:
            labels.append(data["labels"].long())
        feats.append(data[space].float())

    if not feats:
        raise RuntimeError(f"No tensors for space='{space}' were found in {directory}")

    indexes_cat = torch.cat(indexes, dim=0).numpy()
    labels_cat = torch.cat(labels, dim=0).numpy() if labels else None
    feats_cat = torch.cat(feats, dim=0).numpy()

    # Sort by original index to keep deterministic order.
    order = np.argsort(indexes_cat)
    indexes_sorted = indexes_cat[order]
    labels_sorted = labels_cat[order] if labels_cat is not None else None
    feats_sorted = feats_cat[order]

    return FeatureSet(indexes_sorted, labels_sorted, feats_sorted)


def alignment(z1: np.ndarray, z2: np.ndarray) -> float:
    diff = z1 - z2
    return float(np.mean(np.sum(diff * diff, axis=1)))


def uniformity(z: np.ndarray, sample_size: int = 20000, seed: Optional[int] = None) -> float:
    n = z.shape[0]
    if n < 2:
        return math.nan
    rng = np.random.default_rng(seed)
    max_pairs = n * (n - 1)
    m = min(sample_size, max_pairs)

    idx1 = rng.integers(0, n, size=m)
    idx2 = rng.integers(0, n, size=m)
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    if idx1.size == 0:
        return math.nan

    diff = z[idx1] - z[idx2]
    sq_norm = np.sum(diff * diff, axis=1)
    val = np.exp(-2.0 * sq_norm)
    val = np.mean(val)
    return float(np.log(val + 1e-12))


def vicreg_terms(z1: np.ndarray, z2: np.ndarray, gamma: float = 1.0, eps: float = 1e-4) -> Dict[str, float]:
    invariance = float(np.mean(np.sum((z1 - z2) ** 2, axis=1)))

    def variance_penalty(z: np.ndarray) -> Tuple[float, float]:
        std = np.sqrt(np.var(z, axis=0) + eps)
        penalty = np.mean(np.maximum(0.0, gamma - std))
        collapsed = float(np.mean(std < gamma))
        return float(penalty), collapsed

    var1, collapse1 = variance_penalty(z1)
    var2, collapse2 = variance_penalty(z2)
    variance = 0.5 * (var1 + var2)
    collapsed_fraction = 0.5 * (collapse1 + collapse2)

    def covariance_penalty(z: np.ndarray) -> float:
        centered = z - z.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / (z.shape[0] - 1)
        off_diag = cov - np.diag(np.diag(cov))
        return float(np.sum(off_diag ** 2) / z.shape[1])

    cov1 = covariance_penalty(z1)
    cov2 = covariance_penalty(z2)
    covariance = 0.5 * (cov1 + cov2)

    return {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "collapsed_fraction": collapsed_fraction,
    }


def covariance_eigenvalues(z: np.ndarray) -> np.ndarray:
    centered = z - z.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.flip(np.sort(np.real(eigvals)))
    return eigvals


def eigenspectrum_slope(eigvals: np.ndarray, tail_start: int = 5, tail_end_frac: float = 0.5) -> Dict[str, float]:
    d = eigvals.shape[0]
    if d <= tail_start + 1:
        return {"alpha": math.nan, "r2": math.nan}
    tail_end = max(tail_start + 1, int(d * tail_end_frac))
    xs = np.log(np.arange(1, d + 1)[tail_start:tail_end])
    ys = np.log(eigvals[tail_start:tail_end] + 1e-12)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    y_pred = slope * xs + intercept
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {"alpha": float(slope), "r2": float(r2)}


def effective_rank(eigvals: np.ndarray, eps: float = 1e-12) -> float:
    total = np.sum(eigvals)
    if total <= eps:
        return math.nan
    p = eigvals / total
    entropy = -np.sum(p * np.log(p + eps))
    return float(np.exp(entropy))


def participation_ratio(eigvals: np.ndarray, eps: float = 1e-12) -> float:
    numerator = np.sum(eigvals)
    denominator = np.sum(eigvals ** 2) + eps
    if denominator <= eps:
        return math.nan
    return float((numerator ** 2) / denominator)


def variance_explained(eigvals: np.ndarray, thresholds: Sequence[float]) -> Dict[float, int]:
    total = np.sum(eigvals)
    if total <= 0:
        return {float(th): -1 for th in thresholds}
    cumulative = np.cumsum(eigvals)
    results: Dict[float, int] = {}
    for th in thresholds:
        idx = np.searchsorted(cumulative, th * total, side="left")
        results[float(th)] = int(idx + 1)
    return results


def two_nn_intrinsic_dimension(z: np.ndarray, subsample: Optional[int] = None, metric: str = "euclidean") -> float:
    if subsample is not None and subsample < z.shape[0]:
        rng = np.random.default_rng(0)
        perm = rng.choice(z.shape[0], size=subsample, replace=False)
        z = z[perm]

    if z.shape[0] < 3:
        return math.nan

    nn = NearestNeighbors(n_neighbors=3, metric=metric, algorithm="auto")
    nn.fit(z)
    distances, _ = nn.kneighbors(z)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    ratios = r2 / (r1 + 1e-12)
    valid = ratios > 1.0
    ratios = ratios[valid]
    if ratios.size == 0:
        return math.nan
    estimate = (ratios.size - 1) / np.sum(np.log(ratios))
    return float(estimate)


def knn_radius(z: np.ndarray, k: int = 10, metric: str = "euclidean") -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(z)
    distances, _ = nn.kneighbors(z)
    kth = distances[:, -1]
    return float(np.mean(kth))


def hubness(z: np.ndarray, k: int = 10, metric: str = "euclidean") -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(z)
    _, indices = nn.kneighbors(z)
    neighbor_counts = np.bincount(indices[:, 1:].reshape(-1), minlength=z.shape[0])
    mean = np.mean(neighbor_counts)
    std = np.std(neighbor_counts)
    if std == 0:
        return 0.0
    skewness = np.mean(((neighbor_counts - mean) / std) ** 3)
    return float(skewness)


def trustworthiness(reference: np.ndarray, embedding: np.ndarray, n_neighbors: int = 10) -> float:
    return float(sklearn_trustworthiness(reference, embedding, n_neighbors=n_neighbors))


def spectrum_based_metrics(z: np.ndarray) -> Dict[str, float]:
    eigvals = covariance_eigenvalues(z)
    metrics = {}
    metrics.update(eigenspectrum_slope(eigvals))
    metrics["effective_rank"] = effective_rank(eigvals)
    metrics["participation_ratio"] = participation_ratio(eigvals)
    for threshold, k in variance_explained(eigvals, thresholds=(0.9, 0.95)).items():
        metrics[f"topk_{int(threshold*100)}"] = float(k)
    metrics["pr_id"] = metrics["participation_ratio"]
    return metrics


def compute_metric_suite(
    feature_dir: Path,
    space: str,
    sample_uniformity: int = 20000,
    id_subsample: Optional[int] = 5000,
) -> Dict[str, float]:
    feats = load_feature_chunks(feature_dir, space).l2_normalize()
    z1, z2 = feats.paired_views()
    metrics: Dict[str, float] = {}
    metrics["alignment"] = alignment(z1, z2)
    metrics["uniformity"] = uniformity(z1, sample_uniformity)

    if space == "projector":
        vic = vicreg_terms(z1, z2)
        for key, value in vic.items():
            metrics[f"vicreg_{key}"] = value

    # Merge views by stacking for spectrum metrics.
    merged = np.concatenate([z1, z2], axis=0)
    metrics.update(spectrum_based_metrics(merged))
    metrics["two_nn_id"] = two_nn_intrinsic_dimension(merged, subsample=id_subsample)
    metrics["knn_radius"] = knn_radius(merged)
    metrics["hubness"] = hubness(merged)
    return metrics

