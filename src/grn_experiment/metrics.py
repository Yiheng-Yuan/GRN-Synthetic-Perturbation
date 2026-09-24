"""Pure scoring functions; one graph, not one cell, is the study unit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _cells(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("expected a nonempty cells-by-genes array")
    if not np.isfinite(array).all():
        raise ValueError("observations must be finite")
    return array


def sliced_wasserstein(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    n_projections: int = 64,
    seed: int = 0,
) -> float:
    """Sliced 2-Wasserstein distance between *unpaired* cell distributions.

    The projection seed is fixed by the caller for fair model comparisons.
    Different sample sizes are compared on the same quantile grid.
    """

    left, right = _cells(predicted), _cells(observed)
    if left.shape[1] != right.shape[1] or n_projections < 1:
        raise ValueError("gene dimensions must match and projections be positive")
    directions = np.random.default_rng(seed).normal(
        size=(left.shape[1], n_projections)
    )
    directions /= np.maximum(np.linalg.norm(directions, axis=0), 1e-12)
    count = max(left.shape[0], right.shape[0])
    q = (np.arange(count, dtype=np.float64) + 0.5) / count
    a = np.quantile(left @ directions, q, axis=0)
    b = np.quantile(right @ directions, q, axis=0)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def control_adjusted_rmse(
    predicted_treated: np.ndarray,
    predicted_control: np.ndarray,
    observed_treated: np.ndarray,
    observed_control: np.ndarray,
) -> float:
    """RMSE of mean treatment effects, not a cell-wise paired trajectory loss."""

    predicted_effect = _cells(predicted_treated).mean(0) - _cells(
        predicted_control
    ).mean(0)
    observed_effect = _cells(observed_treated).mean(0) - _cells(
        observed_control
    ).mean(0)
    if predicted_effect.shape != observed_effect.shape:
        raise ValueError("gene dimensions must match")
    return float(np.sqrt(np.mean((predicted_effect - observed_effect) ** 2)))


def _off_diagonal(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("expected square gene-by-gene matrix")
    return array[~np.eye(array.shape[0], dtype=bool)]


def edge_average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    """Tie-aware average precision for directed edge *existence*.

    Signed estimates are ranked by absolute magnitude; repression is not a
    negative confidence in edge existence.  Sign correctness is a separate
    metric.
    """

    score, label = _off_diagonal(scores), _off_diagonal(truth) != 0
    if score.size != label.size or not np.isfinite(score).all():
        raise ValueError("invalid edge scores")
    if not label.any():
        return float("nan")
    confidence = np.abs(score)
    order = np.argsort(-confidence, kind="mergesort")
    sorted_score, sorted_label = confidence[order], label[order]
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_score)), score.size - 1]
    cumulative_true = np.cumsum(sorted_label)[distinct_ends]
    predicted_count = distinct_ends + 1
    precision = cumulative_true / predicted_count
    recall = cumulative_true / int(label.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def signed_edge_f1(
    predicted: np.ndarray, truth: np.ndarray, *, threshold: float
) -> float:
    """Macro F1 over activating and repressing edges; zeros are negatives."""

    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    pred, target = _off_diagonal(predicted), _off_diagonal(truth)
    if pred.shape != target.shape:
        raise ValueError("matrix dimensions must match")
    assigned = np.where(np.abs(pred) >= threshold, np.sign(pred), 0)
    scores: list[float] = []
    for sign in (-1, 1):
        tp = np.count_nonzero((assigned == sign) & (np.sign(target) == sign))
        fp = np.count_nonzero((assigned == sign) & (np.sign(target) != sign))
        fn = np.count_nonzero((assigned != sign) & (np.sign(target) == sign))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else float("nan"))
    return float(np.nanmean(scores))


def jacobian_rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Off-diagonal local-effect error across one or many cell states."""

    left, right = np.asarray(predicted), np.asarray(truth)
    if left.shape != right.shape or left.ndim < 2 or left.shape[-1] != left.shape[-2]:
        raise ValueError("matching (..., genes, genes) Jacobians required")
    mask = ~np.eye(left.shape[-1], dtype=bool)
    return float(np.sqrt(np.mean((left[..., mask] - right[..., mask]) ** 2)))


def steady_state_rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    left, right = np.asarray(predicted), np.asarray(truth)
    if left.shape != right.shape or not np.isfinite(left).all():
        raise ValueError("matching finite steady-state vectors required")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def interval_coverage(
    lower: np.ndarray, upper: np.ndarray, truth: np.ndarray
) -> float:
    low, high, value = map(np.asarray, (lower, upper, truth))
    if low.shape != high.shape or low.shape != value.shape or np.any(low > high):
        raise ValueError("invalid interval arrays")
    return float(np.mean((low <= value) & (value <= high)))


def eliminated_fraction(
    registered: Iterable[str], survivors: Iterable[str]
) -> float:
    """Use the *initial registered* structures as the fixed denominator."""

    original, remaining = set(registered), set(survivors)
    if not remaining <= original:
        raise ValueError("survivors must come from the initial registry")
    if not original:
        return float("nan")
    return (len(original) - len(remaining)) / len(original)


@dataclass(frozen=True)
class PairedCI:
    mean: float
    lower: float
    upper: float
    n_networks: int


def paired_network_bootstrap_ci(
    differences: np.ndarray, *, n_bootstrap: int = 2000, seed: int = 0
) -> PairedCI:
    """Resample network rows, keeping both generator families paired.

    `differences[g, f]` is a policy-minus-comparator result on network g
    and simulator family f.  It is never legal to bootstrap 40 family rows as
    though they were 40 independent networks.
    """

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("expected finite networks-by-families differences")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    per_network = values.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_network), size=(n_bootstrap, len(per_network)))
    boot_means = per_network[draws].mean(axis=1)
    return PairedCI(
        mean=float(per_network.mean()),
        lower=float(np.quantile(boot_means, 0.025)),
        upper=float(np.quantile(boot_means, 0.975)),
        n_networks=len(per_network),
    )
