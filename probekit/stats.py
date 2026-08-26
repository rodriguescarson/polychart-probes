"""Uncertainty helpers. The independent unit is always the prompt (or its group), never the token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_safe(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


@dataclass
class BootResult:
    estimate: float
    lo: float
    hi: float
    n_units: int
    n_boot: int
    alpha: float = 0.05

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n_units}, B={self.n_boot})"


def _unit_indices(n: int, groups: Sequence | None):
    """Map resampling units to row indices. With groups, a unit is a group (cluster bootstrap)."""
    if groups is None:
        return [np.array([i]) for i in range(n)]
    groups = np.asarray(groups)
    return [np.where(groups == g)[0] for g in np.unique(groups)]


def bootstrap_metric(metric_fn: Callable[[np.ndarray, np.ndarray], float], y: Sequence, s: Sequence,
                     groups: Sequence | None = None, n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> BootResult:
    """Percentile bootstrap CI for metric_fn(y, s), resampling prompts (or groups) with replacement."""
    y = np.asarray(y)
    s = np.asarray(s)
    units = _unit_indices(len(y), groups)
    rng = np.random.default_rng(seed)
    est = metric_fn(y, s)
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[i] for i in pick])
        samples.append(metric_fn(y[idx], s[idx]))
    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootResult(float(est), float(lo), float(hi), len(units), n_boot, alpha)


def paired_bootstrap_diff(metric_fn: Callable, y: Sequence, s_a: Sequence, s_b: Sequence,
                          groups: Sequence | None = None, n_boot: int = 1000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """CI and two-sided p for metric(A) - metric(B) on the same prompts (resample prompts once, apply to both)."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    units = _unit_indices(len(y), groups)
    rng = np.random.default_rng(seed)
    obs = metric_fn(y, s_a) - metric_fn(y, s_b)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[i] for i in pick])
        diffs.append(metric_fn(y[idx], s_a[idx]) - metric_fn(y[idx], s_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # add-one smoothed two-sided p from the bootstrap sign distribution
    p = 2 * min((np.sum(diffs <= 0) + 1) / (n_boot + 1), (np.sum(diffs >= 0) + 1) / (n_boot + 1))
    return {"diff": float(obs), "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0)),
            "n_units": len(units), "n_boot": n_boot}


def permutation_null(metric_fn: Callable, y: Sequence, s: Sequence, groups: Sequence | None = None,
                     n_perm: int = 1000, seed: int = 0) -> dict:
    """Shuffle labels (at the group level when groups are given) to get the null distribution of the metric."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    obs = metric_fn(y, s)
    nulls = []
    if groups is None:
        for _ in range(n_perm):
            nulls.append(metric_fn(rng.permutation(y), s))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        # one label per group, permuted across groups (keeps within-group structure intact)
        g_label = {g: y[groups == g][0] for g in uniq}
        labels = np.array([g_label[g] for g in uniq])
        for _ in range(n_perm):
            perm = rng.permutation(labels)
            y_perm = np.array([perm[np.where(uniq == g)[0][0]] for g in groups])
            nulls.append(metric_fn(y_perm, s))
    nulls = np.asarray(nulls)
    p = (np.sum(nulls >= obs) + 1) / (n_perm + 1)
    return {"observed": float(obs), "null_mean": float(np.nanmean(nulls)),
            "null_hi95": float(np.nanpercentile(nulls, 95)), "p": float(p), "n_perm": n_perm}
