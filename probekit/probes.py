"""Linear probes on activations, Apollo-style defaults.

Default probe = StandardScaler -> LogisticRegression(C=1/reg_coeff, fit_intercept=False),
reg_coeff 10 (Goldowsky-Dill et al. 2025 config). Variants share the same interface:
`kind="diff_means"` (difference of class means) and `kind="pca"` (first principal
component, sign-oriented to the labels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .activations import ActivationSet
from .stats import BootResult, auroc_safe, bootstrap_metric


@dataclass
class LinearProbe:
    kind: str = "lr"            # lr | diff_means | pca
    reg_coeff: float = 10.0
    fit_intercept: bool = False
    normalize: bool = True
    seed: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        self.scaler_ = StandardScaler().fit(X) if self.normalize else None
        Z = self.scaler_.transform(X) if self.scaler_ else X
        if self.kind == "lr":
            self.clf_ = LogisticRegression(C=1.0 / self.reg_coeff, fit_intercept=self.fit_intercept,
                                           max_iter=2000, random_state=self.seed).fit(Z, y)
            self.w_ = self.clf_.coef_[0].astype(np.float32)
            self.b_ = float(self.clf_.intercept_[0]) if self.fit_intercept else 0.0
        elif self.kind == "diff_means":
            self.w_ = (Z[y == 1].mean(0) - Z[y == 0].mean(0)).astype(np.float32)
            mid = 0.5 * (Z[y == 1].mean(0) + Z[y == 0].mean(0)) @ self.w_
            self.b_ = -float(mid)
        elif self.kind == "pca":
            Zc = Z - Z.mean(0)
            _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
            w = Vt[0]
            if (Z[y == 1] @ w).mean() < (Z[y == 0] @ w).mean():
                w = -w
            self.w_ = w.astype(np.float32)
            self.b_ = -float((Z @ self.w_).mean())  # center the scores
        else:
            raise ValueError(f"unknown probe kind {self.kind!r}")
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(np.asarray(X, dtype=np.float32)) if self.scaler_ else np.asarray(X)
        return Z @ self.w_ + self.b_

    def proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def score_tokens(self, token_acts: np.ndarray) -> np.ndarray:
        """token_acts: [T, d] raw activations -> [T] probe scores."""
        return self.decision(token_acts)

    @property
    def direction_raw(self) -> np.ndarray:
        """Unit-norm probe direction in RAW activation space (undoes the scaler: w.z = (w/scale).x + c)."""
        w = self.w_ / self.scaler_.scale_ if self.scaler_ is not None else self.w_
        return (w / np.linalg.norm(w)).astype(np.float32)


def aggregate(token_scores: np.ndarray, how: str = "mean") -> float:
    """Per-token probe scores -> one per-response score (Apollo uses mean)."""
    t = np.asarray(token_scores, dtype=np.float64)
    if len(t) == 0:
        return float("nan")
    if how == "mean":
        return float(t.mean())
    if how == "max":
        return float(t.max())
    if how == "final":
        return float(t[-1])
    if how == "relu":
        return float(np.maximum(t, 0).mean())
    raise ValueError(f"unknown aggregation {how!r}")


def out_of_fold_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                       n_splits: int = 5, seed: int = 0, **probe_kwargs) -> np.ndarray:
    """Cross-validated decision scores: every prompt scored by a probe that never saw it (or its group)."""
    y = np.asarray(y).astype(int)
    scores = np.full(len(y), np.nan)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = cv.split(X, y)
    for tr, te in splits:
        probe = LinearProbe(seed=seed, **probe_kwargs).fit(X[tr], y[tr])
        scores[te] = probe.decision(X[te])
    return scores


def cv_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, n_splits: int = 5,
             n_boot: int = 1000, seed: int = 0, **probe_kwargs) -> BootResult:
    s = out_of_fold_scores(X, y, groups, n_splits=n_splits, seed=seed, **probe_kwargs)
    return bootstrap_metric(auroc_safe, y, s, groups=groups, n_boot=n_boot, seed=seed)


def layer_sweep(acts: ActivationSet, y: np.ndarray | None = None, groups: np.ndarray | None = None,
                layers: list[int] | None = None, n_splits: int = 5, n_boot: int = 500, seed: int = 0,
                **probe_kwargs) -> pd.DataFrame:
    """Cross-validated AUROC (with bootstrap CI over prompts) at every layer."""
    y = acts.labels() if y is None else np.asarray(y)
    groups = acts.groups() if groups is None else np.asarray(groups)
    layers = acts.layers if layers is None else layers
    rows = []
    for L in layers:
        r = cv_auroc(acts.at(L), y, groups, n_splits=n_splits, n_boot=n_boot, seed=seed, **probe_kwargs)
        rows.append({"layer": L, "auroc": r.estimate, "lo": r.lo, "hi": r.hi, "n": r.n_units})
    return pd.DataFrame(rows)


def fit_at_layer(acts: ActivationSet, layer: int, y: np.ndarray | None = None, **probe_kwargs) -> LinearProbe:
    y = acts.labels() if y is None else np.asarray(y)
    return LinearProbe(**probe_kwargs).fit(acts.at(layer), y)


def transfer_matrix(named_sets: dict[str, tuple[np.ndarray, np.ndarray]], seed: int = 0,
                    **probe_kwargs) -> pd.DataFrame:
    """AUROC of a probe trained on dataset A (rows) evaluated on dataset B (columns).

    Diagonal entries use 5-fold cross-validation so they are honest too.
    """
    names = list(named_sets)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        Xa, ya = named_sets[a]
        probe = LinearProbe(seed=seed, **probe_kwargs).fit(Xa, ya)
        for b in names:
            Xb, yb = named_sets[b]
            if a == b:
                s = out_of_fold_scores(Xb, np.asarray(yb).astype(int), seed=seed, **probe_kwargs)
            else:
                s = probe.decision(Xb)
            out.loc[a, b] = auroc_safe(np.asarray(yb).astype(int), s)
    return out


def recall_at_fpr(scores_pos: np.ndarray, scores_control: np.ndarray, fpr: float = 0.01) -> dict:
    """Threshold at the given false-positive rate on a control set, report recall of positives (Apollo-style)."""
    thr = float(np.quantile(np.asarray(scores_control), 1 - fpr))
    recall = float((np.asarray(scores_pos) > thr).mean())
    return {"threshold": thr, "recall": recall, "fpr": fpr, "n_control": len(scores_control)}


def tpr_at_fpr_curve(y: np.ndarray, s: np.ndarray, fpr_target: float = 0.01) -> dict:
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.searchsorted(fpr, fpr_target, side="right") - 1)
    i = max(i, 0)
    return {"tpr": float(tpr[i]), "fpr": float(fpr[i]), "threshold": float(thr[i])}


def shuffled_label_null(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                        n_perm: int = 10, n_splits: int = 5, seed: int = 0, **probe_kwargs) -> np.ndarray:
    """Train real probes on shuffled labels: the honest floor for the whole pipeline (expect ~0.5 AUROC)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        # constant-label groups (pair designs): permute labels ACROSS groups.
        # mixed-label groups (e.g. one chart series holding both classes): permute WITHIN
        # each group, which preserves per-group class balance and keeps CV folds sane.
        mixed = any(len(np.unique(y[groups == g])) > 1 for g in uniq)
    aurocs = []
    for k in range(n_perm):
        if groups is not None and mixed:
            y_perm = y.copy()
            for g in uniq:
                idx = np.where(groups == g)[0]
                y_perm[idx] = y_perm[idx][rng.permutation(len(idx))]
        elif groups is not None:
            g_label = {g: y[groups == g][0] for g in uniq}
            perm = rng.permutation([g_label[g] for g in uniq])
            y_perm = np.array([perm[list(uniq).index(g)] for g in groups])
        else:
            y_perm = rng.permutation(y)
        s = out_of_fold_scores(X, y_perm, groups, n_splits=n_splits, seed=seed + k, **probe_kwargs)
        aurocs.append(auroc_safe(y_perm, s))
    return np.asarray(aurocs)
