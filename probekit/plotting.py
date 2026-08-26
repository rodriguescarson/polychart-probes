"""Slide-ready figures. Every plot gets a title, labelled axes, and visible uncertainty."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

FIGSIZE = (8, 4.5)  # 16:9

STYLE = {
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "pdf.fonttype": 42,
}

COLORS = {"main": "#1f6f8b", "alt": "#c05746", "control": "#8a8a8a", "null": "#bbbbbb"}


def setup_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def layer_sweep_plot(df: pd.DataFrame, title: str, path: str | Path, ylabel: str = "AUROC (5-fold CV)",
                     chance: float | None = 0.5, extra: dict[str, pd.DataFrame] | None = None) -> Path:
    """df columns: layer, auroc, lo, hi. `extra` overlays more sweeps (e.g. another dataset or probe kind)."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["layer"], df["auroc"], color=COLORS["main"], marker="o", ms=3, label="probe")
    ax.fill_between(df["layer"], df["lo"], df["hi"], color=COLORS["main"], alpha=0.2,
                    label="95% CI (bootstrap over prompts)")
    for i, (name, d2) in enumerate((extra or {}).items()):
        c = list(COLORS.values())[1 + i % 3]
        ax.plot(d2["layer"], d2["auroc"], color=c, marker="s", ms=3, label=name)
        ax.fill_between(d2["layer"], d2["lo"], d2["hi"], color=c, alpha=0.15)
    if chance is not None:
        ax.axhline(chance, color=COLORS["null"], ls="--", lw=1, label="chance")
    ax.set_xlabel("layer (block output index)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    return save(fig, path)


def bar_ci_plot(labels: list[str], values: np.ndarray, lo: np.ndarray, hi: np.ndarray, title: str,
                path: str | Path, ylabel: str = "AUROC", chance: float | None = 0.5) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(labels))
    values, lo, hi = np.asarray(values), np.asarray(lo), np.asarray(hi)
    ax.bar(x, values, color=COLORS["main"], width=0.6)
    ax.errorbar(x, values, yerr=[values - lo, hi - values], fmt="none", ecolor="black", capsize=4, lw=1)
    if chance is not None:
        ax.axhline(chance, color=COLORS["null"], ls="--", lw=1)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel(f"{ylabel} (error bars: 95% bootstrap CI)")
    ax.set_title(title)
    return save(fig, path)


def roc_plot(y: np.ndarray, scores: dict[str, np.ndarray], title: str, path: str | Path) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for i, (name, s) in enumerate(scores.items()):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=name, color=list(COLORS.values())[i % 4])
    ax.plot([0, 1], [0, 1], color=COLORS["null"], ls="--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return save(fig, path)


def score_hist(scores_pos: np.ndarray, scores_neg: np.ndarray, title: str, path: str | Path,
               scores_control: np.ndarray | None = None, threshold: float | None = None,
               pos_label: str = "positive", neg_label: str = "negative") -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.histogram_bin_edges(np.concatenate([scores_pos, scores_neg]), bins=30)
    ax.hist(scores_neg, bins=bins, alpha=0.55, color=COLORS["main"], label=f"{neg_label} (n={len(scores_neg)})")
    ax.hist(scores_pos, bins=bins, alpha=0.55, color=COLORS["alt"], label=f"{pos_label} (n={len(scores_pos)})")
    if scores_control is not None:
        ax.hist(scores_control, bins=bins, alpha=0.4, color=COLORS["control"],
                label=f"control (n={len(scores_control)})")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls=":", lw=1.2, label="threshold @1% FPR")
    ax.set_xlabel("probe score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    return save(fig, path)


def dose_response_plot(df: pd.DataFrame, title: str, path: str | Path,
                       ylabel: str = "behaviour metric") -> Path:
    """df columns: condition, alpha, mean, lo, hi (from steering.dose_response)."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (name, d) in enumerate(df.groupby("condition")):
        c = list(COLORS.values())[i % 4]
        d = d.sort_values("alpha")
        ax.plot(d["alpha"], d["mean"], marker="o", ms=4, color=c, label=name)
        ax.fill_between(d["alpha"], d["lo"], d["hi"], color=c, alpha=0.18)
    ax.set_xlabel("steering strength alpha (added along unit direction)")
    ax.set_ylabel(f"{ylabel} (band: 95% bootstrap CI)")
    ax.set_title(title)
    ax.legend()
    return save(fig, path)


def heatmap(matrix: pd.DataFrame, title: str, path: str | Path, vmin: float = 0.5, vmax: float = 1.0,
            cbar_label: str = "AUROC", fmt: str = "{:.2f}") -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(matrix.columns), 1.0 + 0.7 * len(matrix.index)))
    im = ax.imshow(matrix.values.astype(float), vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            v = float(matrix.values[i, j])
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if v < (vmin + vmax) / 2 else "black", fontsize=10)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
    return save(fig, path)


def token_heatmap(tokens: list[str], scores: np.ndarray, title: str, path: str | Path,
                  max_tokens: int = 60) -> Path:
    """Per-token probe scores rendered as coloured cells (error analysis on one example)."""
    setup_style()
    tokens = tokens[:max_tokens]
    scores = np.asarray(scores)[:max_tokens]
    fig, ax = plt.subplots(figsize=(min(14, 0.24 * len(tokens) + 1), 1.8))
    norm = matplotlib.colors.Normalize(vmin=float(scores.min()), vmax=float(scores.max()))
    cmap = plt.get_cmap("coolwarm")
    for i, (t, s) in enumerate(zip(tokens, scores)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=cmap(norm(s))))
        ax.text(i + 0.5, -0.15, t.replace("Ġ", "").replace("▁", ""), rotation=60,
                ha="right", va="top", fontsize=7)
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(-1.4, 1)
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical",
                 label="probe score", shrink=0.8)
    return save(fig, path)
