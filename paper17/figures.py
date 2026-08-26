"""Paper 17 figures from results JSONs. Single light theme (print), palette validated
via the dataviz skill's validator (5 slots, fixed identity order; contrast WARN relieved
by direct labels).

  uv run python figures.py --results ~/Research/17-chart-severity-probes/results --out ../figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
# fixed identity -> hue (never reordered)
RUNS = [
    ("qwen7b-primary", "Qwen2.5-VL-7B", "#2a78d6"),
    ("draw2", "Qwen-7B, style draw 2", "#eb6834"),
    ("3b", "Qwen2.5-VL-3B", "#1baf7a"),
    ("gemma4b", "Gemma-3-4B", "#eda100"),
    ("72b", "Qwen2.5-VL-72B", "#e87ba4"),
]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 12.5,
    "axes.labelsize": 11, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "pdf.fonttype": 42, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def load_all(res_dir: Path):
    out = {}
    for key, label, color in RUNS:
        f = res_dir / key / "results.json"
        if f.exists():
            out[key] = (json.loads(f.read_text()), label, color)
    return out


def fig_layer_sweep(all_res, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, (r, label, color) in all_res.items():
        auc = r["rq2"]["layer_auc_train"]
        x = np.arange(len(auc)) / (len(auc) - 1)  # relative depth: models differ in layer count
        ax.plot(x, auc, color=color, lw=2, label=label)
        ax.annotate(label, (x[-1], auc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")
    ax.axhline(0.5, color=INK2, ls="--", lw=0.8)
    ax.text(0.005, 0.505, "chance", color=INK2, fontsize=8.5)
    ax.set_xlabel("relative depth (layer / n_layers)")
    ax.set_ylabel("train AUROC, misleading vs faithful (5-fold CV)")
    ax.set_title("Misleading-vs-faithful is decodable at every depth, in every model")
    ax.set_xlim(0, 1.28)
    ax.legend(loc="lower left", fontsize=9)
    save(fig, out, "f1_rq2_layer_sweep")


def fig_rq2_forest(all_res, out):
    rows = [(label, r["rq2"]["test_auroc"], r["rq2"]["ci"], color,
             r["rq2"]["surface_auroc"]["est"])
            for key, (r, label, color) in all_res.items()]
    fig, ax = plt.subplots(figsize=(6.4, 0.66 * len(rows) + 1.4))
    for i, (label, est, ci, color, surf) in enumerate(rows[::-1]):
        y = i
        ax.plot(ci, [y, y], color=color, lw=2)
        ax.plot(est, y, "o", color=color, ms=7)
        ax.plot(surf, y, "s", color=INK2, ms=5, mfc="none")
        ax.annotate(f"{est:.3f}", (est, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.axvline(0.5, color=INK2, ls="--", lw=0.8)
    ax.set_xlabel("test AUROC (95% CI, bootstrap over series)  ·  open square = surface-statistics control")
    ax.set_title("Binary decodability replicates across styles, scales, and families")
    ax.set_xlim(0.45, 1.02)
    save(fig, out, "f2_rq2_forest")


def fig_rq1_bars(all_res, out, sweeps=None):
    labels, vals, colors, ctrl = [], [], [], []
    for key, (r, label, color) in all_res.items():
        labels.append(label)
        vals.append(r["rq1"]["r2"])
        colors.append(color)
        ctrl.append(max(r["rq1"]["surface"]["r2"], r["rq1"]["binary_score_baseline"]["r2"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, vals, width=0.36, color=colors, label="activation probe")
    ax.bar(x + 0.18, ctrl, width=0.36, color=INK2, alpha=0.45, label="best control (surface / binary-bit)")
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.2f}", (xi - 0.18, max(v, 0)), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    if sweeps:
        for xi, key in zip(x, all_res.keys()):
            s = sweeps.get(key)
            if s:
                ax.plot(xi - 0.18, s["test_r2_at_best"], marker="D", ms=6, color=INK,
                        mfc="none", zorder=5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, labels, rotation=12, ha="right", fontsize=9.5)
    ax.set_ylabel("held-out R², exact Lie Factor")
    ax.set_title("Severity is partially encoded, and the margin over controls is wide\n(open diamond: R² at the severity-optimal layer, when it differs)")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, out, "f3_rq1_r2")


def fig_transfer(all_res, out):
    r = all_res["qwen7b-primary"][0]["rq4_transfer"]
    mechs = list(r.keys())
    short = {"aspect_ratio_exaggeration": "aspect", "inverted_y_axis": "inverted",
             "truncated_y_axis": "truncated"}
    M = np.array([[r[a][b] for b in mechs] for a in mechs])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="Blues")
    for i in range(len(mechs)):
        for j in range(len(mechs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > 0.82 else INK)
    ax.set_xticks(range(len(mechs)), [short[m] for m in mechs], fontsize=9.5)
    ax.set_yticks(range(len(mechs)), [short[m] for m in mechs], fontsize=9.5)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("probe trained on")
    ax.set_title("Axis-distortion probes transfer;\nthe aspect-ratio probe does not")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="test AUROC vs faithful", shrink=0.85)
    save(fig, out, "f4_transfer")


def fig_hardneg(all_res, out):
    labels, vals, colors = [], [], []
    for key, (r, label, color) in all_res.items():
        labels.append(label)
        vals.append(r["rq2"]["hard_negative_fpr_at_1pct_faithful"])
        colors.append(color)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.55, color=colors)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9.5)
    ax.axhline(0.01, color=INK2, ls="--", lw=0.8)
    ax.text(len(labels) - 0.45, 0.02, "1% design point", color=INK2, fontsize=8.5)
    ax.set_xticks(x, labels, rotation=12, ha="right", fontsize=9.5)
    ax.set_ylabel("false-positive rate on honest-but-truncated")
    ax.set_title("Every model's probe fires on the look of a lie")
    save(fig, out, "f5_hard_negative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res_dir = Path(args.results).expanduser()
    out = Path(args.out).expanduser() if args.out else res_dir.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    all_res = load_all(res_dir)
    sweeps = {}
    for key, _, _ in RUNS:
        f = res_dir / f"rq1sweep_{key}.json"
        if f.exists():
            sweeps[key] = json.loads(f.read_text())
    print("runs loaded:", list(all_res))
    fig_layer_sweep(all_res, out)
    fig_rq2_forest(all_res, out)
    fig_rq1_bars(all_res, out, sweeps)
    fig_transfer(all_res, out)
    fig_hardneg(all_res, out)


if __name__ == "__main__":
    main()
