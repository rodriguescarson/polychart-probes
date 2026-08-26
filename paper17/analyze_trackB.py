"""Track B frozen analysis. Written and committed BEFORE any activation was extracted
from the real dataset (freeze commit hash is the proof; deviations go in Methods, dated).

Inputs: an ActivationSet saved by run_trackB.py plus the builder manifest.
Discipline encoded here:
  - cherry_picked_window is the TRUE HOLDOUT: excluded from every fit, sweep, and
    number until --open-holdout, which is to be run once, after this file's freeze
    commit exists in git history.
  - Split by SERIES (seed 20260809, 80/20). The probe layer is chosen on the train
    split only; every reported number is test-split, bootstrap CIs over series.
  - Surface controls: (a) pixel-statistics baseline (no model), (b) shuffled-label /
    shuffled-LF nulls (within-series), (c) TruncatedButHonest hard-negative rate.

  uv run python analyze_trackB.py --acts trackB_acts/qwen7b.npz --data trackB_data [--open-holdout]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probekit.activations import ActivationSet
from probekit.probes import LinearProbe, out_of_fold_scores
from probekit.stats import auroc_safe, bootstrap_metric
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SEED = 20260809
HOLDOUT = "cherry_picked_window"
HARD_NEG = "truncated_axis_honest"


def series_split(series: np.ndarray, seed: int = SEED, test_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    uniq = np.sort(np.unique(series))
    test = set(rng.choice(uniq, int(round(len(uniq) * test_frac)), replace=False))
    return np.array([s not in test for s in series]), np.array([s in test for s in series])


def pixel_stats(img_path: Path) -> np.ndarray:
    """No-model surface baseline: per-channel mean/std, grayscale entropy, edge density,
    aspect ratio, ink fraction. Deliberately simple; its job is to be beaten."""
    im = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    g = im.mean(2)
    gx, gy = np.gradient(g)
    edges = np.hypot(gx, gy)
    hist, _ = np.histogram(g, bins=32, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-9)
    ent = float(-(hist * np.log(hist + 1e-12)).sum())
    return np.array([*im.mean((0, 1)), *im.std((0, 1)), ent, float((edges > 0.1).mean()),
                     im.shape[1] / im.shape[0], float((g < 0.85).mean())], dtype=np.float32)


def ridge_r2_mae(Xtr, ytr, Xte, yte, alpha=10.0):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
    return 1 - ss_res / ss_tot, float(np.abs(yte - pred).mean()), pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--open-holdout", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    out = Path(args.out or (Path(args.acts).parent / "analysis"))
    out.mkdir(parents=True, exist_ok=True)

    acts = ActivationSet.load(args.acts)
    meta = acts.meta
    mech = np.array([m["mechanism"] for m in meta])
    series = np.array([m["series"] for m in meta])
    y = np.array([m["label"] for m in meta])
    lf = np.array([m["lie_factor"] if m["lie_factor"] is not None else np.nan for m in meta])

    keep = mech != HOLDOUT if not args.open_holdout else np.ones(len(meta), bool)
    if args.open_holdout:
        print("HOLDOUT OPENED. This run is valid only after the freeze commit of this file.")
    idx = np.where(keep)[0]
    tr_mask, te_mask = series_split(series[idx])
    tr, te = idx[tr_mask], idx[te_mask]
    print(f"n={len(idx)} (excl holdout: {not args.open_holdout}) | train {len(tr)} / test {len(te)} series-split")

    results: dict = {"seed": SEED, "holdout_open": bool(args.open_holdout)}

    # ---- RQ2: binary misleading-vs-faithful, layer chosen on TRAIN only ----
    rq2_pool_tr = tr[(y[tr] == 1) | (mech[tr] == "faithful")]
    rq2_pool_te = te[(y[te] == 1) | (mech[te] == "faithful")]
    layer_auc = []
    for li, L in enumerate(acts.layers):
        s = out_of_fold_scores(acts.X[rq2_pool_tr, li, :], y[rq2_pool_tr],
                               series[rq2_pool_tr], n_splits=5, seed=0)
        layer_auc.append(auroc_safe(y[rq2_pool_tr], s))
    best_li = int(np.argmax(layer_auc))
    best_L = acts.layers[best_li]
    probe = LinearProbe(seed=0).fit(acts.X[rq2_pool_tr, best_li, :], y[rq2_pool_tr])
    s_te = probe.decision(acts.X[rq2_pool_te, best_li, :])
    r = bootstrap_metric(auroc_safe, y[rq2_pool_te], s_te, groups=series[rq2_pool_te], seed=0)
    results["rq2"] = {"layer_auc_train": layer_auc, "best_layer": best_L,
                      "test_auroc": r.estimate, "ci": [r.lo, r.hi], "n_test": int(len(rq2_pool_te))}
    print(f"RQ2 | best layer {best_L} (train) | test AUROC {r}")

    # hard negative: honest-but-truncated scored by the same probe
    hn_te = te[mech[te] == HARD_NEG]
    if len(hn_te):
        thr = float(np.quantile(probe.decision(acts.X[rq2_pool_tr[y[rq2_pool_tr] == 0], best_li, :]), 0.99))
        fp = float((probe.decision(acts.X[hn_te, best_li, :]) > thr).mean())
        results["rq2"]["hard_negative_fpr_at_1pct_faithful"] = fp
        print(f"RQ2 | {HARD_NEG} false-positive rate at 1%-faithful threshold: {fp:.3f} (n={len(hn_te)})")

    # surface-statistics control on the SAME split
    P = np.stack([pixel_stats(data / m["path"]) for m in meta])
    s_px = LinearProbe(seed=0).fit(P[rq2_pool_tr], y[rq2_pool_tr]).decision(P[rq2_pool_te])
    r_px = bootstrap_metric(auroc_safe, y[rq2_pool_te], s_px, groups=series[rq2_pool_te], seed=0)
    results["rq2"]["surface_auroc"] = {"est": r_px.estimate, "ci": [r_px.lo, r_px.hi]}
    print(f"RQ2 | surface-statistics control AUROC {r_px}")

    # ---- RQ1: continuous severity (LF-exact rows only) ----
    lf_ok = idx[~np.isnan(lf[idx])]
    ltr = np.intersect1d(lf_ok, tr); lte = np.intersect1d(lf_ok, te)
    r2, mae, pred = ridge_r2_mae(acts.X[ltr, best_li, :], lf[ltr], acts.X[lte, best_li, :], lf[lte])
    r2_px, mae_px, _ = ridge_r2_mae(P[ltr], lf[ltr], P[lte], lf[lte])
    # binary-score baseline: 1-D map from the RQ2 probe's score to LF
    sc_tr = probe.decision(acts.X[ltr, best_li, :]).reshape(-1, 1)
    sc_te = probe.decision(acts.X[lte, best_li, :]).reshape(-1, 1)
    r2_bin, mae_bin, _ = ridge_r2_mae(sc_tr, lf[ltr], sc_te, lf[lte], alpha=1.0)
    # shuffled-LF null (within series)
    rng = np.random.default_rng(SEED)
    lf_sh = lf.copy()
    for s_ in np.unique(series[ltr]):
        ii = ltr[series[ltr] == s_]
        lf_sh[ii] = lf_sh[ii][rng.permutation(len(ii))]
    r2_null, mae_null, _ = ridge_r2_mae(acts.X[ltr, best_li, :], lf_sh[ltr], acts.X[lte, best_li, :], lf[lte])
    results["rq1"] = {"r2": r2, "mae": mae, "surface": {"r2": r2_px, "mae": mae_px},
                      "binary_score_baseline": {"r2": r2_bin, "mae": mae_bin},
                      "shuffled_lf_null": {"r2": r2_null, "mae": mae_null},
                      "n_train": int(len(ltr)), "n_test": int(len(lte))}
    print(f"RQ1 | ridge R2 {r2:.3f} MAE {mae:.3f} | surface R2 {r2_px:.3f} | "
          f"binary-baseline R2 {r2_bin:.3f} | shuffled-null R2 {r2_null:.3f}")

    # ---- RQ4: mechanism transfer (binary vs faithful, at best_L) ----
    mechs = [m for m in np.unique(mech[idx]) if m not in ("faithful", HARD_NEG)]
    T = {}
    for m_tr in mechs:
        pool_tr = tr[(mech[tr] == m_tr) | (mech[tr] == "faithful")]
        pm = LinearProbe(seed=0).fit(acts.X[pool_tr, best_li, :], y[pool_tr])
        T[m_tr] = {}
        for m_te in mechs:
            pool_te = te[(mech[te] == m_te) | (mech[te] == "faithful")]
            T[m_tr][m_te] = auroc_safe(y[pool_te], pm.decision(acts.X[pool_te, best_li, :]))
    results["rq4_transfer"] = T
    print("RQ4 | transfer matrix:", json.dumps(T, indent=1))

    (out / ("results_holdout.json" if args.open_holdout else "results.json")).write_text(
        json.dumps(results, indent=2, default=float))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
