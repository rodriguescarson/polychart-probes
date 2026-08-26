"""Track A analysis (dated, committed before running): base vs PolyChart-LoRA probes on
Llama-3.3-70B over text-encoded charts.

Endpoints per the frozen design's RQ-A plus the added surface control:
- severity ridge R2 by layer (train-chosen, test-reported, series split seed 20260809),
  base vs LoRA on identical rows;
- probe-direction cosine base-vs-LoRA at matched layers;
- binary AUROC at the best severity layer for context;
- MANDATORY surface-features control: ridge on numbers parsed from the text spec
  (values statistics + stated axis minimum). Track A states the distortion in text,
  so the activation probe must beat this to mean anything; the claim is scoped as
  stated-deception computation.

  python trackA_analyze.py --base acts_base.npz --lora acts_lora.npz --data trackA_data --out trackA_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from probekit.activations import ActivationSet
from probekit.probes import LinearProbe
from probekit.stats import auroc_safe

SEED = 20260809
HOLDOUT = "cherry_picked_window"


def series_split(series, seed=SEED, test_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.sort(np.unique(series))
    test = set(rng.choice(uniq, int(round(len(uniq) * test_frac)), replace=False))
    return np.array([s not in test for s in series]), np.array([s in test for s in series])


def surface_features(text: str) -> np.ndarray:
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", text)]
    vals = np.array(nums[:-2] if len(nums) > 3 else nums, dtype=float)
    axis_min = nums[-2] if "starts at" in text and len(nums) >= 2 else 0.0
    inverted = 1.0 if "inverted" in text else 0.0
    window = 1.0 if "longer series" in text else 0.0
    rng_ = (vals.max() - vals.min()) if len(vals) else 0.0
    return np.array([len(vals), vals.mean() if len(vals) else 0, vals.std() if len(vals) else 0,
                     vals.min() if len(vals) else 0, vals.max() if len(vals) else 0, rng_,
                     axis_min, axis_min / (vals.min() + 1e-9) if len(vals) else 0, inverted, window],
                    dtype=np.float32)


def ridge_r2(Xtr, ytr, Xte, yte, alpha=10.0):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    r2 = 1 - ((yte - pred) ** 2).sum() / (((yte - yte.mean()) ** 2).sum() + 1e-12)
    return float(r2), float(np.abs(yte - pred).mean()), m, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = ActivationSet.load(args.base)
    B = ActivationSet.load(args.lora)
    assert [m["series"] for m in A.meta] == [m["series"] for m in B.meta], "row order mismatch"
    mech = np.array([m["mechanism"] for m in A.meta])
    series = np.array([m["series"] for m in A.meta])
    lf = np.array([m["lie_factor"] if m["lie_factor"] is not None else np.nan for m in A.meta])
    y = np.array([m["label"] for m in A.meta])
    texts = [m["text"] for m in A.meta]

    keep = np.where((mech != HOLDOUT) & ~np.isnan(lf))[0]
    tr_m, te_m = series_split(series[keep])
    tr, te = keep[tr_m], keep[te_m]
    itr_m, iva_m = series_split(series[tr], seed=SEED + 1, test_frac=0.25)
    itr, iva = tr[itr_m], tr[iva_m]

    out = {"layers": A.layers, "n_train": int(len(tr)), "n_test": int(len(te))}
    for name, acts in (("base", A), ("lora", B)):
        val_r2 = []
        for li in range(len(acts.layers)):
            r2, _, _, _ = ridge_r2(acts.X[itr, li, :], lf[itr], acts.X[iva, li, :], lf[iva])
            val_r2.append(r2)
        best_li = int(np.argmax(val_r2))
        r2, mae, _, _ = ridge_r2(acts.X[tr, best_li, :], lf[tr], acts.X[te, best_li, :], lf[te])
        # binary at that layer for context
        pool_tr = tr[(y[tr] == 1) | (mech[tr] == "faithful")]
        pool_te = te[(y[te] == 1) | (mech[te] == "faithful")]
        pb = LinearProbe(seed=0).fit(acts.X[pool_tr, best_li, :], y[pool_tr])
        auc = auroc_safe(y[pool_te], pb.decision(acts.X[pool_te, best_li, :]))
        out[name] = {"val_r2_by_layer": val_r2, "best_layer": acts.layers[best_li],
                     "test_r2": r2, "test_mae": mae, "binary_auroc_at_layer": float(auc)}
        print(f"{name:5} | best layer {acts.layers[best_li]} | test R2 {r2:.3f} MAE {mae:.3f} | AUROC {auc:.3f}")

    # direction cosine at matched layers (severity ridge coefficients in standardised space)
    cosines = {}
    for li, L in enumerate(A.layers):
        _, _, mA, _ = ridge_r2(A.X[tr, li, :], lf[tr], A.X[te, li, :], lf[te])
        _, _, mB, _ = ridge_r2(B.X[tr, li, :], lf[tr], B.X[te, li, :], lf[te])
        a, b_ = mA.coef_, mB.coef_
        cosines[L] = float(a @ b_ / (np.linalg.norm(a) * np.linalg.norm(b_) + 1e-9))
    out["direction_cosine_by_layer"] = cosines
    print("direction cosine base-vs-lora: min {:.2f} max {:.2f}".format(min(cosines.values()), max(cosines.values())))

    P = np.stack([surface_features(t) for t in texts])
    r2s, maes, _, _ = ridge_r2(P[tr], lf[tr], P[te], lf[te])
    out["surface_text_control"] = {"test_r2": r2s, "test_mae": maes}
    print(f"surface-text control | R2 {r2s:.3f} MAE {maes:.3f}")

    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
