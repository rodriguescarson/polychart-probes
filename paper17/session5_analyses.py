"""Session 5 dated analyses (committed before running).

  seeds5    the frozen plan promised 5 probe seeds; delivers seed variance for RQ2
            (AUROC, layer 7-frozen) and RQ1 (ridge R2 at the frozen layer AND the
            severity-optimal layer), seeds 0..4, alongside the bootstrap CIs.
  rq3sign   the dated correction: sign accuracy computed on MISLEADING rows only
            (LF != 1), where sign(truth-1) is defined; the original metric was
            mechanically deflated by exact LF = 1 rows.

  python session5_analyses.py seeds5 --acts qwen7b.npz --out seeds5.json
  python session5_analyses.py rq3sign --acts qwen7b.npz --frozen-results results.json --elicit elicit.json --out rq3sign.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from probekit.activations import ActivationSet
from probekit.probes import out_of_fold_scores
from probekit.stats import auroc_safe

SEED = 20260809
HOLDOUT = "cherry_picked_window"
FROZEN_BINARY_LAYER = 7
SEVERITY_LAYER = 18


def series_split(series, seed=SEED, test_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.sort(np.unique(series))
    test = set(rng.choice(uniq, int(round(len(uniq) * test_frac)), replace=False))
    return np.array([s not in test for s in series]), np.array([s in test for s in series])


def load(acts_path):
    acts = ActivationSet.load(acts_path)
    mech = np.array([m["mechanism"] for m in acts.meta])
    series = np.array([m["series"] for m in acts.meta])
    lf = np.array([m["lie_factor"] if m["lie_factor"] is not None else np.nan for m in acts.meta])
    y = np.array([m["label"] for m in acts.meta])
    paths = np.array([m["path"] for m in acts.meta])
    return acts, mech, series, lf, y, paths


def cmd_seeds5(args):
    acts, mech, series, lf, y, _ = load(args.acts)
    keep = np.where(mech != HOLDOUT)[0]
    tr_m, te_m = series_split(series[keep])
    tr, te = keep[tr_m], keep[te_m]
    li_bin = acts.layers.index(FROZEN_BINARY_LAYER)
    li_sev = acts.layers.index(SEVERITY_LAYER)
    pool_tr = tr[(y[tr] == 1) | (mech[tr] == "faithful")]
    pool_te = te[(y[te] == 1) | (mech[te] == "faithful")]
    lf_tr = tr[~np.isnan(lf[tr])]
    lf_te = te[~np.isnan(lf[te])]
    aucs, r2f, r2s = [], [], []
    for seed in range(5):
        from probekit.probes import LinearProbe
        pb = LinearProbe(seed=seed).fit(acts.X[pool_tr, li_bin, :], y[pool_tr])
        aucs.append(auroc_safe(y[pool_te], pb.decision(acts.X[pool_te, li_bin, :])))
        for li, sink in ((li_bin, r2f), (li_sev, r2s)):
            rng = np.random.default_rng(seed)
            sc = StandardScaler().fit(acts.X[lf_tr, li, :])
            m = Ridge(alpha=10.0, random_state=seed).fit(sc.transform(acts.X[lf_tr, li, :]), lf[lf_tr])
            pred = m.predict(sc.transform(acts.X[lf_te, li, :]))
            sink.append(float(1 - ((lf[lf_te] - pred) ** 2).sum() / (((lf[lf_te] - lf[lf_te].mean()) ** 2).sum() + 1e-12)))
    out = {"rq2_auroc_seeds": aucs, "rq2_auroc_sd": float(np.std(aucs)),
           "rq1_r2_frozen_layer_seeds": r2f, "rq1_r2_frozen_sd": float(np.std(r2f)),
           "rq1_r2_severity_layer_seeds": r2s, "rq1_r2_severity_sd": float(np.std(r2s))}
    print("SEEDS5 |", json.dumps({k: (round(v, 5) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in out.items()}))
    Path(args.out).write_text(json.dumps(out, indent=2))


def cmd_rq3sign(args):
    acts, mech, series, lf, y, paths = load(args.acts)
    res = json.loads(Path(args.frozen_results).read_text())["rq2"]
    li = acts.layers.index(res["best_layer"])
    keep = np.where(mech != HOLDOUT)[0]
    tr_m, te_m = series_split(series[keep])
    tr, te = keep[tr_m], keep[te_m]
    lf_tr = tr[~np.isnan(lf[tr])]
    sc = StandardScaler().fit(acts.X[lf_tr, li, :])
    m = Ridge(alpha=10.0).fit(sc.transform(acts.X[lf_tr, li, :]), lf[lf_tr])
    stated = {r["path"]: r["stated_lf"] for r in json.loads(Path(args.elicit).read_text())}
    rows = [i for i in te if paths[i] in stated and stated[paths[i]] is not None
            and not np.isnan(lf[i]) and abs(lf[i] - 1.0) > 1e-9]  # MISLEADING rows only
    pred = m.predict(sc.transform(acts.X[rows, li, :]))
    mouth = np.array([stated[paths[i]] for i in rows], dtype=float)
    truth = lf[rows]
    out = {"n_misleading": int(len(rows)),
           "sign_acc_probe": float(np.mean(np.sign(pred - 1) == np.sign(truth - 1))),
           "sign_acc_mouth": float(np.mean(np.sign(mouth - 1) == np.sign(truth - 1))),
           "layer": res["best_layer"]}
    print("RQ3SIGN |", json.dumps(out))
    Path(args.out).write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("seeds5", "rq3sign"):
        p = sub.add_parser(name)
        p.add_argument("--acts", required=True)
        p.add_argument("--out", required=True)
        if name == "rq3sign":
            p.add_argument("--frozen-results", required=True)
            p.add_argument("--elicit", required=True)
    args = ap.parse_args()
    {"seeds5": cmd_seeds5, "rq3sign": cmd_rq3sign}[args.cmd](args)


if __name__ == "__main__":
    main()
