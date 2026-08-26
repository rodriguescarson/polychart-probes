"""Session 4 dated analysis extensions (2026-08-25/26). Committed BEFORE running.

Subcommands, all reading the already-saved activations (no new extraction here):

  rq3       probe-vs-mouth on the same test rows: probe LF predictions (ridge at the
            RQ2-chosen layer, trained on train split) vs the model's stated LF
            (elicit_qwen7b.json). Endpoint per the frozen design: both MAEs + the gap
            with a bootstrap CI over series, plus sign agreement.
  rq1sweep  per-layer ridge R2 for severity (LF-exact, holdout sealed) for one acts
            file. DATED DEVIATION: the frozen analysis evaluated RQ1 only at the
            RQ2-chosen layer; Gemma's negative R2 there motivates asking WHERE
            magnitude lives per model. Layer chosen on train, reported on test,
            same series split.
  holdout   the holdout-opening metrics, PREDECLARED here before any holdout row is
            read: cherry LFs span [-322, -0.002], so raw MAE is tail-dominated.
            Declared metrics: Spearman rank correlation (probe prediction vs LF),
            sign accuracy (predicted reversal), winsorized MAE (1st/99th pct), and
            raw MAE reported alongside. Probe = ridge at the RQ2 layer trained on
            ALL non-holdout LF-exact rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from probekit.activations import ActivationSet
from probekit.stats import bootstrap_metric

SEED = 20260809
HOLDOUT = "cherry_picked_window"


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
    paths = np.array([m["path"] for m in acts.meta])
    return acts, mech, series, lf, paths


def fit_ridge(Xtr, ytr, alpha=10.0):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return lambda X: m.predict(sc.transform(X))


def cmd_rq3(args):
    acts, mech, series, lf, paths = load(args.acts)
    res_rq2 = json.loads(Path(args.frozen_results).read_text())["rq2"]
    li = acts.layers.index(res_rq2["best_layer"])
    keep = np.where(mech != HOLDOUT)[0]
    tr_m, te_m = series_split(series[keep])
    tr, te = keep[tr_m], keep[te_m]
    lf_tr = tr[~np.isnan(lf[tr])]
    predict = fit_ridge(acts.X[lf_tr, li, :], lf[lf_tr])

    stated = {r["path"]: r["stated_lf"] for r in json.loads(Path(args.elicit).read_text())}
    rows = [i for i in te if paths[i] in stated and stated[paths[i]] is not None and not np.isnan(lf[i])]
    probe_pred = predict(acts.X[rows, li, :])
    mouth = np.array([stated[paths[i]] for i in rows], dtype=float)
    truth = lf[rows]
    grp = series[rows]
    e_probe, e_mouth = np.abs(probe_pred - truth), np.abs(mouth - truth)
    mae = lambda y, s: float(np.mean(s))
    r_p = bootstrap_metric(mae, np.zeros(len(rows)), e_probe, groups=grp, seed=0)
    r_m = bootstrap_metric(mae, np.zeros(len(rows)), e_mouth, groups=grp, seed=0)
    gap = bootstrap_metric(mae, np.zeros(len(rows)), e_mouth - e_probe, groups=grp, seed=0)
    sign_p = float(np.mean(np.sign(probe_pred - 1) == np.sign(truth - 1)))
    sign_m = float(np.mean(np.sign(mouth - 1) == np.sign(truth - 1)))
    out = {"n": len(rows), "probe_mae": [r_p.estimate, r_p.lo, r_p.hi],
           "mouth_mae": [r_m.estimate, r_m.lo, r_m.hi],
           "gap_mouth_minus_probe": [gap.estimate, gap.lo, gap.hi],
           "sign_acc_probe": sign_p, "sign_acc_mouth": sign_m,
           "layer": res_rq2["best_layer"]}
    print("RQ3 |", json.dumps(out))
    Path(args.out).write_text(json.dumps(out, indent=2))


def cmd_rq1sweep(args):
    acts, mech, series, lf, _ = load(args.acts)
    keep = np.where((mech != HOLDOUT) & ~np.isnan(lf))[0]
    tr_m, te_m = series_split(series[keep])
    tr, te = keep[tr_m], keep[te_m]
    # inner split of train for layer choice
    itr_m, iva_m = series_split(series[tr], seed=SEED + 1, test_frac=0.25)
    itr, iva = tr[itr_m], tr[iva_m]
    r2s = []
    for li in range(len(acts.layers)):
        pred = fit_ridge(acts.X[itr, li, :], lf[itr])(acts.X[iva, li, :])
        ss = 1 - ((lf[iva] - pred) ** 2).sum() / (((lf[iva] - lf[iva].mean()) ** 2).sum() + 1e-12)
        r2s.append(float(ss))
    best_li = int(np.argmax(r2s))
    pred = fit_ridge(acts.X[tr, best_li, :], lf[tr])(acts.X[te, best_li, :])
    r2 = 1 - ((lf[te] - pred) ** 2).sum() / (((lf[te] - lf[te].mean()) ** 2).sum() + 1e-12)
    out = {"val_r2_by_layer": r2s, "best_layer": acts.layers[best_li],
           "test_r2_at_best": float(r2), "test_mae_at_best": float(np.abs(lf[te] - pred).mean()),
           "n_train": int(len(tr)), "n_test": int(len(te))}
    print("RQ1SWEEP |", args.acts, "| best layer", out["best_layer"],
          "| test R2", round(out["test_r2_at_best"], 3), "MAE", round(out["test_mae_at_best"], 3))
    Path(args.out).write_text(json.dumps(out, indent=2))


def cmd_holdout(args):
    acts, mech, series, lf, _ = load(args.acts)
    res_rq2 = json.loads(Path(args.frozen_results).read_text())["rq2"]
    li = acts.layers.index(res_rq2["best_layer"])
    fit_rows = np.where((mech != HOLDOUT) & ~np.isnan(lf))[0]
    predict = fit_ridge(acts.X[fit_rows, li, :], lf[fit_rows])
    hold = np.where((mech == HOLDOUT) & ~np.isnan(lf))[0]
    pred = predict(acts.X[hold, li, :])
    truth = lf[hold]
    lo, hi = np.percentile(truth, [1, 99])
    tw = np.clip(truth, lo, hi)
    rho, p = spearmanr(pred, truth)
    out = {"n": int(len(hold)), "spearman_rho": float(rho), "spearman_p": float(p),
           "sign_acc_reversal": float(np.mean(np.sign(pred - 1) == np.sign(truth - 1))),
           "winsorized_mae": float(np.abs(np.clip(pred, lo, hi) - tw).mean()),
           "raw_mae": float(np.abs(pred - truth).mean()),
           "layer": res_rq2["best_layer"],
           "truth_range": [float(truth.min()), float(truth.max())]}
    print("HOLDOUT |", json.dumps(out))
    Path(args.out).write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rq3", "rq1sweep", "holdout"):
        p = sub.add_parser(name)
        p.add_argument("--acts", required=True)
        p.add_argument("--out", required=True)
        if name != "rq1sweep":
            p.add_argument("--frozen-results", required=True)
        if name == "rq3":
            p.add_argument("--elicit", required=True)
    args = ap.parse_args()
    {"rq3": cmd_rq3, "rq1sweep": cmd_rq1sweep, "holdout": cmd_holdout}[args.cmd](args)


if __name__ == "__main__":
    main()
