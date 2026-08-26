"""RQ5: real published misleading charts, parameter-faithful reconstructions (dated).

The original bitmaps are copyrighted and not redistributed; each case in the
generator's `real_charts.CHARTS` carries verified values and the documented axis
parameters, so we re-render the aired configuration and a faithful zero-based twin.
Per the frozen design: a per-chart table, no confidence intervals at n=5.

  reconstruct:  python rq5_real.py build --out rq5_data
  score (pod):  python rq5_real.py score --acts rq5_acts.npz --train-acts trackB_acts/qwen7b.npz \
                   --frozen-results trackB_acts/analysis/results.json --out rq5_table.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for p in ("/Users/carson/Projects/adaption-hackathon", "/workspace/p17/adaption-hackathon"):
    if Path(p).exists():
        sys.path.insert(0, p)

HOLDOUT = "cherry_picked_window"


def render_case(c: dict, out: Path, aired: bool) -> dict | None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    kind = c["kind"]
    if kind == "bar":
        vals, labels = c["values"], c.get("labels")
        ax.bar(range(len(vals)), vals, color="#4a6fa5", width=0.55)
        ax.set_xticks(range(len(vals)), labels, fontsize=9)
        ax.set_ylim((c["axis_min"], c["axis_max"]) if aired else (0, max(vals) * 1.1))
    elif kind == "line":
        # OWID-sourced series with a documented (absurdly wide) aired axis
        from polychart.sources import fetch
        series = fetch(c["owid_slug"], [c["entity"]], c["year_min"])
        f = series.frame[series.frame["Entity"] == c["entity"]] if "Entity" in series.frame.columns else series.frame
        f = f[(f["Year"] >= c["year_min"]) & (f["Year"] <= c["year_max"])].dropna(subset=[series.value_col])
        ax.plot(f["Year"], f[series.value_col], color="#4a6fa5", lw=2)
        if aired:
            ax.set_ylim(c["axis_min"], c["axis_max"])
    elif kind == "inverted_line":
        years, vals = zip(*c["series"])
        ax.plot(years, vals, color="#4a6fa5", lw=2)
        ax.fill_between(years, vals, max(vals) * 1.05 if aired else 0, color="#4a6fa5", alpha=0.25)
        if aired:
            ax.invert_yaxis()
    else:
        plt.close(fig)
        print(f"  skip {c['id']}: kind={kind} not reconstructable from held data (endpoint-only)")
        return None
    ax.set_title(c["title"][:70], fontsize=9)
    ax.set_ylabel(c.get("unit", ""))
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return {"path": out.name}


def cmd_build(args):
    from polychart.real_charts import CHARTS
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in CHARTS:
        try:
            from polychart.real_charts import _bar_lie_factor
            if c["kind"] == "bar":
                lf = _bar_lie_factor(c["values"], c["axis_min"])
            elif c["kind"] == "inverted_line":
                lf = -1.0  # documented in the Deception Atlas (Reuters inverted axis)
            else:
                lf = c.get("lie_factor")  # climate: understatement, no exact figure held
        except Exception:
            lf = c.get("lie_factor")
        for aired in (True, False):
            name = f"{c['id']}__{'aired' if aired else 'faithful'}.png"
            r = render_case(c, out / "images" / name, aired)
            if r:
                rows.append({"path": f"images/{name}", "case": c["id"], "aired": aired,
                             "mechanism": c["mechanism"], "documented_lf": lf if aired else 1.0,
                             "publication": c.get("publication", ""), "year": c.get("year")})
    (out / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"reconstructed {len(rows)} images from {len(set(r['case'] for r in rows))} cases")
    for r in rows:
        print(" ", r["path"], "lf=", r["documented_lf"])


def cmd_score(args):
    from probekit.activations import ActivationSet
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    train = ActivationSet.load(args.train_acts)
    li = train.layers.index(json.loads(Path(args.frozen_results).read_text())["rq2"]["best_layer"])
    mech = np.array([m["mechanism"] for m in train.meta])
    lf = np.array([m["lie_factor"] if m["lie_factor"] is not None else np.nan for m in train.meta])
    fit = np.where((mech != HOLDOUT) & ~np.isnan(lf))[0]
    sc = StandardScaler().fit(train.X[fit, li, :])
    ridge = Ridge(alpha=10.0).fit(sc.transform(train.X[fit, li, :]), lf[fit])

    acts = ActivationSet.load(args.acts)
    pred = ridge.predict(sc.transform(acts.X[:, li, :]))
    table = []
    for m, p in zip(acts.meta, pred):
        table.append({**{k: m[k] for k in ("case", "aired", "mechanism", "documented_lf")},
                      "probe_lf": round(float(p), 3)})
        print(f"{m['case']:<28} aired={str(m['aired']):<5} doc_lf={m['documented_lf']}  probe_lf={p:.2f}")
    Path(args.out).write_text(json.dumps(table, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", default="rq5_data")
    s = sub.add_parser("score")
    s.add_argument("--acts", required=True); s.add_argument("--train-acts", required=True)
    s.add_argument("--frozen-results", required=True); s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"build": cmd_build, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
