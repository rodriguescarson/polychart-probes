"""Track B dataset builder. Runs BEFORE any analysis; writes images + manifest only.

Real OWID series via the generator's own `sources.fetch` (cached), rendered per
mechanism with style varied independently of distortion (render.py's contract).
Exact lie factors from `severity.compute`: truncated_y_axis, inverted_y_axis,
cherry_picked_window get exact values; faithful and truncated_axis_honest are 1.0;
aspect_ratio has no exact LF (dispatch returns None) and therefore joins RQ2/RQ4
only, never the RQ1 regression. cherry_picked_window is the frozen TRUE HOLDOUT:
the builder writes it, the analysis excludes it until --open-holdout.

  uv run python build_trackB.py --out trackB_data [--limit-slugs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/carson/Projects/adaption-hackathon")

import numpy as np
import pandas as pd

from polychart import build as b
from polychart.distortions import (AspectRatioExaggeration, CherryPickedWindow,
                                   Faithful, InvertedAxis, RenderContext,
                                   TruncatedAxis, TruncatedButHonest,
                                   find_reversing_window)
from polychart.render import render
from polychart.severity import compute as lf_compute
from polychart.sources import fetch

ENTITY_POOL = b.ENTITY_POOL + [
    "Colombia", "Morocco", "Vietnam", "Pakistan", "Ukraine", "Romania",
    "Kazakhstan", "Angola", "Sri Lanka", "Tunisia", "Ecuador", "Jordan",
]
ENTITIES_PER_SLUG = 12


def catalog():
    out, n = [], len(ENTITY_POOL)
    for i, (slug, year_min) in enumerate(b.SLUGS):
        start = (i * 7) % n
        ents = [ENTITY_POOL[(start + k) % n] for k in range(ENTITIES_PER_SLUG)]
        out.append((slug, ents, year_min))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "trackB_data"))
    ap.add_argument("--limit-slugs", type=int, default=None)
    ap.add_argument("--style-suffix", default="", help="independent style draw: same data and distortions, different render style seed")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    cat = catalog()[: args.limit_slugs] if args.limit_slugs else catalog()
    for slug, ents, year_min in cat:
        try:
            series = fetch(slug, ents, year_min)
        except Exception as e:
            skipped.append((slug, f"fetch:{type(e).__name__}"))
            continue
        for entity in ents:
            ent_frame = series.frame[series.frame["Entity"] == entity] if "Entity" in series.frame.columns else series.frame
            ent_frame = ent_frame.dropna(subset=[series.value_col])
            if len(ent_frame) < 8:
                skipped.append((f"{slug}|{entity}", "too_short"))
                continue
            ctx = RenderContext(frame=ent_frame.reset_index(drop=True), value_col=series.value_col,
                                entity=entity, unit=series.unit, short_unit=series.short_unit)
            sid = f"{slug}|{entity}"
            conds: list = [Faithful(), TruncatedButHonest(), TruncatedAxis(),
                           InvertedAxis(), AspectRatioExaggeration()]
            win = find_reversing_window(ctx.frame, series.value_col)
            if win is not None:
                w0, w1 = (win[0], win[1]) if isinstance(win, (tuple, list)) else (None, None)
                if w0 is not None:
                    wframe = ctx.frame[(ctx.frame["Year"] >= w0) & (ctx.frame["Year"] <= w1)]
                    if len(wframe) >= 4:
                        conds.append(("cherry", RenderContext(
                            frame=wframe.reset_index(drop=True), value_col=series.value_col,
                            entity=entity, unit=series.unit, short_unit=series.short_unit,
                            full_frame=ctx.frame)))
            for cond in conds:
                if isinstance(cond, tuple):
                    dist, rctx = CherryPickedWindow(), cond[1]
                else:
                    dist, rctx = cond, ctx
                name = f"{slug}__{entity}__{dist.name}".replace(" ", "-").replace("/", "-")
                p = out / "images" / (name + ".png")
                try:
                    render(rctx, dist, p, seed=name + args.style_suffix)
                    ev = dist.evidence(rctx)
                    lf = lf_compute(dist.name, ev)
                    rows.append({
                        "path": f"images/{p.name}", "series": sid, "slug": slug,
                        "entity": entity, "mechanism": dist.name,
                        "label": int(dist.misleading),
                        "lie_factor": None if lf is None else float(lf.value),
                        "band": None if lf is None else lf.band,
                    })
                except Exception as e:
                    skipped.append((name, f"render:{type(e).__name__}:{str(e)[:60]}"))

    (out / "manifest.json").write_text(json.dumps(rows, indent=0))
    (out / "skipped.json").write_text(json.dumps(skipped, indent=0))
    df = pd.DataFrame(rows)
    print(f"rows: {len(df)} | series: {df['series'].nunique()} | skipped: {len(skipped)}")
    print(df.groupby("mechanism").agg(n=("path", "count"),
                                      lf_exact=("lie_factor", lambda s: s.notna().sum())))
    lf = df["lie_factor"].dropna()
    print(f"LF-exact rows: {len(lf)} | LF range [{lf.min():.2f}, {lf.max():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
