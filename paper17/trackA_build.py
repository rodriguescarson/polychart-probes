"""Track A dataset builder (dated, Session 5): TEXT-encoded charts, same series and
distortions as Track B, in the challenge's own interface style (plotted values plus
how the chart is drawn).

Mechanism notes, stated before any run: aspect-ratio exaggeration has NO textual
signature (its text spec equals the faithful one) and is therefore excluded from
Track A by construction; cherry_picked_window remains the sealed holdout; the
truncated-but-honest hard negative survives in text as a stated non-zero axis with
honest geometry.

  python trackA_build.py --out trackA_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for p in ("/Users/carson/Projects/adaption-hackathon", "/workspace/p17/adaption-hackathon"):
    if Path(p).exists():
        sys.path.insert(0, p)

import numpy as np

from polychart import build as b
from polychart.distortions import (CherryPickedWindow, Faithful, InvertedAxis,
                                   RenderContext, TruncatedAxis, TruncatedButHonest,
                                   find_reversing_window)
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


def text_spec(ctx: RenderContext, dist, unit: str) -> str:
    """The chart as text: values plus how it is drawn. Mirrors the challenge interface."""
    vals = ctx.values
    years = ctx.years.astype(int)
    ev = dist.evidence(ctx)
    lines = [f"Chart of {ctx.entity}, {unit}.", "Plotted values:"]
    lines += [f"  {y}: {v:.2f}" for y, v in zip(years, vals)]
    if dist.name in ("truncated_y_axis", "truncated_axis_honest"):
        ymin = ev.get("axis_min", ev.get("y_min", min(vals)))
        lines.append(f"Drawn as a line chart. The y axis starts at {float(ymin):.2f} (not zero) "
                     f"and ends at {max(vals) * 1.05:.2f}.")
    elif dist.name == "inverted_y_axis":
        lines.append("Drawn as a line chart. The y axis is inverted: larger values appear lower.")
    elif dist.name == "cherry_picked_window":
        lines.append(f"Drawn as a line chart showing only {years.min()} to {years.max()} "
                     f"of a longer series. The y axis starts at zero.")
    else:
        lines.append("Drawn as a line chart. The y axis starts at zero, normal orientation.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "trackA_data"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for slug, ents, year_min in catalog():
        try:
            series = fetch(slug, ents, year_min)
        except Exception:
            skipped += 1
            continue
        for entity in ents:
            f = series.frame[series.frame["Entity"] == entity] if "Entity" in series.frame.columns else series.frame
            f = f.dropna(subset=[series.value_col])
            if len(f) < 8:
                continue
            ctx = RenderContext(frame=f.reset_index(drop=True), value_col=series.value_col,
                                entity=entity, unit=series.unit, short_unit=series.short_unit)
            sid = f"{slug}|{entity}"
            conds = [Faithful(), TruncatedButHonest(), TruncatedAxis(), InvertedAxis()]
            win = find_reversing_window(ctx.frame, series.value_col)
            wctx = None
            if win is not None and not isinstance(win, bool):
                w0, w1 = win[0], win[1]
                wf = ctx.frame[(ctx.frame["Year"] >= w0) & (ctx.frame["Year"] <= w1)]
                if len(wf) >= 4:
                    wctx = RenderContext(frame=wf.reset_index(drop=True), value_col=series.value_col,
                                         entity=entity, unit=series.unit, short_unit=series.short_unit,
                                         full_frame=ctx.frame)
                    conds.append(CherryPickedWindow())
            for dist in conds:
                rctx = wctx if dist.name == "cherry_picked_window" else ctx
                try:
                    dist.apply(_DummyAx(), rctx)
                except Exception:
                    pass
                try:
                    ev = dist.evidence(rctx)
                    lf = lf_compute(dist.name, ev)
                    rows.append({"text": text_spec(rctx, dist, series.unit), "series": sid,
                                 "mechanism": dist.name, "label": int(dist.misleading),
                                 "lie_factor": None if lf is None else float(lf.value)})
                except Exception:
                    skipped += 1

    (out / "manifest.json").write_text(json.dumps(rows, indent=0))
    import collections
    print(f"rows: {len(rows)} | series: {len({r['series'] for r in rows})} | skipped: {skipped}")
    print(dict(collections.Counter(r["mechanism"] for r in rows)))
    lf = [r["lie_factor"] for r in rows if r["lie_factor"] is not None and r["mechanism"] != "cherry_picked_window"]
    print(f"LF-exact (non-holdout): {len(lf)} range [{min(lf):.2f}, {max(lf):.2f}]")
    return 0


class _DummyAx:
    """Distortion.apply draws on a matplotlib Axes; for text we only need evidence()."""

    def __getattr__(self, _):
        return lambda *a, **k: None


if __name__ == "__main__":
    raise SystemExit(main())
