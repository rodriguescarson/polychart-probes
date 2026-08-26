"""RQ3 behavioural elicitation (infrastructure): ask the model for a Lie Factor per chart,
on the TEST SPLIT only (same series split as the frozen analysis, seed 20260809).

Saves raw responses + parsed numbers; the probe-vs-behaviour comparison happens in a
dated analysis step next session. No probe number is touched here.

  uv run python elicit_trackB.py --data trackB_data --model Qwen/Qwen2.5-VL-7B-Instruct --out elicit_qwen7b.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

PROMPT = ("On Tufte's Lie Factor scale (1.0 = honest, higher = the chart exaggerates the "
          "effect, negative = the chart reverses the direction of the effect), how severely "
          "does this chart's design distort the underlying data? Answer with a single number "
          "first, then one short sentence.")

SEED = 20260809


def series_split(series, seed=SEED, test_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.sort(np.unique(series))
    test = set(rng.choice(uniq, int(round(len(uniq) * test_frac)), replace=False))
    return test


def parse_number(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace("**", ""))
    return float(m.group()) if m else None


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    meta = json.loads((data / "manifest.json").read_text())
    test_series = series_split(np.array([m["series"] for m in meta]))
    rows = [m for m in meta if m["series"] in test_series]
    print(f"test-split charts: {len(rows)} of {len(meta)}")

    from probekit.vlm import _batch_inputs, load_vlm
    vlm = load_vlm(args.model)
    tok = vlm.processor.tokenizer if hasattr(vlm.processor, "tokenizer") else vlm.processor

    out_rows, t0 = [], time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        enc = _batch_inputs(vlm, [data / m["path"] for m in batch], PROMPT)
        gen = vlm.model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        n_prompt = enc["input_ids"].shape[1]
        for i, m in enumerate(batch):
            text = tok.decode(gen[i, n_prompt:], skip_special_tokens=True).strip()
            out_rows.append({**{k: m[k] for k in ("path", "series", "mechanism", "label", "lie_factor")},
                             "response": text, "stated_lf": parse_number(text)})
        if start % (args.batch_size * 10) == 0:
            done = start + len(batch)
            print(f"{done}/{len(rows)} ({(time.time()-t0)/max(done,1):.1f}s/chart)")

    Path(args.out).write_text(json.dumps(out_rows, indent=0))
    parsed = sum(1 for r in out_rows if r["stated_lf"] is not None)
    print(f"saved {len(out_rows)} rows ({parsed} parsed) -> {args.out} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
