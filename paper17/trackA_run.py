"""Track A extraction (infrastructure): text-encoded charts through Llama-3.3-70B (4-bit),
optionally with the PolyChart LoRA merged.

  python trackA_run.py --data trackA_data --out acts_base.npz --load-in-4bit [--adapter rodriguescarson/polychart-shown-is-not-supported-lora] [--layer-stride 4]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    meta = json.loads((data / "manifest.json").read_text())
    from probekit.activations import extract
    from probekit.models import load_model
    t = time.time()
    lm = load_model(args.model, load_in_4bit=args.load_in_4bit, adapter=args.adapter)
    print(f"model | {lm.name} +{args.adapter or 'base'} | {lm.n_layers}x{lm.d_model} | {time.time()-t:.0f}s")
    prompts = [lm.chat(m["text"] + "\n\nDescribe what this chart shows.") for m in meta]
    layers = list(range(0, lm.n_layers, args.layer_stride))
    t = time.time()
    acts = extract(lm, prompts, layers=layers, positions="last", meta=meta,
                   batch_size=args.batch_size, max_length=768)
    print(f"acts | X{list(acts.X.shape)} in {time.time()-t:.0f}s")
    acts.save(args.out)
    print("saved ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
