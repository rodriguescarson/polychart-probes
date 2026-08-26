"""Track B extraction runner (pod-side, infrastructure only): manifest -> activations.

  uv run python run_trackB.py --data /workspace/p17/trackB_data --model Qwen/Qwen2.5-VL-7B-Instruct \
      --out /workspace/p17/trackB_acts/qwen7b.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prompt", default="Describe what this chart shows.")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    meta = json.loads((data / "manifest.json").read_text())
    paths = [data / m["path"] for m in meta]
    print(f"charts: {len(paths)} | series: {len({m.get('series', m.get('case','?')) for m in meta})}")

    from probekit.vlm import extract_vlm, load_vlm
    t = time.time()
    vlm = load_vlm(args.model, load_in_4bit=args.load_in_4bit)
    print(f"model | {vlm.name} | {vlm.n_layers}x{vlm.d_model} | {time.time()-t:.0f}s")
    t = time.time()
    acts = extract_vlm(vlm, paths, args.prompt, meta=meta, batch_size=args.batch_size)
    print(f"acts  | X{list(acts.X.shape)} in {time.time()-t:.0f}s")
    acts.save(args.out)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
