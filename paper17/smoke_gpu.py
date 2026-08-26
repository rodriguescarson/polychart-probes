"""Paper 17 GPU smoke: real Qwen2.5-VL-7B extraction over the 120-chart smoke set.

PREREG DISCIPLINE: this run verifies infrastructure only. It computes shapes, timing,
batch reproducibility, and a SHUFFLED-LABEL null (expected ~0.5). It never computes a
true-label metric; the first real-label number happens after the analysis code freezes,
on the real dataset with series splits.

  uv run python smoke_gpu.py --model Qwen/Qwen2.5-VL-7B-Instruct --images smoke_gpu_images
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--images", default=str(Path(__file__).parent / "smoke_gpu_images"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layer-stride", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    img_dir = Path(args.images)
    meta = json.loads((img_dir / "meta.json").read_text())
    paths = [img_dir / m["path"] for m in meta]
    print(f"charts: {len(paths)} | conditions: {sorted({m['mechanism'] for m in meta})}")

    from probekit.vlm import extract_vlm, load_vlm
    t = time.time()
    vlm = load_vlm(args.model)
    print(f"model  | {vlm.name} | {vlm.n_layers} layers x d={vlm.d_model} | {vlm.dtype} on {vlm.device} | {time.time()-t:.1f}s")

    layers = list(range(0, vlm.n_layers, args.layer_stride))
    t = time.time()
    acts = extract_vlm(vlm, paths, "Describe what this chart shows.", layers=layers,
                       meta=meta, batch_size=args.batch_size)
    dt = time.time() - t
    print(f"acts   | X{list(acts.X.shape)} in {dt:.1f}s ({dt/len(paths):.02f}s/chart)")
    acts.save(img_dir.parent / "smoke_acts" / "qwen7b.npz")

    # reproducibility: different batch size, same directions
    sub = paths[:8]
    a2 = extract_vlm(vlm, sub, "Describe what this chart shows.", layers=[layers[len(layers)//2]],
                     batch_size=3, progress=False)
    a1 = acts.X[:8, len(layers)//2, :]
    b = a2.X[:, 0, :]
    cos = np.sum(a1*b, 1) / (np.linalg.norm(a1, axis=1)*np.linalg.norm(b, axis=1) + 1e-8)
    print(f"repro  | min cosine across batch sizes: {cos.min():.5f}")

    # shuffled-label null ONLY (prereg hygiene: no true-label metric before code freeze)
    from probekit.probes import shuffled_label_null
    y = acts.labels()
    groups = acts.groups()
    L = layers[len(layers)//2]
    nulls = shuffled_label_null(acts.at(L), y, groups, n_perm=5)
    print(f"null   | shuffled-label AUROC at layer {L}: {np.mean(nulls):.3f} (expect ~0.5)")

    ok = cos.min() > 0.999 and abs(np.mean(nulls) - 0.5) < 0.15
    print(f"TOTAL {time.time()-t0:.0f}s -> {'SMOKE OK' if ok else 'SMOKE FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
