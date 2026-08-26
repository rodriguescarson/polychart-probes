"""RQ6 (dated, committed before running): steer the severity direction while the model
describes TRUTHFUL charts.

Design per the frozen plan: add alpha times the unit severity direction (ridge
coefficients mapped back to raw activation space) at the severity layer during
generation on faithful charts. Conditions: severity direction, random direction of
matched norm, and the severity direction at a non-target layer. Dose grid spans
fractions of the typical residual norm, both signs. Readouts, declared now:
(1) probe-score shift at the severity layer (quantitative primary);
(2) distortion-assertion rate in the generated descriptions, graded by a fixed
    keyword rule (exaggerat|mislead|distort|truncat|manipulat|deceptiv|inverted),
    with every text saved for a human check;
(3) fluency guard: mean words per description and a repetition flag (any 4-gram
    repeated 3+ times); effects count only where fluency is intact.

  python rq6_steer.py --acts qwen7b.npz --data trackB_data --out rq6.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from probekit.activations import ActivationSet
from probekit.vlm import _batch_inputs, extract_vlm, load_vlm

SEED = 20260809
HOLDOUT = "cherry_picked_window"
SEVERITY_LAYER = 18
NON_TARGET_LAYER = 5
KEYWORDS = re.compile(r"exaggerat|mislead|distort|truncat|manipulat|deceptiv|inverted", re.I)


def series_split(series, seed=SEED, test_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.sort(np.unique(series))
    test = set(rng.choice(uniq, int(round(len(uniq) * test_frac)), replace=False))
    return np.array([s not in test for s in series]), np.array([s in test for s in series])


def severity_direction(acts: ActivationSet, layer_idx: int):
    mech = np.array([m["mechanism"] for m in acts.meta])
    series = np.array([m["series"] for m in acts.meta])
    lf = np.array([m["lie_factor"] if m["lie_factor"] is not None else np.nan for m in acts.meta])
    keep = np.where((mech != HOLDOUT) & ~np.isnan(lf))[0]
    tr_m, _ = series_split(series[keep])
    tr = keep[tr_m]
    sc = StandardScaler().fit(acts.X[tr, layer_idx, :])
    m = Ridge(alpha=10.0).fit(sc.transform(acts.X[tr, layer_idx, :]), lf[tr])
    w_raw = m.coef_ / sc.scale_
    return (w_raw / np.linalg.norm(w_raw)).astype(np.float32)


@torch.no_grad()
def steered_generate(vlm, image_paths, prompt, layer, direction, alpha, max_new_tokens=60):
    d = torch.as_tensor(direction)

    def hook(module, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * d.to(h.dtype).to(h.device)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = vlm.blocks[layer].register_forward_hook(hook) if alpha != 0 else None
    try:
        enc = _batch_inputs(vlm, image_paths, prompt)
        tok = vlm.processor.tokenizer
        gen = vlm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        return [tok.decode(gen[i, n:], skip_special_tokens=True).strip() for i in range(len(image_paths))]
    finally:
        if handle:
            handle.remove()


def fluent(text: str) -> bool:
    words = text.split()
    if len(words) < 8:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    from collections import Counter
    return Counter(grams).most_common(1)[0][1] < 3 if grams else False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n-charts", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = Path(args.data)

    acts = ActivationSet.load(args.acts)
    li = acts.layers.index(SEVERITY_LAYER)
    d_sev = severity_direction(acts, li)
    rng = np.random.default_rng(7)
    d_rand = rng.standard_normal(len(d_sev)).astype(np.float32)
    d_rand /= np.linalg.norm(d_rand)

    mech = np.array([m["mechanism"] for m in acts.meta])
    series = np.array([m["series"] for m in acts.meta])
    _, te_m = series_split(series)
    faithful_te = [i for i in np.where((mech == "faithful") & te_m)[0]][: args.n_charts]
    paths = [data / acts.meta[i]["path"] for i in faithful_te]
    print(f"steering on {len(paths)} held-out faithful charts")

    vlm = load_vlm(args.model)
    # alpha grid from the typical residual norm at the severity layer
    norm = float(np.median(np.linalg.norm(acts.X[faithful_te, li, :], axis=1)))
    alphas = [round(f * norm, 1) for f in (-0.5, -0.25, 0, 0.25, 0.5, 1.0)]
    print(f"residual norm ~{norm:.0f}; alphas {alphas}")

    results, texts_dump = [], {}
    conditions = [("severity", SEVERITY_LAYER, d_sev), ("random", SEVERITY_LAYER, d_rand),
                  ("nontarget", NON_TARGET_LAYER, d_sev)]
    prompt = "Describe what this chart shows."
    for cname, layer, d in conditions:
        for a in alphas:
            texts = []
            for s in range(0, len(paths), 6):
                texts += steered_generate(vlm, paths[s:s + 6], prompt, layer, d, a)
            fl = [fluent(t) for t in texts]
            kw = [bool(KEYWORDS.search(t)) for t in texts]
            kw_fluent = [k for k, f in zip(kw, fl) if f]
            # probe-score shift under the same intervention
            if a == 0:
                sacts = extract_vlm(vlm, paths, prompt, layers=[SEVERITY_LAYER], batch_size=6, progress=False)
            else:
                from probekit.steering import steer as steer_ctx

                class _P:  # adapter for text steer ctx on vlm blocks
                    blocks = vlm.blocks
                with steer_ctx(_P, layer, d, float(a)):
                    sacts = extract_vlm(vlm, paths, prompt, layers=[SEVERITY_LAYER], batch_size=6, progress=False)
            sc = StandardScaler().fit(acts.X[:, li, :])
            proj = float(np.mean(sacts.X[:, 0, :] @ d_sev))
            results.append({"condition": cname, "alpha": a, "keyword_rate": float(np.mean(kw)),
                            "keyword_rate_fluent": float(np.mean(kw_fluent)) if kw_fluent else None,
                            "fluent_rate": float(np.mean(fl)), "mean_proj_on_severity": proj,
                            "n": len(texts)})
            texts_dump[f"{cname}_a{a}"] = texts
            print(f"{cname:10} a={a:>8} | kw {np.mean(kw):.2f} | fluent {np.mean(fl):.2f} | proj {proj:.1f}")

    Path(args.out).write_text(json.dumps({"results": results, "alphas": alphas,
                                          "severity_layer": SEVERITY_LAYER}, indent=2))
    Path(args.out).with_suffix(".texts.json").write_text(json.dumps(texts_dump, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
