"""Residual-stream activation capture.

Convention used everywhere in probekit
--------------------------------------
"layer L" means the output of decoder block L (0-indexed). With HF's
`output_hidden_states=True` that is `hidden_states[L + 1]`; `hidden_states[0]`
is the embedding output. The last entry of `hidden_states` is taken AFTER the
final norm in Llama/Qwen, so it does not equal the raw block output; we always
read block outputs via hooks, never `hidden_states[-1]`.

Token positions
---------------
`positions="last"`  -> the final non-pad token of each sequence (with left padding, index -1)
`positions="mean"`  -> mean over tokens selected by `mask` (default: all non-pad tokens)
`positions="all"`   -> keep every token; returns a ragged list (one [T_i, n_layers, d] array per text)

Masks are boolean arrays over the tokenized sequence, e.g. `response_mask(...)`
gives the Apollo-style "assistant response tokens only" mask.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch

from .models import LoadedModel


def _unwrap(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    return new


@contextmanager
def residual_hooks(lm: LoadedModel, layers: Sequence[int], store: dict[int, torch.Tensor],
                   edit: Callable[[int, torch.Tensor], torch.Tensor] | None = None) -> Iterator[None]:
    """Register forward hooks on the chosen blocks.

    `store[L]` receives the block-L output hidden state (kept on device, original dtype).
    If `edit` is given, it is applied to the hidden state and the edited tensor is what the
    next block sees (this is how steering / ablation are implemented in steering.py).
    """
    handles = []

    def make_hook(layer: int):
        def hook(module, args, output):
            h = _unwrap(output)
            if edit is not None:
                h = edit(layer, h)
            store[layer] = h
            return _rewrap(output, h) if edit is not None else None
        return hook

    try:
        for L in layers:
            handles.append(lm.blocks[L].register_forward_hook(make_hook(L)))
        yield
    finally:
        for h in handles:
            h.remove()


def tokenize_batch(lm: LoadedModel, texts: Sequence[str], max_length: int = 1024):
    return lm.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=not bool(lm.tokenizer.chat_template)).to(lm.device)


def response_mask(lm: LoadedModel, prompt_text: str, full_text: str, max_length: int = 1024) -> np.ndarray:
    """Boolean mask over tokens of `full_text` that belong to the part after `prompt_text`.

    Both strings are tokenized without padding; the mask is aligned to the unpadded token
    sequence, and `extract` re-aligns it to the left-padded batch.
    """
    add_special = not bool(lm.tokenizer.chat_template)
    n_prompt = len(lm.tokenizer(prompt_text, add_special_tokens=add_special, truncation=True, max_length=max_length)["input_ids"])
    n_full = len(lm.tokenizer(full_text, add_special_tokens=add_special, truncation=True, max_length=max_length)["input_ids"])
    m = np.zeros(n_full, dtype=bool)
    m[n_prompt:] = True
    return m


@dataclass
class ActivationSet:
    """X: [n, n_layers_selected, d_model] float32. layers: which block outputs. meta: one dict per row."""
    X: np.ndarray
    layers: list[int]
    meta: list[dict] = field(default_factory=list)
    positions: str = "last"
    model_name: str = ""
    per_token: list[np.ndarray] | None = None  # only when positions == "all": list of [T_i, n_layers, d]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def layer_index(self, layer: int) -> int:
        return self.layers.index(layer)

    def at(self, layer: int) -> np.ndarray:
        """[n, d_model] activations at one layer."""
        return self.X[:, self.layer_index(layer), :]

    def labels(self, key: str = "label") -> np.ndarray:
        return np.array([m[key] for m in self.meta])

    def groups(self, key: str = "group") -> np.ndarray:
        return np.array([m.get(key, i) for i, m in enumerate(self.meta)])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, layers=np.array(self.layers))
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"meta": self.meta, "positions": self.positions, "model_name": self.model_name, "layers": self.layers}))
        if self.per_token is not None:
            np.savez_compressed(path.with_suffix(".tokens.npz"), *self.per_token)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationSet":
        path = Path(path)
        z = np.load(path)
        info = json.loads(path.with_suffix(".meta.json").read_text())
        per_token = None
        tp = path.with_suffix(".tokens.npz")
        if tp.exists():
            t = np.load(tp)
            per_token = [t[k] for k in t.files]
        return cls(X=z["X"], layers=[int(x) for x in z["layers"]], meta=info["meta"],
                   positions=info["positions"], model_name=info["model_name"], per_token=per_token)


@torch.no_grad()
def extract(lm: LoadedModel, texts: Sequence[str], layers: Sequence[int] | None = None,
            positions: str = "last", masks: Sequence[np.ndarray] | None = None,
            meta: Sequence[dict] | None = None, batch_size: int = 8, max_length: int = 1024,
            progress: bool = True) -> ActivationSet:
    """Run the model over `texts` and return residual-stream activations at `layers`.

    masks: optional per-text boolean arrays over the UNPADDED token sequence (see response_mask);
    used by positions="mean" (average over True tokens) and positions="all" (keep only True tokens).
    """
    layers = list(range(lm.n_layers)) if layers is None else list(layers)
    meta = list(meta) if meta is not None else [{} for _ in texts]
    rows: list[np.ndarray] = []
    per_token: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"extract[{positions}]", unit="batch")

    for start in iterator:
        batch = list(texts[start:start + batch_size])
        enc = tokenize_batch(lm, batch, max_length=max_length)
        attn = enc["attention_mask"].bool()  # [B, T], left padded
        store: dict[int, torch.Tensor] = {}
        with residual_hooks(lm, layers, store):
            lm.model(**enc, use_cache=False)
        # stack -> [B, T, n_layers, d] in float32 on CPU
        H = torch.stack([store[L] for L in layers], dim=2).float().cpu()
        B, T = attn.shape
        lengths = attn.sum(1).tolist()
        for i in range(B):
            n_tok = int(lengths[i])
            h_i = H[i, T - n_tok:, :, :]  # drop left padding -> [n_tok, n_layers, d]
            if masks is not None:
                m = np.asarray(masks[start + i], dtype=bool)
                # masks were built on the unpadded, possibly truncated sequence
                m = m[-n_tok:] if len(m) >= n_tok else np.pad(m, (n_tok - len(m), 0))
            else:
                m = np.ones(n_tok, dtype=bool)
            if positions == "last":
                rows.append(h_i[-1].numpy())
            elif positions == "mean":
                sel = h_i[torch.from_numpy(m)]
                rows.append(sel.mean(0).numpy() if len(sel) else h_i[-1].numpy())
            elif positions == "all":
                sel = h_i[torch.from_numpy(m)].numpy()
                per_token.append(sel)
                rows.append(sel.mean(0) if len(sel) else h_i[-1].numpy())
            else:
                raise ValueError(f"unknown positions={positions!r}")
        del store, H

    X = np.stack(rows).astype(np.float32)
    return ActivationSet(X=X, layers=layers, meta=meta, positions=positions, model_name=lm.name,
                         per_token=per_token if positions == "all" else None)


@torch.no_grad()
def check_off_by_one(lm: LoadedModel, text: str = "The capital of France is Paris.") -> dict:
    """Verify hook outputs equal `output_hidden_states` entries shifted by one.

    Returns per-layer max abs difference. Expect ~0 for all layers except possibly the last
    (HF applies the final norm before storing the last hidden state in Llama/Qwen).
    """
    enc = tokenize_batch(lm, [text])
    layers = list(range(lm.n_layers))
    store: dict[int, torch.Tensor] = {}
    with residual_hooks(lm, layers, store):
        out = lm.model(**enc, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states
    diffs = {L: float((store[L].float() - hs[L + 1].float()).abs().max()) for L in layers}
    ok_layers = [L for L in layers[:-1] if diffs[L] < 1e-3]
    return {
        "n_hidden_states": len(hs), "n_layers": lm.n_layers, "max_abs_diff": diffs,
        "all_but_last_match": len(ok_layers) == lm.n_layers - 1,
        "last_layer_matches": diffs[layers[-1]] < 1e-3,
    }
