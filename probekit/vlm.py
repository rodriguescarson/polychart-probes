"""Vision-language extraction: images + one elicitation prompt -> residual-stream activations.

Mirrors models.py/activations.py for VLMs. The residual stream probed is the LANGUAGE
tower's (get_blocks finds model.language_model.layers); positions follow the same
convention as text extraction ("last" = final prompt token, i.e. where the answer would
begin).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from .activations import ActivationSet, residual_hooks
from .models import get_blocks, pick_device, seed_all


@dataclass
class LoadedVLM:
    name: str
    model: torch.nn.Module
    processor: object
    device: str
    blocks: torch.nn.ModuleList
    n_layers: int
    d_model: int
    dtype: torch.dtype


def load_vlm(name: str, dtype: torch.dtype | None = None, device: str | None = None,
             seed: int | None = 0, load_in_4bit: bool = False) -> LoadedVLM:
    device = device or pick_device()
    if dtype is None:
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
    if seed is not None:
        seed_all(seed)
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM
    kwargs = dict(torch_dtype=dtype)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    if device == "cuda" and not load_in_4bit:
        kwargs["device_map"] = "cuda"
    model = AutoVLM.from_pretrained(name, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    proc = AutoProcessor.from_pretrained(name)
    if hasattr(proc, "tokenizer"):
        proc.tokenizer.padding_side = "left"
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
    blocks = get_blocks(model)
    cfg = getattr(model.config, "text_config", model.config)
    d_model = getattr(cfg, "hidden_size")
    return LoadedVLM(name=name, model=model, processor=proc, device=device, blocks=blocks,
                     n_layers=len(blocks), d_model=int(d_model), dtype=dtype)


def _batch_inputs(vlm: LoadedVLM, image_paths: Sequence[str | Path], prompt: str):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                for _ in images]
    texts = [vlm.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages]
    # one image per text, nested: Gemma3's processor requires per-text grouping; Qwen accepts both.
    return vlm.processor(text=texts, images=[[im] for im in images],
                         return_tensors="pt", padding=True).to(vlm.device)


@torch.no_grad()
def extract_vlm(vlm: LoadedVLM, image_paths: Sequence[str | Path], prompt: str,
                layers: Sequence[int] | None = None, positions: str = "last",
                meta: Sequence[dict] | None = None, batch_size: int = 4,
                progress: bool = True) -> ActivationSet:
    """positions: 'last' (final prompt token) or 'mean_text' (mean over the text tokens after the image)."""
    layers = list(range(vlm.n_layers)) if layers is None else list(layers)
    meta = list(meta) if meta is not None else [{} for _ in image_paths]
    rows: list[np.ndarray] = []
    iterator = range(0, len(image_paths), batch_size)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"extract_vlm[{positions}]", unit="batch")
    if positions not in ("last", "mean_text"):
        raise ValueError(f"unknown positions={positions!r}")
    k_tail = 1 if positions == "last" else 24
    for start in iterator:
        batch = list(image_paths[start:start + batch_size])
        enc = _batch_inputs(vlm, batch, prompt)
        # Capture ONLY the needed trailing positions inside the hook. Charts with extreme
        # aspect ratios carry thousands of vision tokens; materialising [B, T, L, d] for
        # them OOM-kills memory-capped containers, and "last"/"mean_text" never needs it.
        # Left padding puts every sequence's real tail at the end, so h[:, -k:, :] is exact.
        store: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(L: int):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                store[L] = h[:, -k_tail:, :].float().cpu()
                return None
            return hook

        try:
            for L in layers:
                handles.append(vlm.blocks[L].register_forward_hook(make_hook(L)))
            vlm.model(**enc, use_cache=False)
        finally:
            for h_ in handles:
                h_.remove()
        H = torch.stack([store.pop(L) for L in layers], dim=2)  # [B, k, L, d]
        for i in range(len(batch)):
            rows.append(H[i].mean(0).numpy() if positions == "mean_text" else H[i, -1].numpy())
        del store, H
    X = np.stack(rows).astype(np.float32)
    return ActivationSet(X=X, layers=layers, meta=meta, positions=positions, model_name=vlm.name)


class _BlocksProxy:
    """Adapter so activations.residual_hooks (which expects .blocks) works for LoadedVLM."""

    def __init__(self, vlm: LoadedVLM):
        self.blocks = vlm.blocks
