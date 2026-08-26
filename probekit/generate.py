"""Batched generation and next-token readouts (also the prompting baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .activations import tokenize_batch
from .models import LoadedModel, seed_all


@dataclass
class GenOut:
    prompt: str
    text: str          # newly generated text only
    full_text: str     # prompt + generation (feed this to extract() with response_mask)
    n_new_tokens: int


@torch.no_grad()
def generate(lm: LoadedModel, prompts: Sequence[str], max_new_tokens: int = 64, do_sample: bool = False,
             temperature: float = 1.0, top_p: float = 1.0, batch_size: int = 8, seed: int = 0,
             max_length: int = 1024) -> list[GenOut]:
    if seed is not None:
        seed_all(seed)
    outs: list[GenOut] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start:start + batch_size])
        enc = tokenize_batch(lm, batch, max_length=max_length)
        gen = lm.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
            pad_token_id=lm.tokenizer.pad_token_id,
        )
        n_prompt = enc["input_ids"].shape[1]  # left padding: all prompts occupy the first n_prompt slots
        for i, p in enumerate(batch):
            new_ids = gen[i, n_prompt:]
            text = lm.tokenizer.decode(new_ids, skip_special_tokens=True)
            outs.append(GenOut(prompt=p, text=text, full_text=p + text, n_new_tokens=int((new_ids != lm.tokenizer.pad_token_id).sum())))
    return outs


@torch.no_grad()
def next_token_logits(lm: LoadedModel, prompts: Sequence[str], batch_size: int = 8,
                      max_length: int = 1024) -> np.ndarray:
    """[n, vocab] logits at the last position of each prompt."""
    rows = []
    for start in range(0, len(prompts), batch_size):
        enc = tokenize_batch(lm, list(prompts[start:start + batch_size]), max_length=max_length)
        out = lm.model(**enc, use_cache=False)
        rows.append(out.logits[:, -1, :].float().cpu())
    return torch.cat(rows).numpy()


def first_token_id(lm: LoadedModel, text: str) -> int:
    ids = lm.tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(ids[0])


def logit_diff(lm: LoadedModel, prompts: Sequence[str], token_a: str = " Yes", token_b: str = " No",
               batch_size: int = 8) -> np.ndarray:
    """logit(token_a) - logit(token_b) at the next-token position. The standard behavioural readout."""
    logits = next_token_logits(lm, prompts, batch_size=batch_size)
    ia, ib = first_token_id(lm, token_a), first_token_id(lm, token_b)
    return logits[:, ia] - logits[:, ib]


def prompting_baseline(lm: LoadedModel, texts: Sequence[str], question: str,
                       batch_size: int = 8) -> np.ndarray:
    """Black-box baseline: ask the model itself and read the Yes/No logit difference.

    Score is comparable to a probe decision score (higher = more Yes).
    """
    prompts = [lm.chat(f"{t}\n\n{question} Answer with Yes or No only.") for t in texts]
    return logit_diff(lm, prompts, batch_size=batch_size)
