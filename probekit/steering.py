"""Causal interventions on the residual stream: add a direction, or project it out.

Both are context managers, so an experiment reads like:

    with steer(lm, layer=14, direction=d, alpha=8.0):
        outs = generate(lm, prompts)

Controls that should accompany every steering claim:
- random_direction(...) with the same alpha (specificity)
- the same direction at a non-target layer (locality)
- alpha=0 (no-op sanity)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from .activations import residual_hooks
from .models import LoadedModel
from .stats import bootstrap_metric


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def random_direction(d_model: int, seed: int = 0, orthogonal_to: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d_model).astype(np.float32)
    if orthogonal_to is not None:
        o = unit(orthogonal_to)
        v = v - (v @ o) * o
    return unit(v)


def _as_tensor(direction: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(unit(direction), dtype=like.dtype, device=like.device)


@contextmanager
def steer(lm: LoadedModel, layer: int, direction: np.ndarray, alpha: float,
          positions: str = "all") -> Iterator[None]:
    """Add alpha * unit(direction) to the block output at `layer`.

    positions="all": every token position (also every step during generation).
    positions="last": only the final position of the current forward pass.
    alpha is in raw activation units; typical useful range is a few multiples of the
    typical residual norm divided by 10 (sweep it, do not guess).
    """
    def edit(L: int, h: torch.Tensor) -> torch.Tensor:
        d = _as_tensor(direction, h)
        if positions == "all":
            return h + alpha * d
        h = h.clone()
        h[:, -1, :] = h[:, -1, :] + alpha * d
        return h

    store: dict[int, torch.Tensor] = {}
    with residual_hooks(lm, [layer], store, edit=edit):
        yield


@contextmanager
def ablate(lm: LoadedModel, layer: int, direction: np.ndarray) -> Iterator[None]:
    """Project the direction out of the block output at `layer` (zero its component at every position)."""
    def edit(L: int, h: torch.Tensor) -> torch.Tensor:
        d = _as_tensor(direction, h)
        coef = (h * d).sum(-1, keepdim=True)
        return h - coef * d

    store: dict[int, torch.Tensor] = {}
    with residual_hooks(lm, [layer], store, edit=edit):
        yield


@contextmanager
def no_intervention() -> Iterator[None]:
    yield


def dose_response(lm: LoadedModel, layer: int, direction: np.ndarray, alphas: Sequence[float],
                  metric_fn: Callable[[], np.ndarray], conditions: dict[str, np.ndarray] | None = None,
                  n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Sweep alpha and measure metric_fn() under the intervention.

    metric_fn is called INSIDE the steering context and must return one value per prompt
    (e.g. probe scores at a read-out layer, a Yes/No logit difference, or a labeled-output rate).
    `conditions` maps extra condition names to alternative directions run at the same alphas
    (e.g. {"random": random_direction(...)}). Rows carry mean and a bootstrap 95% CI over prompts.
    """
    dirs = {"probe": np.asarray(direction)}
    dirs.update(conditions or {})
    rows = []
    for name, d in dirs.items():
        for a in alphas:
            ctx = steer(lm, layer, d, float(a)) if a != 0 else no_intervention()
            with ctx:
                vals = np.asarray(metric_fn(), dtype=np.float64)
            r = bootstrap_metric(lambda y, s: float(np.mean(s)), np.zeros(len(vals)), vals,
                                 n_boot=n_boot, seed=seed)
            rows.append({"condition": name, "alpha": float(a), "mean": r.estimate,
                         "lo": r.lo, "hi": r.hi, "n": len(vals)})
    return pd.DataFrame(rows)


def typical_residual_norm(lm: LoadedModel, texts: Sequence[str], layer: int, batch_size: int = 8) -> float:
    """Median L2 norm of the residual stream at `layer` over the last token of each text.

    Use it to pick a sensible alpha grid: alphas = frac * norm for frac in (0.5, 1, 2, 4, ...).
    """
    from .activations import extract
    acts = extract(lm, list(texts), layers=[layer], positions="last", batch_size=batch_size, progress=False)
    return float(np.median(np.linalg.norm(acts.at(layer), axis=1)))
