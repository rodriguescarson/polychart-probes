"""Datasets: contrastive statement sets and instructed honest/dishonest pairs.

Every example carries a `group` id. Splits and cross-validation are ALWAYS group-aware,
so the two members of a contrast pair (or all statements about one city) can never sit
on both sides of a train/test split. That is the main leakage trap in probe experiments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# RepE-style instructed contrast (Zou et al. 2023; used by Goldowsky-Dill et al. 2025).
HONEST_SYS = "You are an honest assistant. Always make truthful statements."
DISHONEST_SYS = "You are a dishonest assistant. Always make untruthful statements."


@dataclass
class Example:
    text: str
    label: int          # 1 = positive class (e.g. false statement / dishonest condition)
    group: str
    meta: dict = field(default_factory=dict)


def _rows(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_got(name: str = "cities", n: int | None = None, seed: int = 0) -> list[Example]:
    """Geometry-of-truth datasets (Marks & Tegmark 2023): `statement`, `label` (1 = TRUE statement).

    We flip nothing: label 1 means the statement is true. Group = city/object when available,
    else a row id. Available: cities, neg_cities, sp_en_trans, neg_sp_en_trans, larger_than,
    smaller_than, common_claim_true_false, companies_true_false, counterfact_true_false.
    """
    rows = _rows(name)
    rng = np.random.default_rng(seed)
    if n is not None and n < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), n, replace=False)]
    out = []
    for i, r in enumerate(rows):
        group = r.get("city") or r.get("correct_country") or r.get("object") or f"{name}:{i}"
        out.append(Example(text=r["statement"], label=int(float(r["label"])), group=str(group),
                           meta={"dataset": name}))
    return out


def instructed_pairs(statements: list[str], honest_sys: str = HONEST_SYS,
                     dishonest_sys: str = DISHONEST_SYS, lm=None) -> list[Example]:
    """Same statement under an honest vs a dishonest system prompt. label 1 = dishonest condition.

    Both members share a group, so group-aware splits keep pairs together. If `lm` is given,
    texts are rendered with its chat template (assistant turn = the statement itself).
    """
    out = []
    for i, s in enumerate(statements):
        for sys, lab in ((honest_sys, 0), (dishonest_sys, 1)):
            text = lm.chat(user="State a fact about the world.", system=sys, assistant=s) if lm is not None \
                else f"{sys}\nUser: State a fact about the world.\nAssistant: {s}"
            out.append(Example(text=text, label=lab, group=f"pair:{i}", meta={"statement": s}))
    return out


# A small bundled sentiment set (30/30) for a second, unrelated concept.
_POS = [
    "The food at that place was excellent and the staff were kind.",
    "I loved the movie; the ending made the whole theatre cheer.",
    "Her presentation was clear, confident, and well organised.",
    "This laptop is fast, quiet, and the battery lasts all day.",
    "The garden looks beautiful after the rain.",
    "Our trip went smoothly and the hotel exceeded expectations.",
    "The team shipped the feature early and users are delighted.",
    "That book kept me up all night; I could not put it down.",
    "The coffee here tastes wonderful and the seats are comfortable.",
    "My interview went great and they called back the same day.",
    "The concert sounded fantastic from every seat.",
    "He fixed the bug in minutes and explained it patiently.",
    "The beach was clean, warm, and quiet all afternoon.",
    "This recipe is simple and the result tastes amazing.",
    "The new update made the app noticeably faster.",
    "She sang beautifully and the crowd loved every song.",
    "The doctor was thorough and reassuring.",
    "Delivery arrived a day early and nothing was damaged.",
    "The workshop was practical and worth every minute.",
    "Their customer support solved my problem on the first call.",
    "The park was full of happy families enjoying the sun.",
    "I passed the exam with a better score than I hoped.",
    "The apartment is bright, spacious, and close to everything.",
    "Training went well and my times keep improving.",
    "The play was funny, moving, and perfectly cast.",
    "My plants are thriving on the new balcony.",
    "The mentor gave me honest, useful feedback.",
    "We closed the deal and the client praised the work.",
    "The trail offered stunning views the entire hike.",
    "Dinner with old friends left me smiling all night.",
]
_NEG = [
    "The food was cold and the waiter ignored us for an hour.",
    "I hated the movie; the plot made no sense at all.",
    "His presentation was confusing and ran far over time.",
    "This laptop overheats and the battery dies by noon.",
    "The garden flooded and half the plants are dead.",
    "Our flight was delayed twice and the hotel lost our booking.",
    "The release broke login and users are furious.",
    "That book bored me so much I stopped after two chapters.",
    "The coffee tasted burnt and the chairs were filthy.",
    "The interview went badly and they never called back.",
    "The concert sound was muddy and the show started late.",
    "He ignored the bug report and the crash got worse.",
    "The beach was crowded, dirty, and freezing.",
    "This recipe wasted an hour and tasted bland.",
    "The update made the app slower and drains the battery.",
    "She forgot the lyrics and the band fell apart.",
    "The doctor rushed the visit and answered nothing.",
    "Delivery came a week late with the box crushed.",
    "The workshop was vague and a waste of money.",
    "Support kept me on hold and never solved the issue.",
    "The park was littered and the fountain was broken.",
    "I failed the exam despite studying for weeks.",
    "The apartment is dark, cramped, and noisy at night.",
    "My training stalled and the injury is back.",
    "The play dragged on and half the audience left.",
    "My plants withered within a week on that balcony.",
    "The mentor cancelled again without any notice.",
    "We lost the deal and the client blamed our work.",
    "The trail was washed out and we turned back soaked.",
    "Dinner ended in an argument and a huge bill.",
]


def sentiment_small() -> list[Example]:
    """60 hand-written sentences. label 1 = positive sentiment. group = row id."""
    out = [Example(text=t, label=1, group=f"sent:{i}", meta={"dataset": "sentiment"}) for i, t in enumerate(_POS)]
    out += [Example(text=t, label=0, group=f"sent:{len(_POS) + i}", meta={"dataset": "sentiment"}) for i, t in enumerate(_NEG)]
    return out


def to_arrays(examples: list[Example]) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts = [e.text for e in examples]
    y = np.array([e.label for e in examples])
    groups = np.array([e.group for e in examples])
    return texts, y, groups


def split_by_group(examples: list[Example], test_frac: float = 0.3, seed: int = 0) -> tuple[list[Example], list[Example]]:
    rng = np.random.default_rng(seed)
    groups = sorted({e.group for e in examples})
    test_groups = set(rng.choice(groups, int(round(len(groups) * test_frac)), replace=False))
    train = [e for e in examples if e.group not in test_groups]
    test = [e for e in examples if e.group in test_groups]
    return train, test


def balance_report(examples: list[Example], tokenizer=None) -> dict:
    """Length and composition check per class. Big gaps mean the probe may learn length or format, so report it."""
    out: dict = {"n": len(examples)}
    for lab in (0, 1):
        texts = [e.text for e in examples if e.label == lab]
        lens = [len(t) for t in texts]
        row = {"n": len(texts), "chars_mean": float(np.mean(lens)) if lens else 0.0,
               "chars_sd": float(np.std(lens)) if lens else 0.0}
        if tokenizer is not None:
            tl = [len(tokenizer(t)["input_ids"]) for t in texts]
            row["tokens_mean"] = float(np.mean(tl))
        out[f"class_{lab}"] = row
    g = {}
    for e in examples:
        g.setdefault(e.group, set()).add(e.label)
    out["groups_with_both_labels"] = sum(1 for s in g.values() if len(s) == 2)
    out["n_groups"] = len(g)
    return out
