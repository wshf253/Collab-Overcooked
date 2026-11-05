"""Seeded order sequences for continuous-order episodes.

The whole sequence for an episode is drawn up front from the episode seed, so
every condition run with the same seed sees the same orders in the same order
(paired design). Nothing here touches Python's global ``random`` state.
"""
from __future__ import annotations

import random
from typing import Dict, List, Mapping, Sequence, Union

Menu = Union[Sequence[str], Mapping[str, float]]


def normalize_menu(menu: Menu) -> Dict[str, float]:
    """Return {order name: weight} with positive weights only.

    A list or tuple means equal weights. Duplicate names in a list are an error,
    so a typo cannot silently double an order's weight.
    """
    if isinstance(menu, Mapping):
        weights = {str(k): float(v) for k, v in menu.items()}
    else:
        names = [str(n) for n in menu]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate order names in menu: {names}")
        weights = {n: 1.0 for n in names}
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"Menu weights must be non-negative: {weights}")
    weights = {k: w for k, w in weights.items() if w > 0}
    if not weights:
        raise ValueError("Menu has no order with positive weight")
    return weights


def make_order_sequence(menu: Menu, seed: int, length: int, no_repeat: bool = False) -> List[str]:
    """Draw ``length`` orders from ``menu`` with a private RNG seeded by ``seed``.

    ``no_repeat`` forbids the same order twice in a row (needs >= 2 menu items).
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    weights = normalize_menu(menu)
    names = sorted(weights)  # fixed order: the sequence must not depend on dict order
    if no_repeat and len(names) < 2:
        raise ValueError("no_repeat needs at least two orders on the menu")
    rng = random.Random(seed)
    seq: List[str] = []
    for _ in range(length):
        pool = [n for n in names if not (no_repeat and seq and n == seq[-1])]
        seq.append(rng.choices(pool, weights=[weights[n] for n in pool], k=1)[0])
    return seq
