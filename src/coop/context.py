"""Context features (docs/CONTEXT_PARTNER_PLAN_v3.md, 3.2).

Computed on any state: the full state (evaluator) or a player's own view (``coop.view``),
so a player's decisions and the predicates judging them use what that player could see.

| feature | definition |
|---|---|
| ``urgent`` | open orders (index, dish) with at most ``theta`` steps left on their timer |
| ``on_order(who)`` | the dish ``who`` is working on: what it holds belongs to it, or its last choice |
| ``blocked`` | utensil kinds whose every utensil is running or holds finished food |
| ``prep_ready`` | open dishes with a processed item (an intermediate product of the dish) waiting unused, finished in a utensil or on a counter |

``ContextTracker`` turns the features into events: a context *starts* at the step it first
holds (an order becoming urgent, a kind becoming blocked, prepared items appearing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

CONTEXT_RULES = ("default_work", "on_urgent", "on_blocked", "on_prep_ready")


def open_dishes(state, n_open: int) -> List[str]:
    return list(state.order_list[:max(1, n_open)]) if state.order_list else []


def tree(kitchen, dish: str) -> Set[str]:
    """Every item ``dish`` is made from, the dish included."""
    out, todo = set(), [dish]
    while todo:
        x = todo.pop()
        if x in out:
            continue
        out.add(x)
        if x in kitchen.recipes:
            todo += list(kitchen.recipes[x][1])
    return out


def urgent(state, n_open: int, theta: int) -> List[Tuple[int, str]]:
    timers = getattr(state, "order_timers", None)
    if not timers:
        return []
    dishes = open_dishes(state, n_open)
    return [(i, d) for i, d in enumerate(dishes) if i < len(timers) and timers[i] is not None
            and 0 < timers[i] <= theta]


def blocked(state, kitchen) -> List[str]:
    kinds: Dict[str, List[str]] = {}
    for u, k in kitchen.kind_of.items():
        kinds.setdefault(k, []).append(u)
    out = []
    for k, us in sorted(kinds.items()):
        if all(kitchen.utensil_status(state, u)[0] in ("processing", "finished") for u in us):
            out.append(k)
    return out


def prep_ready(state, kitchen, n_open: int) -> List[str]:
    """Open dishes for which a processed item waits unused (not held by anyone)."""
    waiting: List[str] = []
    for u in kitchen.utensils:
        status, product, _, _ = kitchen.utensil_status(state, u)
        if status == "finished" and product:
            waiting.append(product)
    for p, obj in state.objects.items():
        if p not in kitchen.utensil_at and getattr(obj, "name", None):
            waiting.append(obj.name)
    out = []
    for d in open_dishes(state, n_open):
        need = tree(kitchen, d) - {d} - set(kitchen.raw)
        if any(w in need for w in waiting) and d not in out:
            out.append(d)
    return out


def on_order(state, kitchen, n_open: int, index: int, last_dish: Optional[str] = None) -> Optional[str]:
    """The open dish player ``index`` works on: the one its held item belongs to (if only one
    open dish can use it), else its last choice if that dish is still open."""
    dishes = open_dishes(state, n_open)
    held = state.players[index].held_object
    if held is not None and held.name != "dish":
        users = [d for d in dishes if held.name in tree(kitchen, d)]
        if len(set(users)) == 1:
            return users[0]
    return last_dish if last_dish in dishes else None


def features(state, kitchen, n_open: int, theta: int) -> Dict[str, Any]:
    return {"urgent": urgent(state, n_open, theta), "blocked": blocked(state, kitchen),
            "prep_ready": prep_ready(state, kitchen, n_open)}


@dataclass
class ContextTracker:
    """Rising edges of the context features on a sequence of states (one player's view or
    the full state)."""
    kitchen: Any
    n_open: int
    theta: int
    prev: Dict[str, Any] = field(default_factory=lambda: {"urgent": [], "blocked": [], "prep_ready": []})
    now: Dict[str, Any] = field(default_factory=lambda: {"urgent": [], "blocked": [], "prep_ready": []})
    events: List[Dict[str, Any]] = field(default_factory=list)

    def observe(self, state, t: int) -> List[Dict[str, Any]]:
        f = features(state, self.kitchen, self.n_open, self.theta)
        new = []
        dishes = open_dishes(state, self.n_open)
        prev_urgent = {d for _, d in self.prev["urgent"]}
        for i, d in f["urgent"]:
            if d not in prev_urgent:
                new.append({"t": t, "context": "on_urgent", "order": i, "dish": d})
        for k in f["blocked"]:
            if k not in self.prev["blocked"]:
                new.append({"t": t, "context": "on_blocked", "kind": k})
        for d in f["prep_ready"]:
            if d not in self.prev["prep_ready"] and d in dishes:
                new.append({"t": t, "context": "on_prep_ready", "dish": d})
        self.prev = self.now = f
        self.events += new
        return new
