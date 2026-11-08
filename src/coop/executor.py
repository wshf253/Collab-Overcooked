"""High-level actions for the agent, turned into primitive moves.

The action names follow Collab-Overcooked, with no role split: either player
may use any station it can reach.

    pickup(<item>, ingredient_dispenser)   take a raw ingredient (hands empty)
    pickup(dish, dish_dispenser)           take a clean dish (hands empty)
    pickup(<item>, counter)                take <item> from the nearest reachable counter
    pickup(<item>, <utensil>)              take finished food out of a utensil
    put_obj_in_utensil(<utensil>)          put the held item into a utensil that is not running
    cut|cook|bake|stir(<utensil>)          start a loaded utensil (verb must match the utensil)
    fill_dish_with_food(<utensil>)         with a dish in hand, collect food (waits if still cooking)
    deliver_soup()                         serve the held food
    place_obj_on_counter()                 put the held item on a free counter, shared ones first
    wait(<n>)                              stay for n steps (1-20); ends early when the partner
                                           completes an action

An action is checked when it is chosen (``check``) and again on every step
while it runs. It ends as ``done`` when its interaction takes effect, or as
``failed`` with a reason the agent can read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from overcooked_ai_py.mdp.actions import Action

from .env import START
from .kitchen import Kitchen
from .nav import Pos, facing, step_towards

MAX_WAIT = 20
MAX_STEPS = 40          # an action that has not finished after this many steps fails
START_VERBS = ("cut", "cook", "bake", "stir")


@dataclass
class Macro:
    name: str
    args: Tuple[str, ...] = ()
    n: int = 0

    def __str__(self):
        return f"{self.name}({', '.join(self.args) if self.name != 'wait' else self.n})"


def parse_macro(text: str) -> Tuple[Optional[Macro], str]:
    """Parse one action; returns (macro, "") or (None, reason)."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty action"
    s = text.strip().strip("`'\" ;.").replace("’", "'")
    m = re.fullmatch(r"([A-Za-z_]+)\s*\((.*)\)", s)
    if not m:
        return None, f"'{text}' is not of the form name(arguments)"
    name = m.group(1).lower()
    args = tuple(a.strip().strip("'\"").lower() for a in m.group(2).split(",") if a.strip())
    arity = {"pickup": 2, "put_obj_in_utensil": 1, "fill_dish_with_food": 1, "deliver_soup": 0,
             "place_obj_on_counter": 0, "wait": 1, **{v: 1 for v in START_VERBS}}
    if name not in arity:
        return None, f"unknown action '{name}'"
    if len(args) != arity[name]:
        return None, f"{name} takes {arity[name]} argument(s)"
    if name == "wait":
        if not re.fullmatch(r"\d+", args[0]):
            return None, "wait needs a whole number of steps"
        n = int(args[0])
        if not 1 <= n <= MAX_WAIT:
            return None, f"wait needs 1 to {MAX_WAIT} steps"
        return Macro("wait", (), n), ""
    return Macro(name, args), ""


class Executor:
    def __init__(self, kitchen: Kitchen):
        self.k = kitchen
        self.macro: Optional[Macro] = None
        self.steps = 0
        self.pending_interact = False
        self.rejected_reason = ""

    @property
    def busy(self) -> bool:
        return self.macro is not None

    # ------------------------------------------------------------------ #
    # Preconditions
    # ------------------------------------------------------------------ #
    def check(self, m: Macro, state) -> Optional[str]:
        """None if the action can start now, else the reason it cannot."""
        k, me = self.k, state.players[self.k.index]
        held = me.held_object
        if m.name == "wait":
            return None
        if m.name == "pickup":
            if held is not None:
                return f"your hands are full (you hold {held.name}); put it somewhere first"
            item, place = m.args
            if place == "ingredient_dispenser":
                if item not in k.raw:
                    return f"'{item}' is not a raw ingredient"
                return None if self._targets_of("I") else "you cannot reach the ingredient dispenser"
            if place == "dish_dispenser":
                if item != "dish":
                    return "the dish dispenser only gives 'dish'"
                return None if self._targets_of("D") else "you cannot reach the dish dispenser"
            if place == "counter":
                return None if self._counters_with(state, item) else f"no counter you can reach holds {item}"
            if place in k.utensils:
                if not k.mine(k.utensils[place]):
                    return f"you cannot reach {place}"
                status, product, _, _ = k.utensil_status(state, place)
                if status != "finished" or product != item:
                    return f"{place} has no finished {item} ({k.utensil_text(state, place)})"
                if k.need_dish(item):
                    return f"{item} must be served on a dish: use fill_dish_with_food({place})"
                return None
            return f"unknown place '{place}' (use ingredient_dispenser, dish_dispenser, counter or a utensil name)"
        if m.name in ("put_obj_in_utensil", "fill_dish_with_food") or m.name in START_VERBS:
            u = m.args[0]
            if u not in k.utensils:
                return f"unknown utensil '{u}' (utensils: {', '.join(k.utensils)})"
            if not k.mine(k.utensils[u]):
                return f"you cannot reach {u}"
            status, product, _, _ = k.utensil_status(state, u)
            if m.name == "put_obj_in_utensil":
                if held is None:
                    return "you hold nothing"
                if held.name == "dish":
                    return f"use fill_dish_with_food({u}) to collect food with a dish"
                if status not in ("empty", "loaded"):
                    return f"{u} cannot take items now ({k.utensil_text(state, u)})"
                return None
            if m.name == "fill_dish_with_food":
                if held is None or held.name != "dish":
                    return "you need to hold a dish"
                if status not in ("processing", "finished"):
                    return f"{u} has no food to collect ({k.utensil_text(state, u)})"
                return None
            if k.verb_of[u] != m.name:
                return f"{u} is started with {k.verb_of[u]}({u}), not {m.name}"
            if held is not None:
                return f"empty your hands before starting {u} (you hold {held.name})"
            if status != "loaded":
                return f"{u} cannot be started ({k.utensil_text(state, u)})"
            return None
        if m.name == "deliver_soup":
            if held is None:
                return "you hold nothing to deliver"
            return None if self._targets_of("S") else "you cannot reach the serving location"
        if m.name == "place_obj_on_counter":
            if held is None:
                return "you hold nothing"
            return None if self._free_counters(state) else "no free counter you can reach"
        return f"unknown action {m.name}"

    def options(self, state, ingredients: List[str]) -> List[str]:
        """Concrete actions that can start now (``ingredients``: raw ingredients worth listing)."""
        k = self.k
        cands = [f"pickup({i}, ingredient_dispenser)" for i in ingredients]
        cands.append("pickup(dish, dish_dispenser)")
        cands += sorted({f"pickup({state.objects[p].name}, counter)" for p in k.counters()
                         if p in state.objects and k.mine(p)})
        for u in k.utensils:
            status, product, _, _ = k.utensil_status(state, u)
            if status == "finished":
                cands.append(f"pickup({product}, {u})")
            cands += [f"put_obj_in_utensil({u})", f"fill_dish_with_food({u})", f"{k.verb_of[u]}({u})"]
        cands += ["deliver_soup()", "place_obj_on_counter()"]
        out = []
        for c in cands:
            m, _ = parse_macro(c)
            if m is not None and self.check(m, state) is None and c not in out:
                out.append(c)
        return out + [f"wait(1..{MAX_WAIT})"]

    # ------------------------------------------------------------------ #
    # Running an action
    # ------------------------------------------------------------------ #
    def begin(self, m: Macro) -> None:
        self.macro, self.steps, self.pending_interact, self.rejected_reason = m, 0, False, ""

    def note_rejected(self, reason: str) -> None:
        self.rejected_reason = reason

    def update(self, state) -> Optional[Tuple[str, str]]:
        """Call at the start of a step. Returns (status, message) if the action just ended."""
        if self.macro is None:
            return None
        m, me = self.macro, self.k.index
        if self.pending_interact:
            self.pending_interact = False
            done = state.ml_actions[me]
            if done is not None:
                return self._end("done", done)
            reason = self.rejected_reason or "the interaction had no effect"
            return self._end("failed", f"{m}: {reason}")
        if m.name == "wait":
            partner_did = state.ml_actions[self.k.other]
            if self.steps >= m.n:
                return self._end("done", f"waited {self.steps} step(s)")
            if self.steps > 0 and partner_did is not None:
                return self._end("done", f"waited {self.steps} step(s); your partner just did {partner_did}")
            return None
        if self.steps >= MAX_STEPS:
            return self._end("failed", f"{m}: not finished after {MAX_STEPS} steps (path blocked?)")
        reason = self.check(m, state)
        if reason:
            return self._end("failed", f"{m}: {reason}")
        return None

    def step(self, state) -> Tuple[Any, Any]:
        """Primitive (action, parameter) for the running action."""
        m = self.macro
        self.steps += 1
        if m is None or m.name == "wait":
            return Action.STAY, None
        targets, parm = self._targets(m, state)
        me = state.players[self.k.index]
        action = step_towards(self.k.floor, tuple(me.position), me.orientation,
                              tuple(state.players[self.k.other].position), targets)
        if action == Action.INTERACT:
            if m.name == "fill_dish_with_food":
                status, _, _, _ = self.k.utensil_status(state, m.args[0])
                if status != "finished":
                    return Action.STAY, None      # wait by the utensil until the food is ready
            self.pending_interact = True
            return action, parm
        return action, None

    def _end(self, status: str, message: str) -> Tuple[str, str]:
        self.macro, self.pending_interact, self.rejected_reason = None, False, ""
        return status, message

    # ------------------------------------------------------------------ #
    def _targets_of(self, tile: str) -> List[Pos]:
        return [p for p in self.k.cells.get(tile, []) if self.k.mine(p)]

    def _counters_with(self, state, item: str) -> List[Pos]:
        return [p for p in self.k.counters() if self.k.mine(p) and p in state.objects
                and state.objects[p].name == item]

    def _free_counters(self, state) -> List[Pos]:
        free = [p for p in self.k.counters() if self.k.mine(p) and p not in state.objects]
        shared = [p for p in free if self.k.theirs(p)]
        return shared or free

    def _targets(self, m: Macro, state) -> Tuple[List[Pos], Any]:
        if m.name == "pickup":
            item, place = m.args
            if place == "ingredient_dispenser":
                return self._targets_of("I"), item
            if place == "dish_dispenser":
                return self._targets_of("D"), None
            if place == "counter":
                return self._counters_with(state, item), None
            return [self.k.utensils[place]], None
        if m.name in START_VERBS:
            return [self.k.utensils[m.args[0]]], START
        if m.name in ("put_obj_in_utensil", "fill_dish_with_food"):
            return [self.k.utensils[m.args[0]]], None
        if m.name == "deliver_soup":
            return self._targets_of("S"), None
        if m.name == "place_obj_on_counter":
            return self._free_counters(state), None
        return [], None


def facing_cell(state, index: int) -> Pos:
    p = state.players[index]
    return facing(tuple(p.position), p.orientation)
