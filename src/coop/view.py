"""Partial observability: what one player sees and remembers.

A player sees every cell within ``radius`` (Chebyshev distance, as in OvercookedV2's
``agent_view_size``) of its own position. For cells outside the view it remembers
what it saw last, with the time step. The other player is seen only inside the
view; otherwise the last place and held item seen are remembered.

``Belief.update(state, t)`` returns a :class:`ViewState`. It has the attributes the
agents read from the real state (``objects``, ``players``, ``order_list``,
``ml_actions``, ``error_message``, ``timestep``), with hidden information replaced
by memory. Orders stay public (an order display everyone can read). The map
itself is public.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .nav import Pos


def chebyshev(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class SeenPlayer:
    """The other player as last seen (position, orientation, held item)."""

    def __init__(self, position, orientation, held_object, seen_t: Optional[int], visible: bool):
        self.position, self.orientation, self.held_object = tuple(position), tuple(orientation), held_object
        self.seen_t, self.visible = seen_t, visible

    @property
    def pos_and_or(self):
        return self.position, self.orientation

    def has_object(self):
        return self.held_object is not None


class ViewState:
    def __init__(self, objects, players, order_list, ml_actions, error_message, timestep,
                 seen_t: Dict[Pos, Optional[int]], radius: int, me: Pos):
        self.objects, self.players, self.order_list = objects, players, order_list
        self.ml_actions, self.error_message, self.timestep = ml_actions, error_message, timestep
        self.seen_t, self.radius, self._me = seen_t, radius, me

    def visible(self, pos: Pos) -> bool:
        return chebyshev(pos, self._me) <= self.radius

    def last_seen(self, pos: Pos) -> Optional[int]:
        """Time step this cell was last seen (now if visible), None if never."""
        return self.seen_t.get(tuple(pos))


def _projected(mdp, obj, elapsed: int):
    if elapsed <= 0 or obj.state is None or isinstance(obj.state[0], list) or obj.catagory != "semi-finished":
        return obj
    name, n, cooked = obj.state
    total = int(mdp.Recipe.recipe_list[name]["cook_time"])
    if cooked <= 0:
        return obj
    out = obj.deepcopy()
    done = min(total, cooked + elapsed)
    out.state = (name, n, done)
    if done >= total:
        out.name, out.catagory = name, "finished"
    return out


class Belief:
    def __init__(self, mdp, index: int, radius: int):
        if radius < 0:
            raise ValueError("view radius must be >= 0")
        self.index, self.other, self.radius = index, 1 - index, radius
        terrain = mdp.terrain_mtx
        # Cells that can hold an object: counters and stations (everything but floor).
        self.cells = [(x, y) for y, row in enumerate(terrain) for x, c in enumerate(row) if c != " "]
        self.utensil_pos = {u: tuple(mdp.from_utensil_name_to_pos(u)) for u in mdp.utensil_list}
        self.mdp = mdp
        self.memory: Dict[Pos, Any] = {}               # cell -> last seen object (copy)
        self.memory_t: Dict[Pos, int] = {}             # cell -> when that object was seen
        self.seen_t: Dict[Pos, Optional[int]] = {p: None for p in self.cells}
        self.other_seen: Optional[Tuple[Pos, Any, Any, int]] = None   # position, orientation, held, t
        self.other_start: Optional[Tuple[Pos, Any]] = None              # public start position

    def _project(self, obj, elapsed: int):
        return _projected(self.mdp, obj, elapsed)

    def update(self, state, t: int) -> ViewState:
        me = state.players[self.index]
        here = tuple(me.position)
        vis = lambda p: chebyshev(p, here) <= self.radius
        for p in self.cells:
            if vis(p):
                self.seen_t[p] = t
                obj = state.objects.get(p)
                if obj is None:
                    self.memory.pop(p, None)
                else:
                    self.memory[p] = obj.deepcopy()      # a copy: hidden cooking must not leak
                    self.memory_t[p] = t
        # Out of view, a utensil seen running is assumed to keep running by the known
        # cook times (public rules), so a remembered batch "finishes" when it should.
        objects = {p: self._project(o, t - self.memory_t.get(p, t)) for p, o in self.memory.items()}
        for p, o in state.objects.items():           # visible cells: the real objects
            if vis(p):
                objects[p] = o
        other = state.players[self.other]
        if self.other_start is None:
            self.other_start = (tuple(other.position), tuple(other.orientation))
        if vis(tuple(other.position)):
            held = other.held_object.deepcopy() if other.held_object else None
            self.other_seen = (tuple(other.position), tuple(other.orientation), held, t)
            seen_other = SeenPlayer(other.position, other.orientation, other.held_object, t, True)
        elif self.other_seen is not None:
            pos, ori, held, st = self.other_seen
            seen_other = SeenPlayer(pos, ori, held, st, False)
        else:
            # Never seen yet: only the start position is known (the map shows it).
            pos, ori = self.other_start
            seen_other = SeenPlayer(pos, ori, None, None, False)
        players: List[Any] = [None, None]
        players[self.index], players[self.other] = me, seen_other
        ml = list(state.ml_actions)
        if not seen_other.visible:
            ml[self.other] = None                    # actions out of view are not seen
        errors = []
        for msg in state.error_message or []:
            m = re.search(r"items in (\w+) not matching", msg)
            if m is None or m.group(1) not in self.utensil_pos or vis(self.utensil_pos[m.group(1)]):
                errors.append(msg)
        return ViewState(objects, players, state.order_list, ml, errors, state.timestep,
                         dict(self.seen_t), self.radius, here)
