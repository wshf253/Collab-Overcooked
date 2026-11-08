"""Grid navigation shared by the simulated partner and the LLM agent's executor."""
from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Set, Tuple

from overcooked_ai_py.mdp.actions import Action, Direction

DIRECTIONS = (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)
Pos = Tuple[int, int]


def floor_cells(terrain) -> Set[Pos]:
    return {(x, y) for y, row in enumerate(terrain) for x, c in enumerate(row) if c == " "}


def reach(floor: Set[Pos], start: Pos) -> Set[Pos]:
    """Floor cells reachable from ``start`` (the other player is not an obstacle here)."""
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            p = (x + dx, y + dy)
            if p in floor and p not in seen:
                seen.add(p)
                queue.append(p)
    return seen


def touches(pos: Pos, region: Set[Pos]) -> bool:
    """Can a player standing somewhere in ``region`` face ``pos``?"""
    return any((pos[0] + dx, pos[1] + dy) in region for dx, dy in DIRECTIONS)


def step_towards(floor: Set[Pos], position: Pos, orientation, blocked: Optional[Pos],
                 targets: Iterable[Pos]):
    """One primitive action towards facing the nearest target; INTERACT when facing one.

    Returns Action.STAY when no target can be reached (e.g. the other player blocks
    the only path).
    """
    targets: List[Pos] = list(targets)
    if not targets:
        return Action.STAY
    queue, seen = deque([(position, None)]), {position}
    while queue:
        pos, first = queue.popleft()
        for tgt in targets:
            d = (tgt[0] - pos[0], tgt[1] - pos[1])
            if d in DIRECTIONS:
                if first is not None:
                    return first
                return Action.INTERACT if orientation == d else d
        for d in DIRECTIONS:
            p = (pos[0] + d[0], pos[1] + d[1])
            if p in floor and p != blocked and p not in seen:
                seen.add(p)
                queue.append((p, first or d))
    return Action.STAY


def facing(position: Pos, orientation) -> Pos:
    return (position[0] + orientation[0], position[1] + orientation[1])
