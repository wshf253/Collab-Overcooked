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


def access_cells(terrain, floor: Set[Pos]) -> Set[Pos]:
    """Floor cells in front of a station (anything that is not floor or a plain counter)."""
    out = set()
    for (x, y) in floor:
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < len(terrain) and 0 <= nx < len(terrain[ny]) and terrain[ny][nx] not in (" ", "X"):
                out.add((x, y))
    return out


def step_aside(floor: Set[Pos], access: Set[Pos], position: Pos, blocked: Optional[Pos]):
    """Idle players step off cells in front of a station, so they never block one.

    Returns a move towards the nearest cell that is not in front of a station,
    or STAY if already on one (or none can be reached).
    """
    if position not in access:
        return Action.STAY
    queue, seen = deque([(position, None)]), {position}
    while queue:
        pos, first = queue.popleft()
        if pos not in access:
            return first
        for d in DIRECTIONS:
            p = (pos[0] + d[0], pos[1] + d[1])
            if p in floor and p != blocked and p not in seen:
                seen.add(p)
                queue.append((p, first or d))
    return Action.STAY


def steps_to_face(floor: Set[Pos], start: Pos, targets: Iterable[Pos]) -> Optional[int]:
    """Moves from ``start`` until standing next to any target (0 if already there), or None."""
    targets = list(targets)
    if not targets:
        return None
    goal = {(t[0] + dx, t[1] + dy) for t in targets for dx, dy in DIRECTIONS} & floor
    if not goal:
        return None
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        pos, d = queue.popleft()
        if pos in goal:
            return d
        for dx, dy in DIRECTIONS:
            p = (pos[0] + dx, pos[1] + dy)
            if p in floor and p not in seen:
                seen.add(p)
                queue.append((p, d + 1))
    return None


def facing(position: Pos, orientation) -> Pos:
    return (position[0] + orientation[0], position[1] + orientation[1])


class Unstick:
    """Random give-way after a blocked move (seeded, so runs stay reproducible).

    Two players that follow deterministic rules can step into the same cell at
    the same time forever (the env bounces both back). After a move that should
    have changed the position but did not, the player gives way with probability
    1/2: it steps to a random free neighbouring cell or stays this step.
    Turning on the spot (moving towards a station) is not a blocked move.
    """

    def __init__(self, rng):
        self.rng = rng
        self.prev: Optional[Tuple[Pos, bool]] = None    # (position, tried to move onto floor)
        self.give_ways = 0

    def filter(self, floor: Set[Pos], position: Pos, action, blocked: Optional[Pos]):
        stuck = self.prev is not None and self.prev[1] and self.prev[0] == position
        if stuck and action in DIRECTIONS and self.rng.random() < 0.5:
            free = [d for d in DIRECTIONS if (position[0] + d[0], position[1] + d[1]) in floor
                    and (position[0] + d[0], position[1] + d[1]) != blocked]
            action = self.rng.choice(free + [Action.STAY])
            self.give_ways += 1
        moves = action in DIRECTIONS and (position[0] + action[0], position[1] + action[1]) in floor
        self.prev = (position, moves)
        return action
