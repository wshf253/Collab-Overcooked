"""Simulated human partner with configurable behaviour traits.

The partner works on the open order through *jobs*, each tied to one stage of
the work:

=========  ===============================================================
stage      jobs
=========  ===============================================================
prep       bring raw ingredients from the dispenser; load and run the
           chopping board or blender
cook       load and run the pot or oven; move products between utensils
serve      fetch dishes, plate food, deliver
=========  ===============================================================

Traits (see :class:`Traits`) change how jobs are chosen and executed:

* **preference**: which stages it picks first when several jobs are open.
  With ``reluctance > 0`` it waits that many steps for a job of its favourite
  stage before taking another one.
* **speed**: ``step_period`` (it acts on every n-th step and stays otherwise)
  and ``decision_delay`` (idle steps before each new job).
* **reliability**: ``slip_rate`` is the chance that a raw-ingredient job takes a
  wrong ingredient and puts it in the utensil. When the utensil is started, the
  environment discards the wrong batch, and the work has to be redone.
* **commitment**: ``abandon_rate`` is the per-step chance of dropping the current
  job. A held item is put on the nearest free counter, and the job is not taken
  up again for ``abandon_cooldown`` steps.
* **order habit** (with several open orders): ``order_focus`` picks which open
  order's jobs come first: the oldest (``first``), the newest (``last``), or one
  for the dish it delivered last (``same_dish``, else the oldest).
  ``favourite_dishes`` puts orders for those dishes before all others. With
  ``order_commit`` it commits to one open order (chosen by that habit) and only
  works on its dish until it is delivered; after ``COMMIT_PATIENCE`` steps with
  nothing to do on it, it helps elsewhere until its order has work again.

With no job (or while waiting for a preferred one) the partner steps off cells in
front of a station, and after a blocked move it gives way at random (seeded).

Every step is logged with the true job and status (``trace``), and slips and
abandoned jobs are logged as events (``events``). This is ground truth for
evaluation only and must never be shown to the agent being tested.

The partner reads the full environment state. It acts with primitive actions,
using the same interaction rules as every other agent.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from overcooked_ai_py.mdp.actions import Action

from .env import START
from .view import Belief
from .nav import (Pos, Unstick, access_cells, facing, floor_cells, reach, step_aside, step_towards,
                  steps_to_face, touches)

STAGES = ("prep", "cook", "serve")
HELD_WAIT = 5      # steps to wait holding an item with nowhere to put it before setting it down
COMMIT_PATIENCE = 8  # order_commit: idle steps on the own order before helping with another
ORDER_FOCUS = ("first", "last", "same_dish")
UTENSIL_STAGE = {"chopping_board": "prep", "blender": "prep", "pot": "cook", "oven": "cook"}


@dataclass(frozen=True)
class Traits:
    preference: Tuple[str, ...] = ()   # stages, most preferred first; unlisted stages rank last
    reluctance: int = 0                # steps to wait for a job of preference[0] before another
    step_period: int = 1               # act on every n-th step (1 = full speed)
    decision_delay: int = 0            # idle steps before starting each new job
    slip_rate: float = 0.0             # chance a raw-ingredient job fetches a wrong ingredient
    abandon_rate: float = 0.0          # per-step chance of dropping the current job
    abandon_cooldown: int = 15         # steps before an abandoned (or left) job may be taken again
    yield_tasks: bool = False          # give way when the other player is heading for the same task
    avoid_stations: Tuple[str, ...] = ()   # utensil kinds it does not use (if the other player can)
    unfinished_rate: float = 0.0       # chance of walking away from starting a loaded utensil or
                                       # taking finished food out of one
    order_focus: str = "first"         # with several open orders: work on the oldest ("first"),
                                       # the newest ("last"), or the dish delivered last ("same_dish")
    favourite_dishes: Tuple[str, ...] = ()  # orders for these dishes first (listed order), then
                                            # order_focus decides
    order_commit: bool = False         # stick to one open order (by the habit above) until delivered

    def __post_init__(self):
        unknown = set(self.preference) - set(STAGES)
        if unknown or len(set(self.preference)) != len(self.preference):
            raise ValueError(f"preference must list distinct stages from {STAGES}: {self.preference}")
        if self.reluctance and not self.preference:
            raise ValueError("reluctance needs a preference")
        if self.step_period < 1 or self.decision_delay < 0 or self.reluctance < 0 \
                or self.abandon_cooldown < 0:
            raise ValueError("step_period >= 1; decision_delay, reluctance, abandon_cooldown >= 0")
        for name in ("slip_rate", "abandon_rate", "unfinished_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.order_focus not in ORDER_FOCUS:
            raise ValueError(f"order_focus must be one of {ORDER_FOCUS}")
        if len(set(self.favourite_dishes)) != len(self.favourite_dishes):
            raise ValueError("favourite_dishes must be distinct")
        unknown = set(self.avoid_stations) - set(UTENSIL_STAGE)
        if unknown:
            raise ValueError(f"avoid_stations must be utensil kinds from {sorted(UTENSIL_STAGE)}: {unknown}")


# Partner types, one trait changed at a time from "steady". All are stable: the same
# traits on every day.
PERSONAS: Dict[str, Traits] = {
    "steady": Traits(),
    "prep_first": Traits(preference=("prep", "cook", "serve"), reluctance=8),
    "serve_first": Traits(preference=("serve", "cook", "prep"), reluctance=8),
    "slow": Traits(step_period=2, decision_delay=2),
    # Gives way on tasks: if the other player is heading for the same job, it picks another.
    "yielding": Traits(yield_tasks=True),
    # Never uses the chopping board when the other player can reach one.
    "avoids_board": Traits(avoid_stations=("chopping_board",)),
    # Half the time walks away from starting a loaded utensil or from taking finished food
    # out, and leaves it for 15 steps: unfinished work stays visible.
    "leaves_unfinished": Traits(unfinished_rate=0.5),
    # Kept for later tests; replaced in the default lists because their traits were hard
    # to see (sloppy) or looked like normal hand-overs (flaky).
    "sloppy": Traits(slip_rate=0.3),
    "flaky": Traits(abandon_rate=0.08),
    # Order habits: only visible with several open orders. Each commits to one open order
    # and works on it until it is delivered (every type above does whatever is needed
    # next, oldest order first).
    "oldest_first": Traits(order_commit=True),
    "newest_first": Traits(order_focus="last", order_commit=True),
    # baked_carrot_soup: the one dish that needs the oven; it is in the default menu.
    "favourite_dish": Traits(favourite_dishes=("baked_carrot_soup",), order_commit=True),
    # After delivering a dish, takes an open order for the same dish if there is one.
    "repeats_dish": Traits(order_focus="same_dish", order_commit=True),
}

# The partner types experiments use by default.
DEFAULT_PERSONAS = ["steady", "prep_first", "serve_first", "slow", "yielding", "avoids_board",
                    "leaves_unfinished"]
# For runs with several open orders: the oldest-first baseline and the three order habits.
ORDER_PERSONAS = ["steady", "oldest_first", "newest_first", "favourite_dish", "repeats_dish"]


@dataclass(frozen=True)
class Job:
    kind: str                 # bring | start | clear | unload | plate | supply_dish | put_down
    item: str                 # item the job is about ("dish" for supply_dish)
    stage: str                # prep | cook | serve | other
    dest: Optional[str] = None    # bring: product the item goes into (None = deliver the order)
    pos: Optional[Pos] = None     # station, counter or dispenser cell the job starts from
    order: int = 0                # which open order (0 = oldest) the job serves; not part of the key

    @property
    def key(self) -> tuple:
        # A bring keeps its identity when the item moves (dispenser -> hand -> counter).
        if self.kind == "bring":
            return (self.kind, self.item, self.dest)
        return (self.kind, self.item, self.dest, self.pos)

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "item": self.item, "stage": self.stage, "dest": self.dest,
                "order": self.order}


class SimulatedHuman:
    """Scripted partner. Call ``reset`` once per episode, then ``act`` every step."""

    def __init__(self, traits: Traits = Traits(), name: str = "custom"):
        self.traits, self.name = traits, name

    @classmethod
    def persona(cls, name: str) -> "SimulatedHuman":
        return cls(PERSONAS[name], name)

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, mdp, index: int, seed: int, view_radius: Optional[int] = None) -> None:
        """``view_radius``: play from a partial view (see coop.view); None = full view."""
        self.mdp, self.index = mdp, index
        self.belief = Belief(mdp, index, view_radius) if view_radius is not None else None
        # A string seed is hashed with SHA-512 by `random`: reproducible across runs.
        self.rng = random.Random(f"{seed}:{index}:{self.name}")
        # Separate stream for giving way, so slips and abandons do not depend on traffic.
        self.unstick = Unstick(random.Random(f"{seed}:{index}:{self.name}:nav"))
        terrain = mdp.terrain_mtx
        self.terrain = terrain
        self.floor = floor_cells(terrain)
        self.access = access_cells(terrain, self.floor)
        self.cells: Dict[str, List[Pos]] = {}
        for y, row in enumerate(terrain):
            for x, c in enumerate(row):
                self.cells.setdefault(c, []).append((x, y))
        self.recipes: Dict[str, Tuple[str, Counter]] = {}
        for kind, products in mdp.recipes.items():
            if kind == "none":
                continue
            for product, spec in products.items():
                self.recipes[product] = (kind, Counter(spec["recipe"]))
        self.stations = {kind: self.cells.get(spec["symbol"], [])
                         for kind, spec in mdp.utensils.items() if kind != "none"}
        self.kind_at = {p: kind for kind, ps in self.stations.items() for p in ps}
        self.raw = list(mdp.default_ingredients)
        self.job: Optional[Job] = None
        self.job_t = 0
        self.delay_left = 0
        self.waited = 0
        self.cooldown: Dict[tuple, int] = {}
        self.slip: Optional[str] = None       # wrong ingredient carried for the current job
        self.slip_decided = False
        self.trace: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self._prev_other: Optional[Pos] = None
        self._last_other: Optional[Pos] = None
        self.explore_target: Optional[Pos] = None
        self.held_wait = 0
        self.last_dish: Optional[str] = None     # dish this player delivered last (same_dish)
        self.committed: Optional[str] = None     # order_commit: dish of the order it works on
        self.commit_waited = 0
        self._held_name: Optional[str] = None

    def act(self, state, t: int) -> Tuple[Any, Any]:
        """Return (primitive action, interaction parameter) for this step."""
        if self.belief is not None:
            state = self.belief.update(state, t)
        self._look(state)
        if t > 0 and (state.ml_actions[self.index] or "").startswith("deliver"):
            self.last_dish = self._held_name
        self._held_name = self.me.held_object.name if self.me.held_object is not None else None
        jobs = self._jobs()
        if self.job is not None:
            current = next((j for j in jobs if j.key == self.job.key), None)
            if current is None:
                self._end_job()       # finished, or no longer possible
            else:
                self.job = current    # same job; the item may have moved
        if (self.job is not None and self.me.held_object is None
                and self.cooldown.get(self.job.key, -1) > t):
            self._end_job()           # set down on purpose (see _carry); come back to it later
        if (self.job is not None and self.traits.yield_tasks and self.me.held_object is None
                and self._claimed(self.job)):
            self.events.append({"t": t, "type": "yield", "job": self.job.describe()})
            self._end_job()           # the other player is going for it: leave it to them

        if t % self.traits.step_period != 0:
            return self._log(t, "off_beat")

        if (self.job is not None and self.job.kind != "put_down" and t > self.job_t
                and self.traits.abandon_rate and self.rng.random() < self.traits.abandon_rate):
            self.events.append({"t": t, "type": "abandon", "job": self.job.describe(),
                                "holding": self.me.held_object.name if self.me.held_object else None})
            self.cooldown[self.job.key] = t + self.traits.abandon_cooldown
            self._end_job()

        if self.job is None:
            job, why = self._choose(jobs, t)
            if job is None:
                if self.belief is not None and why == "no_job":
                    # Partial view, nothing known to do: go and look at the station seen longest ago.
                    return self._log(t, "looking", self._explore())
                # Idle: step out from in front of a station so the other player can use it.
                aside = step_aside(self.floor, self.access, tuple(self.me.position), tuple(self.other.position))
                return self._log(t, why, aside)
            self.job, self.job_t, self.delay_left = job, t, self.traits.decision_delay
        if self.delay_left > 0:
            self.delay_left -= 1
            return self._log(t, "deciding")
        action, parm = self._execute(t)
        action = self.unstick.filter(self.floor, tuple(self.me.position), action, tuple(self.other.position))
        return self._log(t, "act", action, parm)

    def summary(self) -> Dict[str, Any]:
        status = Counter(r["status"] for r in self.trace)
        acting = [r for r in self.trace if r["status"] == "act" and r["job"]]
        stages = Counter(r["job"]["stage"] for r in acting)
        n = len(self.trace) or 1
        return {
            "name": self.name,
            "traits": asdict(self.traits),
            "steps": len(self.trace),
            "status_share": {k: v / n for k, v in sorted(status.items())},
            "stage_share_when_acting": {k: v / (len(acting) or 1) for k, v in sorted(stages.items())},
            "slips": sum(e["type"] == "slip" for e in self.events),
            "abandons": sum(e["type"] == "abandon" for e in self.events),
            "yields": sum(e["type"] == "yield" for e in self.events),
            "left_unfinished": sum(e["type"] == "left_unfinished" for e in self.events),
            "give_ways": self.unstick.give_ways,
        }

    # ------------------------------------------------------------------ #
    # Choosing jobs
    # ------------------------------------------------------------------ #
    def _end_job(self):
        self.job, self.slip, self.slip_decided, self.delay_left = None, None, False, 0
        self.held_wait = 0

    def _rank(self, job: Job) -> Tuple[int, int, int, int]:
        pref = self.traits.preference
        stage = pref.index(job.stage) if job.stage in pref else len(pref)
        orders = self.state.order_list or []
        dish = orders[job.order] if job.order < len(orders) else None
        fav = self.traits.favourite_dishes
        favourite = fav.index(dish) if dish in fav else len(fav)
        focus = self.traits.order_focus
        same = 0 if focus != "same_dish" or dish == self.last_dish else 1
        return stage, favourite, same, (-job.order if focus == "last" else job.order)

    def _explore(self):
        """Walk towards the station seen longest ago, keeping that target until it is in view
        (re-choosing every step makes a player turn back and forth on the spot)."""
        stations = [p for p in list(self.kind_at) + [q for t in "IDS" for q in self.cells.get(t, [])]
                    if self._mine(p)]
        if not stations:
            return Action.STAY
        if self.explore_target is None or self.state.visible(self.explore_target):
            unseen = [p for p in stations if not self.state.visible(p)]
            if not unseen:
                self.explore_target = None
                return step_aside(self.floor, self.access, tuple(self.me.position), tuple(self.other.position))
            self.explore_target = min(unseen, key=lambda p: (self.state.last_seen(p) is not None,
                                                             self.state.last_seen(p) or 0))
        action = step_towards(self.floor, self.me.position, self.me.orientation, self.other.position,
                              [self.explore_target])
        return Action.STAY if action == Action.INTERACT else action

    def _choose(self, jobs: List[Job], t: int) -> Tuple[Optional[Job], str]:
        open_jobs = [j for j in jobs if self.cooldown.get(j.key, -1) <= t]
        held = self.me.held_object
        if held is not None:
            # Hands are full: carry on with what is held, or put it down.
            use = [j for j in open_jobs if self._uses(j, held)]
            if use:
                return min(use, key=self._rank), ""
            return Job("put_down", held.name, "other"), ""
        if not open_jobs:
            return None, "no_job"
        if self.traits.order_commit:
            target = self._commit_target()
            own = [j for j in open_jobs if target in self._job_dishes.get(j.key, ())]
            if own:
                open_jobs, self.commit_waited = own, 0
            elif self.commit_waited < COMMIT_PATIENCE:
                self.commit_waited += 1
                return None, "waiting_on_order"
            # else: patience used up, help with another order until its own has work again
        if self.traits.yield_tasks:
            free = [j for j in open_jobs if not self._claimed(j)]
            if not free:
                return None, "yielding"
            if min(open_jobs, key=self._rank) not in free:
                self.events.append({"t": t, "type": "yield", "job": min(open_jobs, key=self._rank).describe()})
            open_jobs = free
        best = min(open_jobs, key=self._rank)   # min keeps enumeration order on ties
        while self.traits.unfinished_rate and self._finishing(best):
            if self.rng.random() >= self.traits.unfinished_rate:
                break
            # Walk away from it; someone else may finish it (or this partner, later).
            self.events.append({"t": t, "type": "left_unfinished", "job": best.describe()})
            self.cooldown[best.key] = t + self.traits.abandon_cooldown
            open_jobs = [j for j in open_jobs if j.key != best.key]
            if not open_jobs:
                return None, "left_unfinished"
            best = min(open_jobs, key=self._rank)
        pref = self.traits.preference
        if pref and best.stage != pref[0] and self.waited < self.traits.reluctance:
            self.waited += 1
            return None, "reluctant"
        self.waited = 0
        return best, ""

    def _dish(self, job: Job) -> Optional[str]:
        orders = self.state.order_list or []
        return orders[job.order] if job.order < len(orders) else None

    def _commit_target(self) -> Optional[str]:
        """The dish of the order this player is committed to; a new one is picked by its
        habit once the old one is no longer open."""
        n = max(1, getattr(self.mdp, "num_open_orders", 1))
        open_dishes = list(self.state.order_list[:n]) if self.state.order_list else []
        if self.committed not in open_dishes:
            fav, focus = self.traits.favourite_dishes, self.traits.order_focus

            def key(i):
                d = open_dishes[i]
                return (fav.index(d) if d in fav else len(fav),
                        0 if focus != "same_dish" or d == self.last_dish else 1,
                        -i if focus == "last" else i)
            self.committed = open_dishes[min(range(len(open_dishes)), key=key)] if open_dishes else None
            self.commit_waited = 0
        return self.committed

    def _finishing(self, job: Job) -> bool:
        """Jobs that finish a step someone started: starting a loaded utensil, or taking
        finished food out of one."""
        return job.kind == "start" or (job.kind == "bring" and job.pos in self.kind_at)

    def _job_targets(self, job: Job) -> List[Pos]:
        if job.kind == "bring":
            return self.cells.get("I", []) if job.pos is None else [job.pos]
        if job.kind == "supply_dish":
            dishes = [p for p, o in self.objects.items() if o.name == "dish"]
            return dishes or self.cells.get("D", [])
        return [job.pos] if job.pos is not None else []

    def _claimed(self, job: Job) -> bool:
        """Does the other player look like they are going for this job?

        True if they stand facing its station, or if they are closer to it than me
        and got closer since the last step. Only public positions are used.
        """
        targets = self._job_targets(job)
        if not targets or self.other.held_object is not None and job.kind != "plate":
            return False
        other = tuple(self.other.position)
        if facing(other, self.other.orientation) in targets:
            return True
        d_other = steps_to_face(self.floor, other, targets)
        d_me = steps_to_face(self.floor, tuple(self.me.position), targets)
        prev = self._prev_other
        if d_other is None or prev is None or d_me is None:
            return False
        d_prev = steps_to_face(self.floor, prev, targets)
        return d_other < d_me and d_prev is not None and d_other < d_prev

    @staticmethod
    def _uses(job: Job, held) -> bool:
        if held.name == "dish":
            return job.kind in ("plate", "supply_dish")
        return job.kind == "bring" and job.item == held.name

    # ------------------------------------------------------------------ #
    # What the open order still needs (from the full state)
    # ------------------------------------------------------------------ #
    def _look(self, state):
        self.state = state
        self.objects = state.objects
        self.me = state.players[self.index]
        self.other = state.players[1 - self.index]
        self._prev_other = self._last_other
        self._last_other = tuple(self.other.position)
        self.my_reach = reach(self.floor, self.me.position)
        self.other_reach = reach(self.floor, self.other.position)

    def _mine(self, pos: Pos) -> bool:
        return touches(pos, self.my_reach)

    def _theirs(self, pos: Pos) -> bool:
        return touches(pos, self.other_reach)

    def _need_dish(self, item: str) -> bool:
        return bool(self.mdp.need_dish.get(item, 0))

    def _stage_of(self, kind: str) -> str:
        return UTENSIL_STAGE.get(kind, "cook")

    def _jobs(self) -> List[Job]:
        orders = self.state.order_list
        if not orders:
            return []
        out, keys = [], set()
        self._job_dishes: Dict[tuple, set] = {}     # job key -> dishes of the open orders it serves
        for i, order in enumerate(orders[:max(1, getattr(self.mdp, "num_open_orders", 1))]):
            for job in self._needs(order, None, frozenset()):
                self._job_dishes.setdefault(job.key, set()).add(order)
                if job.key not in keys:
                    keys.add(job.key)
                    out.append(replace(job, order=i) if i else job)
        if self.traits.avoid_stations:
            # Soft: a station kind is avoided only if the other player can use one of them.
            avoided = {k for k in self.traits.avoid_stations
                       if any(self._theirs(p) for p in self.stations.get(k, []))}
            out = [j for j in out if not (self._station_kinds(j) & avoided)]
        return out

    def _station_kinds(self, job: Job) -> set:
        kinds = set()
        if job.pos is not None and job.pos in self.kind_at:
            kinds.add(self.kind_at[job.pos])
        if job.kind == "bring" and job.dest is not None:
            kinds.add(self.recipes[job.dest][0])
        return kinds

    def _bring(self, item: str, dest: Optional[str], pos: Optional[Pos]) -> Job:
        if dest is None:
            stage = "serve"
        elif item not in self.recipes:
            stage = "prep"
        else:
            stage = self._stage_of(self.recipes[dest][0])
        return Job("bring", item, stage, dest, pos)

    def _select_station(self, product: str) -> Tuple[Optional[Pos], str]:
        """Station for ``product``: one already processing it, else a compatible partial
        batch, else an empty one (each group in my-reachable-first order)."""
        kind, recipe = self.recipes[product]
        busy, partial, empty = [], [], []
        for p in self.stations.get(kind, []):
            obj = self.objects.get(p)
            if obj is None:
                empty.append(p)
            elif isinstance(obj.state[0], list):
                if not (Counter(obj.state[0]) - recipe):
                    partial.append(p)
            elif obj.state[0] == product:
                busy.append(p)
        for group, status in ((busy, "busy"), (partial, "open"), (empty, "open")):
            if group:
                return sorted(group, key=lambda p: not self._mine(p))[0], status
        return None, ""

    def _dish_available(self) -> bool:
        held = self.me.held_object
        if held is not None and held.name == "dish":
            return True
        dishes = [p for p, o in self.objects.items() if o.name == "dish"]
        return any(self._mine(p) for p in dishes) or any(self._mine(p) for p in self.cells.get("D", []))

    def _dish_job(self, item: str, pos: Pos) -> List[Job]:
        """A dish for ``item`` at station ``pos`` (cooking or finished)."""
        other_held = self.other.held_object
        if other_held is not None and other_held.name == "dish":
            return []       # the other player already has a dish for it
        if self._mine(pos):
            return [Job("plate", item, "serve", None, pos)] if self._dish_available() else []
        # I cannot reach the food: hand a dish over unless one is already on its way.
        if any(o.name == "dish" and self._theirs(p) for p, o in self.objects.items()):
            return []
        return [Job("supply_dish", "dish", "serve", None, pos)] if self._dish_available() else []

    def _can_place(self, dest: Optional[str]) -> bool:
        """Can I put an item for ``dest`` where it goes myself?"""
        if dest is None:
            return any(self._mine(p) for p in self.cells.get("S", []))
        station, status = self._select_station(dest)
        return station is not None and status == "open" and self._mine(station)

    def _needs(self, item: str, dest: Optional[str], seen: frozenset) -> List[Job]:
        if item in seen:
            return []
        seen = seen | {item}
        held, other_held = self.me.held_object, self.other.held_object
        if held is not None and held.name == item:
            return [self._bring(item, dest, None)]
        if other_held is not None and other_held.name == item:
            return []           # the partner is carrying it
        existing = [(p, o) for p, o in self.objects.items() if o.name == item]
        if existing:
            out: List[Job] = []
            for p, o in existing:
                on_counter = self.terrain[p[1]][p[0]] == "X"
                if dest is None and self._need_dish(item) and o.catagory != "with_dish":
                    if not on_counter:      # unplated food left on a counter cannot be plated
                        out += self._dish_job(item, p)
                    continue
                if not self._mine(p):
                    continue
                if on_counter and self._theirs(p) and not self._can_place(dest):
                    continue    # already handed over; the partner's side moves it on
                out.append(self._bring(item, dest, p))
            return out
        if item not in self.recipes:        # raw ingredient
            if any(self._mine(p) for p in self.cells.get("I", [])):
                return [self._bring(item, dest, None)]
            return []
        kind, recipe = self.recipes[item]
        station, status = self._select_station(item)
        if station is None:
            # Every station for this product is in use. Free one if possible, and meanwhile
            # prepare inputs that are made on other stations (bringing an input itself would
            # leave it nowhere to go). Without this, two open orders can wait on each other's
            # utensils forever.
            out = self._free_station_jobs(kind)
            for child in dict.fromkeys(recipe.elements()):
                if child in self.recipes and not self._exists(child):
                    out += self._needs(child, item, seen)
            return out
        if status == "busy":
            if dest is None and self._need_dish(item):
                return self._dish_job(item, station)
            return []
        obj = self.objects.get(station)
        contents = Counter(obj.state[0]) if obj is not None else Counter()
        missing = recipe - contents
        if not missing:
            return [Job("start", item, self._stage_of(kind), None, station)] if self._mine(station) else []
        out = []
        for child in recipe:                # recipe order, each missing input once
            if missing[child] > 0:
                out += self._needs(child, item, seen)
        return out

    def _exists(self, item: str) -> bool:
        """Is ``item`` anywhere: on a counter or utensil, being made, or in someone's hands?"""
        if any(p.held_object is not None and p.held_object.name == item for p in (self.me, self.other)):
            return True
        for o in self.objects.values():
            if o.name == item or (o.state is not None and o.state[0] == item):
                return True
        return False

    def _needed_items(self) -> set:
        """Every item in the recipe trees of the open orders."""
        out: set = set()

        def walk(item):
            if item in out:
                return
            out.add(item)
            if item in self.recipes:
                for child in self.recipes[item][1]:
                    walk(child)
        for order in self.state.order_list[:max(1, getattr(self.mdp, "num_open_orders", 1))]:
            walk(order)
        return out

    def _free_station_jobs(self, kind: str) -> List[Job]:
        """Every station of ``kind`` is blocked: clear a wrong batch or unload finished food."""
        out = []
        recipes = [r for _, (k, r) in self.recipes.items() if k == kind]
        for p in self.stations.get(kind, []):
            obj = self.objects.get(p)
            if obj is None or not self._mine(p):
                continue
            if isinstance(obj.state[0], list) and obj.state[2] == 0:
                contents = Counter(obj.state[0])
                if not any(not (contents - r) for r in recipes):
                    # Starting a batch that matches no recipe makes the environment discard it.
                    out.append(Job("clear", str(obj.state[0]), self._stage_of(kind), None, p))
            elif obj.catagory == "finished" and obj.name not in self._needed_items():
                # Only food no open order needs (with several open orders, food blocking
                # this order's station may be another order's dish).
                out.append(Job("unload", obj.name, self._stage_of(kind), None, p))
        return out

    # ------------------------------------------------------------------ #
    # Executing the current job with primitive actions
    # ------------------------------------------------------------------ #
    def _execute(self, t: int) -> Tuple[Any, Any]:
        job, held = self.job, self.me.held_object
        if job.kind == "put_down":
            return self._go(self._free_counters(shared=False)), None
        if job.kind == "bring":
            if held is not None and held.name in (job.item, self.slip):
                return self._carry(job), None
            if held is not None:
                return self._go(self._free_counters(shared=False)), None
            if job.pos is None:
                action = self._go(self.cells.get("I", []))
                return action, (self._ingredient(job, t) if action == Action.INTERACT else None)
            return self._go([job.pos]), None
        if job.kind in ("start", "clear", "unload"):
            if held is not None:
                return self._go(self._free_counters(shared=False)), None
            action = self._go([job.pos])
            return action, (START if action == Action.INTERACT and job.kind != "unload" else None)
        if job.kind in ("plate", "supply_dish"):
            if held is not None and held.name == "dish":
                if job.kind == "supply_dish":
                    return self._handoff(), None
                obj = self.objects.get(job.pos)
                action = self._go([job.pos])
                if action == Action.INTERACT and (obj is None or obj.catagory != "finished"):
                    return Action.STAY, None    # wait next to the utensil until the food is ready
                return action, None
            if held is not None:
                return self._go(self._free_counters(shared=False)), None
            dishes = [p for p, o in self.objects.items() if o.name == "dish" and self._mine(p)]
            return self._go(dishes or self.cells.get("D", [])), None
        raise ValueError(f"unknown job kind {job.kind}")

    def _ingredient(self, job: Job, t: int) -> str:
        """The raw ingredient taken at the dispenser; decides a slip once per job."""
        if not self.slip_decided:
            self.slip_decided = True
            wrong = self._wrong_ingredients(job)
            if wrong and self.traits.slip_rate and self.rng.random() < self.traits.slip_rate:
                self.slip = self.rng.choice(wrong)
                self.events.append({"t": t, "type": "slip", "job": job.describe(), "took": self.slip})
        return self.slip or job.item

    def _wrong_ingredients(self, job: Job) -> List[str]:
        """Ingredients that would spoil the batch at the destination (no recipe accepts it).
        Slips only happen when I put the item into the utensil myself."""
        if job.dest is None or not self._can_place(job.dest):
            return []
        kind, _ = self.recipes[job.dest]
        station, _ = self._select_station(job.dest)
        obj = self.objects.get(station)
        contents = Counter(obj.state[0]) if obj is not None else Counter()
        recipes = [r for _, (k, r) in self.recipes.items() if k == kind]
        return [y for y in self.raw if y != job.item
                and not any(not ((contents + Counter([y])) - r) for r in recipes)]

    def _carry(self, job: Job):
        if job.dest is None:
            serve = [p for p in self.cells.get("S", []) if self._mine(p)]
            return self._go(serve) if serve else self._handoff()
        station, status = self._select_station(job.dest)
        if station is None or status != "open":
            # Nowhere to put it yet. After a few steps, set it down so the hands are free
            # (two players waiting with full hands for one busy utensil would never plate).
            self.held_wait += 1
            if self.held_wait > HELD_WAIT and self._free_counters(shared=False):
                self.cooldown[job.key] = self.state.timestep + HELD_WAIT
                return self._go(self._free_counters(shared=False))
            return Action.STAY
        self.held_wait = 0
        return self._go([station]) if self._mine(station) else self._handoff()

    def _free_counters(self, shared: bool) -> List[Pos]:
        return [p for p in self.cells.get("X", []) if p not in self.objects and self._mine(p)
                and (not shared or self._theirs(p))]

    def _handoff(self):
        return self._go(self._free_counters(shared=True) or self._free_counters(shared=False))

    def _go(self, targets: List[Pos]):
        """One primitive step towards facing the nearest target; INTERACT when facing it.

        If every target is cut off (the other player stands in the only way), step
        out from in front of any station with probability 1/2 instead of freezing there.
        """
        action = step_towards(self.floor, self.me.position, self.me.orientation,
                              self.other.position, targets)
        if action == Action.STAY and targets and self.unstick.rng.random() < 0.5:
            # Random, so two players cutting each other off do not step aside (and back) in step.
            return step_aside(self.floor, self.access, tuple(self.me.position), tuple(self.other.position))
        return action

    def _log(self, t: int, status: str, action: Any = Action.STAY, parm: Any = None):
        if status != "act":
            action = self.unstick.filter(self.floor, tuple(self.me.position), action,
                                         tuple(self.other.position))
        self.trace.append({"t": t, "status": status,
                           "job": self.job.describe() if self.job else None,
                           "dish": self._dish(self.job) if self.job else None,   # dish of the job's order
                           "committed": self.committed,                         # order_commit only
                           "action": action if isinstance(action, str) else list(action),
                           "parameter": parm})
        return action, parm
