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

Every step is logged with the true job and status (``trace``), and slips and
abandoned jobs are logged as events (``events``). This is ground truth for
evaluation only and must never be shown to the agent being tested.

The partner reads the full environment state. It acts with primitive actions,
using the same interaction rules as every other agent.
"""
from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from overcooked_ai_py.mdp.actions import Action, Direction

from .env import START

STAGES = ("prep", "cook", "serve")
UTENSIL_STAGE = {"chopping_board": "prep", "blender": "prep", "pot": "cook", "oven": "cook"}
DIRECTIONS = (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)
Pos = Tuple[int, int]


@dataclass(frozen=True)
class Traits:
    preference: Tuple[str, ...] = ()   # stages, most preferred first; unlisted stages rank last
    reluctance: int = 0                # steps to wait for a job of preference[0] before another
    step_period: int = 1               # act on every n-th step (1 = full speed)
    decision_delay: int = 0            # idle steps before starting each new job
    slip_rate: float = 0.0             # chance a raw-ingredient job fetches a wrong ingredient
    abandon_rate: float = 0.0          # per-step chance of dropping the current job
    abandon_cooldown: int = 15         # steps before an abandoned job may be taken again

    def __post_init__(self):
        unknown = set(self.preference) - set(STAGES)
        if unknown or len(set(self.preference)) != len(self.preference):
            raise ValueError(f"preference must list distinct stages from {STAGES}: {self.preference}")
        if self.reluctance and not self.preference:
            raise ValueError("reluctance needs a preference")
        if self.step_period < 1 or self.decision_delay < 0 or self.reluctance < 0 \
                or self.abandon_cooldown < 0:
            raise ValueError("step_period >= 1; decision_delay, reluctance, abandon_cooldown >= 0")
        for name in ("slip_rate", "abandon_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


# Starting set of partner types, one trait changed at a time from "steady".
PERSONAS: Dict[str, Traits] = {
    "steady": Traits(),
    "prep_first": Traits(preference=("prep", "cook", "serve"), reluctance=8),
    "serve_first": Traits(preference=("serve", "cook", "prep"), reluctance=8),
    "slow": Traits(step_period=2, decision_delay=2),
    "sloppy": Traits(slip_rate=0.3),
    "flaky": Traits(abandon_rate=0.08),
}


@dataclass(frozen=True)
class Job:
    kind: str                 # bring | start | clear | unload | plate | supply_dish | put_down
    item: str                 # item the job is about ("dish" for supply_dish)
    stage: str                # prep | cook | serve | other
    dest: Optional[str] = None    # bring: product the item goes into (None = deliver the order)
    pos: Optional[Pos] = None     # station, counter or dispenser cell the job starts from

    @property
    def key(self) -> tuple:
        # A bring keeps its identity when the item moves (dispenser -> hand -> counter).
        if self.kind == "bring":
            return (self.kind, self.item, self.dest)
        return (self.kind, self.item, self.dest, self.pos)

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "item": self.item, "stage": self.stage, "dest": self.dest}


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
    def reset(self, mdp, index: int, seed: int) -> None:
        self.mdp, self.index = mdp, index
        # A string seed is hashed with SHA-512 by `random`: reproducible across runs.
        self.rng = random.Random(f"{seed}:{index}:{self.name}")
        terrain = mdp.terrain_mtx
        self.terrain = terrain
        self.floor = {(x, y) for y, row in enumerate(terrain) for x, c in enumerate(row) if c == " "}
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

    def act(self, state, t: int) -> Tuple[Any, Any]:
        """Return (primitive action, interaction parameter) for this step."""
        self._look(state)
        jobs = self._jobs()
        if self.job is not None:
            current = next((j for j in jobs if j.key == self.job.key), None)
            if current is None:
                self._end_job()       # finished, or no longer possible
            else:
                self.job = current    # same job; the item may have moved

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
                return self._log(t, why)
            self.job, self.job_t, self.delay_left = job, t, self.traits.decision_delay
        if self.delay_left > 0:
            self.delay_left -= 1
            return self._log(t, "deciding")
        action, parm = self._execute(t)
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
        }

    # ------------------------------------------------------------------ #
    # Choosing jobs
    # ------------------------------------------------------------------ #
    def _end_job(self):
        self.job, self.slip, self.slip_decided, self.delay_left = None, None, False, 0

    def _rank(self, job: Job) -> int:
        pref = self.traits.preference
        return pref.index(job.stage) if job.stage in pref else len(pref)

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
        best = min(open_jobs, key=self._rank)   # min keeps enumeration order on ties
        pref = self.traits.preference
        if pref and best.stage != pref[0] and self.waited < self.traits.reluctance:
            self.waited += 1
            return None, "reluctant"
        self.waited = 0
        return best, ""

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
        self.my_reach = self._reach(self.me.position)
        self.other_reach = self._reach(self.other.position)

    def _reach(self, start: Pos) -> set:
        seen, queue = {start}, deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS:
                p = (x + dx, y + dy)
                if p in self.floor and p not in seen:
                    seen.add(p)
                    queue.append(p)
        return seen

    @staticmethod
    def _touches(pos: Pos, reach: set) -> bool:
        return any((pos[0] + dx, pos[1] + dy) in reach for dx, dy in DIRECTIONS)

    def _mine(self, pos: Pos) -> bool:
        return self._touches(pos, self.my_reach)

    def _theirs(self, pos: Pos) -> bool:
        return self._touches(pos, self.other_reach)

    def _need_dish(self, item: str) -> bool:
        return bool(self.mdp.need_dish.get(item, 0))

    def _stage_of(self, kind: str) -> str:
        return UTENSIL_STAGE.get(kind, "cook")

    def _jobs(self) -> List[Job]:
        orders = self.state.order_list
        if not orders:
            return []
        out, keys = [], set()
        for job in self._needs(orders[0], None, frozenset()):
            if job.key not in keys:
                keys.add(job.key)
                out.append(job)
        return out

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
        if self._mine(pos):
            return [Job("plate", item, "serve", None, pos)] if self._dish_available() else []
        # I cannot reach the food: hand a dish over unless one is already on its way.
        other_held = self.other.held_object
        if other_held is not None and other_held.name == "dish":
            return []
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
            return self._free_station_jobs(kind)
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
            elif obj.catagory == "finished":
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
            return Action.STAY          # nowhere to put it yet
        return self._go([station]) if self._mine(station) else self._handoff()

    def _free_counters(self, shared: bool) -> List[Pos]:
        return [p for p in self.cells.get("X", []) if p not in self.objects and self._mine(p)
                and (not shared or self._theirs(p))]

    def _handoff(self):
        return self._go(self._free_counters(shared=True) or self._free_counters(shared=False))

    def _go(self, targets: List[Pos]):
        """One primitive step towards facing the nearest target; INTERACT when facing it."""
        targets = list(targets)
        if not targets:
            return Action.STAY
        start, blocked = self.me.position, self.other.position
        queue, seen = deque([(start, None)]), {start}
        while queue:
            pos, first = queue.popleft()
            for tgt in targets:
                d = (tgt[0] - pos[0], tgt[1] - pos[1])
                if d in DIRECTIONS:
                    if first is not None:
                        return first
                    return Action.INTERACT if self.me.orientation == d else d
            for d in DIRECTIONS:
                p = (pos[0] + d[0], pos[1] + d[1])
                if p in self.floor and p != blocked and p not in seen:
                    seen.add(p)
                    queue.append((p, first or d))
        return Action.STAY

    def _log(self, t: int, status: str, action: Any = Action.STAY, parm: Any = None):
        self.trace.append({"t": t, "status": status,
                           "job": self.job.describe() if self.job else None,
                           "action": action if isinstance(action, str) else list(action),
                           "parameter": parm})
        return action, parm
