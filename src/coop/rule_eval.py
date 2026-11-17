"""Rule predicates: did the partner follow its persona rules, and did the ego respect them?

Everything is computed from what happened in the kitchen, step by step:

* **Events.** After each step, each player's completed action (``ml_actions``) becomes an
  ``Event``: verb (pickup / put / start / fill / deliver / place), target (utensil,
  dispenser, counter, serving), item, product, the menu dishes it belongs to, and whether
  it was a *choice* (taken with empty hands: the start of a new piece of work).
* **Tracker.** Shared state built from the events: who loaded or started each utensil,
  who put each counter item down, where held items came from, how long a complete batch
  or finished food has waited, each player's current order, new orders. The same tracker
  is used by the evaluator (full state), by the rule-following scripted players
  (``coop.rule_partner``) and by the LLM partner's reminders (its own view).
* **Predicates** (one class per library rule, ``CHECKS``):
  ``judge`` returns judgments on this step's events: ``partner`` = adherent (did the
  partner follow the rule), ``ego`` = aligned (did the ego respect it); a step without
  an applicable event is no opportunity, so a rule with no opportunity is undefined,
  not perfect. ``opportunity`` says whether the rule applies now, from the partner's
  view, with the reminder line for the LLM partner.

``RuleEvaluator`` runs the tracker and the predicates over an episode and counts
opportunities and successes per rule and side. Definitions: docs/coop-setting.md,
"Persona rules".
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .rules import Rule, counter_groups, render

Pos = Tuple[int, int]
FALLBACK_WAIT = 20        # split_dish / serves(ego): finish yourself after this many steps
SUPPLY_WAIT = 15          # wants_supplies: fetch it yourself after this many steps
TAKEOVER_WINDOW = 30      # on_takeover: the reaction must come within this many steps
COMMIT_WAIT = 5           # waits_for_commit
CONTENTION_WINDOW = 15    # on_contention: give up resolving after this many steps
VERBS = {"cook": "start", "cut": "start", "bake": "start", "stir": "start"}


@dataclass
class Event:
    t: int
    who: str                      # ego | partner
    verb: str                     # pickup | put | start | fill | deliver | place
    target: str                   # utensil name | ingredient_dispenser | dish_dispenser | counter | serving
    pos: Optional[Pos]
    item: Optional[str]           # what was picked / put / started / plated / delivered
    dishes: Tuple[str, ...]       # menu dishes the item belongs to
    choice: bool                  # taken with empty hands
    source: Optional[str] = None  # pickup: where from (dispenser / counter / utensil name)

    @property
    def utensil(self) -> Optional[str]:
        return self.target if self.target not in ("ingredient_dispenser", "dish_dispenser", "counter",
                                                  "serving") else None

    def on(self, dish: str) -> bool:
        return dish in self.dishes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Judgment:
    t: int
    rule: str
    side: str          # partner (adherent) | ego (aligned)
    value: bool
    note: str = ""
    seen: bool = True  # partner side: the partner was within the ego's view (exposure)


class KitchenInfo:
    """Menu-level facts: which dishes each item belongs to, finishing products."""

    def __init__(self, kitchen, menu):
        self.k = kitchen
        self.menu = list(menu)
        self.dishes_of: Dict[str, set] = {}
        for d in self.menu:
            for item in self._tree(d):
                self.dishes_of.setdefault(item, set()).add(d)

    def _tree(self, item: str) -> List[str]:
        out = [item]
        if item in self.k.recipes:
            for child in self.k.recipes[item][1]:
                out += self._tree(child)
        return out

    def dishes(self, item: Optional[str]) -> Tuple[str, ...]:
        return tuple(sorted(self.dishes_of.get(item or "", ())))

    def kind(self, utensil: str) -> str:
        return self.k.kind_of[utensil]

    def product_of(self, kind: str, contents: List[str]) -> Optional[str]:
        c = Counter(contents)
        for p, (k, inputs) in self.k.recipes.items():
            if k == kind and Counter(inputs) == c:
                return p
        return None

    def missing(self, kind: str, contents: List[str], open_dishes: List[str]) -> List[str]:
        """Raw items still needed to complete a partial batch (recipes in the open orders'
        trees first, else any recipe of the kind)."""
        c = Counter(contents)
        wanted = {i for d in open_dishes for i in self._tree(d)}
        cands = [(p, Counter(inputs)) for p, (k, inputs) in self.k.recipes.items()
                 if k == kind and not (c - Counter(inputs)) and Counter(inputs) != c]
        cands.sort(key=lambda x: x[0] not in wanted)
        if not cands:
            return []
        return sorted((cands[0][1] - c).elements())

    def loadable(self, state, utensil: str, item: Optional[str]) -> bool:
        """Could ``item`` go into ``utensil`` now (empty, or a partial batch it fits)?"""
        status, _, contents, _ = self.k.utensil_status(state, utensil)
        if status == "empty":
            return True
        if status != "loaded" or item is None:
            return False
        c = Counter(contents) + Counter([item])
        kind = self.kind(utensil)
        return any(k == kind and not (c - Counter(inputs)) for _, (k, inputs) in self.k.recipes.items())


def facing(player) -> Pos:
    return (player.position[0] + player.orientation[0], player.position[1] + player.orientation[1])


def open_orders(state, n_open: int) -> List[str]:
    return list(state.order_list[:max(1, n_open)]) if state.order_list else []


def parse_events(t: int, pre, post, info: KitchenInfo, roles: Dict[int, str]) -> List[Event]:
    """Events completed during step t (``post.ml_actions`` against the state before it)."""
    k = info.k
    out = []
    for i, done in enumerate(post.ml_actions or []):
        if not done or i not in roles:
            continue
        m = re.fullmatch(r"(\w+)\((.*)\)", done.replace(" ", ""))
        if not m:
            continue
        name, args = m.group(1), [a for a in m.group(2).split(",") if a]
        p = pre.players[i]
        held = p.held_object.name if p.held_object is not None else None
        cell = facing(p)
        sym = k.terrain[cell[1]][cell[0]] if 0 <= cell[1] < len(k.terrain) and 0 <= cell[0] < len(k.terrain[0]) else ""
        target = k.utensil_at.get(cell) or {"I": "ingredient_dispenser", "D": "dish_dispenser", "S": "serving",
                                            "X": "counter"}.get(sym, "")
        verb = VERBS.get(name, name)
        item: Optional[str] = None
        source = None
        if name == "pickup":
            item, source = (args + [None, None])[:2]
            verb = "pickup"
        elif name == "put_obj_in_utensil":
            verb, item = "put", held
        elif verb == "start":
            u = args[0] if args else target
            status, product, contents, _ = k.utensil_status(pre, u) if u in k.utensils else ("", None, [], 0)
            item = info.product_of(k.kind_of.get(u, ""), contents) if status == "loaded" else product
        elif name == "fill_dish_with_food":
            verb = "fill"
            u = args[0] if args else target
            _, product, _, _ = k.utensil_status(pre, u) if u in k.utensils else ("", None, [], 0)
            item = product
        elif name == "deliver_soup":
            verb, item = "deliver", held
        elif name == "place_obj_on_counter":
            verb, item = "place", held
        else:
            continue
        out.append(Event(t, roles[i], verb, target, cell, item, info.dishes(item), held is None, source))
    return out


def serve_stage(ev: Event, info: KitchenInfo) -> bool:
    if ev.verb in ("fill", "deliver"):
        return True
    if ev.verb == "pickup" and ev.item == "dish":
        return True
    return ev.verb == "pickup" and ev.utensil is not None and ev.item in info.menu


def serve_instance(ev: Event, info: KitchenInfo) -> bool:
    """One serving step: plating (filling a dish, or taking a dish-less dish out) or delivering."""
    return ev.verb in ("fill", "deliver") or (ev.verb == "pickup" and ev.utensil is not None
                                               and ev.item in info.menu)


def finishing(ev: Event, dish: str) -> bool:
    """The last steps of ``dish``: starting its final utensil, plating it, delivering it."""
    if ev.item != dish:
        return False
    return ev.verb in ("start", "fill", "deliver") or (ev.verb == "pickup" and ev.utensil is not None)


# ---------------------------------------------------------------------- #
# Tracker
# ---------------------------------------------------------------------- #
class Tracker:
    """Shared kitchen memory built from events (see module doc). ``roles`` maps player
    index to "ego" / "partner"."""

    def __init__(self, kitchen, menu, roles: Dict[int, str], n_open: int = 1):
        self.info = KitchenInfo(kitchen, menu)
        self.k = kitchen
        self.roles = roles
        self.index = {v: i for i, v in roles.items()}
        self.n_open = n_open
        self.loaders: Dict[str, List[Tuple[str, str]]] = {}     # utensil -> [(who, item)] since emptied
        self.starter: Dict[str, str] = {}
        self.counter_by: Dict[Pos, str] = {}                   # counter cell -> who put the item there
        self.held_from: Dict[str, Optional[str]] = {"ego": None, "partner": None}
        self.complete_since: Dict[str, int] = {}               # complete, unstarted batch
        self.ready_since: Dict[str, int] = {}                  # finished food in the utensil
        self.order_of: Dict[str, Optional[str]] = {"ego": None, "partner": None}
        self.last_choice: Dict[str, int] = {"ego": -10 ** 6, "partner": -10 ** 6}
        self.new_orders: List[Dict[str, Any]] = []             # {"t", "dish", "ego_empty"}
        self.takeovers: List[Dict[str, Any]] = []              # ego took over partner items: {"t", "dishes"}
        self.events: List[Event] = []
        self.t = -1
        self.started = False

    # -- queries ----------------------------------------------------------
    def owner(self, utensil: str) -> Optional[str]:
        if self.loaders.get(utensil):
            return self.loaders[utensil][0][0]
        return self.starter.get(utensil)

    def pending(self, state) -> Dict[str, Tuple[List[str], int]]:
        """Partner batches still incomplete: utensil -> (missing raw items, since)."""
        out = {}
        dishes = open_orders(state, self.n_open)
        for u in self.k.utensils:
            if self.owner(u) != "partner":
                continue
            status, _, contents, _ = self.k.utensil_status(state, u)
            if status != "loaded":
                continue
            miss = self.info.missing(self.k.kind_of[u], contents, dishes)
            if miss:
                first = min((t for t in [self._first_load_t.get(u)] if t is not None), default=self.t)
                out[u] = (miss, first)
        return out

    def waited(self, since: Dict[str, int], u: str, t: int) -> int:
        return t - since[u] if u in since else 0

    # -- update -----------------------------------------------------------
    def start(self, state) -> None:
        """Call with the first state of the episode (the starting orders count as new)."""
        self._first_load_t: Dict[str, int] = {}
        for d in open_orders(state, self.n_open):
            self.new_orders.append({"t": 0, "dish": d, "ego_empty": True, "ego_done": False,
                                    "partner_done": False, "partner_free": True})
        self.started = True

    def observe(self, t: int, pre, post) -> List[Event]:
        if not self.started:
            self.start(pre)
        events = parse_events(t, pre, post, self.info, self.roles)
        self.t = t
        for ev in events:
            self._apply(ev, pre)
        self._refresh(t + 1, pre, post)
        self.events.extend(events)
        return events

    def _apply(self, ev: Event, pre) -> None:
        u = ev.utensil
        if ev.verb == "put" and u:
            self.loaders.setdefault(u, []).append((ev.who, ev.item or ""))
            self._first_load_t.setdefault(u, ev.t)
        elif ev.verb == "start" and u:
            self.starter[u] = ev.who
        elif ev.verb == "place" and ev.pos is not None:
            self.counter_by[ev.pos] = ev.who
        elif ev.verb == "pickup":
            if ev.source == "counter" and ev.pos is not None:
                self.held_from[ev.who] = self.counter_by.pop(ev.pos, None)
            elif ev.source in ("ingredient_dispenser", "dish_dispenser"):
                self.held_from[ev.who] = "dispenser"
            else:
                self.held_from[ev.who] = "utensil"
        if ev.choice and ev.item != "dish":
            self.last_choice[ev.who] = ev.t
            self._mark_commit(ev)
            matched = {o for o in open_orders(pre, self.n_open) if o in ev.dishes}
            if len(matched) == 1:
                self.order_of[ev.who] = matched.pop()

    def _mark_commit(self, ev: Event) -> None:
        if ev.who == "ego":
            for o in self.new_orders:
                if o["t"] <= ev.t and not o.get("ego_t"):
                    o["ego_t"] = ev.t

    def _refresh(self, t: int, pre, post) -> None:
        for u in self.k.utensils:
            status, _, _, _ = self.k.utensil_status(post, u)
            if status == "empty":
                self.loaders.pop(u, None)
                self.starter.pop(u, None)
                self._first_load_t.pop(u, None)
            if status == "loaded" and self.info.product_of(self.k.kind_of[u], self.k.utensil_status(post, u)[2]):
                self.complete_since.setdefault(u, t)
            else:
                self.complete_since.pop(u, None)
            if status == "finished":
                self.ready_since.setdefault(u, t)
            else:
                self.ready_since.pop(u, None)
        for who, i in self.index.items():
            if post.players[i].held_object is None:
                self.held_from[who] = None
        before, after = open_orders(pre, self.n_open), open_orders(post, self.n_open)
        if after != before:
            fresh = list(after)
            for d in before:
                if d in fresh:
                    fresh.remove(d)
            ego = self.index.get("ego")
            empty = ego is None or post.players[ego].held_object is None
            # waits_for_commit is about choosing new work: the partner was free if it held
            # nothing and had no open order of its own (not in the middle of something)
            par = self.index.get("partner")
            free = par is None or (post.players[par].held_object is None
                                   and self.order_of.get("partner") not in after)
            for d in fresh:
                self.new_orders.append({"t": t, "dish": d, "ego_empty": empty, "ego_done": False,
                                        "partner_done": False, "partner_free": free})
        self.new_orders = [o for o in self.new_orders if t - o["t"] <= 60]
        self.takeovers = [x for x in self.takeovers if t - x["t"] <= TAKEOVER_WINDOW]


# ---------------------------------------------------------------------- #
# Jobs from one player's perspective (what is needed now), for "was there work for X"
# ---------------------------------------------------------------------- #
class JobView:
    def __init__(self, mdp, index: int):
        from .partner import SimulatedHuman, Traits
        self.h = SimulatedHuman(Traits(), "jobview")
        self.h.reset(mdp, index, 0)
        self._cache: Tuple[Any, list] = (None, [])

    def jobs(self, state) -> List[Tuple[Any, set]]:
        if self._cache[0] is not state:
            self.h._look(state)
            js = self.h._jobs()
            self._cache = (state, [(j, set(self.h._job_dishes.get(j.key, ()))) for j in js])
        return self._cache[1]

    def has(self, state, pred) -> bool:
        return any(pred(j, ds) for j, ds in self.jobs(state))


def job_finishing(job, dishes: set, dish: str) -> bool:
    if dish not in dishes:
        return False
    if job.kind == "start" and job.item == dish:
        return True
    if job.kind in ("plate", "supply_dish") and job.item == dish:
        return True
    return job.kind == "bring" and job.item == dish and job.dest is None


def partner_allows(job, dishes: set, rules: List[Rule], tracker: "Tracker", now: int) -> bool:
    """Would the partner's own rules let it take this job now? (split_dish leaves the
    finishing steps; serves(who=ego) leaves serving; each until its fallback.) Used by the
    twin to choose and by the predicates to ask "did the partner have work on X"."""
    for r in rules:
        if r.id == "split_dish" and job_finishing(job, dishes, r.p["dish"]):
            since = tracker.complete_since if job.kind == "start" else tracker.ready_since
            u = tracker.k.utensil_at.get(tuple(job.pos)) if job.pos is not None else None
            if not (u in since and now - since[u] >= FALLBACK_WAIT):
                return False
        if r.id == "serves" and r.p["who"] == "ego" and job.stage == "serve":
            if not any(now - x >= FALLBACK_WAIT for x in tracker.ready_since.values()):
                return False
    return True


def earlier(rules: List[Rule], first: str, second: str) -> bool:
    """Does a rule with id ``first`` come before one with id ``second`` in the card?"""
    ids = [r.id for r in rules]
    return first in ids and second in ids and ids.index(first) < ids.index(second)


def takeover_overridden(rules: List[Rule], dish: str, committed: Optional[str]) -> bool:
    """on_takeover gives way to an earlier order_priority when the taken dish is the
    partner's committed order (rules apply in card order)."""
    return earlier(rules, "order_priority", "on_takeover") and committed is not None and dish == committed


# ---------------------------------------------------------------------- #
# Predicates
# ---------------------------------------------------------------------- #
class Ctx:
    """What a predicate may look at for one step."""

    def __init__(self, t, pre, post, events, tracker: Tracker, jobs: Dict[str, JobView], intents=None,
                 rules: Optional[List[Rule]] = None):
        self.t, self.pre, self.post, self.events = t, pre, post, events
        self.tr, self.info, self.k = tracker, tracker.info, tracker.k
        self.jobs = jobs
        self.intents = intents or {}
        self.rules = rules or []
        self.open = open_orders(pre, tracker.n_open)

    def has_job(self, who: str, pred) -> bool:
        """Was there a job for ``who`` matching ``pred`` (for the partner: one its own rules allow)?"""
        jv = self.jobs.get(who)
        if jv is None:
            return True
        if who == "partner":
            return jv.has(self.pre, lambda j, ds: pred(j, ds) and partner_allows(j, ds, self.rules, self.tr, self.t))
        return jv.has(self.pre, pred)


class Check:
    def __init__(self, rule: Rule, library, kitchen):
        self.rule, self.p, self.lib, self.k = rule, rule.p, library, kitchen
        self.label = rule.label()

    def J(self, ctx, side, value, note="") -> Judgment:
        return Judgment(ctx.t, self.label, side, bool(value), note)

    def judge(self, ctx: Ctx) -> List[Judgment]:
        return []

    def opportunity(self, view, tracker: Tracker, me: int, target: Optional[str] = None) -> Optional[str]:
        """Reminder line if the rule applies now (``view``: the partner's view; ``me``: its index)."""
        return None

    def reminder(self, kitchen, **extra) -> str:
        return render(self.rule, "reminder", self.lib, kitchen, {k: str(v) for k, v in extra.items()})


class OwnStation(Check):
    def __init__(self, *a):
        super().__init__(*a)
        self.s = self.p["station"]
        self.kind = self.s.rstrip("0123456789")
        self.others = [u for u in self.k.utensils if self.k.kind_of[u] == self.kind and u != self.s]

    def judge(self, ctx):
        out = []
        for ev in ctx.events:
            u = ev.utensil
            if not u or ctx.k.kind_of.get(u) != self.kind:
                continue
            if ev.who == "partner" and ev.verb == "put":
                ok = u == self.s or not ctx.info.loadable(ctx.pre, self.s, ev.item)
                out.append(self.J(ctx, "partner", ok, f"loaded {u}"))
            elif ev.who == "ego" and ev.verb in ("put", "start", "fill", "pickup"):
                if u != self.s:
                    out.append(self.J(ctx, "ego", True, f"{ev.verb} at {u}"))
                elif not (ev.verb == "put" and not any(ctx.info.loadable(ctx.pre, o, ev.item) for o in self.others)):
                    out.append(self.J(ctx, "ego", False, f"{ev.verb} at {u}"))
        return out

    def opportunity(self, view, tracker, me, target=None):
        held = view.players[me].held_object
        info = tracker.info
        goes_here = held is not None and any(k == self.kind and held.name in inputs
                                             for _, (k, inputs) in info.k.recipes.items())
        status = info.k.utensil_status(view, self.s)[0]
        if goes_here or (held is None and status == "finished"):
            return self.reminder(info.k)
        return None


class SoloDish(Check):
    def judge(self, ctx):
        d = self.p["dish"]
        if d not in ctx.open:
            return []
        out = []
        for ev in ctx.events:
            if ev.item == "dish":
                continue
            if ev.who == "partner" and ev.choice and ev.verb in ("pickup", "start"):
                if ctx.has_job("partner", lambda j, ds: d in ds):
                    out.append(self.J(ctx, "partner", ev.on(d), f"{ev.verb} {ev.item}"))
            elif ev.who == "ego":
                if ev.on(d):
                    out.append(self.J(ctx, "ego", False, f"touched {d}: {ev.verb} {ev.item}"))
                elif ev.choice:
                    out.append(self.J(ctx, "ego", True, f"{ev.verb} {ev.item}"))
        return out

    def opportunity(self, view, tracker, me, target=None):
        d = self.p["dish"]
        if d in open_orders(view, tracker.n_open) and view.players[me].held_object is None \
                and tracker.order_of.get("partner") != d:
            return self.reminder(tracker.k)
        return None


class SplitDish(Check):
    """Each finishing step of the dish (starting its final utensil, plating; delivery is not
    judged separately). partner: adherent if the ego did it, or the partner only after the
    fallback wait. What the partner works on otherwise is order choice, not this rule."""

    def judge(self, ctx):
        d = self.p["dish"]
        out = []
        for ev in ctx.events:
            if finishing(ev, d):
                out.append(self.J(ctx, "ego", ev.who == "ego", f"{ev.verb} {d} by {ev.who}"))
                if ev.verb == "deliver":
                    continue
                if ev.who == "ego":
                    out.append(self.J(ctx, "partner", True, f"{ev.verb} {d} left to the partner"))
                else:
                    since = ctx.tr.complete_since if ev.verb == "start" else ctx.tr.ready_since
                    waited = ctx.tr.waited(since, ev.utensil, ctx.t) if ev.utensil else 0
                    out.append(self.J(ctx, "partner", waited >= FALLBACK_WAIT, f"{ev.verb} {d} after {waited}"))
        return out

    def opportunity(self, view, tracker, me, target=None):
        d = self.p["dish"]
        k = tracker.k
        finish_ready = any(k.utensil_status(view, u)[0] == "finished" and k.utensil_status(view, u)[1] == d
                           for u in k.utensils) or any(
            k.utensil_status(view, u)[0] == "loaded"
            and tracker.info.product_of(k.kind_of[u], k.utensil_status(view, u)[2]) == d for u in k.utensils)
        return self.reminder(k) if finish_ready else None


class OrderPriority(Check):
    """The partner commits to an order and stays with it until it is served. A *pick* is
    a choice event on a different order than the player's current one.

    partner: when its committed order is gone, the next pick must be the {which} open
    order (adherent); a pick of another order while the committed one is open counts only
    if that order still had work for the partner (the fallback allows helping otherwise).
    Picks of a dish that an earlier rule of the card claims (solo_dish / split_dish) are
    not judged: rules apply in card order.
    ego: a pick is aligned if it is not the order the partner is committed to.
    """

    RECENT = 10     # a pick may follow a decision made a few steps earlier

    def __init__(self, *a):
        super().__init__(*a)
        self.committed: Optional[str] = None
        self.targets: List[Tuple[int, str]] = []

    def _claimed_earlier(self, ctx, dish) -> bool:
        for r in ctx.rules:
            if r == self.rule:
                return False
            if r.id in ("solo_dish", "split_dish") and r.p["dish"] == dish:
                return True
        return False

    def judge(self, ctx):
        if len(ctx.open) < 2:
            return []
        which = self.p["which"]
        target = ctx.open[0] if which == "oldest" else ctx.open[-1]
        self.targets = [(t, d) for t, d in self.targets if ctx.t - t <= self.RECENT] + [(ctx.t, target)]
        recent = {d for _, d in self.targets if d in ctx.open}
        if self.committed not in ctx.open:
            self.committed = None
        out = []
        for ev in ctx.events:
            if not ev.choice or ev.item == "dish":
                continue
            picked = [o for o in ctx.open if o in ev.dishes]
            if len(set(picked)) != 1 or picked[0] == ctx.tr.order_of.get(ev.who):
                continue
            dish = picked[0]
            if ev.who == "partner":
                if self._claimed_earlier(ctx, dish):
                    continue
                if self.committed is None:
                    if ctx.has_job("partner", lambda j, ds: bool(ds & recent)):
                        out.append(self.J(ctx, "partner", dish in recent, f"picked {dish} (target {target})"))
                    self.committed = dish
                elif dish != self.committed:
                    c = self.committed
                    taken = any(c in x["dishes"] for x in ctx.tr.takeovers)
                    if taken and earlier(ctx.rules, "on_takeover", "order_priority"):
                        self.committed = dish
                        continue
                    out.append(self.J(ctx, "partner", not ctx.has_job("partner", lambda j, ds: c in ds),
                                      f"left {c} for {dish}"))
            else:
                theirs = self.committed or target
                if ctx.has_job("ego", lambda j, ds: bool(ds - {theirs})):
                    out.append(self.J(ctx, "ego", dish != theirs, f"picked {dish} (partner's {theirs})"))
        return out

    def opportunity(self, view, tracker, me, target=None):
        opened = open_orders(view, tracker.n_open)
        if view.players[me].held_object is None and len(opened) >= 2 and tracker.order_of.get("partner") not in opened:
            return self.reminder(tracker.k)
        return None


class Serves(Check):
    def judge(self, ctx):
        who = self.p["who"]
        out = []
        for ev in ctx.events:
            if serve_instance(ev, ctx.info):
                want = "partner" if who == "self" else "ego"
                out.append(self.J(ctx, "ego", ev.who == want, f"{ev.verb} {ev.item} by {ev.who}"))
            if ev.who == "partner" and ev.choice and ev.verb in ("pickup", "start") and \
                    ctx.has_job("partner", lambda j, ds: j.stage == "serve"):
                if who == "self":
                    # "you always plate and deliver": judged while food is ready, not while it cooks
                    if self._ready(ctx):
                        out.append(self.J(ctx, "partner", serve_stage(ev, ctx.info), f"{ev.verb} {ev.item}"))
                else:
                    late = any(ctx.t - s >= FALLBACK_WAIT for s in ctx.tr.ready_since.values())
                    out.append(self.J(ctx, "partner", (not serve_stage(ev, ctx.info)) or late,
                                      f"{ev.verb} {ev.item}"))
        return out

    def _ready(self, ctx) -> bool:
        split = {r.p["dish"] for r in ctx.rules if r.id == "split_dish"}
        for u in ctx.k.utensils:
            status, product, _, _ = ctx.k.utensil_status(ctx.pre, u)
            if status == "finished" and product in ctx.info.menu and product not in split:
                return True
        return False

    def opportunity(self, view, tracker, me, target=None):
        k = tracker.k
        held = view.players[me].held_object
        if (held is None or held.name == "dish") and any(
                k.utensil_status(view, u)[0] == "finished" and k.utensil_status(view, u)[1] in tracker.info.menu
                for u in k.utensils):
            return self.reminder(k)
        return None


class WantsSupplies(Check):
    def judge(self, ctx):
        out = []
        pend = ctx.tr.pending(ctx.pre)
        for ev in ctx.events:
            if ev.who == "partner" and ev.verb == "pickup" and ev.source in ("ingredient_dispenser", "counter"):
                for u, (miss, since) in pend.items():
                    if ev.item in miss:
                        ok = ev.source == "counter" or ctx.t - since >= SUPPLY_WAIT
                        out.append(self.J(ctx, "partner", ok, f"took {ev.item} from {ev.source}"))
                        break
            if ev.verb == "put" and ev.utensil in pend and ev.item in pend[ev.utensil][0]:
                by_ego = ev.who == "ego" or ctx.tr.held_from.get(ev.who) == "ego"
                out.append(self.J(ctx, "ego", by_ego, f"{ev.item} into {ev.utensil} supplied by "
                                  f"{'ego' if by_ego else 'partner'}"))
        return out

    def opportunity(self, view, tracker, me, target=None):
        pend = tracker.pending(view)
        if pend:
            u, (miss, _) = next(iter(pend.items()))
            return self.reminder(tracker.k, station=u, missing=", ".join(miss))
        return None


class HandoverSpot(Check):
    def __init__(self, *a):
        super().__init__(*a)
        self.cells = set(counter_groups(self.k).get(self.p["group"], []))

    def judge(self, ctx):
        out = []
        free = [c for c in self.cells if c not in ctx.pre.objects]
        for ev in ctx.events:
            if ev.verb != "place":
                continue
            if ev.who == "partner":
                out.append(self.J(ctx, "partner", ev.pos in self.cells or not free, f"placed at {ev.pos}"))
            elif free:
                out.append(self.J(ctx, "ego", ev.pos in self.cells, f"placed at {ev.pos}"))
        return out

    def opportunity(self, view, tracker, me, target=None):
        held = view.players[me].held_object
        if held is None or held.name in tracker.info.menu or held.name == "dish":
            return None
        info = tracker.info
        if any(info.loadable(view, u, held.name) and any(held.name in inputs for _, (kd, inputs) in info.k.recipes.items()
                                                          if kd == info.kind(u)) for u in info.k.utensils):
            return None     # it can go where it belongs: no reason to put it down
        return self.reminder(tracker.k)


def wanted_by_card(ev: Event, rules: List[Rule], tracker: Tracker, state) -> bool:
    """Interactions with the partner's things that another of its rules asks for."""
    for r in rules:
        if r.id == "split_dish" and finishing(ev, r.p["dish"]):
            return True
        if r.id == "serves" and r.p["who"] == "ego" and serve_instance(ev, tracker.info):
            return True
        if r.id == "wants_supplies" and ev.verb == "put":
            pend = tracker.pending(state)
            if ev.utensil in pend and ev.item in pend[ev.utensil][0]:
                return True
    return False


def is_takeover(ev: Event, ctx_tracker: Tracker, state, rules: List[Rule]) -> bool:
    if ev.who != "ego":
        return False
    hand = set()
    for r in rules:
        if r.id == "handover_spot":
            hand = set(counter_groups(ctx_tracker.k).get(r.p["group"], []))
    if ev.verb == "pickup" and ev.source == "counter":
        return ctx_tracker.counter_by.get(ev.pos) == "partner" and ev.pos not in hand
    u = ev.utensil
    if u and ev.verb in ("pickup", "fill", "put", "start") and ctx_tracker.owner(u) == "partner":
        return not wanted_by_card(ev, rules, ctx_tracker, state)
    return False


class OnTakeover(Check):
    def __init__(self, *a):
        super().__init__(*a)
        self.waiting: List[Dict[str, Any]] = []

    def judge(self, ctx):
        out = []
        for ev in ctx.events:
            touches = (ev.utensil and ev.verb in ("pickup", "fill", "put", "start")
                       and ctx.k.utensil_status(ctx.pre, ev.utensil)[0] != "empty") or \
                      (ev.verb == "pickup" and ev.source == "counter")
            if ev.who == "ego" and touches:
                taken = is_takeover(ev, ctx.tr, ctx.pre, ctx.rules)
                out.append(self.J(ctx, "ego", not taken, f"{ev.verb} {ev.item} at {ev.target}"))
                if taken and ev.dishes:
                    self.waiting.append({"t": ctx.t, "dishes": set(ev.dishes)})
            elif ev.who == "partner" and ev.choice and ev.item != "dish" and ev.dishes:
                for w in list(self.waiting):
                    self.waiting.remove(w)
                    if ctx.t - w["t"] > TAKEOVER_WINDOW:
                        continue
                    if any(takeover_overridden(ctx.rules, d, ctx.tr.order_of.get("partner")) for d in w["dishes"]):
                        continue
                    if not ctx.has_job("partner", lambda j, ds: bool(ds) and not ds <= w["dishes"]):
                        continue        # fallback: no other order to work on
                    out.append(self.J(ctx, "partner", not (set(ev.dishes) & w["dishes"]),
                                      f"after takeover of {sorted(w['dishes'])}: {ev.verb} {ev.item}"))
        self.waiting = [w for w in self.waiting if ctx.t - w["t"] <= TAKEOVER_WINDOW]
        return out

    def opportunity(self, view, tracker, me, target=None):
        if tracker.takeovers:
            return self.reminder(tracker.k, dish=", ".join(sorted(tracker.takeovers[-1]["dishes"])))
        return None


class OnContention(Check):
    """Both players head for the same utensil (``Ctx.intents``: who -> utensil name) while
    both are within 3 cells of it. Resolved by who uses it first (within 15 steps):

    yields:  adherent if the ego used it first, or the partner did after waiting
             YIELD_PATIENCE steps (its fallback); aligned if the ego used it first.
    insists: adherent if the partner used it first, or still had it as its target when the
             ego used it (it did not give way); aligned if the partner used it first.
    """

    NEAR = 3
    YIELD_PATIENCE = 5

    def __init__(self, *a):
        super().__init__(*a)
        self.active: Optional[Dict[str, Any]] = None

    def _near(self, ctx, who, u) -> bool:
        i = ctx.tr.index.get(who)
        if i is None:
            return False
        pos, q = ctx.k.utensils[u], ctx.pre.players[i].position
        return max(abs(q[0] - pos[0]), abs(q[1] - pos[1])) <= self.NEAR

    def _loads(self, ctx, who, u) -> bool:
        """``who`` carries something that can go into ``u`` now (a race to load it)."""
        held = ctx.pre.players[ctx.tr.index[who]].held_object
        return held is not None and ctx.info.loadable(ctx.pre, u, held.name)

    def judge(self, ctx):
        ego_t, par_t = ctx.intents.get("ego"), ctx.intents.get("partner")
        if self.active is None:
            if ego_t and ego_t == par_t and self._near(ctx, "ego", ego_t) and self._near(ctx, "partner", ego_t) \
                    and all(self._loads(ctx, who, ego_t) for who in ("ego", "partner")):
                self.active = {"t": ctx.t, "u": ego_t}
            return []
        u = self.active["u"]
        taker = next((ev.who for ev in ctx.events if ev.utensil == u), None)
        if taker is None:
            if ctx.t - self.active["t"] > CONTENTION_WINDOW:
                self.active = None
            return []
        waited = ctx.t - self.active["t"]
        self.active = None
        if self.p["mode"] == "yields":
            adherent = taker == "ego" or waited >= self.YIELD_PATIENCE
            aligned = taker == "ego"
        else:
            adherent = taker == "partner" or par_t == u
            aligned = taker == "partner"
        note = f"{u}: {taker} used it first after {waited} steps (partner target then: {par_t})"
        return [self.J(ctx, "partner", adherent, note), self.J(ctx, "ego", aligned, note)]

    def opportunity(self, view, tracker, me, target=None):
        if not target or target not in tracker.k.utensils:
            return None
        other = view.players[1 - me]
        if getattr(other, "visible", True):
            pos = tracker.k.utensils[target]
            if max(abs(other.position[0] - pos[0]), abs(other.position[1] - pos[1])) <= 2:
                return self.reminder(tracker.k, station=target)
        return None


class WaitsForCommit(Check):
    def judge(self, ctx):
        out = []
        for ev in ctx.events:
            if not ev.choice or ev.item == "dish":
                continue
            for o in ctx.tr.new_orders:
                if o["t"] > ev.t:
                    continue
                if ev.who == "ego" and not o["ego_done"]:
                    o["ego_done"] = True
                    if o["ego_empty"]:
                        out.append(self.J(ctx, "ego", ev.t - o["t"] <= COMMIT_WAIT,
                                          f"committed {ev.t - o['t']} steps after {o['dish']} opened"))
                elif ev.who == "partner" and not o["partner_done"]:
                    o["partner_done"] = True
                    if not o.get("partner_free", True):
                        continue        # busy when the order came in: carrying on is not covered
                    ego_first = o.get("ego_t") is not None and o["ego_t"] <= ev.t
                    out.append(self.J(ctx, "partner", ev.t - o["t"] >= COMMIT_WAIT or ego_first,
                                      f"started {ev.t - o['t']} steps after {o['dish']} opened"))
        return out

    def opportunity(self, view, tracker, me, target=None):
        t = getattr(view, "timestep", tracker.t + 1)
        if view.players[me].held_object is not None:
            return None
        for o in tracker.new_orders:
            if 0 <= t - o["t"] < COMMIT_WAIT and o.get("ego_t") is None and o.get("partner_free", True):
                return self.reminder(tracker.k)
        return None


CHECKS = {"own_station": OwnStation, "solo_dish": SoloDish, "split_dish": SplitDish,
          "order_priority": OrderPriority, "serves": Serves, "wants_supplies": WantsSupplies,
          "handover_spot": HandoverSpot, "on_takeover": OnTakeover, "on_contention": OnContention,
          "waits_for_commit": WaitsForCommit}


def make_checks(rules: List[Rule], library, kitchen) -> List[Check]:
    return [CHECKS[r.id](r, library, kitchen) for r in rules]


# ---------------------------------------------------------------------- #
# Episode evaluator
# ---------------------------------------------------------------------- #
class RuleEvaluator:
    """Adherence (partner) and alignment (ego) per rule over one episode, on the full state."""

    def __init__(self, rules: List[Rule], library, kitchen, menu, ego: int, partner: int, n_open: int, mdp,
                 view_radius: Optional[int] = None):
        self.rules = rules
        self.ego, self.partner, self.view_radius = ego, partner, view_radius
        self.tracker = Tracker(kitchen, menu, {ego: "ego", partner: "partner"}, n_open)
        self.checks = make_checks(rules, library, kitchen)
        self.jobs = {"partner": JobView(mdp, partner), "ego": JobView(mdp, ego)}
        self.judgments: List[Judgment] = []

    def observe(self, t: int, pre, post, intents: Optional[Dict[str, Optional[str]]] = None) -> List[Judgment]:
        if not self.tracker.started:
            self.tracker.start(pre)
        events = parse_events(t, pre, post, self.tracker.info, self.tracker.roles)
        ctx = Ctx(t, pre, post, events, self.tracker, self.jobs, intents, self.rules)
        out: List[Judgment] = []
        for c in self.checks:
            out += c.judge(ctx)
        if self.view_radius is not None:
            a, b = pre.players[self.ego].position, pre.players[self.partner].position
            seen = max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= self.view_radius
            for j in out:
                if j.side == "partner":
                    j.seen = seen
        # Takeovers are part of the shared memory (the partner reacts to them).
        for ev in events:
            if is_takeover(ev, self.tracker, pre, self.rules) and ev.dishes:
                self.tracker.takeovers.append({"t": t, "dishes": set(ev.dishes)})
        self.tracker.observe(t, pre, post)      # update after judging (judgments see the pre-step memory)
        self.judgments += out
        return out

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in self.rules:
            lab = r.label()
            rows = {}
            for side, key in (("partner", "adherent"), ("ego", "aligned")):
                js = [j for j in self.judgments if j.rule == lab and j.side == side]
                rows[side] = {"opportunities": len(js), key: sum(j.value for j in js),
                              "rate": (sum(j.value for j in js) / len(js)) if js else None}
            # Exposure: the partner's opportunities that happened within the ego's view.
            rows["exposure"] = sum(1 for j in self.judgments if j.rule == lab and j.side == "partner" and j.seen)
            out[lab] = rows
        return out


def intent_target(player, kitchen) -> Optional[str]:
    """The utensil a player is heading for now (LLM players: the current macro; scripted
    players: the current job), or None."""
    ex = getattr(player, "executor", None) or getattr(getattr(player, "agent", None), "executor", None)
    if ex is not None:
        m = getattr(ex, "macro", None)
        if m is not None:
            for a in m.args:
                if a in kitchen.utensils:
                    return a
        return None
    p = getattr(player, "player", player)
    job = getattr(p, "job", None)
    if job is None:
        return None
    pos = job.pos
    if job.kind == "bring" and job.dest is not None and getattr(p, "me", None) is not None \
            and p.me.held_object is not None:
        try:
            pos, _ = p._select_station(job.dest)
        except Exception:
            pos = None
    return kitchen.utensil_at.get(tuple(pos)) if pos is not None else None
