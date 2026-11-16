"""Rule-following scripted players built on ``SimulatedHuman`` (by subclass; the test
personas in ``coop.partner`` are untouched).

* ``mode="partner"``: the **rule-based twin** of a persona card. It follows the card's
  rules deterministically (tendencies with their fallbacks), so its adherence checks the
  predicates in ``coop.rule_eval`` (it should be >= 95% by construction). Used for the
  gates and for tests without a model.
* ``mode="ego"``: a scripted ego that does what the card **wants** from the other cook
  (the aligned ego of Gate 2; with another card's rules, the misaligned ego).

Shared priority (docs/PERSONA_PARTNER_PLAN_v2.md, 4.2): a valid action and no deadlock,
then finishing the current commitment (what it holds), then the rules in card order, then
"do what is needed next, oldest order first".
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .kitchen import Kitchen
from .partner import SimulatedHuman, Traits
from .rule_eval import (COMMIT_WAIT, FALLBACK_WAIT, SUPPLY_WAIT, TAKEOVER_WINDOW, Tracker, is_takeover,
                        job_finishing, parse_events, partner_allows,
                        takeover_overridden)
from .rules import Rule, counter_groups

Pos = Tuple[int, int]
YIELD_PATIENCE = 5        # on_contention(yields): steps to give way before carrying on


def rule_traits(rules: List[Rule], mode: str) -> Traits:
    """Order commitment is a rule key (``RulePlayer._rule_key``), so it takes its place in
    card order; the base order focus only breaks ties."""
    focus = "first"
    for r in rules:
        if r.id == "order_priority":
            focus = "first" if _which(r, mode) == "oldest" else "last"
    # Giving way is station-level (RulePlayer._execute), not the base player's task-level
    # yielding, which would override higher rules (e.g. serves(self)).
    return Traits(order_focus=focus)


def _which(rule: Rule, mode: str) -> str:
    w = rule.p["which"]
    return w if mode == "partner" else ("newest" if w == "oldest" else "oldest")


def _yields(rule: Rule, mode: str) -> bool:
    """The partner yields under on_contention(yields); the aligned ego under (insists)."""
    return (rule.p["mode"] == "yields") == (mode == "partner")


class RulePlayer(SimulatedHuman):
    def __init__(self, rules: List[Rule], menu: List[str], mode: str = "partner", name: str = "rule"):
        if mode not in ("partner", "ego"):
            raise ValueError("mode is partner (twin) or ego (does what the card wants)")
        super().__init__(rule_traits(rules, mode), name)
        self.rules, self.menu, self.mode = list(rules), list(menu), mode
        self.by_id = {r.id: r for r in self.rules}

    def reset(self, mdp, index: int, seed: int, view_radius: Optional[int] = None) -> None:
        super().reset(mdp, index, seed, view_radius)
        self.rt: Optional[Tracker] = None
        self._prev_view = None
        self.avoid_until: Dict[str, int] = {}      # on_takeover (twin): dish -> step
        self.kitchen: Optional[Kitchen] = None
        self.order_dish: Optional[str] = None       # order_priority: the order I am committed to

    # -- watching ----------------------------------------------------------
    def _look(self, state):
        super()._look(state)
        if self.rt is None:
            self.kitchen = Kitchen(self.mdp, self.index, state)
            roles = {self.index: self.mode, 1 - self.index: "ego" if self.mode == "partner" else "partner"}
            self.rt = Tracker(self.kitchen, self.menu, roles, max(1, getattr(self.mdp, "num_open_orders", 1)))
            self.rt.start(state)
        prev = self._prev_view
        t = getattr(state, "timestep", None)
        if prev is not None and t is not None and t > getattr(prev, "timestep", -1):
            events = parse_events(t - 1, prev, state, self.rt.info, self.rt.roles)
            for ev in events:
                if is_takeover(ev, self.rt, prev, self.rules if self.mode == "partner" else self._partner_rules()) \
                        and ev.dishes:
                    self.rt.takeovers.append({"t": t - 1, "dishes": set(ev.dishes)})
                    if self.mode == "partner" and "on_takeover" in self.by_id:
                        for d in ev.dishes:
                            if not takeover_overridden(self.rules, d, self.order_dish):
                                self.avoid_until[d] = t - 1 + TAKEOVER_WINDOW
            self.rt.observe(t - 1, prev, state)
        self._prev_view = state

    def _partner_rules(self) -> List[Rule]:
        return self.rules      # ego mode: the card being respected is the partner's

    # -- helpers -------------------------------------------------------------
    def _dishes(self, job) -> set:
        return set(getattr(self, "_job_dishes", {}).get(job.key, ()))

    def _now(self) -> int:
        return getattr(self.state, "timestep", 0)

    def _late(self, since: Dict[str, int], pos: Optional[Pos], wait: int) -> bool:
        if pos is None or self.kitchen is None:
            return False
        u = self.kitchen.utensil_at.get(tuple(pos))
        return u is not None and u in since and self._now() - since[u] >= wait

    def _supply_job(self, job) -> bool:
        if job.kind != "bring":
            return False
        pend = self.rt.pending(self.state) if self.rt else {}
        return any(job.item in miss for miss, _ in pend.values())

    def _waiting_for_commit(self) -> bool:
        if self.mode != "partner" or "waits_for_commit" not in self.by_id or self.me.held_object is not None:
            return False
        now = self._now()
        return any(0 <= now - o["t"] < COMMIT_WAIT and o.get("ego_t") is None for o in self.rt.new_orders)

    def _supply_wait(self) -> Optional[Dict[str, Any]]:
        if self.mode != "partner" or "wants_supplies" not in self.by_id or self.me.held_object is not None:
            return None
        for u, (miss, since) in self.rt.pending(self.state).items():
            if self._now() - since < SUPPLY_WAIT:
                return {"u": u, "missing": miss}
        return None

    # -- job selection -------------------------------------------------------
    def _jobs(self):
        jobs = super()._jobs()
        if self.rt is None:
            return jobs
        if self._waiting_for_commit():
            return []
        r = self.by_id.get("order_priority")
        open_ = list(self.state.order_list[:max(1, getattr(self.mdp, "num_open_orders", 1))]) \
            if self.state.order_list else []
        if r is not None and self.order_dish not in open_ and open_:
            self.order_dish = open_[0] if _which(r, self.mode) == "oldest" else open_[-1]
        out = []
        for j in jobs:
            if self._keep(j):
                out.append(j)
        wait = self._supply_wait()
        if wait is not None:
            pos = self.kitchen.utensils[wait["u"]]
            out = [j for j in out if (j.kind == "bring" and j.item in wait["missing"] and j.pos is not None)
                   or (j.kind == "start" and j.pos == pos)]
        if self.mode == "partner" and self.avoid_until:
            now = self._now()
            avoided = {d for d, until in self.avoid_until.items() if now <= until}
            rest = [j for j in out if not (self._dishes(j) and self._dishes(j) <= avoided)]
            if rest:
                out = rest
        # Rules come before the job in hand (unless it is held): drop the current job when a
        # job a higher rule prefers is available, so the choice reflects the card's order.
        if self.job is not None and self.me.held_object is None and out:
            cur = next((j for j in out if j.key == self.job.key), None)
            if cur is not None and self._rule_key(min(out, key=self._rank)) < self._rule_key(cur):
                out = [j for j in out if j.key != cur.key]
        return out

    def _keep(self, job) -> bool:
        ds = self._dishes(job)
        if self.mode == "partner":
            return partner_allows(job, ds, self.rules, self.rt, self._now())
        for r in self.rules:
            p = r.p
            if self.mode == "partner":
                continue
            else:
                if r.id == "own_station" and job.pos is not None and \
                        self.kitchen.utensil_at.get(tuple(job.pos)) == p["station"]:
                    return False
                if r.id == "solo_dish" and p["dish"] in ds:
                    return False
                if r.id == "serves" and p["who"] == "self" and job.stage == "serve":
                    if not any(self._now() - s >= FALLBACK_WAIT for s in self.rt.ready_since.values()):
                        return False
                if r.id == "on_takeover" and job.pos is not None:
                    u = self.kitchen.utensil_at.get(tuple(job.pos))
                    wanted = any(job_finishing(job, ds, q.p["dish"]) for q in self.rules if q.id == "split_dish") \
                        or any(q.id == "serves" and q.p["who"] == "ego" and job.stage == "serve" for q in self.rules)
                    if u and self.rt.owner(u) == "partner" and not wanted:
                        return False
        return True

    def _rule_key(self, job) -> Tuple[int, ...]:
        ds = self._dishes(job)
        key = []
        for r in self.rules:
            p = r.p
            v = 0
            if self.mode == "partner":
                if r.id == "solo_dish":
                    v = 0 if p["dish"] in ds else 1
                elif r.id == "split_dish":
                    v = 0 if p["dish"] in ds and not job_finishing(job, ds, p["dish"]) else 1
                elif r.id == "serves" and p["who"] == "self":
                    v = 0 if job.stage == "serve" else 1
            else:
                if r.id == "split_dish":
                    v = 0 if job_finishing(job, ds, p["dish"]) else 1
                elif r.id == "serves" and p["who"] == "ego":
                    v = 0 if job.stage == "serve" else 1
                elif r.id == "wants_supplies":
                    v = 0 if self._supply_job(job) else 1
            if r.id == "order_priority":
                v = 0 if self.order_dish is not None and self.order_dish in ds else 1
            key.append(v)
        return tuple(key)

    def _rank(self, job):
        return self._rule_key(job) + tuple(super()._rank(job))

    def _choose(self, jobs, t):
        if self.rt is not None and self._waiting_for_commit():
            return None, "waiting_for_commit"
        job, why = super()._choose(jobs, t)
        if job is None and self.rt is not None and self._supply_wait() is not None:
            return None, "waiting_for_supplies"
        return job, why

    # -- on_contention: give way when the other player is about to use my station ----
    def _execute(self, t: int):
        r = self.by_id.get("on_contention")
        if r is not None and _yields(r, self.mode) and self.kitchen is not None:
            from .rule_eval import intent_target
            u = intent_target(self, self.kitchen)
            if u is not None:
                pos = self.kitchen.utensils[u]
                d = lambda q: max(abs(q[0] - pos[0]), abs(q[1] - pos[1]))
                me, other = tuple(self.me.position), tuple(self.other.position)
                if getattr(self.other, "visible", True) and d(other) <= 2 and d(other) <= d(me):
                    # Step back, but only for a while (two players waiting by one station
                    # would never move): after YIELD_PATIENCE steps, carry on.
                    self.yield_steps = getattr(self, "yield_steps", 0) + 1
                    if self.yield_steps <= YIELD_PATIENCE:
                        from overcooked_ai_py.mdp.actions import Action
                        return Action.STAY, None
                else:
                    self.yield_steps = 0
        return super()._execute(t)

    # -- where -----------------------------------------------------------------
    def _select_station(self, product: str):
        station, status = super()._select_station(product)
        r = next((q for q in self.rules if q.id == "own_station"
                  and q.p["station"].rstrip("0123456789") == self.recipes[product][0]), None)
        if r is None or self.kitchen is None or status == "busy":
            return station, status
        own = self.kitchen.utensils[r.p["station"]]
        kind, recipe = self.recipes[product]
        openish = []
        for p in self.stations.get(kind, []):
            obj = self.objects.get(p)
            if obj is None or (isinstance(obj.state[0], list) and not (Counter(obj.state[0]) - recipe)):
                openish.append(p)
        if self.mode == "partner":
            if own in openish:
                return own, "open"
            return station, status
        rest = [p for p in openish if p != own]
        if station == own and rest:
            return rest[0], "open"
        return station, status

    def _free_counters(self, shared: bool):
        free = super()._free_counters(shared)
        r = self.by_id.get("handover_spot")
        if r is None or self.kitchen is None:
            return free
        group = set(counter_groups(self.kitchen).get(r.p["group"], []))
        mine = [p for p in free if tuple(p) in group]
        return mine or free

    def summary(self) -> Dict[str, Any]:
        out = super().summary()
        out["rules"] = [r.label() for r in self.rules]
        out["mode"] = self.mode
        return out


# ---------------------------------------------------------------------- #
# Scripted egos for Gate 2: aligned (own card's wants) / misaligned (another card's)
# ---------------------------------------------------------------------- #
from .agents import Agent, register_agent  # noqa: E402


class RuleEgoAgent(Agent):
    """A scripted ego that does what a card wants from the other cook (``RulePlayer`` in
    ego mode). The runner resolves which card (``align: own | other``) and passes its
    rules; the agent itself is told nothing else about the partner."""
    name = "rule_ego"

    def __init__(self, rules: List[Dict[str, Any]], menu: List[str], card: str = "", align: str = "own"):
        from .rules import load_library, parse_rule
        lib = load_library()
        self.rules = [parse_rule(r, lib) for r in rules]
        self.menu, self.card, self.align = list(menu), card, align
        self.memory: Dict[str, Any] = {}

    def begin_episode(self, mdp, index, context):
        self.player = RulePlayer(self.rules, self.menu, "ego", f"rule_ego:{self.card}")
        self.player.reset(mdp, index, context["episode_seed"], view_radius=context.get("view_radius"))

    def act(self, state, t):
        return self.player.act(state, t)

    def episode_log(self):
        return {"summary": self.player.summary(), "card": self.card, "align": self.align}


register_agent("rule_ego", RuleEgoAgent)
