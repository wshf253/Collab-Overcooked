"""Context-dependent scripted players (docs/CONTEXT_PARTNER_PLAN_v3.md, sections 4 and 6).

``ContextPlayer`` extends the rule-based player (``coop.rule_partner.RulePlayer``) with the
context rules (``default_work``, ``on_urgent``, ``on_blocked``, ``on_prep_ready``). Every
player decides on its own view (``coop.view``) and the context features of ``coop.context``.

Policies:

* ``card`` (the **twin** of a context card): does what each mode says, in card order, with
  no fallback. One global **deadlock guard**: if neither player has completed an interaction
  for ``guard_steps`` steps while an order is open, it does the next needed step once (logged
  as ``guard``; a failure count, not part of the persona).
* ``proactive`` (ego): at each context acts at once on the card's predicted response, i.e.
  does what the card's ``wants`` line says. ``mismatched`` is ``proactive`` with the opposite
  card (the caller passes that card).
* reactive egos: ``steady`` (the base player), ``always_take_urgent`` / ``never_take_urgent``,
  and ``wait_and_see`` (after a context starts, waits ``k`` steps, then acts on what it saw).

How the modes act on jobs (a job is one step of the base player's plan):

| rule | partner (twin) | ego, proactive for that card |
|---|---|---|
| default_work prep | only prep jobs: fetch raw ingredients, cut / blend; raw items for pots and ovens are put on a counter | no prep jobs: take items from counters and utensils, cook, plate, deliver |
| default_work whole | only jobs of its order (oldest open order the other is not on), kept until served | only jobs of its own order: the newest open order the partner is not on |
| on_urgent takes | only jobs of the urgent order (unless_ego_on_it=yes: not if the other is on it) | no jobs of the urgent order (unless it is already on it) |
| on_urgent keeps | as default | only jobs of the urgent order |
| on_blocked clears | only jobs that take finished food out of the blocked kind | no such jobs |
| on_blocked waits | no such jobs and none that need that kind | only such jobs |
| on_prep_ready switches | only cook / serve jobs of dishes with prepared items waiting | no such jobs |
| on_prep_ready continues | as default | only such jobs |

Rules apply in card order: the first rule whose context holds and that names jobs ("only")
decides; "no ... jobs" rules remove jobs and let later rules apply.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .context import ContextTracker, on_order, open_dishes, tree
from .partner import Job
from .rule_partner import RulePlayer
from .rules import CONTEXT_RULE_IDS, Rule

PREP_KINDS = {"chopping_board", "blender"}
COOK_KINDS = {"pot", "oven"}
GUARD_STEPS = 30
EGO_POLICIES = ("proactive", "steady", "always_take_urgent", "never_take_urgent", "wait_and_see")


class ContextPlayer(RulePlayer):
    def __init__(self, rules: List[Rule], menu: List[str], mode: str = "partner", policy: str = "card",
                 k: int = 0, guard_steps: int = GUARD_STEPS, name: str = "context"):
        super().__init__(rules, menu, mode, name)
        if mode == "partner" and policy != "card":
            raise ValueError("the partner plays its card (policy 'card')")
        if mode == "ego" and policy not in EGO_POLICIES:
            raise ValueError(f"ego policy must be one of {EGO_POLICIES}")
        self.policy, self.k, self.guard_steps = policy, int(k), int(guard_steps)
        self.ctx_rules = [r for r in self.rules if r.id in CONTEXT_RULE_IDS]

    # -- episode ------------------------------------------------------------
    def reset(self, mdp, index: int, seed: int, view_radius: Optional[int] = None) -> None:
        super().reset(mdp, index, seed, view_radius)
        self.theta = getattr(mdp, "urgent_steps", None) or 10
        self.n_open = max(1, getattr(mdp, "num_open_orders", 1))
        self.commit: Optional[str] = None          # default_work whole: the order it owns
        self.last_progress = 0                     # last step anyone completed an interaction
        self.guard_active = False
        self.guards: List[Dict[str, Any]] = []
        self.ctx_tracker: Optional[ContextTracker] = None
        self.waiting: Dict[str, Any] = {}          # wait_and_see: event being watched, until
        self.seen: Dict[Tuple, bool] = {}          # wait_and_see: what it decided per context
        self.decision_log: List[Dict[str, Any]] = []

    def act(self, state, t: int):
        # The guard is a runner mechanism: it looks at the true state, not the player's view.
        if any(state.ml_actions or []):
            self.last_progress = t
            self.guard_active = False
        if (self.mode == "partner" and not self.guard_active and state.order_list
                and t - self.last_progress >= self.guard_steps):
            self.guard_active = True
            self.last_progress = t
            self.guards.append({"t": t})
            self.events.append({"t": t, "type": "guard"})
        return super().act(state, t)

    def _look(self, state):
        super()._look(state)
        if self.ctx_tracker is None:
            self.ctx_tracker = ContextTracker(self.kitchen, self.n_open, self.theta)
        t = getattr(state, "timestep", 0)
        for ev in self.ctx_tracker.observe(state, t):
            if self.policy == "wait_and_see" and self.k > 0 and not self.waiting:
                self.waiting = {"event": ev, "until": t + self.k}

    # -- context ------------------------------------------------------------
    def _other(self) -> str:
        return "ego" if self.mode == "partner" else "partner"

    def _other_on(self) -> Optional[str]:
        return on_order(self.state, self.kitchen, self.n_open, 1 - self.index, self.rt.order_of.get(self._other()))

    def _self_on(self) -> Optional[str]:
        return on_order(self.state, self.kitchen, self.n_open, self.index, self.rt.order_of.get(self.mode))

    def _context(self) -> Dict[str, Any]:
        dishes = open_dishes(self.state, self.n_open)
        need = {self.kitchen.recipes[x][0] for d in dishes for x in tree(self.kitchen, d) if x in self.kitchen.recipes}
        now = self.ctx_tracker.now
        return {"urgent": [d for _, d in now["urgent"]],
                "blocked": [k for k in now["blocked"] if k in need],
                "prep_ready": list(now["prep_ready"]), "open": dishes}

    def _theirs_item(self, job: Job) -> bool:
        """The job starts from an item the other cook made or put down (its utensil batch or
        its counter item): a cook doing its own order start to finish does not take it."""
        if job.pos is None:
            return False
        other = self._other()
        u = self.kitchen.utensil_at.get(tuple(job.pos))
        if u is not None:
            return self.rt.owner(u) == other
        return self.rt.counter_by.get(tuple(job.pos)) == other

    # -- job classes ---------------------------------------------------------
    def _dest_kind(self, job: Job) -> Optional[str]:
        return self.recipes[job.dest][0] if job.kind == "bring" and job.dest in self.recipes else None

    def _is_prep(self, job: Job) -> bool:
        if job.kind == "bring":
            dk = self._dest_kind(job)
            if dk in PREP_KINDS:
                return True
            if dk in COOK_KINDS:
                # fetching a raw ingredient from a dispenser is prep; carrying one (in hand or
                # from a counter) into a pot or oven is loading it, i.e. cooking
                held = self.me.held_object is not None and self.me.held_object.name == job.item
                return (job.item not in self.recipes and not held
                        and (job.pos is None or self.terrain[job.pos[1]][job.pos[0]] == "I"))
            return False
        if job.kind == "start" and job.pos in self.kind_at:
            return self.kind_at[job.pos] in PREP_KINDS
        return False

    def _clears(self, job: Job, kinds: List[str], others_only: bool = False) -> bool:
        """Taking finished food out of a utensil of a blocked kind (``others_only``: not from
        its own batch; finishing its own order is not clearing)."""
        if not (job.pos is not None and self.kind_at.get(job.pos) in kinds
                and job.kind in ("bring", "plate", "unload", "clear", "supply_dish")):
            return False
        if others_only:
            u = self.kitchen.utensil_at.get(tuple(job.pos))
            return self.rt.owner(u) != self.mode
        return True

    def _needs_kind(self, job: Job, kinds: List[str]) -> bool:
        return self._dest_kind(job) in kinds or (job.kind == "start" and self.kind_at.get(job.pos) in kinds)

    def _serves(self, job: Job, dish: str) -> bool:
        return dish in self._dishes(job)

    def _cooks(self, job: Job, dishes: List[str]) -> bool:
        return any(self._serves(job, d) for d in dishes) and not self._is_prep(job)

    # -- the policy ------------------------------------------------------------
    def _no_cook(self) -> bool:
        """The prep partner does not load pots or ovens (it puts their raw items on a counter)."""
        return (self.mode == "partner" and self.policy == "card" and not self.guard_active
                and any(r.id == "default_work" and r.p["mode"] == "prep" for r in self.ctx_rules)
                and getattr(self, "_regime", "default") == "default")

    def _can_place(self, dest):
        if dest in self.recipes and self.recipes[dest][0] in COOK_KINDS and self._no_cook():
            return False
        return super()._can_place(dest)

    def _carry(self, job: Job):
        if job.dest in self.recipes and self.recipes[job.dest][0] in COOK_KINDS and self._no_cook():
            return self._handoff()
        return super()._carry(job)

    def _pick_commit(self, ctx, newest: bool) -> Optional[str]:
        if self.commit in ctx["open"]:
            return self.commit
        other = self._other_on()
        cands = [d for d in ctx["open"] if d != other] or list(ctx["open"])
        self.commit = (cands[-1] if newest else cands[0]) if cands else None
        return self.commit

    def _base(self, jobs: List[Job], ctx, rule: Optional[Rule], ego: bool) -> List[Job]:
        if rule is None:
            return jobs
        if rule.p["mode"] == "prep":
            return [j for j in jobs if self._is_prep(j) != ego]
        own = self._pick_commit(ctx, newest=ego)
        return [j for j in jobs if own is not None and self._serves(j, own) and not self._theirs_item(j)]

    def _apply(self, jobs: List[Job], ctx, rules: List[Rule], ego: bool) -> Tuple[List[Job], str]:
        """Card order: the first applicable "only" rule decides; "no" rules remove jobs."""
        default = next((r for r in rules if r.id == "default_work"), None)
        current = self._base(jobs, ctx, default, ego)
        for r in rules:
            m = r.p.get("mode")
            only, drop = None, None
            if r.id == "on_urgent" and ctx["urgent"]:
                u = ctx["urgent"][0]
                exempt = r.opt("unless_ego_on_it") == "yes" and (
                    (self._other_on() == u) if not ego else (self._self_on() == u))
                take = (m == "takes") != ego          # the partner takes, or the ego takes for "keeps"
                if m == "takes" and exempt:
                    take = ego                        # the ego is on it: the ego finishes it
                if take:
                    only = [j for j in jobs if self._serves(j, u)]
                elif ego or m == "takes":
                    drop = [j for j in current if self._serves(j, u)]
            elif r.id == "on_blocked" and ctx["blocked"]:
                kinds = ctx["blocked"]
                clear = (m == "clears") != ego
                if clear:
                    only = [j for j in jobs if self._clears(j, kinds)]
                else:
                    drop = [j for j in current if self._clears(j, kinds, others_only=True)
                            or (m == "waits" and self._needs_kind(j, kinds))]
            elif r.id == "on_prep_ready" and ctx["prep_ready"]:
                cook = (m == "switches") != ego
                if cook:
                    only = [j for j in jobs if self._cooks(j, ctx["prep_ready"])]
                elif ego:
                    drop = [j for j in current if self._cooks(j, ctx["prep_ready"])]
            if only:
                return only, r.id
            if drop:
                current = [j for j in current if j not in drop]
        return current, "default_work" if default is not None else "none"

    def _reactive(self, jobs: List[Job], ctx) -> Tuple[List[Job], str]:
        p, t = self.policy, self._now()
        if p in ("always_take_urgent", "never_take_urgent") and ctx["urgent"]:
            u = ctx["urgent"][0]
            if p == "always_take_urgent":
                own = [j for j in jobs if self._serves(j, u)]
                return (own or jobs), "on_urgent"
            return [j for j in jobs if not self._serves(j, u)], "on_urgent"
        if p == "wait_and_see":
            if self.waiting and t < self.waiting["until"]:
                return [], "waiting"
            if self.waiting:                         # the wait is over: decide on what it saw
                ev, self.waiting = self.waiting["event"], {}
                if ev["context"] == "on_urgent":
                    self.seen[("on_urgent", ev["dish"])] = self._other_on() != ev["dish"]     # take it?
                elif ev["context"] == "on_blocked":
                    self.seen[("on_blocked", ev["kind"])] = ev["kind"] in ctx["blocked"]      # still full: clear
                elif ev["context"] == "on_prep_ready":
                    self.seen[("on_prep_ready", ev["dish"])] = ev["dish"] in ctx["prep_ready"]  # untouched: cook
            for (c, x), act in list(self.seen.items()):
                if c == "on_urgent" and x in ctx["urgent"]:
                    sel = [j for j in jobs if self._serves(j, x) == act]
                    return (sel or jobs), c
                if c == "on_blocked" and x in ctx["blocked"] and act:
                    sel = [j for j in jobs if self._clears(j, [x])]
                    if sel:
                        return sel, c
                if c == "on_prep_ready" and x in ctx["prep_ready"]:
                    sel = [j for j in jobs if self._cooks(j, [x]) == act]
                    if sel:
                        return sel, c
        return jobs, "none"

    def _jobs(self):
        jobs = super()._jobs()
        if self.rt is None or not self.state.order_list:
            return jobs
        ctx = self._context()
        if self.mode == "partner":
            if self.guard_active:
                self._regime = "guard"
                return jobs
            sel, why = self._apply(jobs, ctx, self.ctx_rules, ego=False)
        elif self.policy == "proactive":
            sel, why = self._apply(jobs, ctx, self.ctx_rules, ego=True)
        else:
            sel, why = self._reactive(jobs, ctx)
        self._regime = "default" if why in ("default_work", "none") else why
        self._last_ctx = (why, ctx)
        return sel

    def _choose(self, jobs, t):
        if self.policy == "wait_and_see" and self.waiting and t < self.waiting["until"] and self.me.held_object is None:
            return None, "waiting_to_see"
        job, why = super()._choose(jobs, t)
        if job is not None and job is not getattr(self, "_logged_job", None):
            rule, ctx = getattr(self, "_last_ctx", ("none", {}))
            self.decision_log.append({"t": t, "rule": rule, "job": job.describe(),
                                      "urgent": ctx.get("urgent"), "blocked": ctx.get("blocked"),
                                      "prep_ready": ctx.get("prep_ready")})
            self._logged_job = job
        return job, why

    def summary(self) -> Dict[str, Any]:
        out = super().summary()
        out.update(policy=self.policy, k=self.k, guards=len(self.guards))
        return out


# ---------------------------------------------------------------------- #
# Runner agent: scripted context egos (Gate 2 with the LLM partner, plan v3 7)
# ---------------------------------------------------------------------- #
from .agents import Agent, register_agent  # noqa: E402


class ContextEgoAgent(Agent):
    """A scripted context ego in the runner. ``policy``: proactive (``align: own`` = this
    partner's card, ``other`` = mismatched), always_take_urgent, never_take_urgent,
    wait_and_see (``k``)."""
    name = "context_ego"

    def __init__(self, rules: List[Dict[str, Any]], menu: List[str], policy: str = "proactive", k: int = 0,
                 card: str = "", align: str = "own"):
        from .rules import load_library, parse_rule
        lib = load_library()
        self.rules = [parse_rule(r, lib) for r in rules] if policy == "proactive" else []
        self.menu, self.policy, self.k, self.card, self.align = list(menu), policy, int(k), card, align
        self.memory: Dict[str, Any] = {}

    def begin_episode(self, mdp, index, context):
        self.player = ContextPlayer(self.rules, self.menu, "ego", self.policy, k=self.k, name=f"context_ego:{self.policy}")
        self.player.reset(mdp, index, context["episode_seed"], view_radius=context.get("view_radius"))

    def act(self, state, t):
        return self.player.act(state, t)

    @property
    def job(self):
        return self.player.job

    @property
    def job_t(self):
        return self.player.job_t

    def episode_log(self):
        return {"summary": self.player.summary(), "card": self.card, "align": self.align, "policy": self.policy,
                "decisions": self.player.decision_log}


register_agent("context_ego", ContextEgoAgent)
