"""Checks for the context rules (docs/CONTEXT_PARTNER_PLAN_v3.md, section 4).

One checker covers all context rules of a card, because card order decides which rule governs
a decision. Every judgment is about a *choice* (an action started with empty hands; for the
partner also loading a pot or oven), judged in the context the chooser saw **in its own view
at the step it decided** (``Ctx.decided``; the evaluator keeps each player's view and context
history, ``RuleEvaluator.ctx_hist``).

partner (adherent), at a decision where:

* an "only" rule's context holds (first in card order): ``on_urgent(takes)`` (urgent order,
  not exempt) -> the choice serves the urgent order; ``on_blocked(clears)`` -> it takes
  finished food out of a blocked kind; ``on_prep_ready(switches)`` -> it cooks / serves a dish
  whose prepared items wait;
* otherwise, every rule whose context holds is judged: ``keeps`` -> it does not switch to
  cooking or serving the urgent order (unless that is its own order); ``waits`` -> it neither
  clears nor uses a blocked kind; ``continues`` -> it does not cook a dish whose prepared items
  wait; and with no context at all, ``default_work``: ``prep`` -> a prep step (fetch from a
  dispenser, load or start a board or blender); ``whole`` -> it serves its own order;
  ``station`` (plan v4) -> a station step (load or start a pot or oven, take food out of one,
  plate) and not a fetch, a cut or a delivery; with ``station`` loads and deliveries are judged
  too (they are not choices).

ego (aligned), with the first rule (card order) whose context holds, else ``default_work``:
the complement of the partner's mode (the card's ``wants`` line), e.g. ``takes`` -> it does not
serve the urgent order (unless it is already on it, with ``unless_ego_on_it: yes``), ``keeps``
-> it serves the urgent order; ``prep`` -> no prep step; ``whole`` -> it does not serve the
partner's order.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .context import open_dishes, tree
from .rule_eval import Check, Event, Judgment
from .rules import CONTEXT_RULE_IDS, Rule, render

PREP_KINDS = {"chopping_board", "blender"}
COOK_KINDS = {"pot", "oven"}
EMPTY = {"urgent": [], "blocked": [], "prep_ready": []}


class ContextRules(Check):
    def __init__(self, rules: List[Rule], library, kitchen, style: bool = False):
        self.style = style            # plan v4 cards: the judging fixes below apply to them only
        self.rules = [r for r in rules if r.id in CONTEXT_RULE_IDS]
        self.lib, self.k = library, kitchen
        self.rule = self.rules[0] if self.rules else None
        self.label = "context"
        self.by_id = {r.id: r for r in self.rules}
        self.own: Optional[str] = None      # default_work(whole): the order the partner took

    # -- event classes -----------------------------------------------------
    def kind(self, u: Optional[str]) -> Optional[str]:
        return self.k.kind_of.get(u) if u else None

    def prep_step(self, ev: Event) -> bool:
        return (ev.verb == "pickup" and ev.source == "ingredient_dispenser") or \
            (ev.verb in ("start", "put") and self.kind(ev.utensil) in PREP_KINDS)

    def clearing(self, ev: Event, kinds: List[str], ctx=None, others_only: bool = False) -> bool:
        """Taking finished food out of a utensil of a blocked kind. ``others_only``: not the
        chooser's own batch (finishing its own order is not clearing someone's utensil)."""
        if not (ev.verb in ("pickup", "fill") and self.kind(ev.utensil) in kinds):
            return False
        return not (others_only and ctx is not None and ctx.tr.owner(ev.utensil) == ev.who)

    def cook_serve_step(self, ev: Event) -> bool:
        """Cooking or serving: loading or starting a pot or oven, taking food out of one,
        plating, delivering. Moving prepared items (off a board, onto a counter) is not."""
        return (ev.verb in ("put", "start") and self.kind(ev.utensil) in COOK_KINDS) or ev.verb in ("fill", "deliver") \
            or (ev.verb == "pickup" and self.kind(ev.utensil) in COOK_KINDS)

    def station_step(self, ev: Event) -> bool:
        """The station cook's work (plan v4): loading or starting a pot or oven, taking food out
        of one, plating."""
        from .context_partner import STATION_PLATES
        return (ev.verb in ("put", "start", "pickup") and self.kind(ev.utensil) in COOK_KINDS) or \
            (ev.verb == "fill" and STATION_PLATES)

    def non_station_step(self, ev: Event) -> bool:
        return self.prep_step(ev) or ev.verb == "deliver"

    def workflow(self) -> Optional[str]:
        r = self.by_id.get("default_work")
        return r.p["mode"] if r is not None else None

    def uses_kind(self, ev: Event, kinds: List[str]) -> bool:
        return ev.verb in ("put", "start") and self.kind(ev.utensil) in kinds

    @staticmethod
    def serves(ev: Event, dish: Optional[str]) -> bool:
        return dish is not None and dish in ev.dishes

    def cook_of(self, ev: Event, dishes: List[str]) -> bool:
        """Cooking or serving one of ``dishes``; taking food out of a pot or oven (to free it for
        them) counts too."""
        if self.style and ev.verb == "pickup" and self.kind(ev.utensil) in COOK_KINDS:
            return True
        return any(d in ev.dishes for d in dishes) and not self.prep_step(ev)

    # -- context at the decision ---------------------------------------------
    def features(self, ctx, who: str) -> Dict[str, Any]:
        hist = (getattr(ctx, "ctx_hist", None) or {}).get(who) or {}
        d = ctx.decided(who)
        if not hist:
            return dict(EMPTY, open=open_dishes(ctx.pre, ctx.tr.n_open))
        key = max((s for s in hist if s <= d), default=None)
        f = dict(hist[key] if key is not None else hist[min(hist)])
        dishes = f.get("open") or open_dishes(ctx.pre, ctx.tr.n_open)
        need = {self.k.recipes[x][0] for dd in dishes for x in tree(self.k, dd) if x in self.k.recipes}
        f["blocked"] = [k for k in f["blocked"] if k in need]
        f["urgent"] = [d for _, d in f["urgent"]] if f["urgent"] and isinstance(f["urgent"][0], tuple) else f["urgent"]
        return f

    def J(self, ctx, rule: Rule, side: str, value: bool, note: str) -> Judgment:
        return Judgment(ctx.t, rule.label(), side, bool(value), note)

    # -- judging -----------------------------------------------------------------
    @staticmethod
    def decision(ev: Event) -> bool:
        """A choice of work: started with empty hands (a dish pickup only says that something
        will be plated, so the plating itself, ``fill``, is judged instead)."""
        return (ev.choice and ev.item != "dish") or ev.verb == "fill"

    def _job_kind(self, j) -> Optional[str]:
        return self.k.kind_of.get(self.k.utensil_at.get(tuple(j.pos))) if j.pos is not None else None

    def _available(self, ctx, rule_id: str, f) -> bool:
        """Did the partner have a job the "only" rule names? If not it acts on its next rule,
        as the twin does."""
        if rule_id == "on_urgent":
            return ctx.has_job("partner", lambda j, ds: f["urgent"][0] in ds)
        if rule_id == "on_blocked":
            return ctx.has_job("partner", lambda j, ds: self._job_kind(j) in f["blocked"]
                               and j.kind in ("bring", "plate", "unload", "clear", "supply_dish"))
        return ctx.has_job("partner", lambda j, ds: any(d in ds for d in f["prep_ready"]) and j.stage != "prep")

    def judge(self, ctx) -> List[Judgment]:
        if not self.rules:
            return []
        out: List[Judgment] = []
        if "default_work" in self.by_id and self.by_id["default_work"].p["mode"] == "whole":
            self._update_own(ctx)
        station = self.workflow() == "station"
        for ev in ctx.events:
            prep = "default_work" in self.by_id and self.by_id["default_work"].p["mode"] == "prep"
            extra = station and (ev.verb == "deliver" or (ev.verb == "put" and self.kind(ev.utensil) in COOK_KINDS))
            if ev.who == "partner" and (self.decision(ev) or extra
                                        or (prep and ev.verb == "put" and self.kind(ev.utensil) in COOK_KINDS)):
                out += self._partner(ctx, ev)
            elif ev.who == "ego" and (self.decision(ev) or extra):
                out += self._ego(ctx, ev)
        return out

    def _update_own(self, ctx) -> None:
        """default_work(whole), every step: when its order has closed, the partner takes the
        oldest open order the ego is not working on, as its own view shows it (what the rule
        prescribes, and what the twin does)."""
        h = (getattr(ctx, "ctx_hist", None) or {}).get("partner", {}).get(ctx.t)
        if h is None:
            return
        open_ = h.get("open") or []
        if self.own not in open_:
            declared = ((getattr(ctx, "intents", None) or {}).get("orders") or {}).get("partner")
            if declared in open_:
                self.own = declared       # the order it says it took (LLM partner: its stated goal)
            else:
                cands = [d for d in open_ if d != h.get("other_on")] or list(open_)
                self.own = cands[0] if cands else None

    def _partner(self, ctx, ev: Event) -> List[Judgment]:
        f = self.features(ctx, "partner")
        U, B, P = f["urgent"], f["blocked"], f["prep_ready"]
        whole = "default_work" in self.by_id and self.by_id["default_work"].p["mode"] == "whole"
        mine = self.own if whole and self.own in f["open"] else None
        ego_on = f.get("other_on", ctx.tr.order_of.get("ego"))       # as the partner's view shows it
        what = f"{ev.verb} {ev.item} at {ev.target}"
        for r in self.rules:
            m = r.p.get("mode")
            if r.id == "on_urgent" and U and m == "takes" and not (r.opt("unless_ego_on_it") == "yes" and ego_on == U[0]) \
                    and self._available(ctx, r.id, f):
                return [self.J(ctx, r, "partner", self.serves(ev, U[0]), f"{what} (urgent {U[0]})")]
            if r.id == "on_blocked" and B and m == "clears" and self._available(ctx, r.id, f):
                return [self.J(ctx, r, "partner", self.clearing(ev, B), f"{what} (blocked {', '.join(B)})")]
            if r.id == "on_prep_ready" and P and m == "switches" and self._available(ctx, r.id, f):
                return [self.J(ctx, r, "partner", self.cook_of(ev, P), f"{what} (prepared items for {', '.join(P)})")]
        out = []
        for r in self.rules:
            m = r.p.get("mode")
            if r.id == "on_urgent" and U:                 # keeps (or takes while the ego is on it)
                if self.workflow() == "station":          # keeps working the station: no fetch, cut or serve for it
                    bad = self.serves(ev, U[0]) and self.non_station_step(ev)
                else:
                    bad = self.serves(ev, U[0]) and not self.prep_step(ev) and U[0] != mine
                out.append(self.J(ctx, r, "partner", not bad, f"{what} (urgent {U[0]})"))
            elif r.id == "on_blocked" and B:              # waits
                bad = self.clearing(ev, B, ctx, others_only=True) or self.uses_kind(ev, B)
                out.append(self.J(ctx, r, "partner", not bad, f"{what} (blocked {', '.join(B)})"))
            elif r.id == "on_prep_ready" and P:           # continues: does not cook / serve them
                bad = self.cook_serve_step(ev) and any(d in ev.dishes for d in P)
                out.append(self.J(ctx, r, "partner", not bad, f"{what} (prepared items for {', '.join(P)})"))
        if not (U or B or P) and "default_work" in self.by_id:
            r = self.by_id["default_work"]
            if r.p["mode"] == "prep":
                # preparing is adherent, cooking or serving is not; moving prepared items is neither
                if self.prep_step(ev) or self.cook_serve_step(ev):
                    out.append(self.J(ctx, r, "partner", self.prep_step(ev), what))
            elif r.p["mode"] == "station":
                if self.station_step(ev) or self.non_station_step(ev):
                    out.append(self.J(ctx, r, "partner", self.station_step(ev), what))
            else:
                out.append(self.J(ctx, r, "partner", mine is None or self.serves(ev, mine), f"{what} (own order {mine})"))
        return out

    def _ego(self, ctx, ev: Event) -> List[Judgment]:
        f = self.features(ctx, "ego")
        U, B, P = f["urgent"], f["blocked"], f["prep_ready"]
        what = f"{ev.verb} {ev.item} at {ev.target}"
        for r in self.rules:
            m = r.p.get("mode")
            if r.id == "on_urgent" and U:
                u = U[0]
                if m == "keeps":
                    return [self.J(ctx, r, "ego", self.serves(ev, u), f"{what} (urgent {u})")]
                on_it = r.opt("unless_ego_on_it") == "yes" and ctx.tr.order_of.get("ego") == u
                return [self.J(ctx, r, "ego", self.serves(ev, u) == on_it, f"{what} (urgent {u})")]
            if r.id == "on_blocked" and B:
                cleared = self.clearing(ev, B, ctx, others_only=(m == "clears"))
                return [self.J(ctx, r, "ego", cleared == (m == "waits"), f"{what} (blocked {', '.join(B)})")]
            if r.id == "on_prep_ready" and P:
                return [self.J(ctx, r, "ego", self.cook_of(ev, P) == (m == "continues"),
                               f"{what} (prepared items for {', '.join(P)})")]
        r = self.by_id.get("default_work")
        if r is None:
            return []
        if r.p["mode"] == "prep":
            return [self.J(ctx, r, "ego", not self.prep_step(ev), what)]
        if r.p["mode"] == "station":
            if self.station_step(ev) or self.non_station_step(ev):
                return [self.J(ctx, r, "ego", not self.station_step(ev), what)]
            return []
        theirs = self.own if self.own in f["open"] else None
        ok = theirs is None or not self.serves(ev, theirs)
        return [self.J(ctx, r, "ego", ok, f"{what} (partner's order {theirs})")]

    # -- reminders (LLM partner, plan v3 W4) --------------------------------------
    def opportunity(self, view, tracker, me, target=None):
        """The reminder of the first rule (card order) whose context holds in the partner's
        own view now. ``tracker.ctx_now``: the features kept by the partner every step (so
        "prepared items waiting" counts waiting time). Nothing in the default context: the
        habit is in the system prompt."""
        f = getattr(tracker, "ctx_now", None)
        if f is None:
            return None
        dishes = open_dishes(view, tracker.n_open)
        need = {self.k.recipes[x][0] for d in dishes for x in tree(self.k, d) if x in self.k.recipes}
        U = [d for _, d in f["urgent"]]
        B = [k for k in f["blocked"] if k in need]
        P = list(f["prep_ready"])
        for r in self.rules:
            extra = None
            if r.id == "on_urgent" and U:
                extra = {"dish": U[0]}
            elif r.id == "on_blocked" and B:
                extra = {"kind": B[0].replace("_", " ")}
            elif r.id == "on_prep_ready" and P:
                extra = {"dish": P[0]}
            if extra is not None:
                self.label = r.label()
                return render(r, "reminder", self.lib, self.k, extra)
        r = self.by_id.get("default_work")
        if r is None:
            return None
        held = view.players[me].held_object
        if r.p["mode"] == "prep":
            # when it could cook or serve: holding a processed item or a dish, or food is ready
            food = any(self.k.utensil_status(view, u)[0] == "finished" and self.k.kind_of[u] in COOK_KINDS
                       for u in self.k.utensils)
            if (held is not None and (held.name == "dish" or held.name not in self.k.raw)) or (held is None and food):
                self.label = r.label()
                return render(r, "reminder", self.lib, self.k)
            return None
        if r.p["mode"] == "station":                       # when choosing, or holding food to serve
            if held is None or held.catagory == "with_dish" or held.name in self.k.recipes and not any(
                    held.name in self.k.recipes[x][1] for x in self.k.recipes):
                self.label = r.label()
                return render(r, "reminder", self.lib, self.k)
            return None
        if held is None:                                   # whole: when choosing what to do next
            self.label = r.label()
            return render(r, "reminder", self.lib, self.k, {"order": getattr(tracker, "own_order", None) or "your order"})
        return None
