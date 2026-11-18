"""LLM persona partner (docs/PERSONA_PARTNER_PLAN_v2.md, section 5).

Shares the ego's stack (``LLMAgent``: observation text, macro actions, executor,
retry and fallback) and adds:

* a **persona**: backstory (voice only) and the card's active rules in the second person,
  with their fallbacks, then the shared priority order;
* an **intention state**: each reply is ``{"goal", "order", "reconsider_when", "action"}``;
  the previous goal is shown at the next decision, so the partner keeps or revises it
  (reset at each episode start);
* **opportunity reminders** (``remind``, default on): at each decision, one line per active
  rule whose opportunity holds now, computed from the partner's own view with the shared
  ``coop.rule_eval.Tracker``. Every reminder is logged;
* ``enforce`` (action filtering) is only a hook: results with it would be a separate,
  labelled simulator variant (plan 5). Not implemented.

No ToM block, no memory across episodes, no probe. A card with no rules is the neutral
partner (the competence baseline). The partner never sees the ego's prompts or memory.
``LLMPartnerPlayer`` adapts it to the runner's partner interface
(``reset(mdp, index, seed, view_radius)``, ``act``, ``observe``, ``summary``, ``events``, ``trace``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .llm_agent import LLMAgent
from .rule_eval import Tracker, is_takeover, make_checks, parse_events, takeover_overridden
from .rules import CONTEXT_RULE_IDS, Card, render

PRIORITY = ("How you decide, in this order: (1) choose a valid action and never block the kitchen; "
            "(2) finish what you are carrying or have started; (3) follow your habits above, in the "
            "order listed; (4) otherwise do what is needed next, oldest order first.")
NEUTRAL_PRIORITY = ("How you decide: choose a valid action and never block the kitchen; finish what you are "
                    "carrying or have started; otherwise do what is needed next, oldest order first.")
INTENTION_REPLY = ('{"goal": "<what you are doing now and why, a few words>", "order": "<the open order you are '
                   'working on: its dish name, or \\"none\\">", "reconsider_when": "<what would make you change '
                   'your goal>", "action": "<one action>"}')
PARTNER_INTRO = ("You are one of two cooks in a kitchen game (an adapted Overcooked). You play a person "
                 "with their own way of working, described below; stay in character. You and your partner "
                 "have the same abilities: either of you can use any station you can reach. You cannot talk "
                 "to your partner; there are no messages.")


def macro_stage(macro: str) -> str:
    """Canonical stage of a macro action (for the partner trace)."""
    m = re.fullmatch(r"(\w+)\((.*)\)", macro or "")
    if not m:
        return "other"
    name, args = m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]
    target = args[-1] if args else ""
    if name == "wait":
        return "wait"
    if name == "place_obj_on_counter":
        return "handover"
    if name in ("deliver_soup", "fill_dish_with_food") or (name == "pickup" and args and args[0] == "dish"):
        return "serve"
    if name in ("cut", "stir") or target.startswith(("chopping_board", "blender")) or target == "ingredient_dispenser":
        return "prep"
    if name in ("cook", "bake") or target.startswith(("pot", "oven")):
        return "cook"
    return "other"


class LLMPartner(LLMAgent):
    name = "llm_partner"

    def __init__(self, backend: Dict[str, Any], card: Card, library: Dict[str, Any],
                 complexity: Optional[int] = None, remind: bool = True, enforce: bool = False,
                 max_tokens: int = 400, partner_log_lines: int = 15, retry_invalid: bool = True,
                 useful_starts_only: bool = False):
        if enforce:
            raise NotImplementedError("partner.enforce is only a hook (plan section 5): implement it if "
                                      "Gate 1 fails, and report its results as a separate simulator variant")
        super().__init__(backend, tom=False, memory=False, probe=False, max_tokens=max_tokens,
                         partner_log_lines=partner_log_lines, retry_invalid=retry_invalid,
                         useful_starts_only=useful_starts_only)
        self.card, self.library = card, library
        self.complexity = complexity
        self.rules = card.active(complexity)
        self.remind = remind
        self.INTRO = PARTNER_INTRO

    # -- episode -------------------------------------------------------------
    def begin_episode(self, mdp, index, context):
        super().begin_episode(mdp, index, context)
        self.intention: Optional[Dict[str, Any]] = None
        self.trace: List[Dict[str, Any]] = []
        self.goal_events: List[Dict[str, Any]] = []      # (LLMAgent.events holds the public kitchen events)
        self.reminder_log: List[Dict[str, Any]] = []
        self.rt: Optional[Tracker] = None
        self.checks = None
        self._prev_view = None
        self.order_dish: Optional[str] = None
        # plan v3: context features every step, and the global deadlock guard
        self.ctx_tracker = None
        self.last_progress = 0
        self.guard_active = False
        self.guards: List[Dict[str, Any]] = []

    # -- prompts ---------------------------------------------------------------
    def _reply_template(self) -> str:
        return INTENTION_REPLY

    def _persona_parts(self) -> List[str]:
        k = self.kitchen
        parts = []
        if self.card.backstory:
            parts.append(f"Who you are: {self.card.backstory}")
        if self.rules:
            lines = []
            for i, r in enumerate(self.rules, 1):
                text = render(r, "text", self.library, k)
                fb = render(r, "fallback", self.library, k)
                lines.append(f"{i}. {text}" + (f" ({fb})" if fb else ""))
            parts.append("Your habits (you always work this way; they are tendencies, not prohibitions):\n"
                         + "\n".join(lines))
            parts.append(PRIORITY)
        else:
            parts.append(NEUTRAL_PRIORITY)
        parts.append("Before each action, state your current goal. Your previous goal is shown to you at the "
                     "next decision: keep it or change it.")
        return parts

    GUARD_STEPS = 30

    def act(self, state, t):
        # Deadlock guard (plan v3, 4): a runner mechanism on the true state, not a habit. If
        # neither cook has completed an interaction for GUARD_STEPS steps while an order is
        # open, the partner is told once to do the next needed step whatever its habits say.
        if any(state.ml_actions or []):
            self.last_progress, self.guard_active = t, False
        if any(r.id in CONTEXT_RULE_IDS for r in self.rules) and not self.guard_active and state.order_list \
                and t - self.last_progress >= self.GUARD_STEPS:
            self.guard_active, self.last_progress = True, t
            self.guards.append({"t": t})
            self.goal_events.append({"t": t, "type": "guard"})
        return super().act(state, t)

    def _observation(self, state, t) -> str:
        text = super()._observation(state, t)
        extra = []
        if self.guard_active:
            extra.append("Nothing has been done in the kitchen for a long time: do the next needed step now, "
                         "whatever your usual way of working is.")
        if self.intention is not None:
            it = self.intention
            extra.append(f"Your current goal (set at t={it['t']}): {it['goal'] or '-'}; order: {it['order'] or '-'}; "
                         f"reconsider when: {it['reconsider_when'] or '-'}.")
        self._reminders_now = self._reminders(state, t) if self.remind else []
        for line in self._reminders_now:
            extra.append(f"Reminder: {line}")
        return text + ("\n" + "\n".join(extra) if extra else "")

    # -- watching (own view) -----------------------------------------------------
    def _watch(self, state, t):
        super()._watch(state, t)
        if not self.rules:
            return
        if self.rt is None:
            k = self.kitchen
            self.rt = Tracker(k, self.context["menu"], {k.index: "partner", k.other: "ego"},
                              self.context.get("open_orders", 1))
            self.rt.start(state)
            self.checks = make_checks(self.rules, self.library, k)
            if any(r.id in CONTEXT_RULE_IDS for r in self.rules):
                from .context import ContextTracker
                self.ctx_tracker = ContextTracker(k, self.context.get("open_orders", 1),
                                                  getattr(self.mdp, "urgent_steps", None) or 10)
        if self.ctx_tracker is not None:
            self.ctx_tracker.observe(state, t)
            self.rt.ctx_now = self.ctx_tracker.now
            self.rt.own_order = (self.intention or {}).get("order") if self.intention else None
        prev = self._prev_view
        if prev is not None and t > 0:
            for ev in parse_events(t - 1, prev, state, self.rt.info, self.rt.roles):
                if is_takeover(ev, self.rt, prev, self.rules) and ev.dishes and \
                        not any(takeover_overridden(self.rules, d, self.rt.order_of.get("partner")) for d in ev.dishes):
                    self.rt.takeovers.append({"t": t - 1, "dishes": set(ev.dishes)})
            self.rt.observe(t - 1, prev, state)
        self._prev_view = state

    def _reminders(self, state, t) -> List[str]:
        if not self.rules or self.rt is None:
            return []
        target = None
        m = self.executor.macro if self.executor is not None else None
        if m is not None:
            target = next((a for a in m.args if a in self.kitchen.utensils), None)
        out = []
        for c in self.checks:
            line = c.opportunity(state, self.rt, self.kitchen.index, target)
            if line:
                out.append(line)
                self.reminder_log.append({"t": t, "rule": c.label})
        return out

    # -- replies -------------------------------------------------------------------
    def _on_reply(self, parsed, record, t):
        if not isinstance(parsed, dict):
            return
        goal = {k: str(parsed.get(k, "")).strip()[:200] for k in ("goal", "order", "reconsider_when")}
        if any(goal.values()):
            record["intention"] = goal

    def _decide(self, state, t):
        super()._decide(state, t)
        rec = self.decisions[-1]
        if rec.get("intention"):
            prev = self.intention
            self.intention = dict(rec["intention"], t=t)
            if prev is None or prev.get("goal") != self.intention["goal"]:
                self.goal_events.append({"t": t, "type": "goal", **rec["intention"]})
        self.trace.append({"t": t, "goal": (self.intention or {}).get("goal"),
                           "order": (self.intention or {}).get("order"), "macro": rec.get("chosen"),
                           "stage": macro_stage(rec.get("chosen", "")),
                           "reminders": [r["rule"] for r in self.reminder_log if r["t"] == t]})

    def summary(self) -> Dict[str, Any]:
        stages: Dict[str, int] = {}
        for tr in self.trace:
            stages[tr["stage"]] = stages.get(tr["stage"], 0) + 1
        reminders: Dict[str, int] = {}
        for r in self.reminder_log:
            reminders[r["rule"]] = reminders.get(r["rule"], 0) + 1
        return {"name": self.card.name, "type": "llm", "rules": [r.label() for r in self.rules],
                "decisions": len(self.decisions), "invalid_actions": self.invalid, "fallbacks": self.fallbacks,
                "usage": self.usage.as_dict(), "stages": stages, "reminders": reminders,
                "goal_changes": len(self.goal_events),
                **({"guards": len(self.guards)} if any(r.id in CONTEXT_RULE_IDS for r in self.rules) else {})}


class LLMPartnerPlayer:
    """Runner adapter: the partner interface on top of ``LLMPartner``."""

    def __init__(self, agent: LLMPartner, context: Dict[str, Any]):
        self.agent, self.base_context = agent, dict(context)
        self.name = agent.card.name

    def reset(self, mdp, index: int, seed: int, view_radius: Optional[int] = None) -> None:
        ctx = dict(self.base_context, index=index, partner_index=1 - index, episode_seed=seed,
                   view_radius=view_radius)
        self.agent.begin_episode(mdp, index, ctx)

    def act(self, state, t):
        return self.agent.act(state, t)

    def observe(self, t, reward, info):
        self.agent.observe(t, reward, info)

    @property
    def executor(self):
        return self.agent.executor

    @property
    def trace(self):
        return self.agent.trace

    @property
    def events(self):
        return self.agent.goal_events

    def summary(self) -> Dict[str, Any]:
        return self.agent.summary()

    def log(self) -> Dict[str, Any]:
        """Partner-side record (saved next to the episode, never shown to the ego)."""
        a = self.agent
        return {"system_prompt": a.system, "calls": a.calls, "decisions": a.decisions, "reminders": a.reminder_log,
                # its own record of the episode (for the day's feedback): what it did, what it saw
                "own_actions": [list(x) for x in a.own_log], "seen_partner_actions": [list(x) for x in a.partner_log],
                "events": list(a.events), "score": a.score}
