"""Multi-day runs: the same partner for several days, the ego's memory carried over.

A *run* is one condition (ego agent settings), one partner persona and one run
seed, played for ``days`` days of ``episodes_per_day`` continuous-order
episodes. The ego's memory carries over from episode to episode and from day
to day (see ``coop.agents``). Every (condition, persona, run seed) is its own
run with its own memory.

Orders for (run seed, day, episode) are the same for every persona and
condition. Each (persona, run seed) is also played by a reference ego
(``reference_agent``, by default the scripted no-memory player). The table
reports each condition's score minus the reference score on the same episodes,
which removes how hard that day's orders were.

When the ego answers a partner probe (``coop.llm_agent``), the answer is scored
against the persona's true profile (``coop.profile``).

    PYTHONPATH=src:lib/overcooked_ai python -m coop.days configs/coop/days_smoke.yaml --out outputs/days_smoke

Output layout::

    OUT/config.json                          resolved config (a resume must match it)
    OUT/results.json                         every episode's metrics + condition x persona x day table
    OUT/runs/<condition>/<persona>/seed<k>/  (condition "_reference" = the reference ego)
        run.json                             this run's episodes
        error.log                            traceback, if the run failed
        day01/ep01/
            summary.json                     orders, deliveries, score, probe score (what happened)
            steps.jsonl                      per-step actions, positions, holdings, events
            partner_truth.json               partner persona, traits, true jobs, slips/abandons
            agent_log.json                   ego-side record (LLM calls, decisions, probe answer)
            memory_after.json                ego memory after this episode (and day)

``partner_truth.json`` is for evaluation only; nothing in it is passed to the ego.
A run resumes from the last episode with both ``summary.json`` and
``memory_after.json``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from . import llm_agent  # noqa: F401  (registers the "llm" agent type)
from .agents import Agent, make_agent
from .env import ContinuousOrderEnv
from .llm import build_backend
from .orders import normalize_menu
from .partner import PERSONAS, SimulatedHuman, Traits
from .kitchen import Kitchen
from .profile import MODEL_SECTIONS, avoid_options, score_profile, truth_profile
from . import rule_partner  # noqa: F401  (registers the "rule_ego" agent type)
from .llm_partner import LLMPartner, LLMPartnerPlayer
from .rule_eval import RuleEvaluator, decided_at, intent_target
from .rule_partner import RulePlayer
from .rules import NEUTRAL, load_cards, load_library, render
from . import feedback as day_feedback_mod
from . import probe2

REFERENCE = "_reference"
CONFIG_KEYS = {
    "name", "layout", "menu", "horizon", "wrong_delivery_penalty", "no_repeat", "days",
    "episodes_per_day", "ego_index", "personas", "custom_personas", "run_seeds", "agent",
    "conditions", "reference_agent", "open_orders", "view_radius", "partner", "deadlines",
    "schedule", "order_lists", "eval_episodes", "knowledge_probe",
}
# Condition keys the runner handles (not passed to the agent).
RUNNER_KEYS = ("memory_source", "feedback", "oracle")
FEEDBACK_LEVELS = ("none", "specific")
PARTNER_TYPES = ("scripted", "rule", "llm")


@dataclass
class DaysConfig:
    name: str = "days"
    layout: str = "new_env"
    menu: Union[List[str], Dict[str, float]] = field(default_factory=lambda: [
        "boiled_egg", "boiled_mushroom", "boiled_corn_slices", "baked_carrot_soup",
        "sliced_eggplant_and_chickpea_stew"])
    horizon: int = 400
    wrong_delivery_penalty: float = 20
    no_repeat: bool = False
    days: int = 5
    episodes_per_day: int = 3
    ego_index: int = 0                  # player slot of the ego; the partner takes the other
    open_orders: int = 1                # orders open at once (any of them may be served)
    view_radius: Optional[int] = None   # partial view for every player (None = full view)
    # Order timers with expiry (plan v3, 3.1): {slack, reference: {dish: steps}, expiry_penalty,
    # urgent_steps}; None (default) = no timers, the original behaviour.
    deadlines: Optional[Dict[str, Any]] = None
    personas: List[str] = field(default_factory=lambda: ["steady"])
    custom_personas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_seeds: List[int] = field(default_factory=lambda: [0])
    # Base ego settings; each condition overrides some keys (nested dicts merge one level).
    agent: Dict[str, Any] = field(default_factory=lambda: {"type": "scripted", "persona": "steady"})
    conditions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"main": {}})
    # Played on the same episodes for the paired comparison; null disables it.
    reference_agent: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"type": "scripted", "persona": "steady"})
    # Who the partner is (docs/PERSONA_PARTNER_PLAN_v2.md, section 5). None = the scripted
    # SimulatedHuman personas (unchanged behaviour). {type: rule | llm, cards: <yaml>,
    # complexity: K, remind: bool, enforce: bool, backend: {...}, max_tokens: n}: personas
    # are card names (or "neutral") played by the rule-based twin or the LLM persona partner.
    partner: Optional[Dict[str, Any]] = None
    # Day structure (plan sections 6-9); the defaults keep the old behaviour.
    schedule: Optional[Dict[Any, Any]] = None   # day -> dishes of that day's interaction episodes, or "all"
    order_lists: str = "random"                 # random | controlled (blocks with every dish once)
    eval_episodes: int = 0                      # held-out episodes on frozen memory: day 0 and after each day
    knowledge_probe: bool = False               # one multiple-choice question per library rule, per day

    def __post_init__(self):
        if self.days < 1 or self.episodes_per_day < 1 or self.horizon < 1:
            raise ValueError("days, episodes_per_day and horizon must be >= 1")
        if self.ego_index not in (0, 1):
            raise ValueError("ego_index must be 0 or 1")
        if self.open_orders < 1:
            raise ValueError("open_orders must be >= 1")
        if self.view_radius is not None and self.view_radius < 0:
            raise ValueError("view_radius must be >= 0 (or null for a full view)")
        if not self.personas or len(set(self.personas)) != len(self.personas):
            raise ValueError(f"personas must be a non-empty list of distinct names: {self.personas}")
        if not self.run_seeds or len(set(self.run_seeds)) != len(self.run_seeds):
            raise ValueError(f"run_seeds must be a non-empty list of distinct ints: {self.run_seeds}")
        clash = set(self.custom_personas) & set(PERSONAS)
        if clash:
            raise ValueError(f"custom_personas reuse built-in names: {sorted(clash)}")
        ptype = self.partner_type()
        if ptype not in PARTNER_TYPES:
            raise ValueError(f"partner.type must be one of {PARTNER_TYPES}, got {ptype!r}")
        if ptype == "scripted":
            for p in self.personas:
                persona_traits(p, self.custom_personas)     # raises on unknown names or bad traits
        else:
            if not self.partner.get("cards"):
                raise ValueError("partner.cards must name a persona-card file")
            cards = self.cards()
            for p in self.personas:
                if p not in cards:
                    raise ValueError(f"persona {p!r} is not a card in {self.partner['cards']}: {sorted(cards)}")
            k = self.partner.get("complexity")
            if k is not None and (not isinstance(k, int) or k < 0):
                raise ValueError("partner.complexity must be a non-negative int (or null for all rules)")
            if ptype == "llm" and not isinstance(self.partner.get("backend"), dict):
                raise ValueError("partner.type llm needs partner.backend")
        for name in list(self.conditions) + ([REFERENCE] if self.reference_agent is not None else []):
            if self.condition_agent(name).get("type") == "rule_ego" and ptype == "scripted":
                raise ValueError(f"condition {name!r}: a rule_ego needs persona cards (partner.type rule or llm)")
        menu = normalize_menu(self.menu)
        with contextlib.redirect_stdout(io.StringIO()):
            mdp = OvercookedGridworld.from_layout_name(self.layout)
        unknown = sorted(set(menu) - set(mdp.Recipe.recipe_list))
        if unknown:
            raise ValueError(f"Orders not producible in layout '{self.layout}': {unknown}")
        if self.order_lists not in ("random", "controlled"):
            raise ValueError("order_lists must be 'random' or 'controlled'")
        if self.eval_episodes < 0:
            raise ValueError("eval_episodes must be >= 0")
        for day in range(1, self.days + 1):
            unknown = sorted(set(self.menu_for(day)) - set(mdp.Recipe.recipe_list))
            if unknown:
                raise ValueError(f"schedule day {day}: orders not producible in '{self.layout}': {unknown}")
        if not self.conditions:
            raise ValueError("conditions must name at least one condition")
        for name in self.conditions:
            run = self.condition_runner(name)
            if run.get("feedback", "none") not in FEEDBACK_LEVELS:
                raise ValueError(f"condition {name!r}: feedback must be one of {FEEDBACK_LEVELS}")
            if run.get("feedback", "none") != "none" and self.partner_type() != "llm":
                raise ValueError(f"condition {name!r}: daily feedback comes from the LLM partner (partner.type llm)")
            if run.get("oracle") and self.partner_type() == "scripted":
                raise ValueError(f"condition {name!r}: oracle needs persona cards")
        for name in self.conditions:
            if not name or name.startswith("_") or "/" in name:
                raise ValueError(f"bad condition name {name!r}")
            if "type" not in self.condition_agent(name):
                raise ValueError(f"condition {name!r}: the agent needs a 'type'")
        if self.reference_agent is not None and "type" not in self.reference_agent:
            raise ValueError("reference_agent needs a 'type'")
        for name in self.conditions:
            src = self.memory_source(name)
            if src is None:
                continue
            if src.get("type") not in ("swap", "placebo"):
                raise ValueError(f"condition {name!r}: memory_source type must be 'swap' or 'placebo'")
            if src.get("donor", "card") not in ("card", "seed"):
                raise ValueError(f"condition {name!r}: memory_source donor must be 'card' or 'seed'")
            if src.get("donor") == "seed" and len(self.run_seeds) < 2:
                raise ValueError(f"condition {name!r}: a donor seed needs at least two run_seeds")
            origin = src.get("from")
            if origin not in self.conditions or origin == name or self.memory_source(origin):
                raise ValueError(f"condition {name!r}: memory_source.from must name another condition "
                                 "of this config that has no memory_source itself")
            if not self.condition_agent(origin).get("memory"):
                raise ValueError(f"condition {name!r}: the source condition {origin!r} keeps no memory")
            if not self.condition_agent(name).get("memory"):
                raise ValueError(f"condition {name!r}: set memory: true so the given notes are shown")
            if src["type"] == "swap" and src.get("donor", "card") == "card":
                donors = self.donors(name)
                for p in self.personas:
                    if donors.get(p) not in self.personas or donors[p] == p:
                        raise ValueError(f"condition {name!r}: persona {p!r} needs a different donor "
                                         f"persona from this config (got {donors.get(p)!r})")

    def partner_type(self) -> str:
        return (self.partner or {}).get("type", "scripted")

    def cards(self):
        if getattr(self, "_cards", None) is None:
            path = Path(self.partner["cards"])
            self._cards = load_cards(path if path.is_absolute() else REPO / path)
        return self._cards

    def card_files(self) -> List[str]:
        """Rule library and card file: part of the code version when the partner uses cards."""
        if self.partner_type() == "scripted":
            return []
        return ["configs/coop/rules/library.yaml", str(self.partner["cards"])]

    def condition_agent(self, name: str) -> Dict[str, Any]:
        if name == REFERENCE:
            return copy.deepcopy(self.reference_agent)
        spec = copy.deepcopy(self.agent)
        for key, value in (self.conditions[name] or {}).items():
            if key in RUNNER_KEYS:
                continue
            if isinstance(value, dict) and isinstance(spec.get(key), dict):
                spec[key] = {**spec[key], **value}
            else:
                spec[key] = copy.deepcopy(value)
        return spec

    def condition_runner(self, name: str) -> Dict[str, Any]:
        """The runner's keys of a condition (memory_source, feedback, oracle)."""
        if name == REFERENCE:
            return {}
        return {k: v for k, v in (self.conditions.get(name) or {}).items() if k in RUNNER_KEYS}

    def menu_for(self, day: int) -> List[str]:
        """Dishes of a day's interaction episodes (schedule), else the whole menu."""
        full = list(normalize_menu(self.menu))
        if not self.schedule:
            return full
        dishes = self.schedule.get(day, self.schedule.get(str(day), "all"))
        return full if dishes in ("all", None) else list(dishes)

    def memory_source(self, name: str) -> Optional[Dict[str, Any]]:
        """Control conditions: memory taken from another condition's runs, not from this run."""
        if name == REFERENCE:
            return None
        return (self.conditions.get(name) or {}).get("memory_source")

    def donors(self, name: str) -> Dict[str, str]:
        """For a swap: whose memory each persona gets (default: the next persona in the list)."""
        src = self.memory_source(name) or {}
        if src.get("donors"):
            return dict(src["donors"])
        ps = self.personas
        return {p: ps[(i + 1) % len(ps)] for i, p in enumerate(ps)}


def persona_traits(name: str, custom: Dict[str, Dict[str, Any]]) -> Traits:
    if name in PERSONAS:
        return PERSONAS[name]
    if name in custom:
        spec = dict(custom[name])
        if "preference" in spec:
            spec["preference"] = tuple(spec["preference"])
        return Traits(**spec)
    raise ValueError(f"Unknown persona {name!r}; built-in: {sorted(PERSONAS)}; custom: {sorted(custom)}")


def load_config(path: Union[str, Path], **overrides) -> DaysConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    unknown = set(raw) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    picked = overrides.pop("conditions", None)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    if picked:
        all_conditions = raw.get("conditions") or {"main": {}}
        missing = [c for c in picked if c not in all_conditions]
        if missing:
            raise ValueError(f"Unknown conditions {missing}; config has {sorted(all_conditions)}")
        raw["conditions"] = {c: all_conditions[c] for c in picked}
    return DaysConfig(**raw)


def episode_seed(run_seed: int, day: int, episode: int) -> int:
    """Stable seed for (run seed, day, episode), the same for every persona and condition."""
    digest = hashlib.sha256(f"coop-days:{run_seed}:{day}:{episode}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------- #
# One episode
# ---------------------------------------------------------------------- #
def _player_view(p) -> Dict[str, Any]:
    held = p.held_object
    return {"pos": list(p.position), "dir": list(p.orientation),
            "held": None if held is None else {"name": held.name, "category": held.catagory}}


def _jsonable_action(a):
    return a if isinstance(a, str) else list(a)


def play_episode(cfg: DaysConfig, agent: Agent, persona: str, run_seed: int, day: int,
                 episode: int, ep_dir: Path, kind: str = "interaction") -> Dict[str, Any]:
    """One episode. ``kind="eval"``: a held-out episode (plan 6): the same seed every day,
    the whole menu, and the agent does not learn from it (no end_episode)."""
    if kind == "eval":
        seed = episode_seed(run_seed, 0, 100 + episode)
        menu = cfg.menu
    else:
        seed = episode_seed(run_seed, day, episode)
        menu = cfg.menu_for(day) if cfg.schedule else cfg.menu
    ego, other = cfg.ego_index, 1 - cfg.ego_index
    env = ContinuousOrderEnv(menu, layout=cfg.layout, horizon=cfg.horizon,
                             wrong_delivery_penalty=cfg.wrong_delivery_penalty, no_repeat=cfg.no_repeat,
                             open_orders=cfg.open_orders, order_mode=cfg.order_lists,
                             deadlines=cfg.deadlines)
    start = env.reset(seed)
    context = {"day": day, "episode": episode, "episode_seed": seed, "layout": cfg.layout,
               "horizon": cfg.horizon, "menu": normalize_menu(menu), "index": ego,
               "partner_index": other, "wrong_delivery_penalty": cfg.wrong_delivery_penalty,
               "open_orders": cfg.open_orders, "view_radius": cfg.view_radius}
    partner, rules, traits = make_partner(cfg, persona, context)
    partner.reset(env.mdp, other, seed, view_radius=cfg.view_radius)    # same view rules for everyone
    evaluator = None
    if rules is not None:
        evaluator = RuleEvaluator(rules, load_library(), Kitchen(env.mdp, ego, start), normalize_menu(menu),
                                  ego, other, cfg.open_orders, env.mdp, view_radius=cfg.view_radius)
    agent.begin_episode(env.mdp, ego, context)
    ep_dir.mkdir(parents=True, exist_ok=True)
    with open(ep_dir / "steps.jsonl", "w") as steps:
        while not env.done:
            t, state = env.t, env.state
            moves = [None, None]
            moves[ego] = agent.act(state, t)
            moves[other] = partner.act(state, t)
            order = env.current_order
            pre = state
            intents = None
            if evaluator is not None:
                kit = evaluator.tracker.k
                intents = {"ego": intent_target(agent, kit), "partner": intent_target(partner, kit),
                           "decided": {"ego": decided_at(agent), "partner": decided_at(partner)}}
            state, reward, _, info = env.step([m[0] for m in moves], [m[1] for m in moves])
            agent.observe(t, reward, info)
            if hasattr(partner, "observe"):
                partner.observe(t, reward, info)
            if evaluator is not None:
                evaluator.observe(t, pre, state, intents)
            steps.write(json.dumps({
                "t": t, "order": order,
                "actions": [_jsonable_action(m[0]) for m in moves], "params": [m[1] for m in moves],
                "completed": list(state.ml_actions), "reward": reward, "score": env.score,
                "players": [_player_view(p) for p in state.players],
                "deliveries": [asdict(d) for d in info["deliveries"]], "rejected": info["rejected"],
            }) + "\n")
    summary = dict(env.summary(), day=day, episode=episode, run_seed=run_seed, persona=persona,
                   ego_index=ego)
    if kind != "interaction":
        summary["kind"] = kind
    # The ego learns from what happened in the kitchen, never from the partner's
    # persona, traits or true jobs.
    public = {k: v for k, v in summary.items() if k != "persona"}
    if kind == "interaction":
        agent.end_episode(dict(public, context=context))
    agent_log = agent.episode_log()
    probe = agent_log.get("probe") if isinstance(agent_log, dict) else None
    psummary = partner.summary()
    if traits is not None:
        # Scripted personas. Truth for this episode: avoids is asked only over utensils the
        # orders needed and both players can reach (from the map and orders, not the agent).
        dishes = [o["name"] for o in summary["orders"]]
        avoid = avoid_options(Kitchen(env.mdp, ego, start), dishes)
        truth = truth_profile(traits, avoid, dishes if cfg.open_orders > 1 else None)
        if probe and probe.get("partner_profile"):
            summary["probe"] = score_profile(probe["partner_profile"], truth,
                                             probe.get("stage_shares"),
                                             psummary["stage_share_when_acting"])
    if isinstance(agent_log, dict) and "usage" in agent_log:
        summary["llm_usage"] = dict(agent_log["usage"], invalid_actions=agent_log.get("invalid_actions"),
                                    fallbacks=agent_log.get("fallbacks"))
    if cfg.partner_type() != "scripted":
        summary["partner_type"] = cfg.partner_type()
        summary["rules"] = evaluator.summary()
        if cfg.partner_type() == "llm":
            summary["partner_llm"] = {k: psummary.get(k) for k in
                                      ("usage", "invalid_actions", "fallbacks", "decisions", "reminders",
                                       "goal_changes")}
        with open(ep_dir / "rule_events.jsonl", "w") as f:
            for j in evaluator.judgments:
                f.write(json.dumps(asdict(j)) + "\n")
    (ep_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if traits is not None:
        truth_doc = {"persona": persona, "index": other, "traits": asdict(traits),
                     "profile": truth, "avoid_options": avoid, "summary": psummary,
                     "events": partner.events, "trace": partner.trace}
    else:
        truth_doc = {"persona": persona, "index": other, "type": cfg.partner_type(),
                     "card": cfg.cards()[persona].to_dict(cfg.partner.get("complexity")),
                     "summary": psummary, "events": partner.events, "trace": partner.trace}
    (ep_dir / "partner_truth.json").write_text(json.dumps(truth_doc, default=str))
    if hasattr(partner, "log"):
        (ep_dir / "partner_log.json").write_text(json.dumps(partner.log(), default=str))
    (ep_dir / "agent_log.json").write_text(json.dumps(agent_log))
    return summary


def make_partner(cfg: DaysConfig, persona: str, context: Dict[str, Any]):
    """(partner player, active rules or None, scripted traits or None) for this persona."""
    ptype = cfg.partner_type()
    if ptype == "scripted":
        traits = persona_traits(persona, cfg.custom_personas)
        return SimulatedHuman(traits, persona), None, traits
    card = cfg.cards()[persona]
    k = cfg.partner.get("complexity")
    rules = card.active(k)
    if ptype == "rule":
        return RulePlayer(rules, normalize_menu(cfg.menu), "partner", persona), rules, None
    p = cfg.partner
    agent = LLMPartner(p["backend"], card, load_library(), complexity=k, remind=p.get("remind", True),
                       enforce=p.get("enforce", False), max_tokens=p.get("max_tokens", 400))
    ctx = {key: v for key, v in context.items() if key not in ("index", "partner_index")}
    return LLMPartnerPlayer(agent, ctx), rules, None


def oracle_kitchen(cfg: DaysConfig) -> Kitchen:
    """The layout's kitchen from the ego's slot (for rendering rule texts)."""
    env = ContinuousOrderEnv(cfg.menu, layout=cfg.layout, open_orders=cfg.open_orders)
    with contextlib.redirect_stdout(io.StringIO()):
        start = env.reset(0)
    return Kitchen(env.mdp, cfg.ego_index, start)


def resolve_agent_spec(cfg: DaysConfig, condition: str, persona: str) -> Dict[str, Any]:
    """The ego spec for one run. A ``rule_ego`` gets the rules of its card: ``align: own``
    (this persona's card), ``other`` (the next persona's card, or ``card: <name>``)."""
    spec = cfg.condition_agent(condition)
    if cfg.condition_runner(condition).get("oracle"):
        # The one exception to "the ego never sees the card": what the partner wants from it.
        card = cfg.cards()[persona]
        lib = load_library()
        spec["oracle_wants"] = [render(r, "wants", lib, oracle_kitchen(cfg)) for r in card.active(cfg.partner.get("complexity"))]
    if spec.get("type") != "rule_ego":
        return spec
    align = spec.get("align", "own")
    cards = cfg.cards()
    if spec.get("card"):
        name = spec["card"]
    elif align == "own":
        name = persona
    elif align == "other":
        ps = [p for p in cfg.personas if p != NEUTRAL] or cfg.personas
        name = ps[(ps.index(persona) + 1) % len(ps)] if persona in ps else ps[0]
    else:
        raise ValueError(f"rule_ego align must be own or other, got {align!r}")
    rules = cards[name].active(cfg.partner.get("complexity"))
    return {"type": "rule_ego", "rules": [r.to_dict() for r in rules], "menu": normalize_menu(cfg.menu),
            "card": name, "align": align}


# ---------------------------------------------------------------------- #
# One run (condition x persona x run seed)
# ---------------------------------------------------------------------- #
def run_dir_for(out: Path, condition: str, persona: str, run_seed: int) -> Path:
    return out / "runs" / condition / persona / f"seed{run_seed}"


def ep_dir_for(run_dir: Path, day: int, episode: int) -> Path:
    return run_dir / f"day{day:02d}" / f"ep{episode:02d}"


PLACEBO_SENTENCES = (
    "Orders are opened one at a time and the next one appears after a correct delivery.",
    "Raw ingredients come from the ingredient dispenser and dishes from the dish dispenser.",
    "A utensil has to be started after it is loaded, and it takes a few steps to finish.",
    "Some dishes are served on a dish and others are served without one.",
    "Each counter holds one item, and items can be left on counters.",
    "A correct delivery gives 20 points and a wrong delivery costs points.",
)
# With several open orders the first sentence above is false; this one replaces it there.
PLACEBO_MANY_ORDERS = "Orders are listed oldest first, and a correct delivery opens a new one."


def placebo_sentences(open_orders: int = 1) -> tuple:
    return PLACEBO_SENTENCES if open_orders <= 1 else (PLACEBO_MANY_ORDERS,) + PLACEBO_SENTENCES[1:]


def placebo_text(length: int, open_orders: int = 1) -> str:
    """Neutral kitchen facts (all already in the rules) of about ``length`` characters."""
    sentences = placebo_sentences(open_orders)
    out, i = "", 0
    while len(out) < length:
        out = (out + " " if out else "") + sentences[i % len(sentences)]
        i += 1
    cut = out[:length]
    return cut.rsplit(" ", 1)[0] if " " in cut and len(out) > length else cut


def placebo_model(model: str, open_orders: int = 1, sections=None) -> str:
    """A partner-model file with the same headings and about the same length, holding
    neutral kitchen facts instead of anything about the partner."""
    sections = tuple(sections or MODEL_SECTIONS)
    base = placebo_sentences(open_orders)
    body = max(len(model) - sum(len(h) + 2 for h in sections), 3 * len(base[0]))
    sentences = [s if s.endswith(".") else s + "." for s in re.split(r"(?<=\.) ", placebo_text(body, open_orders)) if s]
    parts = [sentences[i::len(sections)] or [base[i % len(base)]] for i in range(len(sections))]
    return "\n\n".join(h + "\n" + "\n".join(f"- {s}" for s in p) for h, p in zip(sections, parts))


def _previous(cfg: DaysConfig, day: int, ep: int) -> Optional[tuple]:
    if ep > 1:
        return day, ep - 1
    if day > 1:
        return day - 1, cfg.episodes_per_day
    return None


def given_memory(cfg: DaysConfig, out: Path, condition: str, persona: str, run_seed: int,
                 day: int, ep: int) -> Dict[str, Any]:
    """Memory a control condition starts episode (day, ep) with.

    swap:    the notes (or partner-model file) the source condition had built with a
             different partner (the donor persona) after the same number of episodes.
    placebo: the notes (or file) the source condition had built with this partner,
             replaced by neutral text of the same length, and with no profile.
    """
    src = cfg.memory_source(condition)
    daily = cfg.condition_agent(src["from"]).get("memory_mode") == "model_daily"
    if daily:
        # Daily file: replaced at the start of each day with the donor's file after the
        # previous day; the run keeps its own notes and receives its own partner's feedback.
        if ep != 1:
            return {}
        prev = (day - 1, cfg.episodes_per_day) if day > 1 else None
    else:
        prev = _previous(cfg, day, ep)
    if prev is None:
        return {"partner_model": None, "memory_source": src["type"]} if daily else \
            {"notes": [], "partner_model": None, "memory_source": src["type"]}
    donor, donor_seed = persona, run_seed
    if src["type"] == "swap":
        if src.get("donor", "card") == "seed":
            seeds = cfg.run_seeds
            donor_seed = seeds[(seeds.index(run_seed) + 1) % len(seeds)]
        else:
            donor = cfg.donors(condition)[persona]
    path = ep_dir_for(run_dir_for(out, src["from"], donor, donor_seed), *prev) / "memory_after.json"
    if not path.exists():
        raise FileNotFoundError(f"{condition}: needs {path} (run the source condition first)")
    source = json.loads(path.read_text())
    if daily:
        model = source.get("partner_model")
        if src["type"] == "placebo" and model:
            from .llm_agent import DAILY_SECTIONS
            model = placebo_model(model, cfg.open_orders, DAILY_SECTIONS)
        return {"partner_model": model, "memory_source": src["type"],
                "source": {"condition": src["from"], "persona": donor, "seed": donor_seed, "after": list(prev)}}
    notes = source.get("notes") or []
    model = source.get("partner_model")
    if src["type"] == "placebo":
        notes = [{"day": n["day"], "episode": n["episode"],
                  "profile": {k: None for k in (n.get("profile") or {})},
                  "notes": placebo_text(len(n.get("notes") or ""), cfg.open_orders)} for n in notes]
        model = placebo_model(model, cfg.open_orders) if model else model
    return {"notes": copy.deepcopy(notes), "partner_model": model, "memory_source": src["type"],
            "source": {"condition": src["from"], "persona": donor, "after": list(prev)}}


def _done(ep_dir: Path) -> bool:
    return (ep_dir / "summary.json").exists() and (ep_dir / "memory_after.json").exists()


def eval_dir_for(run_dir: Path, day: int, episode: int) -> Path:
    return run_dir / f"day{day:02d}" / f"eval{episode:02d}"


def run_one(cfg_dict: Dict[str, Any], out: str, condition: str, persona: str, run_seed: int,
            quiet: bool = True) -> Dict[str, Any]:
    """Play (or resume) every episode of one run. Picklable entry point for workers.

    With ``eval_episodes`` / ``knowledge_probe``: held-out episodes and the knowledge probe on
    a frozen copy of the ego's memory before day 1 (day 0) and after each day's end. With a
    condition's ``feedback``: the partner's day feedback, given to ``agent.end_day``.
    Feedback, consolidation, probe and evaluation are each written once and read back on
    resume.
    """
    cfg = DaysConfig(**cfg_dict)
    run_dir = run_dir_for(Path(out), condition, persona, run_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes: List[Dict[str, Any]] = []
    evals: List[Dict[str, Any]] = []
    resumed = 0
    sink = (lambda: io.StringIO()) if quiet else (lambda: sys.stdout)
    try:
        spec = resolve_agent_spec(cfg, condition, persona)
        runner = cfg.condition_runner(condition)
        fb_level = runner.get("feedback", "none")
        agent = make_agent(spec)
        agent.set_memory({})
        # Does the ego's memory change across days? If not, held-out results are the same
        # every day: evaluate once (day 0).
        cross_day = bool(spec.get("memory")) or fb_level != "none"
        exposure: Dict[str, int] = {}

        def evaluate(day: int) -> None:
            if day > 0 and not cross_day:
                return
            snap = agent.get_memory()
            for e in range(1, cfg.eval_episodes + 1):
                d = eval_dir_for(run_dir, day, e)
                if (d / "summary.json").exists():
                    evals.append(_metrics(json.loads((d / "summary.json").read_text())))
                    continue
                if d.exists():
                    shutil.rmtree(d)
                agent.set_memory(snap)
                with contextlib.redirect_stdout(sink()):
                    summary = play_episode(cfg, agent, persona, run_seed, day, e, d, kind="eval")
                evals.append(_metrics(summary))
            agent.set_memory(snap)
            probe_path = run_dir / f"day{day:02d}" / "knowledge_probe.json"
            if cfg.knowledge_probe and hasattr(agent, "knowledge_probe") and not probe_path.exists() \
                    and getattr(agent, "kitchen", None) is not None and cfg.partner_type() != "scripted":
                card = cfg.cards()[persona]
                rules = card.active(cfg.partner.get("complexity"))
                res = probe2.run(agent, cfg, rules, dict(exposure))
                probe_path.parent.mkdir(parents=True, exist_ok=True)
                probe_path.write_text(json.dumps(dict(res, day=day), indent=1))
            agent.set_memory(snap)

        if cfg.eval_episodes or cfg.knowledge_probe:
            evaluate(0)
        for day in range(1, cfg.days + 1):
            day_results = []
            for ep in range(1, cfg.episodes_per_day + 1):
                ep_dir = ep_dir_for(run_dir, day, ep)
                last = ep == cfg.episodes_per_day
                if _done(ep_dir):
                    summary = json.loads((ep_dir / "summary.json").read_text())
                    agent.set_memory(json.loads((ep_dir / "memory_after.json").read_text()))
                    resumed += 1
                else:
                    if ep_dir.exists():
                        shutil.rmtree(ep_dir)        # partial episode from an interrupted run
                    if last:
                        for name in ("feedback.json", "consolidation.json"):
                            (run_dir / f"day{day:02d}" / name).unlink(missing_ok=True)
                    if cfg.memory_source(condition):
                        # Control: the memory comes from another run, not from this run's own notes.
                        given = given_memory(cfg, Path(out), condition, persona, run_seed, day, ep)
                        if given:
                            agent.set_memory(dict(agent.get_memory(), **given))
                    with contextlib.redirect_stdout(sink()):  # the legacy MDP prints every step
                        summary = play_episode(cfg, agent, persona, run_seed, day, ep, ep_dir)
                    if last:
                        feedback = None
                        if fb_level != "none":
                            feedback = day_feedback_mod.for_day(cfg, persona, run_dir, day, run_seed)
                        if feedback is None:
                            agent.end_day(day, day_results + [summary])
                        else:
                            agent.end_day(day, day_results + [summary], feedback=feedback["text"])
                        log = getattr(agent, "day_log", None)
                        if log:
                            (run_dir / f"day{day:02d}" / "consolidation.json").write_text(
                                json.dumps(log, indent=1, default=str))
                    (ep_dir / "memory_after.json").write_text(json.dumps(agent.get_memory()))
                for rule, row in (summary.get("rules") or {}).items():
                    exposure[rule] = exposure.get(rule, 0) + row.get("exposure", 0)
                day_results.append(summary)
                episodes.append(_metrics(summary))
            if cfg.eval_episodes or cfg.knowledge_probe:
                evaluate(day)
        status = "ok"
        (run_dir / "error.log").unlink(missing_ok=True)
    except Exception:
        status = "error"
        (run_dir / "error.log").write_text(traceback.format_exc())
    record = {"condition": condition, "persona": persona, "run_seed": run_seed, "status": status,
              "episodes_resumed": resumed, "episodes": episodes}
    if evals:
        record["eval_episodes"] = evals
    (run_dir / "run.json").write_text(json.dumps(record, indent=1))
    return record


def _metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("day", "episode", "score", "orders_completed", "wrong_deliveries",
            "rejected_interactions", "mean_order_steps", "seed")
    out = {k: summary.get(k) for k in keys}
    probe = summary.get("probe")
    if probe:
        out["probe_accuracy"] = probe["accuracy"]
        out["probe_correct"] = probe["correct"]
    usage = summary.get("llm_usage")
    if usage:
        out["llm_calls"] = usage.get("calls")
        out["invalid_actions"] = usage.get("invalid_actions")
    return out


# ---------------------------------------------------------------------- #
# All runs of a config
# ---------------------------------------------------------------------- #
# Files whose changes alter how episodes are played (analysis/screening code is left out).
CODE_FILES = ["src/coop/" + f for f in ("agents.py", "days.py", "env.py", "executor.py", "kitchen.py",
                                        "llm.py", "llm_agent.py", "nav.py", "orders.py", "partner.py",
                                        "profile.py", "view.py", "rules.py", "rule_eval.py",
                                        "rule_partner.py", "llm_partner.py")] + ["lib/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py"]
REPO = Path(__file__).resolve().parents[2]


def write_run_meta(cfg: DaysConfig, out: Path) -> None:
    """Models, serving, decoding and every prompt template with its hash (plan invariant 8)."""
    from .llm_agent import GRAMMAR, HANDOVER_RULES, LLMAgent
    from .llm_partner import INTENTION_REPLY, NEUTRAL_PRIORITY, PARTNER_INTRO, PRIORITY
    from .profile import PROFILE_GUIDE
    env_keys = ["ICRL_LLM_MODEL", "ICRL_LLM_BASE_URL", "COOP_CHAT_TEMPLATE_KWARGS", "ICRL_PARTNER_MODEL",
                "ICRL_PARTNER_BASE_URL", "COOP_PARTNER_CHAT_TEMPLATE_KWARGS", "COOP_SERVING_VERSION",
                "LLM_MODEL_PATH"]
    templates = {"ego_intro": LLMAgent.INTRO, "grammar": GRAMMAR, "handover_rules": HANDOVER_RULES,
                 "probe_guide": PROFILE_GUIDE, "partner_intro": PARTNER_INTRO, "partner_priority": PRIORITY,
                 "partner_neutral_priority": NEUTRAL_PRIORITY, "partner_reply": INTENTION_REPLY}
    if cfg.partner_type() != "scripted":
        templates["rule_library"] = (REPO / "configs/coop/rules/library.yaml").read_text()
        cards = Path(cfg.partner["cards"])
        templates["persona_cards"] = (cards if cards.is_absolute() else REPO / cards).read_text()
    meta = {"written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ego_backends": {c: cfg.condition_agent(c).get("backend") for c in cfg.conditions
                             if cfg.condition_agent(c).get("type") == "llm"},
            "partner": cfg.partner, "env": {k: os.environ.get(k) for k in env_keys if os.environ.get(k)},
            "prompt_templates": {k: {"sha256": hashlib.sha256(v.encode()).hexdigest()[:16], "text": v}
                                 for k, v in templates.items()}}
    path = out / "run_meta.json"
    history = json.loads(path.read_text()).get("history", []) if path.exists() else []
    old = json.loads(path.read_text()) if path.exists() else None
    if old is not None and {k: old.get(k) for k in ("partner", "prompt_templates", "env")} != \
            {k: meta.get(k) for k in ("partner", "prompt_templates", "env")}:
        history.append({k: old.get(k) for k in ("written", "env")})
    meta["history"] = history
    path.write_text(json.dumps(meta, indent=1))


def code_version(layout: str, extra: Optional[List[str]] = None) -> str:
    """Hash of the code and layout that play episodes, to stop a resume from mixing versions.
    ``extra``: files a config's partner depends on (rule library, persona cards)."""
    h = hashlib.sha256()
    files = CODE_FILES + [f"lib/overcooked_ai/overcooked_ai_py/data/layouts/{layout}.layout"] + list(extra or [])
    for rel in files:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    for f in sorted((REPO / "src" / "prompts" / "recipe").glob("*.txt")):
        h.update(f.read_bytes())
    for f in sorted((REPO / "src" / "prompts" / "recipe_coop").glob("*.txt")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def run_all(cfg: DaysConfig, out: Union[str, Path], workers: int = 1, log=print,
            allow_code_change: bool = False) -> Dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        new = json.loads(json.dumps(cfg_dict))
        # Keys added to DaysConfig after the folder was started, still at their default value
        # (e.g. partner: null, order_lists: random, eval_episodes: 0), are the same config.
        defaults = json.loads(json.dumps(asdict(DaysConfig())))
        for key in [k for k in new if k not in old and new[k] == defaults.get(k)]:
            del new[key]
        if old != new:
            raise ValueError(f"{cfg_path} holds a different config; use a new --out to start over")
    else:
        cfg_path.write_text(json.dumps(cfg_dict, indent=1))
    version = code_version(cfg.layout, cfg.card_files())
    ver_path = out / "code_version.json"
    if ver_path.exists():
        recorded = json.loads(ver_path.read_text())
        if recorded.get("hash") != version and not allow_code_change:
            raise ValueError(
                f"The code that plays episodes changed since {out} was started ({recorded.get('hash')} -> "
                f"{version}); finishing it would mix two versions. Use a new --out, or pass "
                "--allow-code-change if the change cannot affect results.")
        if recorded.get("hash") != version:
            log(f"warning: code changed ({recorded.get('hash')} -> {version}); continuing as asked")
            recorded.setdefault("history", []).append(version)
            recorded["hash"] = version
            ver_path.write_text(json.dumps(recorded, indent=1))
    else:
        ver_path.write_text(json.dumps({"hash": version, "files": CODE_FILES}, indent=1))
    conditions = list(cfg.conditions) + ([REFERENCE] if cfg.reference_agent is not None else [])
    for c in conditions:      # fail before any run when a backend is misconfigured (e.g. no server URL)
        spec = cfg.condition_agent(c)
        if spec.get("type") == "llm":
            build_backend(spec.get("backend") or {}, f"condition {c}")
    if cfg.partner_type() == "llm":
        build_backend(cfg.partner.get("backend") or {}, "partner")
    write_run_meta(cfg, out)
    # Control conditions read memory from finished runs of their source condition: second phase.
    phases = [[c for c in conditions if not cfg.memory_source(c)],
              [c for c in conditions if cfg.memory_source(c)]]
    total = len(conditions) * len(cfg.personas) * len(cfg.run_seeds)
    records = []
    started = time.time()
    for phase in phases:
        jobs = [(c, p, s) for c in phase for p in cfg.personas for s in cfg.run_seeds]
        if not jobs:
            continue
        if workers <= 1:
            for c, p, s in jobs:
                records.append(run_one(cfg_dict, str(out), c, p, s))
                log(_progress(records[-1], len(records), total, started))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_one, cfg_dict, str(out), c, p, s) for c, p, s in jobs]
                for f in as_completed(futures):
                    records.append(f.result())
                    log(_progress(records[-1], len(records), total, started))
    records.sort(key=lambda r: (conditions.index(r["condition"]), cfg.personas.index(r["persona"]),
                                r["run_seed"]))
    results = {"config": cfg_dict, "runs": records, "table": day_table(cfg, records)}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


def _progress(rec, done, total, started) -> str:
    return (f"[{done}/{total}] {rec['condition']} {rec['persona']} seed{rec['run_seed']}: {rec['status']}, "
            f"{len(rec['episodes'])} episodes ({rec['episodes_resumed']} resumed), "
            f"{time.time() - started:.0f}s")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def day_table(cfg: DaysConfig, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """condition -> persona -> day -> means over run seeds and the day's episodes.

    ``vs_reference`` is the mean of (score - reference score) over episodes both
    played, i.e. on identical orders.
    """
    ref_score = {}
    for r in records:
        if r["condition"] == REFERENCE and r["status"] == "ok":
            for e in r["episodes"]:
                ref_score[(r["persona"], r["run_seed"], e["day"], e["episode"])] = e["score"]
    table: Dict[str, Any] = {}
    for cond in list(cfg.conditions) + ([REFERENCE] if cfg.reference_agent is not None else []):
        table[cond] = {}
        for persona in cfg.personas:
            eps = [(r["run_seed"], e) for r in records
                   if r["condition"] == cond and r["persona"] == persona and r["status"] == "ok"
                   for e in r["episodes"]]
            rows = {}
            for day in range(1, cfg.days + 1):
                d = [(s, e) for s, e in eps if e["day"] == day]
                if not d:
                    continue
                diffs = [e["score"] - ref_score[(persona, s, day, e["episode"])] for s, e in d
                         if (persona, s, day, e["episode"]) in ref_score]
                row = {"episodes": len(d),
                       "score": _mean(e["score"] for _, e in d),
                       "orders": _mean(e["orders_completed"] for _, e in d),
                       "wrong": _mean(e["wrong_deliveries"] for _, e in d),
                       "vs_reference": _mean(diffs) if cond != REFERENCE else None,
                       "probe_accuracy": _mean(e.get("probe_accuracy") for _, e in d)}
                probes = [e["probe_correct"] for _, e in d if e.get("probe_correct")]
                if probes:
                    fields = sorted({k for p in probes for k in p})
                    row["probe_field_accuracy"] = {k: _mean(float(p[k]) for p in probes if k in p)
                                                   for k in fields}
                rows[str(day)] = row
            table[cond][persona] = rows
    return table


def format_table(cfg: DaysConfig, table) -> str:
    lines = ["mean score per day; [score minus reference ego on the same episodes]; "
             "{partner probe accuracy}"]
    for cond, personas in table.items():
        lines.append(f"\n{cond}")
        lines.append("  persona       " + "".join(f"{'day ' + str(d):>26}" for d in range(1, cfg.days + 1)))
        for persona, rows in personas.items():
            cells = []
            for d in range(1, cfg.days + 1):
                r = rows.get(str(d))
                if not r:
                    cells.append("-")
                    continue
                cell = f"{r['score']:.1f}"
                if r["vs_reference"] is not None:
                    cell += f" [{r['vs_reference']:+.1f}]"
                if r["probe_accuracy"] is not None:
                    cell += f" {{{r['probe_accuracy']:.2f}}}"
                cells.append(cell)
            lines.append(f"  {persona:<14}" + "".join(f"{c:>26}" for c in cells))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-day runs with a simulated human partner.")
    ap.add_argument("config", help="YAML or JSON config (see configs/coop/)")
    ap.add_argument("--out", required=True, help="output directory; rerun with the same one to resume")
    ap.add_argument("--workers", type=int, default=1, help="runs in parallel processes")
    ap.add_argument("--conditions", nargs="+", help="run only these conditions of the config")
    ap.add_argument("--personas", nargs="+")
    ap.add_argument("--run-seeds", nargs="+", type=int)
    ap.add_argument("--days", type=int)
    ap.add_argument("--episodes-per-day", type=int)
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--layout", help="layout file name: new_env (split), open_env, open2_env (open kitchens)")
    ap.add_argument("--open-orders", type=int, help="orders open at once")
    ap.add_argument("--view-radius", type=int, help="partial view for every player")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume even though the episode-playing code changed since the run started")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, conditions=args.conditions, personas=args.personas,
                      run_seeds=args.run_seeds, days=args.days, episodes_per_day=args.episodes_per_day,
                      horizon=args.horizon, layout=args.layout, open_orders=args.open_orders,
                      view_radius=args.view_radius)
    results = run_all(cfg, args.out, args.workers, log=lambda s: print(s, file=sys.stderr, flush=True),
                      allow_code_change=args.allow_code_change)
    failed = [r for r in results["runs"] if r["status"] != "ok"]
    print(format_table(cfg, results["table"]))
    print(f"wrote {Path(args.out) / 'results.json'}")
    if failed:
        print(f"{len(failed)} run(s) failed; see runs/<condition>/<persona>/seed<k>/error.log",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
