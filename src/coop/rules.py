"""Persona rules: the library, persona cards, contradiction checks and text rendering.

A persona card (configs/coop/personas/*.yaml) is an ordered list of rules from the
library (configs/coop/rules/library.yaml) plus a backstory for voice. At complexity K
the first K rules are active. The rules' predicates (opportunity / adherent / aligned)
live in ``coop.rule_eval``; the rule-following scripted players in ``coop.rule_partner``.
See docs/PERSONA_PARTNER_PLAN_v2.md, section 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
LIBRARY_PATH = REPO / "configs" / "coop" / "rules" / "library.yaml"
CATEGORIES = ("station", "dish", "order", "stage", "hand-over", "reaction", "initiative", "context")
# Rules that may appear at most once per card (own_station: once per station kind).
SINGLE = {"order_priority", "serves", "wants_supplies", "handover_spot", "on_takeover", "on_contention",
          "waits_for_commit", "default_work", "on_urgent", "on_blocked", "on_prep_ready"}
# Context rules (docs/CONTEXT_PARTNER_PLAN_v3.md, 4): what the partner does in a context, no fallback.
CONTEXT_RULE_IDS = ("default_work", "on_urgent", "on_blocked", "on_prep_ready")
# Keys of a library entry that may carry a per-parameter variant suffix (text_self, ...).
TEXT_FIELDS = ("text", "fallback", "wants", "reminder")
# Optional parameters (the library lists them under ``options``); the first value is the
# default. A default is not stored in the rule, so labels and texts of cards that do not set
# the option are unchanged ("waits for its things" variants, 2026-11-17).
OPTIONS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "own_station": {"when_taken": ("switch", "wait")},
    "split_dish": {"patience": ("20", "40", "60")},
    "serves": {"patience": ("20", "40", "60")},
    "waits_for_commit": {"patience": ("5", "10")},
    "on_urgent": {"unless_ego_on_it": ("no", "yes")},
}
OWN_STATION_PATIENCE = 20     # own_station(when_taken=wait): use the other station after this many steps


def _yaml(path: Path) -> Any:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def load_library(path: Path = LIBRARY_PATH) -> Dict[str, Dict[str, Any]]:
    lib = _yaml(path) or {}
    for rid, spec in lib.items():
        if spec.get("category") not in CATEGORIES:
            raise ValueError(f"rule {rid}: unknown category {spec.get('category')!r}")
    return lib


@dataclass(frozen=True)
class Rule:
    id: str
    params: Tuple[Tuple[str, str], ...] = ()

    @property
    def p(self) -> Dict[str, str]:
        return dict(self.params)

    def opt(self, name: str) -> str:
        """An optional parameter's value (its default when the card does not set it)."""
        return self.p.get(name, OPTIONS[self.id][name][0])

    def label(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params)
        return f"{self.id}({args})" if args else self.id

    def to_dict(self) -> Dict[str, Any]:
        return {self.id: self.p}


@dataclass
class Card:
    name: str
    backstory: str = ""
    rules: List[Rule] = field(default_factory=list)
    # Two-axis cards (docs/STYLE_PARTNER_PLAN_v4.md, 2): latent labels {workflow, adherence},
    # per context {mode, adherence, identifiable}, and the backstory variants. None for rule cards.
    latent: Optional[Dict[str, Any]] = None
    contexts: Optional[Dict[str, Dict[str, Any]]] = None
    backstories: Optional[Dict[str, str]] = None

    def active(self, complexity: Optional[int] = None) -> List[Rule]:
        return list(self.rules if complexity is None else self.rules[:complexity])

    def to_dict(self, complexity: Optional[int] = None) -> Dict[str, Any]:
        out = {"name": self.name, "backstory": self.backstory, "complexity": complexity,
               "rules": [r.to_dict() for r in self.active(complexity)]}
        if self.latent is not None:          # two-axis cards only: rule cards keep their shape
            out.update(latent=self.latent, contexts=self.contexts, backstories=self.backstories)
        return out


NEUTRAL = "neutral"


def parse_rule(entry: Any, library: Dict[str, Dict[str, Any]]) -> Rule:
    if isinstance(entry, str):
        rid, params = entry, {}
    elif isinstance(entry, dict) and len(entry) == 1:
        rid, params = next(iter(entry.items()))
        params = params or {}
    else:
        raise ValueError(f"a rule is 'id' or {{id: {{param: value}}}}, got {entry!r}")
    if rid not in library:
        raise ValueError(f"unknown rule {rid!r}; library has {sorted(library)}")
    allowed = library[rid].get("params") or {}
    options = {k: [str(x) for x in v] for k, v in (library[rid].get("options") or {}).items()}
    if options != {k: list(v) for k, v in OPTIONS.get(rid, {}).items()}:
        raise ValueError(f"rule {rid}: library options {options} differ from coop.rules.OPTIONS")
    required = set(allowed)
    if not required <= set(params) or not set(params) <= required | set(options):
        raise ValueError(f"rule {rid} takes parameters {sorted(allowed)} (optional: {sorted(options)}), "
                         f"got {sorted(params)}")
    for k, v in params.items():
        values = [str(x) for x in (allowed[k] if k in allowed else options[k])]
        if str(v) not in values:
            raise ValueError(f"rule {rid}: {k}={v!r} is not one of {values}")
    kept = {k: str(v) for k, v in params.items() if not (k in options and str(v) == options[k][0])}
    return Rule(rid, tuple(sorted(kept.items())))


def check_rules(rules: List[Rule]) -> List[str]:
    """Contradictions within a card (any prefix of a valid card is valid too)."""
    errors: List[str] = []
    seen: Dict[str, Rule] = {}
    kinds: Dict[str, Rule] = {}
    dishes: Dict[str, Rule] = {}
    for r in rules:
        if r.id in SINGLE:
            if r.id in seen:
                errors.append(f"{r.id} appears twice ({seen[r.id].label()}, {r.label()})")
            seen[r.id] = r
        if r.id == "own_station":
            kind = r.p["station"].rstrip("0123456789")
            if kind in kinds:
                errors.append(f"two own_station rules for {kind} ({kinds[kind].label()}, {r.label()})")
            kinds[kind] = r
        if r.id in ("solo_dish", "split_dish"):
            d = r.p["dish"]
            if d in dishes:
                errors.append(f"{dishes[d].label()} and {r.label()} on the same dish")
            dishes[d] = r
    ids = {r.id: r for r in rules}
    serves = ids.get("serves")
    if serves and serves.p["who"] == "ego" and "solo_dish" in ids:
        errors.append("serves(who=ego) contradicts solo_dish (a solo dish is served by the partner)")
    if serves and serves.p["who"] == "self" and "split_dish" in ids:
        errors.append("serves(who=self) contradicts split_dish (the other cook serves a split dish)")
    if "wants_supplies" in ids and "solo_dish" in ids:
        errors.append("wants_supplies contradicts solo_dish (a solo dish takes no help)")
    return errors


# ---------------------------------------------------------------------- #
# Two-axis cards (docs/STYLE_PARTNER_PLAN_v4.md, 2): workflow x plan adherence
# ---------------------------------------------------------------------- #
WORKFLOWS = ("prep", "whole", "station")
ADHERENCE = ("rigid", "flexible")
STYLE_CONTEXTS = ("on_urgent", "on_blocked", "on_prep_ready")
ADHERENCE_MODES = {
    "rigid": {"on_urgent": {"mode": "keeps"}, "on_blocked": {"mode": "waits"}, "on_prep_ready": {"mode": "continues"}},
    "flexible": {"on_urgent": {"mode": "takes", "unless_ego_on_it": "yes"}, "on_blocked": {"mode": "clears"},
                 "on_prep_ready": {"mode": "switches"}},
}
# (workflow, context) pairs where the workflow fixes the response whatever the adherence, so the
# adherence is not identifiable there: the station cook clears pots and ovens and cooks prepared
# items as its own work. Those context rules are left out of the card (the workflow governs) and
# the pair is marked not_identifiable.
FIXED_BY_WORKFLOW = {("station", "on_blocked"): "clears and cooks as its own work",
                     ("station", "on_prep_ready"): "cooks prepared items as its own work"}
BACKSTORY_VARIANTS = ("style_only", "style_habits")


def expand_style(name: str, spec: Dict[str, Any], library: Dict[str, Dict[str, Any]]) -> Card:
    """A two-axis card -> its context rules, latent labels and per-context truth. Raises on
    contradictions (unknown values, an explicit adherence the workflow overrides, both forms)."""
    if "rules" in spec:
        raise ValueError(f"card {name}: give either rules or workflow/adherence, not both")
    wf, adh = spec.get("workflow"), spec.get("adherence")
    if wf not in WORKFLOWS:
        raise ValueError(f"card {name}: workflow must be one of {WORKFLOWS}, got {wf!r}")
    if isinstance(adh, str):
        if adh not in ADHERENCE:
            raise ValueError(f"card {name}: adherence must be one of {ADHERENCE} or a per-context map")
        per = {c: adh for c in STYLE_CONTEXTS}
    elif isinstance(adh, dict):
        if set(adh) != set(STYLE_CONTEXTS):
            raise ValueError(f"card {name}: a per-context adherence names exactly {STYLE_CONTEXTS}, got {sorted(adh)}")
        bad = {c: v for c, v in adh.items() if v not in ADHERENCE}
        if bad:
            raise ValueError(f"card {name}: adherence values must be {ADHERENCE}, got {bad}")
        fixed = [c for c in STYLE_CONTEXTS if (wf, c) in FIXED_BY_WORKFLOW]
        if fixed:
            raise ValueError(f"card {name}: the {wf} workflow fixes {fixed}; give one adherence value, not a map")
        per = dict(adh)
    else:
        raise ValueError(f"card {name}: adherence is missing")
    rules = [parse_rule({"default_work": {"mode": wf}}, library)]
    contexts: Dict[str, Dict[str, Any]] = {}
    for c in STYLE_CONTEXTS:
        if (wf, c) in FIXED_BY_WORKFLOW:
            contexts[c] = {"mode": None, "adherence": None, "identifiable": False,
                           "note": FIXED_BY_WORKFLOW[(wf, c)]}
            continue
        mode = ADHERENCE_MODES[per[c]][c]
        rules.append(parse_rule({c: dict(mode)}, library))
        contexts[c] = {"mode": mode["mode"], "adherence": per[c], "identifiable": True}
    errors = check_rules(rules)
    if errors:
        raise ValueError(f"card {name}: " + "; ".join(errors))
    stories = spec.get("backstory") or {}
    if isinstance(stories, str):
        stories = {"style_only": stories}
    unknown = set(stories) - set(BACKSTORY_VARIANTS)
    if unknown:
        raise ValueError(f"card {name}: backstory variants are {BACKSTORY_VARIANTS}, got {sorted(unknown)}")
    stories = {k: str(v).strip() for k, v in stories.items()}
    latent = {"workflow": wf, "adherence": adh if isinstance(adh, str) else dict(adh)}
    return Card(name, stories.get("style_only", ""), rules, latent=latent, contexts=contexts, backstories=stories)


def flip_adherence(card: Card, contexts: Optional[List[str]] = None, workflow: Optional[str] = None,
                   library: Optional[Dict[str, Dict[str, Any]]] = None) -> Card:
    """The same card with the adherence flipped in ``contexts`` (all identifiable ones by
    default) and optionally another ``workflow``: the cards that wrong-belief egos act on."""
    if card.latent is None:
        raise ValueError("flip_adherence needs a two-axis card")
    library = library or load_library()
    adh = card.latent["adherence"]
    per = {c: adh for c in STYLE_CONTEXTS} if isinstance(adh, str) else dict(adh)
    flip = {"rigid": "flexible", "flexible": "rigid"}
    for c in (contexts if contexts is not None else STYLE_CONTEXTS):
        per[c] = flip[per[c]]
    wf = workflow or card.latent["workflow"]
    keep = {c: v for c, v in per.items() if (wf, c) not in FIXED_BY_WORKFLOW}
    vals = set(keep.values())
    adh_new: Any = vals.pop() if len(vals) == 1 else {c: per[c] for c in STYLE_CONTEXTS}
    return expand_style(f"{card.name}~{wf}/{adh_new if isinstance(adh_new, str) else 'mixed'}",
                        {"workflow": wf, "adherence": adh_new}, library)


def load_cards(path: Path, library: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Card]:
    library = library or load_library()
    raw = _yaml(path) or {}
    cards: Dict[str, Card] = {}
    for name, spec in raw.items():
        if name == NEUTRAL:
            raise ValueError(f"'{NEUTRAL}' is built in (a card with no rules)")
        if "workflow" in spec or "adherence" in spec:
            cards[name] = expand_style(name, spec, library)
            continue
        rules = [parse_rule(e, library) for e in (spec.get("rules") or [])]
        errors = check_rules(rules)
        if errors:
            raise ValueError(f"card {name}: " + "; ".join(errors))
        cards[name] = Card(name, str(spec.get("backstory", "")).strip(), rules)
    cards[NEUTRAL] = Card(NEUTRAL, "", [])
    return cards


# ---------------------------------------------------------------------- #
# Text
# ---------------------------------------------------------------------- #
def counter_groups(kitchen) -> Dict[str, List[Tuple[int, int]]]:
    """Counters both players can use, split into thirds across the kitchen's width."""
    width = len(kitchen.terrain[0])
    groups: Dict[str, List[Tuple[int, int]]] = {"left": [], "center": [], "right": []}
    for p in kitchen.counters():
        third = min(2, p[0] * 3 // width)
        groups[("left", "center", "right")[third]].append(tuple(p))
    return groups


def _other_station(station: str, kitchen) -> str:
    kind = station.rstrip("0123456789")
    others = [u for u in kitchen.utensils if u.rstrip("0123456789") == kind and u != station]
    return others[0] if others else f"another {kind.replace('_', ' ')}"


def rule_vars(rule: Rule, kitchen=None, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    v = dict(rule.p)
    for name, values in OPTIONS.get(rule.id, {}).items():
        v.setdefault(name, values[0])
    if rule.id == "own_station":
        v["own_patience"] = str(OWN_STATION_PATIENCE)
    if rule.id == "on_urgent":
        yes = v.get("unless_ego_on_it") == "yes"
        v["unless"] = ", unless your partner is already working on it" if yes else ""
        v["unless_wants"] = " (unless you are already on it: then you finish it and I keep going)" if yes else ""
    if "station" in v:
        v["kind"] = v["station"].rstrip("0123456789").replace("_", " ")
        v["other"] = _other_station(v["station"], kitchen) if kitchen is not None else "the other one"
    if "which" in v:
        v["other_order"] = "newest" if v["which"] == "oldest" else "oldest"
    if "group" in v:
        cells = counter_groups(kitchen).get(v["group"], []) if kitchen is not None else []
        v["group_cells"] = ", ".join(str(c) for c in cells) or "none"
    v.update(extra or {})
    return v


class _Default(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def render(rule: Rule, field_name: str, library: Dict[str, Dict[str, Any]], kitchen=None,
           extra: Optional[Dict[str, str]] = None) -> str:
    """One text field of a rule with its parameters filled in. Fields can have a variant
    per parameter value (``text_self`` for serves(who=self))."""
    spec = library[rule.id]
    key = field_name
    for value in rule.p.values():
        if f"{field_name}_{value}" in spec:
            key = f"{field_name}_{value}"
    text = spec.get(key, "")
    return str(text).format_map(_Default(rule_vars(rule, kitchen, extra))).strip()


def category(rule: Rule, library: Dict[str, Dict[str, Any]]) -> str:
    return library[rule.id]["category"]
