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
CATEGORIES = ("station", "dish", "order", "stage", "hand-over", "reaction", "initiative")
# Rules that may appear at most once per card (own_station: once per station kind).
SINGLE = {"order_priority", "serves", "wants_supplies", "handover_spot", "on_takeover", "on_contention",
          "waits_for_commit"}
# Keys of a library entry that may carry a per-parameter variant suffix (text_self, ...).
TEXT_FIELDS = ("text", "fallback", "wants", "reminder")


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

    def active(self, complexity: Optional[int] = None) -> List[Rule]:
        return list(self.rules if complexity is None else self.rules[:complexity])

    def to_dict(self, complexity: Optional[int] = None) -> Dict[str, Any]:
        return {"name": self.name, "backstory": self.backstory, "complexity": complexity,
                "rules": [r.to_dict() for r in self.active(complexity)]}


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
    if set(params) != set(allowed):
        raise ValueError(f"rule {rid} takes parameters {sorted(allowed)}, got {sorted(params)}")
    for k, v in params.items():
        if v not in allowed[k]:
            raise ValueError(f"rule {rid}: {k}={v!r} is not one of {allowed[k]}")
    return Rule(rid, tuple(sorted((k, str(v)) for k, v in params.items())))


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


def load_cards(path: Path, library: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Card]:
    library = library or load_library()
    raw = _yaml(path) or {}
    cards: Dict[str, Card] = {}
    for name, spec in raw.items():
        if name == NEUTRAL:
            raise ValueError(f"'{NEUTRAL}' is built in (a card with no rules)")
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
