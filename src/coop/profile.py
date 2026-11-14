"""Partner profile: the traits an agent is asked to judge, and their true values.

Probe rules (2026-11-12):

* ``avoids`` is asked only over utensils that this episode's orders needed and that
  both players can reach (``avoid_options``). With no such utensil the field is not
  asked and not scored. Avoiding a utensil only the partner can reach is impossible,
  and a utensil no order needs is unused by everyone.
* ``speed`` is about pace while working. The probe shows an observed pace line
  (``PaceCounter``): the agent only sees the kitchen at its decision points, so it
  cannot see pace otherwise. Waiting for preferred work is a preference, not slowness.
* ``commitment`` is about visibly unfinished work (``unfinished_rate``). Abandoning a
  carried item (``abandon_rate``) puts it on a counter, which looks exactly like a
  hand-over, so it is not scored (truth ``None``).

* ``orders`` (the order habit) is asked only with several open orders, over
  ``order_options``: oldest / newest open order, the dish served last, a favourite
  dish (one option per dish of the episode), or no fixed habit. A partner that does
  whatever is needed next (no ``order_commit``) has no fixed habit.

Fields whose truth is ``None`` are left out of the score.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

PROFILE_FIELDS = {
    "prefers": ["prep", "cook", "serve", "no clear preference"],
    "avoids": ["chopping board", "pot", "oven", "blender", "nothing"],
    "gives_way": ["gives way to you", "does not give way"],
    "speed": ["normal", "slow"],
    "reliability": ["reliable", "makes mistakes"],
    "commitment": ["finishes tasks", "leaves tasks unfinished"],
}
STATION_NAMES = {"chopping_board": "chopping board", "pot": "pot", "oven": "oven", "blender": "blender"}
STATION_KINDS = {v: k for k, v in STATION_NAMES.items()}
# Answers under earlier option names, mapped to the current ones when old runs are rescored.
LEGACY_ANSWERS = {"commitment": {"abandons tasks": "leaves tasks unfinished"}}


# Partner-model file (LLM agent, memory_mode "model"): one markdown file the agent
# rewrites after every episode, shown in every later episode.
MODEL_SECTIONS = ("## How they choose what to do (with evidence)", "## How they react to me",
                  "## What I should do with them", "## Open questions")


def model_template() -> str:
    return "\n\n".join(f"{h}\n- (nothing yet)" for h in MODEL_SECTIONS)


def needed_kinds(kitchen, dishes: Iterable[str]) -> set:
    """Utensil kinds needed to make these dishes (every step of every recipe)."""
    out: set = set()

    def walk(item):
        if item in kitchen.recipes:
            kind, inputs = kitchen.recipes[item]
            out.add(kind)
            for child in inputs:
                walk(child)
    for d in dishes:
        walk(d)
    return out


def avoid_options(kitchen, dishes: Iterable[str]) -> List[str]:
    """Utensils the ``avoids`` field asks about: needed by these dishes, and at least one
    of that kind reachable by both players. Public information (map and orders)."""
    need = needed_kinds(kitchen, dishes)
    shared = {kitchen.kind_of[u] for u, p in kitchen.utensils.items()
              if kitchen.mine(p) and kitchen.theirs(p)}
    return [name for kind, name in STATION_NAMES.items() if kind in need and kind in shared]


ORDER_HABITS = {"first": "its oldest open order", "last": "the newest open order",
                "same_dish": "the dish it served last"}
NO_ORDER_HABIT = "no fixed habit"


def order_options(dishes: Iterable[str]) -> List[str]:
    """Options of the ``orders`` field for an episode with these dishes ordered."""
    return (list(ORDER_HABITS.values()) + [f"favourite dish: {d}" for d in dict.fromkeys(dishes)]
            + [NO_ORDER_HABIT])


def probe_fields(avoid_opts: Optional[List[str]] = None,
                 order_opts: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """The fields and options asked in one probe (``avoids`` and ``orders`` depend on the
    episode; ``orders`` is asked only when ``order_opts`` is given)."""
    fields = dict(PROFILE_FIELDS)
    if avoid_opts is not None:
        if avoid_opts:
            fields["avoids"] = list(avoid_opts) + ["nothing"]
        else:
            del fields["avoids"]
    if order_opts:
        fields["orders"] = list(order_opts)
    return fields


class PaceCounter:
    """How often the partner, having just done something (moved, turned, picked up or
    put down), did something again on the very next step. Counted only over steps where
    the partner was in view; a partner at full pace does so most of the time while busy,
    one that moves every other step never does."""

    def __init__(self):
        self.prev = None          # (position, orientation, held) last step, None if not in view
        self.acted = None         # whether the partner acted on the last step (None: unknown)
        self.pairs = self.again = 0

    def update(self, player, visible: bool = True):
        held = player.held_object.name if player.held_object is not None else None
        sig = (tuple(player.position), tuple(player.orientation), held) if visible else None
        acted = None if sig is None or self.prev is None else sig != self.prev
        if acted is not None and self.acted:
            self.pairs += 1
            self.again += acted
        self.prev, self.acted = sig, acted

    def share(self) -> Optional[float]:
        return self.again / self.pairs if self.pairs else None

    def text(self) -> str:
        if self.pairs < 5:
            return "Your partner's pace: you did not see enough of them to tell."
        return (f"Your partner's pace (from what you saw): after doing something (moving, turning, "
                f"picking up or putting down), they did something again on the very next step "
                f"{100 * self.again / self.pairs:.0f}% of the time ({self.pairs} times seen).")


# Shown with the probe, identical in every condition: what each answer means.
PROFILE_GUIDE = (
    "What the answers mean:\n"
    '- prefers: the stage your partner chose clearly more often than the orders required. Answer '
    '"no clear preference" if they mostly did whatever was needed at the moment.\n'
    '- avoids: one of the listed utensils (the orders needed them and both of you can reach them) '
    'that your partner never used; "nothing" if they used all of them.\n'
    '- gives_way: "gives way to you" if, when you both went for the same task, your partner usually '
    "turned to something else.\n"
    '- speed: "slow" if your partner moves and works at a slower pace than full speed (see the pace '
    "line) or pauses before every new task. Standing and waiting for a kind of work they like, while "
    "you do the rest, is a preference, not slowness.\n"
    '- reliability: "makes mistakes" if items your partner put into utensils matched no recipe and '
    "were thrown away.\n"
    '- commitment: "leaves tasks unfinished" if your partner often left a job half done and walked '
    "away to something else: a utensil they loaded and did not start, or finished food they did not "
    "take out. Putting an item on a counter is a normal hand-over, not a sign of this.\n"
    "- stage_shares: your estimate of how your partner's work split between prep, cook and serve "
    "(percentages that add up to 100).")

# Added to the guide only when the orders field is asked (several open orders), so the
# one-order probe is word for word the one used before order habits existed.
ORDERS_GUIDE = (
    '- orders (only with several open orders): which open order your partner worked on. "its '
    'oldest open order" / "the newest open order" if they stuck to one order, picked that way, until '
    'it was served; "the dish it served last" if after serving a dish they took an order for the same '
    'dish; "favourite dish: <dish>" if they went for orders of one dish whenever there was one; "no '
    'fixed habit" if they did whatever was needed next for any order.\n')


def profile_guide(fields: Dict[str, List[str]]) -> str:
    """The answer guide for the fields asked in one probe."""
    if "orders" not in fields:
        return PROFILE_GUIDE
    at = PROFILE_GUIDE.index("- commitment:")
    return PROFILE_GUIDE[:at] + ORDERS_GUIDE + PROFILE_GUIDE[at:]


def _trait(traits, name, default=None):
    return traits.get(name, default) if isinstance(traits, dict) else getattr(traits, name, default)


def order_truth(traits, dishes: Iterable[str]) -> str:
    """The order habit an observer should name, given the dishes ordered this episode
    (a favourite dish that was never ordered cannot be seen)."""
    if not _trait(traits, "order_commit", False):
        return NO_ORDER_HABIT
    dishes = set(dishes)
    fav = [d for d in (_trait(traits, "favourite_dishes", ()) or ()) if d in dishes]
    if fav:
        return f"favourite dish: {fav[0]}"
    return ORDER_HABITS[_trait(traits, "order_focus", "first") or "first"]


def truth_profile(traits, avoid_opts: Optional[List[str]] = None,
                  order_dishes: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """The answer a perfect observer should give for a partner with these traits (a
    ``Traits`` or its dict). ``avoid_opts``: the utensils ``avoids`` asked about
    (``avoid_options``); None = all utensils (no episode context). ``order_dishes``:
    the dishes ordered in an episode with several open orders (``orders`` is scored
    only then). ``None`` values are not scored."""
    avoid = [STATION_NAMES[k] for k in _trait(traits, "avoid_stations", ()) or ()]
    if avoid_opts is None:
        avoids: Optional[str] = avoid[0] if avoid else "nothing"
    elif not avoid_opts:
        avoids = None
    else:
        avoids = next((a for a in avoid if a in avoid_opts), "nothing")
    pref = _trait(traits, "preference", ()) or ()
    if (_trait(traits, "unfinished_rate", 0) or 0) > 0:
        commitment: Optional[str] = "leaves tasks unfinished"
    elif (_trait(traits, "abandon_rate", 0) or 0) > 0:
        commitment = None
    else:
        commitment = "finishes tasks"
    return {
        "prefers": pref[0] if pref else "no clear preference",
        "avoids": avoids,
        "gives_way": "gives way to you" if _trait(traits, "yield_tasks", False) else "does not give way",
        "speed": ("slow" if _trait(traits, "step_period", 1) > 1 or _trait(traits, "decision_delay", 0) > 0
                  else "normal"),
        "reliability": "makes mistakes" if (_trait(traits, "slip_rate", 0) or 0) > 0 else "reliable",
        "commitment": commitment,
        "orders": order_truth(traits, order_dishes) if order_dishes else None,
    }


def clean_profile(raw: Any, fields: Optional[Dict[str, List[str]]] = None) -> Dict[str, Optional[str]]:
    """Normalise an answered profile; values outside the allowed options become None."""
    out: Dict[str, Optional[str]] = {}
    raw = raw if isinstance(raw, dict) else {}
    for key, options in (fields or PROFILE_FIELDS).items():
        v = str(raw.get(key, "")).strip().lower()
        out[key] = next((o for o in options if v == o), None)
    return out


def legacy_answer(answer: Dict[str, Optional[str]], avoid_opts: Optional[List[str]]) -> Dict[str, Optional[str]]:
    """An answer given under earlier probe options, read under the current ones: renamed
    options are mapped, and an ``avoids`` answer naming a utensil outside ``avoid_opts``
    (not needed, or not reachable by both) counts as "nothing"."""
    out = dict(answer)
    for f, mapping in LEGACY_ANSWERS.items():
        if out.get(f) in mapping:
            out[f] = mapping[out[f]]
    if avoid_opts is not None and out.get("avoids") in STATION_KINDS and out["avoids"] not in avoid_opts:
        out["avoids"] = "nothing"
    return out


def clean_shares(raw: Any) -> Optional[Dict[str, float]]:
    """Stage shares as percentages summing to 100, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    vals = {}
    for k in ("prep", "cook", "serve"):
        try:
            v = float(raw.get(k, 0))
        except (TypeError, ValueError):
            return None
        if v < 0:
            return None
        vals[k] = v
    total = sum(vals.values())
    if total <= 0:
        return None
    return {k: 100.0 * v / total for k, v in vals.items()}


def score_profile(answer: Dict[str, Optional[str]], truth: Dict[str, str],
                  shares: Optional[Dict[str, float]] = None,
                  true_shares: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """``true_shares``: fractions the partner actually spent on each stage this episode."""
    correct = {k: answer.get(k) == v for k, v in truth.items() if v is not None}
    out = {"answer": answer, "truth": truth, "correct": correct,
           "accuracy": sum(correct.values()) / len(correct) if correct else None}
    if shares is not None and true_shares:
        truth_pct = {k: 100.0 * true_shares.get(k, 0.0) for k in ("prep", "cook", "serve")}
        out["shares"] = shares
        out["true_shares"] = truth_pct
        out["share_error"] = sum(abs(shares[k] - truth_pct[k]) for k in truth_pct) / 3
    return out
