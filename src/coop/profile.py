"""Partner profile: the four traits an agent is asked to judge, and their true values."""
from __future__ import annotations

from typing import Any, Dict, Optional

PROFILE_FIELDS = {
    "prefers": ["prep", "cook", "serve", "no clear preference"],
    "avoids": ["chopping board", "pot", "oven", "blender", "nothing"],
    "gives_way": ["gives way to you", "does not give way"],
    "speed": ["normal", "slow"],
    "reliability": ["reliable", "makes mistakes"],
    "commitment": ["finishes tasks", "abandons tasks"],
}
STATION_NAMES = {"chopping_board": "chopping board", "pot": "pot", "oven": "oven", "blender": "blender"}


# Shown with the probe, identical in every condition: what each answer means.
PROFILE_GUIDE = (
    "What the answers mean:\n"
    '- prefers: the stage your partner chose clearly more often than the orders required. Answer '
    '"no clear preference" if they mostly did whatever was needed at the moment.\n'
    '- avoids: a utensil your partner never used although using it would have helped; "nothing" if '
    "there was none.\n"
    '- gives_way: "gives way to you" if, when you both went for the same task, your partner usually '
    "turned to something else.\n"
    '- speed: "slow" if your partner often stood still for several steps with nothing to do that you '
    "could see, or completed clearly fewer actions than the work allowed.\n"
    '- reliability: "makes mistakes" if items your partner put into utensils matched no recipe and '
    "were thrown away.\n"
    '- commitment: "abandons tasks" if your partner often put down what they were carrying where it was '
    "not needed, or left a job half done (for example a loaded utensil they did not start, or "
    "finished food they did not take out) and switched to something else.\n"
    "- stage_shares: your estimate of how your partner's work split between prep, cook and serve "
    "(percentages that add up to 100).")


def truth_profile(traits) -> Dict[str, str]:
    """The answer a perfect observer should give for a partner with these traits."""
    avoid = getattr(traits, "avoid_stations", ())
    return {
        "prefers": traits.preference[0] if traits.preference else "no clear preference",
        "avoids": STATION_NAMES[avoid[0]] if avoid else "nothing",
        "gives_way": "gives way to you" if getattr(traits, "yield_tasks", False) else "does not give way",
        "speed": "slow" if traits.step_period > 1 or traits.decision_delay > 0 else "normal",
        "reliability": "makes mistakes" if traits.slip_rate > 0 else "reliable",
        "commitment": ("abandons tasks" if traits.abandon_rate > 0 or getattr(traits, "unfinished_rate", 0) > 0
                       else "finishes tasks"),
    }


def clean_profile(raw: Any) -> Dict[str, Optional[str]]:
    """Normalise an answered profile; values outside the allowed options become None."""
    out: Dict[str, Optional[str]] = {}
    raw = raw if isinstance(raw, dict) else {}
    for key, options in PROFILE_FIELDS.items():
        v = str(raw.get(key, "")).strip().lower()
        out[key] = next((o for o in options if v == o), None)
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
    correct = {k: answer.get(k) == v for k, v in truth.items()}
    out = {"answer": answer, "truth": truth, "correct": correct,
           "accuracy": sum(correct.values()) / len(correct)}
    if shares is not None and true_shares:
        truth_pct = {k: 100.0 * true_shares.get(k, 0.0) for k in ("prep", "cook", "serve")}
        out["shares"] = shares
        out["true_shares"] = truth_pct
        out["share_error"] = sum(abs(shares[k] - truth_pct[k]) for k in truth_pct) / 3
    return out
