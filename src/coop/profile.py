"""Partner profile: the four traits an agent is asked to judge, and their true values."""
from __future__ import annotations

from typing import Any, Dict, Optional

PROFILE_FIELDS = {
    "prefers": ["prep", "cook", "serve", "no clear preference"],
    "speed": ["normal", "slow"],
    "reliability": ["reliable", "makes mistakes"],
    "commitment": ["finishes tasks", "abandons tasks"],
}


def truth_profile(traits) -> Dict[str, str]:
    """The answer a perfect observer should give for a partner with these traits."""
    return {
        "prefers": traits.preference[0] if traits.preference else "no clear preference",
        "speed": "slow" if traits.step_period > 1 or traits.decision_delay > 0 else "normal",
        "reliability": "makes mistakes" if traits.slip_rate > 0 else "reliable",
        "commitment": "abandons tasks" if traits.abandon_rate > 0 else "finishes tasks",
    }


def clean_profile(raw: Any) -> Dict[str, Optional[str]]:
    """Normalise an answered profile; values outside the allowed options become None."""
    out: Dict[str, Optional[str]] = {}
    raw = raw if isinstance(raw, dict) else {}
    for key, options in PROFILE_FIELDS.items():
        v = str(raw.get(key, "")).strip().lower()
        out[key] = next((o for o in options if v == o), None)
    return out


def score_profile(answer: Dict[str, Optional[str]], truth: Dict[str, str]) -> Dict[str, Any]:
    correct = {k: answer.get(k) == v for k, v in truth.items()}
    return {"answer": answer, "truth": truth, "correct": correct,
            "accuracy": sum(correct.values()) / len(correct)}
