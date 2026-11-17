"""The LLM partner's feedback at the end of each day (docs/PERSONA_PARTNER_PLAN_v2.md, section 8).

Called after the day's last interaction episode, before the ego's consolidation. The
partner model gets its card (backstory and active rules), the day's scores and its own
log of the day (its actions, the actions of the ego it saw, public events), and writes at
most 60 words in the persona's voice about what happened. It may ask for a change; it must
not list its rules. Checks, stored with the text:

* ``recitation``: rule texts or rule ids found in the feedback (string check);
* ``words``: length; claims about events are kept with the prompt for a manual sample
  (unsupported claims are judged by hand, not here).

Written once per day to ``runs/<cond>/<persona>/seed<k>/dayDD/feedback.json`` and read back
on resume.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import build_backend, parse_json
from .rules import load_library, render

MAX_WORDS = 60
LOG_LINES = 40


def _episode_logs(run_dir: Path, day: int, episodes: int) -> List[Dict[str, Any]]:
    out = []
    for ep in range(1, episodes + 1):
        d = run_dir / f"day{day:02d}" / f"ep{ep:02d}"
        log = json.loads((d / "partner_log.json").read_text()) if (d / "partner_log.json").exists() else {}
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        out.append({"episode": ep, "score": summary.get("score"), "orders": summary.get("orders_completed"),
                    "own": log.get("own_actions") or [], "seen": log.get("seen_partner_actions") or [],
                    "events": log.get("events") or []})
    return out


def prompt(card, rules, library, kitchen, day: int, logs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    habits = "\n".join(f"- {render(r, 'text', library, kitchen)}" for r in rules) or "- (no particular habits)"
    system = ("You are a cook in a kitchen game, working with a partner you cannot talk to during work. "
              f"Who you are: {card.backstory or 'an ordinary cook.'}\nHow you like to work:\n{habits}\n"
              "At the end of each day you may tell your partner one thing, in your own voice.")
    parts = [f"DAY_FEEDBACK\nDay {day} is over. Your record of the day:"]
    for lg in logs:
        own = "\n".join(f"  t={x[0]}: {x[1]} -> {x[2]}" for x in lg["own"][-LOG_LINES:]) or "  (none)"
        seen = "\n".join(f"  t={x[0]}: {x[1]}" for x in lg["seen"][-LOG_LINES:]) or "  (none)"
        events = "\n".join(f"  {e}" for e in lg["events"][-15:]) or "  (none)"
        parts.append(f"Episode {lg['episode']}: team score {lg['score']}, orders completed {lg['orders']}.\n"
                     f" What you did:\n{own}\n What you saw your partner do:\n{seen}\n Events:\n{events}")
    parts.append(f"Tell your partner, in at most {MAX_WORDS} words and in your own voice, what you thought of "
                 "today's work together. You may ask them to do something differently. Only mention things "
                 "that happened in your record. Do not list your habits or rules.\n"
                 'Reply with JSON only: {"feedback": "<what you say>"}')
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def checks(text: str, rules, library, kitchen) -> Dict[str, Any]:
    low = text.lower()
    recited = []
    for r in rules:
        words = render(r, "text", library, kitchen).lower().split()
        for i in range(max(0, len(words) - 7)):
            if " ".join(words[i:i + 8]) in low:          # 8 consecutive words of a rule's text
                recited.append(r.label())
                break
        if re.search(rf"\b{re.escape(r.id)}\b", low):
            recited.append(r.id)
    return {"words": len(text.split()), "recitation": sorted(set(recited))}


def for_day(cfg, persona: str, run_dir: Path, day: int, run_seed: int) -> Optional[Dict[str, Any]]:
    """The partner's feedback for this run's day (generated once, then read back)."""
    path = run_dir / f"day{day:02d}" / "feedback.json"
    if path.exists():
        return json.loads(path.read_text())
    from .days import oracle_kitchen
    library = load_library()
    card = cfg.cards()[persona]
    rules = card.active(cfg.partner.get("complexity"))
    kitchen = oracle_kitchen(cfg)
    logs = _episode_logs(run_dir, day, cfg.episodes_per_day)
    messages = prompt(card, rules, library, kitchen, day, logs)
    backend = build_backend(cfg.partner["backend"], "partner feedback")
    r = backend.chat(messages, max_tokens=200, seed=run_seed * 1000 + day)
    parsed = parse_json(r.text) or {}
    text = parsed.get("feedback") if isinstance(parsed.get("feedback"), str) else ""
    words = text.split()
    if len(words) > MAX_WORDS * 1.5:
        text = " ".join(words[:int(MAX_WORDS * 1.5)])
    out = {"day": day, "text": text.strip(), "ok": bool(text.strip()), "checks": checks(text, rules, library, kitchen),
           "prompt": messages, "reply": r.text, "error": r.error}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out
