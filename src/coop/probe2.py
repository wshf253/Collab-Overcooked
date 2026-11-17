"""Knowledge probe (docs/PERSONA_PARTNER_PLAN_v2.md, section 9).

After each day's consolidation, on the frozen memory, the ego answers one multiple-choice
question per library rule, including rules this persona does not have (correct answer:
the "no preference" option). ``own_station`` is asked once per station kind that has two
stations. The probe is a forked call: its answers never enter memory or later prompts.

Scored against the card's active rules, reported separately for rules exposed so far
(the partner had an opportunity inside the ego's view), rules not yet exposed, and rules
the persona lacks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rules import Rule, counter_groups

NO = "no preference"


def questions(kitchen, menu: List[str]) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    kinds: Dict[str, List[str]] = {}
    for u in kitchen.utensils:
        kinds.setdefault(kitchen.kind_of[u], []).append(u)
    for kind, us in sorted(kinds.items()):
        if len(us) >= 2:
            qs.append({"id": f"own_station:{kind}", "rule": "own_station",
                       "text": f"When both {kind.replace('_', ' ')}s are free, which one does your partner use?",
                       "options": sorted(us) + [NO]})
    dishes = list(menu)
    qs += [
        {"id": "solo_dish", "rule": "solo_dish", "options": dishes + ["none"],
         "text": "Is there a dish your partner always makes alone, from start to finish, without wanting help?"},
        {"id": "split_dish", "rule": "split_dish", "options": dishes + ["none"],
         "text": "Is there a dish your partner prepares and loads but leaves for you to start, plate and serve?"},
        {"id": "order_priority", "rule": "order_priority", "options": ["oldest", "newest", NO],
         "text": "When your partner chooses a new order to work on, which open order does it take?"},
        {"id": "serves", "rule": "serves",
         "options": ["my partner always serves", "my partner never serves (I do)", NO],
         "text": "Who plates and delivers the food?"},
        {"id": "wants_supplies", "rule": "wants_supplies",
         "options": ["yes, it waits for me to bring them", "no, it fetches them itself"],
         "text": "When your partner's batch is missing an ingredient, does it stay and wait for you to bring it?"},
        {"id": "handover_spot", "rule": "handover_spot", "options": ["left", "center", "right", NO],
         "text": "Where does your partner put items it passes on (left, center or right counters)?"},
        {"id": "on_takeover", "rule": "on_takeover",
         "options": ["it leaves that order to me and switches", "it carries on"],
         "text": "If you take over something your partner started, what does it do?"},
        {"id": "on_contention", "rule": "on_contention",
         "options": ["it steps back", "it carries on", "no clear pattern"],
         "text": "When you and your partner head for the same station, what does your partner do?"},
        {"id": "waits_for_commit", "rule": "waits_for_commit",
         "options": ["yes, it waits a few steps to see what I start", "no, it starts at once"],
         "text": "When a new order comes in, does your partner wait to see what you start?"},
    ]
    return qs


def truth(q: Dict[str, Any], rules: List[Rule]) -> Dict[str, Any]:
    """The correct option and whether the persona has the rule behind the question."""
    rid = q["rule"]
    for r in rules:
        if r.id != rid:
            continue
        p = r.p
        if rid == "own_station":
            if p["station"].rstrip("0123456789") == q["id"].split(":")[1]:
                return {"answer": p["station"], "has": True, "rule": r.label()}
            continue
        answer = {"solo_dish": p.get("dish"), "split_dish": p.get("dish"), "order_priority": p.get("which"),
                  "serves": "my partner always serves" if p.get("who") == "self" else "my partner never serves (I do)",
                  "wants_supplies": q["options"][0], "handover_spot": p.get("group"),
                  "on_takeover": q["options"][0],
                  "on_contention": "it steps back" if p.get("mode") == "yields" else "it carries on",
                  "waits_for_commit": q["options"][0]}[rid]
        return {"answer": answer, "has": True, "rule": r.label()}
    default = {"solo_dish": "none", "split_dish": "none", "wants_supplies": q["options"][1],
               "on_takeover": q["options"][1], "on_contention": "no clear pattern",
               "waits_for_commit": q["options"][1]}.get(rid, NO)
    return {"answer": default, "has": False, "rule": None}


def score(qs: List[Dict[str, Any]], answers: Dict[str, Any], rules: List[Rule],
          exposure: Dict[str, int]) -> Dict[str, Any]:
    rows = []
    for q in qs:
        t = truth(q, rules)
        given = str(answers.get(q["id"], "")).strip()
        given = next((o for o in q["options"] if o.lower() == given.lower()), None)
        if not t["has"]:
            group = "absent"
        else:
            group = "exposed" if exposure.get(t["rule"], 0) > 0 else "not_exposed"
        rows.append({"id": q["id"], "answer": given, "truth": t["answer"], "correct": given == t["answer"],
                     "group": group, "rule": t["rule"]})
    acc = {}
    for g in ("exposed", "not_exposed", "absent"):
        sel = [r for r in rows if r["group"] == g]
        acc[g] = {"n": len(sel), "accuracy": (sum(r["correct"] for r in sel) / len(sel)) if sel else None}
    acc["all"] = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None}
    return {"rows": rows, "accuracy": acc}


def run(agent, cfg, rules: List[Rule], exposure: Dict[str, int]) -> Dict[str, Any]:
    """Ask the ego (frozen memory) and score the answers."""
    from .orders import normalize_menu
    qs = questions(agent.kitchen, list(normalize_menu(cfg.menu)))
    res = agent.knowledge_probe(qs)
    out = score(qs, res.get("answers") or {}, rules, exposure)
    out.update(questions=qs, prompt=res.get("prompt"), reply=res.get("reply"), error=res.get("error"),
               exposure=exposure)
    return out
