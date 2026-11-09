"""Analyse a coop.days output directory (LLM ego): where score is lost, what the probe says.

    PYTHONPATH=src:lib/overcooked_ai python -m coop.analyze outputs/days_llm [--out outputs/days_llm/analysis.json]

Reports, per condition and persona:

* score minus the reference ego on identical episodes (mean ± standard error),
  first day vs last day, and each condition minus ``direct`` on identical episodes;
* the partner probe: accuracy per field and the most common answer;
* how the ego spent its decisions: share of ``wait``, chosen wait length, how
  waits ended (full length / early because the partner acted), and **waits that
  ran their full length while finished food sat in a utensil the ego can reach**
  (seen in the next decision's prompt);
* LLM health: calls per episode, invalid replies, fallbacks, thinking or
  truncated replies, latency.

Reads only files that coop.days writes; nothing is re-run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REFERENCE = "_reference"
FIELDS = ("prefers", "speed", "reliability", "commitment")


def _mean_se(xs: List[float]) -> Dict[str, Optional[float]]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "se": None}
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
    return {"n": len(xs), "mean": statistics.fmean(xs), "se": se}


def load_episodes(out: Path) -> List[Dict[str, Any]]:
    eps = []
    for summary in sorted(out.glob("runs/*/*/seed*/day*/ep*/summary.json")):
        d = summary.parent
        cond = d.parents[3].name
        s = json.loads(summary.read_text())
        log_path = d / "agent_log.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        eps.append({"condition": cond, "persona": s["persona"], "seed": s["run_seed"], "day": s["day"],
                    "episode": s["episode"], "summary": s, "log": log})
    return eps


def _reachable_utensils(system_prompt: str) -> List[str]:
    return re.findall(r"^\s+(\w+) at \(\d+, \d+\) \(.\): start it with .*reachable by (?:you|both of you)$",
                      system_prompt or "", flags=re.M)


def _reason_kind(error: str) -> str:
    """Group executor/parser messages: drop item and utensil names, keep the reason."""
    e = re.sub(r"\(.*\)", "(...)", error)
    e = re.sub(r"'[^']*'", "'X'", e)
    e = re.sub(r"\b(pot|oven|chopping_board|blender)\d+\b", "U", e)
    e = re.sub(r"\b(?:no counter you can reach holds|no finished|you hold|has no finished) \w+",
               lambda m: m.group(0).rsplit(" ", 1)[0] + " X", e)
    e = re.sub(r"^\w+ must be served on a dish", "X must be served on a dish", e)
    return e[:90]


def _action_kind(action: Any) -> str:
    if not isinstance(action, str) or not action.strip():
        return "<no action>"
    a = action.strip()
    m = re.match(r"\s*(\w+)\s*\((.*)\)", a)
    if not m:
        return "<not name(args)>: " + a[:30]
    name, args = m.group(1), [x.strip() for x in m.group(2).split(",")]
    if name == "pickup" and len(args) == 2:
        place = re.sub(r"\d+$", "", args[1])
        return f"pickup(_, {place})"
    if name == "wait":
        return f"wait({args[0][:8]})"
    return f"{name}({re.sub(r'[0-9]+$', '', args[0]) if args and args[0] else ''})"


def decision_stats(log: Dict[str, Any]) -> Dict[str, Any]:
    decisions = log.get("decisions") or []
    calls = [c for c in log.get("calls") or [] if c.get("kind") == "decide"]
    chosen = [d.get("chosen", "") for d in decisions]
    waits = [c for c in chosen if c.startswith("wait(")]
    lengths = [int(re.search(r"\d+", w).group()) for w in waits if re.search(r"\d+", w)]
    mine = _reachable_utensils(log.get("system_prompt", ""))
    full = early = missed_ready = 0
    for c in calls:
        m = re.search(r"Result of your last action: done: waited (\d+) step\(s\)(; your partner just did)?",
                      c.get("user", ""))
        if not m:
            continue
        if m.group(2):
            early += 1
            continue
        full += 1
        utensils = re.search(r"^Utensils: (.*)$", c["user"], flags=re.M)
        if utensils and any(re.search(rf"\b{re.escape(u)}: \S+ is ready to take out", utensils.group(1))
                            for u in mine):
            missed_ready += 1
    first = [d["attempts"][0] for d in decisions if d.get("attempts")]
    reasons, bad_actions, fallback_actions = Counter(), Counter(), Counter()
    for d in decisions:
        for a in d.get("attempts") or []:
            if a.get("error"):
                reasons[_reason_kind(a["error"])] += 1
                bad_actions[_action_kind(a.get("action"))] += 1
        if d.get("attempts") and all(a.get("error") for a in d["attempts"]):
            fallback_actions[" -> ".join(_action_kind(a.get("action")) for a in d["attempts"])] += 1
    return {"decisions": len(decisions),
            "invalid_reasons": reasons, "invalid_actions": bad_actions, "fallback_pairs": fallback_actions, "wait_share": len(waits) / len(chosen) if chosen else None,
            "mean_wait_chosen": statistics.fmean(lengths) if lengths else None,
            "waits_full": full, "waits_ended_by_partner": early,
            "full_waits_with_ready_food": missed_ready,
            "first_reply_invalid": sum(bool(a.get("error")) for a in first),
            "action_mix": Counter(c.split("(")[0] for c in chosen)}


def analyse(out: Path) -> Dict[str, Any]:
    eps = load_episodes(out)
    ref = {(e["persona"], e["seed"], e["day"], e["episode"]): e["summary"]["score"]
           for e in eps if e["condition"] == REFERENCE}
    by_key = {(e["condition"], e["persona"], e["seed"], e["day"], e["episode"]): e for e in eps}
    conditions = sorted({e["condition"] for e in eps if e["condition"] != REFERENCE},
                        key=lambda c: (c != "direct", c))
    personas = sorted({e["persona"] for e in eps})
    days = sorted({e["day"] for e in eps})
    report: Dict[str, Any] = {"conditions": {}, "overall": {}}
    for cond in conditions:
        report["conditions"][cond] = {}
        for persona in personas:
            mine = [e for e in eps if e["condition"] == cond and e["persona"] == persona]
            if not mine:
                continue
            key = lambda e: (e["persona"], e["seed"], e["day"], e["episode"])
            diffs = [e["summary"]["score"] - ref[key(e)] for e in mine if key(e) in ref]
            first = [e["summary"]["score"] - ref[key(e)] for e in mine if e["day"] == days[0] and key(e) in ref]
            last = [e["summary"]["score"] - ref[key(e)] for e in mine if e["day"] == days[-1] and key(e) in ref]
            vs_direct = []
            if cond != "direct":
                for e in mine:
                    d = by_key.get(("direct",) + key(e))
                    if d:
                        vs_direct.append(e["summary"]["score"] - d["summary"]["score"])
            probes = [e["summary"].get("probe") for e in mine if e["summary"].get("probe")]
            probe = {}
            if probes:
                for f in FIELDS:
                    answers = Counter(str(p["answer"].get(f)) for p in probes)
                    probe[f] = {"accuracy": statistics.fmean(float(p["correct"][f]) for p in probes),
                                "truth": probes[0]["truth"][f], "answers": dict(answers.most_common())}
            ds = [decision_stats(e["log"]) for e in mine]
            usage = [e["log"].get("usage") or {} for e in mine]
            report["conditions"][cond][persona] = {
                "episodes": len(mine),
                "vs_reference": _mean_se(diffs),
                "vs_reference_first_day": _mean_se(first),
                "vs_reference_last_day": _mean_se(last),
                "vs_direct": _mean_se(vs_direct) if cond != "direct" else None,
                "probe": probe,
                "wait_share": _mean_se([d["wait_share"] for d in ds])["mean"],
                "mean_wait_chosen": _mean_se([d["mean_wait_chosen"] for d in ds])["mean"],
                "waits_full_per_episode": _mean_se([d["waits_full"] for d in ds])["mean"],
                "waits_ended_by_partner_per_episode": _mean_se([d["waits_ended_by_partner"] for d in ds])["mean"],
                "full_waits_with_ready_food_per_episode": _mean_se([d["full_waits_with_ready_food"] for d in ds])["mean"],
                "action_mix": dict(sum((d["action_mix"] for d in ds), Counter())),
                "invalid_reasons": dict(sum((d["invalid_reasons"] for d in ds), Counter()).most_common()),
                "invalid_actions": dict(sum((d["invalid_actions"] for d in ds), Counter()).most_common()),
                "fallback_pairs": dict(sum((d["fallback_pairs"] for d in ds), Counter()).most_common()),
                "llm_calls_per_episode": _mean_se([u.get("calls") for u in usage])["mean"],
                "invalid_per_episode": _mean_se([e["log"].get("invalid_actions") for e in mine])["mean"],
                "fallbacks_per_episode": _mean_se([e["log"].get("fallbacks") for e in mine])["mean"],
                "thinking": sum(u.get("thinking", 0) for u in usage),
                "truncated": sum(u.get("truncated", 0) for u in usage),
                "errors": sum(u.get("errors", 0) for u in usage),
                "wrong_deliveries_per_episode": _mean_se([e["summary"]["wrong_deliveries"] for e in mine])["mean"],
            }
        allc = [e for e in eps if e["condition"] == cond]
        k = lambda e: (e["persona"], e["seed"], e["day"], e["episode"])
        report["overall"][cond] = {
            "vs_reference_by_day": {str(d): _mean_se([e["summary"]["score"] - ref[k(e)] for e in allc
                                                      if e["day"] == d and k(e) in ref]) for d in days},
            "vs_direct": _mean_se([e["summary"]["score"] - by_key[("direct",) + k(e)]["summary"]["score"]
                                   for e in allc if ("direct",) + k(e) in by_key]) if cond != "direct" else None,
        }
    report["dishes"] = dish_table(eps)
    return report


def dish_table(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """condition -> dish -> mean steps from issue to correct delivery, over completed orders.

    Also the open order at the end of each episode is counted as unfinished, so a
    dish the ego cannot finish shows up even though it has no duration.
    """
    out: Dict[str, Any] = defaultdict(dict)
    acc: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    unfinished: Dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        for o in e["summary"].get("orders", []):
            if o.get("completed_t") is not None:
                acc[e["condition"]][o["name"]].append(o["completed_t"] - o["issued_t"] + 1)
            else:
                unfinished[e["condition"]][o["name"]] += 1
    for cond, dishes in acc.items():
        for dish, ds in dishes.items():
            out[cond][dish] = {"completed": len(ds), "mean_steps": statistics.fmean(ds),
                               "unfinished_at_end": unfinished[cond][dish]}
    for cond, c in unfinished.items():
        for dish, n in c.items():
            out[cond].setdefault(dish, {"completed": 0, "mean_steps": None, "unfinished_at_end": n})
    return dict(out)


def _fmt(ms, signed=True):
    if not ms or ms["mean"] is None:
        return "-"
    s = f"{ms['mean']:+.1f}" if signed else f"{ms['mean']:.1f}"
    return s + (f"±{ms['se']:.1f}" if ms.get("se") is not None else "")


def format_report(rep: Dict[str, Any]) -> str:
    lines = ["== score minus reference on identical episodes (mean±SE over episodes) ==",
             f"{'condition':<12}" + "".join(f"{'day ' + d:>14}" for d in next(iter(rep['overall'].values()))["vs_reference_by_day"])
             + f"{'vs direct':>14}"]
    for cond, o in rep["overall"].items():
        lines.append(f"{cond:<12}" + "".join(f"{_fmt(v):>14}" for v in o["vs_reference_by_day"].values())
                     + f"{_fmt(o['vs_direct']):>14}")
    lines.append("\n== per persona: vs reference (all days | first day -> last day), vs direct ==")
    for cond, rows in rep["conditions"].items():
        lines.append(f"{cond}")
        for persona, r in rows.items():
            lines.append(f"  {persona:<12} {_fmt(r['vs_reference']):>12} | {_fmt(r['vs_reference_first_day']):>12}"
                         f" -> {_fmt(r['vs_reference_last_day']):<12} vs direct {_fmt(r['vs_direct'])}")
    lines.append("\n== partner probe: accuracy per field (most common answer; truth) ==")
    for cond, rows in rep["conditions"].items():
        lines.append(f"{cond}")
        for persona, r in rows.items():
            cells = []
            for f in FIELDS:
                p = r["probe"].get(f)
                if p:
                    top = next(iter(p["answers"]))
                    cells.append(f"{f} {p['accuracy']:.2f} ('{top}'; truth '{p['truth']}')")
            lines.append(f"  {persona:<12} " + " | ".join(cells))
    lines.append("\n== how decisions were spent (per episode) ==")
    lines.append(f"  {'condition/persona':<24}{'wait%':>6}{'wait n':>7}{'full':>6}{'early':>6}"
                 f"{'full+ready':>11}{'calls':>7}{'invalid':>8}{'fallbk':>7}{'wrong':>6}")
    for cond, rows in rep["conditions"].items():
        for persona, r in rows.items():
            def n(x, f="{:.1f}"):
                return f.format(x) if x is not None else "-"
            lines.append(f"  {cond + '/' + persona:<24}{n(r['wait_share'] and 100 * r['wait_share'], '{:.0f}'):>6}"
                         f"{n(r['mean_wait_chosen']):>7}{n(r['waits_full_per_episode']):>6}"
                         f"{n(r['waits_ended_by_partner_per_episode']):>6}"
                         f"{n(r['full_waits_with_ready_food_per_episode']):>11}{n(r['llm_calls_per_episode'], '{:.0f}'):>7}"
                         f"{n(r['invalid_per_episode']):>8}{n(r['fallbacks_per_episode']):>7}"
                         f"{n(r['wrong_deliveries_per_episode']):>6}")
    dishes = rep.get("dishes") or {}
    conds = [c for c in list(rep["conditions"]) + [REFERENCE] if c in dishes]
    names = sorted({d for c in conds for d in dishes[c]})
    if names:
        lines.append("\n== steps per completed order, by dish (completed orders in brackets) ==")
        lines.append(f"  {'dish':<36}" + "".join(f"{c:>16}" for c in conds))
        for d in names:
            cells = []
            for c in conds:
                r = dishes[c].get(d)
                cells.append(f"{r['mean_steps']:.1f} ({r['completed']})" if r and r["mean_steps"] is not None
                             else "-")
            lines.append(f"  {d:<36}" + "".join(f"{x:>16}" for x in cells))
    lines.append("\n== why replies were invalid (all personas; share of invalid replies) ==")
    for cond, rows in rep["conditions"].items():
        reasons, actions, pairs = Counter(), Counter(), Counter()
        for r in rows.values():
            reasons.update(r["invalid_reasons"])
            actions.update(r["invalid_actions"])
            pairs.update(r["fallback_pairs"])
        total = sum(reasons.values())
        lines.append(f"{cond}: {total} invalid replies, {sum(pairs.values())} fallbacks")
        for k, v in reasons.most_common(6):
            lines.append(f"  {v / total:>5.0%}  {k}")
        lines.append("  invalid actions: " + ", ".join(f"{k} {v / total:.0%}" for k, v in actions.most_common(6)))
        if pairs:
            lines.append("  fallbacks (first -> retry): " + ", ".join(f"{k} x{v}" for k, v in pairs.most_common(4)))
    health = [(c, p, r) for c, rows in rep["conditions"].items() for p, r in rows.items()
              if r["thinking"] or r["truncated"] or r["errors"]]
    lines.append("\n== LLM health ==")
    lines += [f"  {c}/{p}: thinking {r['thinking']}, truncated {r['truncated']}, errors {r['errors']}"
              for c, p, r in health] or ["  no thinking, truncated or failed replies"]
    lines.append("\nfull+ready = waits that ran their full length while finished food sat in a utensil "
                 "the ego can reach (from the next prompt)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, help="also write the full report as JSON")
    args = ap.parse_args(argv)
    rep = analyse(args.run_dir)
    if not rep["conditions"]:
        raise SystemExit(f"no LLM-condition episodes under {args.run_dir}/runs")
    print(format_report(rep))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
