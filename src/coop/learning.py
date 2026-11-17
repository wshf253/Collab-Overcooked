"""Learning curves for multi-day partner learning (docs/PERSONA_PARTNER_PLAN_v2.md, sections 6, 9, 14).

    PYTHONPATH=src:lib/overcooked_ai python -m coop.learning outputs/days_persona [--out file.json]

The unit is the run (persona x seed). **Score is the primary outcome, alignment the
secondary one** (decided 2026-11-17). Reports, per condition and day 0..D:

* **held-out score** minus the scripted reference on the same held-out episodes (the same
  episodes every day, so the curve is paired across days); conditions without memory across
  days were evaluated once (day 0), and that value is used for every day;
* **alignment** with the partner's rules on the held-out episodes (mean over evaluable
  rules), and the partner's adherence;
* **contrasts** paired over runs: each condition minus tom_struct and minus memory_fb, per
  day and averaged over days 1..D, also per persona; for score (primary) and alignment
  (secondary);
* **interaction episodes**: each condition minus tom_struct on the day's own episodes (the
  same episode seeds in every condition). ``oracle`` minus ``tom_struct`` here is the score
  value of knowing the partner for this ego, on 3 episodes a day instead of 2 held-out ones;
* **knowledge probe** accuracy for rules exposed so far, not yet exposed, and absent;
* **exposure**: cumulative opportunities per rule inside the ego's view;
* **feedback** (words, rule recitation) and **consolidation** (trimmed lines);
* **infrastructure**: invalid replies and fallbacks per episode, ego and partner.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analyze import _mean_se
from .persona_check import composite

REFERENCE = "_reference"
BASELINES = ("tom_struct", "memory_fb")


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def collect(out: Path) -> Dict[str, Any]:
    evals: Dict[Tuple[str, str, int, int, int], Dict[str, Any]] = {}
    inter: Dict[Tuple[str, str, int, int, int], Dict[str, Any]] = {}
    probes: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    feedback: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    consol: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    for run in sorted(out.glob("runs/*/*/seed*")):
        cond, persona, seed = run.parent.parent.name, run.parent.name, int(run.name[4:])
        for day_dir in sorted(run.glob("day*")):
            day = int(day_dir.name[3:])
            for e in sorted(day_dir.glob("eval*")):
                s = _load(e / "summary.json")
                if s:
                    evals[(cond, persona, seed, day, int(e.name[4:]))] = s
            for e in sorted(day_dir.glob("ep*")):
                s = _load(e / "summary.json")
                if s:
                    inter[(cond, persona, seed, day, int(e.name[2:]))] = s
            for name, store in (("knowledge_probe.json", probes), ("feedback.json", feedback),
                                ("consolidation.json", consol)):
                d = _load(day_dir / name)
                if d:
                    store[(cond, persona, seed, day)] = d
    cfg = _load(out / "config.json") or {}
    return {"evals": evals, "inter": inter, "probes": probes, "feedback": feedback, "consol": consol,
            "days": int(cfg.get("days", 0)) or max([k[3] for k in evals] or [0])}


def held_out(data: Dict[str, Any]) -> Dict[Tuple[str, str, int], Dict[int, Dict[str, Optional[float]]]]:
    """(condition, persona, seed) -> day -> mean over held-out episodes of score minus
    reference, alignment and adherence (memoryless conditions: day 0 repeated)."""
    evals, days = data["evals"], data["days"]
    ref = {(p, s, e): v["score"] for (c, p, s, d, e), v in evals.items() if c == REFERENCE}
    acc: Dict[Tuple[str, str, int], Dict[int, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for (c, p, s, d, e), v in evals.items():
        if c == REFERENCE:
            continue
        r = acc[(c, p, s)][d]
        if (p, s, e) in ref:
            r["vs_ref"].append(v["score"] - ref[(p, s, e)])
        r["score"].append(v["score"])
        al, _ = composite(v, "ego")
        ad, _ = composite(v, "partner")
        if al is not None:
            r["alignment"].append(al)
        if ad is not None:
            r["adherence"].append(ad)
    out: Dict[Tuple[str, str, int], Dict[int, Dict[str, Optional[float]]]] = {}
    for run, by_day in acc.items():
        rows = {d: {m: statistics.fmean(v) if v else None for m, v in ms.items()} for d, ms in by_day.items()}
        if set(rows) == {0}:            # evaluated once: no memory across days
            rows = {d: dict(rows[0]) for d in range(0, days + 1)}
        out[run] = rows
    return out


def interaction(data: Dict[str, Any]) -> Dict[Tuple[str, str, int], Dict[int, Dict[str, Optional[float]]]]:
    """(condition, persona, seed) -> day -> mean score and alignment over the day's interaction
    episodes (seeds are shared across conditions, so differences are paired)."""
    acc: Dict[Tuple[str, str, int], Dict[int, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    ref = {(p, s, d, e): v["score"] for (c, p, s, d, e), v in data["inter"].items() if c == REFERENCE}
    for (c, p, s, d, e), v in data["inter"].items():
        if c == REFERENCE:
            continue
        r = acc[(c, p, s)][d]
        r["score"].append(v["score"])
        if (p, s, d, e) in ref:
            r["vs_ref"].append(v["score"] - ref[(p, s, d, e)])
        al, _ = composite(v, "ego")
        if al is not None:
            r["alignment"].append(al)
    return {run: {d: {m: statistics.fmean(v) if v else None for m, v in ms.items()} for d, ms in by_day.items()}
            for run, by_day in acc.items()}


def curves(runs, metric: str, scale: float = 1.0) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for (c, p, s), by_day in runs.items():
        for d, r in by_day.items():
            out[c].setdefault(d, [])
            if r.get(metric) is not None:
                out[c][d].append(scale * r[metric])
    return {c: {d: _mean_se(v) for d, v in sorted(ds.items())} for c, ds in out.items()}


def contrasts(runs, base: str, days: int, metric: str = "vs_ref", scale: float = 1.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    conds = sorted({c for c, _, _ in runs})
    if base not in conds:
        return out
    for c in conds:
        if c == base:
            continue
        per_day, avg, per_persona = defaultdict(list), [], defaultdict(list)
        for (cc, p, s), by_day in runs.items():
            if cc != c or (base, p, s) not in runs:
                continue
            b = runs[(base, p, s)]
            diffs = []
            for d, r in by_day.items():
                if d in b and r.get(metric) is not None and b[d].get(metric) is not None:
                    x = scale * (r[metric] - b[d][metric])
                    per_day[d].append(x)
                    if d >= 1:
                        diffs.append(x)
            if diffs:
                avg.append(statistics.fmean(diffs))
                per_persona[p].append(statistics.fmean(diffs))
        out[c] = {"days": {d: _mean_se(v) for d, v in sorted(per_day.items())},
                  f"days_1_{days}": _mean_se(avg), "personas": {p: _mean_se(v) for p, v in per_persona.items()}}
    return out


def probe_table(data) -> Dict[str, Dict[int, Dict[str, Any]]]:
    acc: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (c, p, s, d), pr in data["probes"].items():
        for g, row in (pr.get("accuracy") or {}).items():
            if row.get("accuracy") is not None:
                acc[c][d][g].append(row["accuracy"])
    return {c: {d: {g: _mean_se(v)["mean"] for g, v in gs.items()} for d, gs in sorted(ds.items())}
            for c, ds in acc.items()}


def exposure_table(data) -> Dict[str, Dict[int, float]]:
    """rule -> day -> mean over runs of cumulative exposure up to that day (interaction episodes)."""
    per_run: Dict[Tuple[str, str, int], Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for (c, p, s, d, e), v in data["inter"].items():
        if c == REFERENCE:
            continue
        for rule, r in (v.get("rules") or {}).items():
            per_run[(c, p, s)][rule][d] = per_run[(c, p, s)][rule].get(d, 0) + r.get("exposure", 0)
    days = data["days"]
    out: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for run, rules in per_run.items():
        for rule, by_day in rules.items():
            total = 0
            for d in range(1, days + 1):
                total += by_day.get(d, 0)
                out[rule][d].append(total)
    return {r: {d: statistics.fmean(v) for d, v in ds.items()} for r, ds in out.items()}


def infra(data) -> Dict[str, Dict[str, Optional[float]]]:
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (c, p, s, d, e), v in list(data["inter"].items()) + list(data["evals"].items()):
        u = v.get("llm_usage") or {}
        pl = v.get("partner_llm") or {}
        for k, x in (("ego_invalid", u.get("invalid_actions")), ("ego_fallbacks", u.get("fallbacks")),
                     ("partner_invalid", pl.get("invalid_actions")), ("partner_fallbacks", pl.get("fallbacks"))):
            if x is not None:
                acc[c][k].append(x)
    return {c: {k: statistics.fmean(v) for k, v in ks.items()} for c, ks in acc.items()}


def report(out: Path) -> Dict[str, Any]:
    data = collect(out)
    runs = held_out(data)
    fb: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"days": 0, "ok": 0, "words": [], "recited": 0})
    for (c, p, s, d), f in data["feedback"].items():
        r = fb[c]
        r["days"] += 1
        r["ok"] += bool(f.get("ok"))
        r["words"].append((f.get("checks") or {}).get("words", 0))
        r["recited"] += bool((f.get("checks") or {}).get("recitation"))
    trims: Dict[str, List[int]] = defaultdict(list)
    for (c, p, s, d), x in data["consol"].items():
        trims[c].append(len(x.get("trimmed") or []))
    return {
        "folder": str(out), "days": data["days"],
        "held_out_vs_reference": curves(runs, "vs_ref"),
        "interaction_vs_reference": curves(interaction(data), "vs_ref"),
        "alignment": curves(runs, "alignment", 100),
        "adherence": curves(runs, "adherence", 100),
        "contrasts": {b: contrasts(runs, b, data["days"]) for b in BASELINES},
        "alignment_contrasts": {b: contrasts(runs, b, data["days"], "alignment", 100) for b in BASELINES},
        "interaction_contrasts": {
            "score": contrasts(interaction(data), "tom_struct", data["days"], "score"),
            "alignment": contrasts(interaction(data), "tom_struct", data["days"], "alignment", 100)},
        "probe": probe_table(data),
        "exposure": exposure_table(data),
        "feedback": {c: {"days": r["days"], "ok": r["ok"], "mean_words": statistics.fmean(r["words"]) if r["words"] else None,
                         "recited_days": r["recited"]} for c, r in fb.items()},
        "trimmed_lines_per_day": {c: statistics.fmean(v) for c, v in trims.items() if v},
        "infra": infra(data),
    }


def format_report(rep: Dict[str, Any]) -> str:
    f = lambda ms: "-" if not ms or ms.get("mean") is None else (
        f"{ms['mean']:+.1f}" + (f"±{ms['se']:.1f}" if ms.get("se") is not None else ""))
    days = list(range(0, rep["days"] + 1))
    head = f"{'condition':<14}" + "".join(f"{'day ' + str(d):>12}" for d in days)
    lines = [f"== {rep['folder']} ==", "score is the primary outcome, alignment the secondary one"]
    avg_key = "days_1_%d" % rep["days"]

    def contrast_block(title, cs, ds):
        if not cs:
            return
        lines.extend([f"\n== {title} ==",
                      f"{'condition':<14}" + "".join(f"{'day ' + str(d):>12}" for d in ds) + f"{'days 1-' + str(rep['days']):>14}"])
        for c, r in cs.items():
            lines.append(f"{c:<14}" + "".join(f"{f(r['days'].get(d)):>12}" for d in ds) + f"{f(r[avg_key]):>14}")
            lines.append(f"{'':<14}per persona: " + ", ".join(f"{p} {f(v)}" for p, v in sorted(r["personas"].items())))

    if any(v for v in rep["held_out_vs_reference"].values()):
        lines += ["\n== PRIMARY: held-out score minus reference (mean±SE over runs) ==", head]
        for c, by_day in sorted(rep["held_out_vs_reference"].items()):
            lines.append(f"{c:<14}" + "".join(f"{f(by_day.get(d)):>12}" for d in days))
    for base, cs in rep["contrasts"].items():
        contrast_block(f"PRIMARY: held-out score, condition minus {base} (paired over runs)", cs, days)
    inter_days = [d for d in days if d >= 1]
    inter_ref = rep.get("interaction_vs_reference") or {}
    if any(v for v in inter_ref.values()):
        lines += ["\n== PRIMARY: interaction-episode score minus reference (mean±SE over runs) ==",
                  f"{'condition':<14}" + "".join(f"{'day ' + str(d):>12}" for d in inter_days)]
        for c, by_day in sorted(inter_ref.items()):
            lines.append(f"{c:<14}" + "".join(f"{f(by_day.get(d)):>12}" for d in inter_days))
    contrast_block("PRIMARY: interaction-episode score, condition minus tom_struct (paired; "
                   "oracle - tom_struct = score value of knowing the partner)",
                   rep.get("interaction_contrasts", {}).get("score"), inter_days)
    for title, key in (("secondary: alignment with the partner's rules on held-out episodes (%)", "alignment"),
                       ("partner adherence on held-out episodes (%)", "adherence")):
        if not any(v for v in rep[key].values()):
            continue
        lines += [f"\n== {title} ==", head]
        for c, by_day in sorted(rep[key].items()):
            lines.append(f"{c:<14}" + "".join(f"{f(by_day.get(d)):>12}" for d in days))
    for base, cs in rep.get("alignment_contrasts", {}).items():
        contrast_block(f"secondary: held-out alignment (points), condition minus {base}", cs, days)
    contrast_block("secondary: interaction-episode alignment (points), condition minus tom_struct",
                   rep.get("interaction_contrasts", {}).get("alignment"), inter_days)
    if rep["probe"]:
        lines.append("\n== knowledge probe accuracy (exposed / not yet exposed / absent rules) ==")
        pct = lambda x: "-" if x is None else f"{100 * x:.0f}"
        for c, by_day in sorted(rep["probe"].items()):
            cells = [f"d{d} {pct(g.get('exposed'))}/{pct(g.get('not_exposed'))}/{pct(g.get('absent'))}"
                     for d, g in sorted(by_day.items())]
            lines.append(f"{c:<14}" + "  ".join(cells))
    if rep["exposure"]:
        lines.append("\n== cumulative exposure (partner's opportunities in the ego's view, mean over runs) ==")
        for rule, by_day in sorted(rep["exposure"].items()):
            lines.append(f"  {rule:<52}" + " ".join(f"d{d} {v:.1f}" for d, v in sorted(by_day.items())))
    if rep["feedback"]:
        lines.append("\n== daily feedback ==")
        for c, r in sorted(rep["feedback"].items()):
            w = "-" if r["mean_words"] is None else f"{r['mean_words']:.0f}"
            lines.append(f"  {c:<14} days {r['days']}, written {r['ok']}, mean words {w}, days with rule recitation {r['recited_days']}")
    if rep["trimmed_lines_per_day"]:
        lines.append("\n== partner file: lines trimmed to the section budgets per day ==")
        for c, v in sorted(rep["trimmed_lines_per_day"].items()):
            lines.append(f"  {c:<14} {v:.1f}")
    if rep["infra"]:
        lines.append("\n== infrastructure (per episode) ==")
        for c, r in sorted(rep["infra"].items()):
            lines.append(f"  {c:<14} " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(r.items())))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rep = report(args.run_dir)
    print(format_report(rep))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
