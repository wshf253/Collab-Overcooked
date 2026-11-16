"""Gates before any learning run (docs/PERSONA_PARTNER_PLAN_v2.md, section 10).

Gate 1, is the partner usable? (scripted steady ego, each card, the twin or the LLM partner)

    python -m coop.persona_check gate1 outputs/gate1_llm [--twin outputs/gate1_twin]
        [--calibrate 0 1] [--judge 2 3] [--bar 0.7] [--out outputs/gate1_llm/gate1.json]

* competence: score with each card minus the neutral partner (same model, no rules) on the
  same episodes, mean ± SE over seeds;
* adherence per rule: adherent / opportunities, on the calibration seeds (to set the bar)
  and on the judge seeds (judged against the bar, frozen beforehand);
* distinctness: leave-one-seed-out nearest-centroid classifier over the cards, from
  per-episode features of the partner's behaviour (loads per utensil, stage shares, order
  choice, hand-over spots, wait share); accuracy and confusion matrix;
* consistency: adherence by episode within the day (it should not fall);
* infrastructure: the partner's invalid replies and fallbacks per episode.

Gate 2, does knowing the partner matter? (scripted egos: aligned / generic / misaligned)

    python -m coop.persona_check gate2 outputs/gate2_llm [--out ...]

* aligned − generic and aligned − misaligned, for score and for alignment (mean over the
  card's evaluable rules of aligned / opportunities), paired over runs (card × seed), on
  the first and second half of the seeds separately and pooled, and per card.
  For the rule-based twin the same table comes from ``python -m coop.voi --cards ...``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analyze import _layout_items, _mean_se, action_stage, load_episodes, order_stats
from .rules import NEUTRAL

LEFT_CENTER_RIGHT = ("left", "center", "right")


# ---------------------------------------------------------------------- #
# Per-episode quantities
# ---------------------------------------------------------------------- #
def rule_rates(summary: Dict[str, Any], side: str) -> Dict[str, Tuple[int, int]]:
    """rule -> (successes, opportunities) for the partner (adherent) or ego (aligned)."""
    key = "adherent" if side == "partner" else "aligned"
    out = {}
    for rule, r in (summary.get("rules") or {}).items():
        out[rule] = (r[side][key], r[side]["opportunities"])
    return out


def composite(summary: Dict[str, Any], side: str) -> Tuple[Optional[float], float]:
    """Mean over evaluable rules of the rate, and the share of rules that were evaluable."""
    rates = [s / n for s, n in rule_rates(summary, side).values() if n]
    total = len(summary.get("rules") or {})
    return (statistics.fmean(rates) if rates else None), (len(rates) / total if total else 0.0)


def features(e: Dict[str, Any], raw: set, finals: set, width: int) -> Dict[str, float]:
    """Behaviour of the partner in one episode (for the distinctness classifier)."""
    ego = e["summary"].get("ego_index", 0)
    other = 1 - ego
    loads: Counter = Counter()
    stages: Counter = Counter()
    places: Counter = Counter()
    n = 0
    with open(e["dir"] / "steps.jsonl") as f:
        for line in f:
            s = json.loads(line)
            a = s["completed"][other]
            if not a:
                continue
            n += 1
            stages[action_stage(a, raw, finals)] += 1
            m = re.fullmatch(r"put_obj_in_utensil\((\w+)\)", a.replace(" ", ""))
            if m:
                loads[m.group(1)] += 1
            if a.startswith("place_obj_on_counter"):
                p = s["players"][other]
                pos, ori = p.get("position"), p.get("orientation")
                if pos and ori:
                    x = pos[0] + ori[0]
                    places[LEFT_CENTER_RIGHT[min(2, x * 3 // width)]] += 1
    f: Dict[str, float] = {}
    tl = sum(loads.values()) or 1
    for u, c in loads.items():
        f[f"load:{u}"] = c / tl
    for st in ("prep", "cook", "serve", "handover"):
        f[f"stage:{st}"] = stages[st] / (n or 1)
    tp = sum(places.values()) or 1
    for g in LEFT_CENTER_RIGHT:
        f[f"place:{g}"] = places[g] / tp
    os_ = order_stats(e["summary"]) or {}
    f["order:newer"] = (os_.get("partner") or {}).get("newer_share") or 0.0
    f["actions"] = n / max(1, e["summary"].get("steps", 300)) * 10
    try:
        truth = json.loads((e["dir"] / "partner_truth.json").read_text())
        tr = truth.get("trace") or []
        if tr and "stage" in tr[0]:
            f["wait"] = sum(t.get("stage") == "wait" for t in tr) / len(tr)
        elif tr:
            f["wait"] = sum(t.get("status") != "act" for t in tr) / len(tr)
    except (OSError, ValueError):
        pass
    return f


def classify(rows: List[Tuple[str, int, Dict[str, float]]]) -> Dict[str, Any]:
    """Leave-one-seed-out nearest-centroid classifier; rows: (card, seed, features)."""
    keys = sorted({k for _, _, f in rows for k in f})
    seeds = sorted({s for _, s, _ in rows})
    cards = sorted({c for c, _, _ in rows})
    confusion = {c: Counter() for c in cards}
    correct = total = 0
    for held in seeds:
        train = [(c, [f.get(k, 0.0) for k in keys]) for c, s, f in rows if s != held]
        test = [(c, [f.get(k, 0.0) for k in keys]) for c, s, f in rows if s == held]
        if not train or not test:
            continue
        cols = list(zip(*[v for _, v in train]))
        mu = [statistics.fmean(c) for c in cols]
        sd = [statistics.pstdev(c) or 1.0 for c in cols]
        z = lambda v: [(x - m) / s for x, m, s in zip(v, mu, sd)]
        cent = {}
        for c in cards:
            vs = [z(v) for cc, v in train if cc == c]
            if vs:
                cent[c] = [statistics.fmean(col) for col in zip(*vs)]
        for c, v in test:
            zv = z(v)
            pred = min(cent, key=lambda k: sum((a - b) ** 2 for a, b in zip(zv, cent[k])))
            confusion[c][pred] += 1
            correct += pred == c
            total += 1
    return {"accuracy": correct / total if total else None, "episodes": total, "chance": 1 / len(cards) if cards else None,
            "confusion": {c: dict(v) for c, v in confusion.items()}, "features": keys}


# ---------------------------------------------------------------------- #
# Gate 1
# ---------------------------------------------------------------------- #
def gate1(out: Path, calibrate: List[int], judge: List[int], bar: float) -> Dict[str, Any]:
    eps = load_episodes(out)
    cfg = json.loads((out / "config.json").read_text())
    raw, finals = _layout_items(out)
    width = 9
    try:
        from overcooked_ai_py.data.layouts import read_layout_dict
        width = len(read_layout_dict(cfg["layout"])["grid"].strip().splitlines()[0].strip())
    except Exception:
        pass
    cards = [p for p in cfg["personas"] if p != NEUTRAL]
    by = {(e["persona"], e["seed"], e["day"], e["episode"]): e for e in eps}
    rep: Dict[str, Any] = {"folder": str(out), "partner": (cfg.get("partner") or {}).get("type"),
                           "calibrate_seeds": calibrate, "judge_seeds": judge, "bar": bar, "cards": {}}
    for card in cards:
        mine = [e for e in eps if e["persona"] == card]
        if not mine:
            continue
        diffs: Dict[int, List[float]] = defaultdict(list)
        for e in mine:
            n = by.get((NEUTRAL, e["seed"], e["day"], e["episode"]))
            if n is not None:
                diffs[e["seed"]].append(e["summary"]["score"] - n["summary"]["score"])
        rules: Dict[str, Any] = {}
        for rule in (mine[0]["summary"].get("rules") or {}):
            row = {}
            for label, seeds in (("calibrate", calibrate), ("judge", judge), ("all", None)):
                sel = [e for e in mine if seeds is None or e["seed"] in seeds]
                s = sum(rule_rates(e["summary"], "partner")[rule][0] for e in sel)
                n = sum(rule_rates(e["summary"], "partner")[rule][1] for e in sel)
                row[label] = {"adherent": s, "opportunities": n, "rate": s / n if n else None}
            by_ep = {}
            for k in sorted({e["episode"] for e in mine}):
                sel = [e for e in mine if e["episode"] == k]
                s = sum(rule_rates(e["summary"], "partner")[rule][0] for e in sel)
                n = sum(rule_rates(e["summary"], "partner")[rule][1] for e in sel)
                by_ep[str(k)] = s / n if n else None
            row["by_episode"] = by_ep
            j = row["judge"]["rate"]
            row["pass"] = None if j is None else j >= bar
            rules[rule] = row
        llm = [e["summary"].get("partner_llm") or {} for e in mine]
        rep["cards"][card] = {
            "episodes": len(mine),
            "score": _mean_se([e["summary"]["score"] for e in mine]),
            "vs_neutral": _mean_se([statistics.fmean(v) for v in diffs.values()]),
            "rules": rules,
            "invalid_per_episode": _mean_se([x.get("invalid_actions") for x in llm if x])["mean"],
            "fallbacks_per_episode": _mean_se([x.get("fallbacks") for x in llm if x])["mean"],
        }
    neutral = [e for e in eps if e["persona"] == NEUTRAL]
    rep["neutral_score"] = _mean_se([e["summary"]["score"] for e in neutral])
    rows = [(e["persona"], e["seed"], features(e, raw, finals, width)) for e in eps if e["persona"] in cards]
    rep["distinctness"] = classify(rows)
    return rep


def format_gate1(rep: Dict[str, Any], twin: Optional[Dict[str, Any]] = None) -> str:
    f = lambda ms: "-" if not ms or ms.get("mean") is None else (
        f"{ms['mean']:+.1f}" + (f"±{ms['se']:.1f}" if ms.get("se") is not None else ""))
    pct = lambda x: "-" if x is None else f"{100 * x:.0f}%"
    lines = [f"== Gate 1: {rep['folder']} (partner: {rep['partner']}) ==",
             f"neutral partner score {f(rep['neutral_score'])}; bar {pct(rep['bar'])} on judge seeds "
             f"{rep['judge_seeds']} (calibration seeds {rep['calibrate_seeds']})"]
    for card, c in rep["cards"].items():
        infra = ""
        if c.get("invalid_per_episode") is not None:
            infra = f"; partner invalid {c['invalid_per_episode']:.1f} / fallbacks {c['fallbacks_per_episode']:.1f} per ep"
        lines.append(f"\n{card}: score {f(c['score'])}, vs neutral {f(c['vs_neutral'])}{infra}")
        lines.append(f"  {'rule':<50}{'calibrate':>16}{'judge':>16}{'pass':>6}{'by episode':>18}"
                     + ("{:>14}".format("twin (all)") if twin else ""))
        for rule, r in c["rules"].items():
            cal, jud = r["calibrate"], r["judge"]
            ep = " ".join(pct(v) for v in r["by_episode"].values())
            tw = ""
            if twin:
                t = ((twin.get("cards") or {}).get(card) or {}).get("rules", {}).get(rule, {}).get("all", {})
                tw = f"{pct(t.get('rate')):>6} of {t.get('opportunities', 0):<4}"
            lines.append(f"  {rule:<50}{pct(cal['rate']):>7} of {cal['opportunities']:<5}"
                         f"{pct(jud['rate']):>7} of {jud['opportunities']:<5}"
                         f"{'-' if r['pass'] is None else ('yes' if r['pass'] else 'NO'):>6}{ep:>18}{tw:>14}")
    d = rep["distinctness"]
    lines.append(f"\ndistinctness (leave-one-seed-out, {d['episodes']} episodes): accuracy {pct(d['accuracy'])} "
                 f"(chance {pct(d['chance'])})")
    for c, row in d["confusion"].items():
        lines.append(f"  {c:<8} -> " + ", ".join(f"{k} {v}" for k, v in sorted(row.items())))
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Gate 2
# ---------------------------------------------------------------------- #
EGOS = ("aligned", "generic", "misaligned")


def gate2_rows(eps: List[Dict[str, Any]]) -> Dict[Tuple[str, str, int], Dict[str, Optional[float]]]:
    """(ego condition, card, seed) -> mean score, alignment and ego actions per episode."""
    acc: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for e in eps:
        key = (e["condition"], e["persona"], e["seed"])
        acc[key]["score"].append(e["summary"]["score"])
        al, _ = composite(e["summary"], "ego")
        if al is not None:
            acc[key]["alignment"].append(al)
        ad, _ = composite(e["summary"], "partner")
        if ad is not None:
            acc[key]["adherence"].append(ad)
    return {k: {m: statistics.fmean(v) if v else None for m, v in d.items()} for k, d in acc.items()}


def gate2_table(rows, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    cards = sorted({c for _, c, _ in rows})
    out: Dict[str, Any] = {"means": {}, "diffs": {}}
    for ego in EGOS:
        for m in ("score", "alignment"):
            vals = [r[m] for (g, c, s), r in rows.items() if g == ego and (seeds is None or s in seeds)
                    and r.get(m) is not None]
            out["means"][f"{ego}:{m}"] = _mean_se(vals)
    for other in ("generic", "misaligned"):
        for m in ("score", "alignment"):
            d, per = [], defaultdict(list)
            for (g, c, s), r in rows.items():
                if g != "aligned" or (seeds is not None and s not in seeds):
                    continue
                o = rows.get((other, c, s))
                if o and r.get(m) is not None and o.get(m) is not None:
                    d.append(r[m] - o[m])
                    per[c].append(r[m] - o[m])
            out["diffs"][f"aligned-{other}:{m}"] = {"all": _mean_se(d), "cards": {c: _mean_se(v) for c, v in per.items()}}
    out["cards"] = cards
    return out


def gate2(out: Path) -> Dict[str, Any]:
    eps = load_episodes(out)
    rows = gate2_rows(eps)
    seeds = sorted({s for _, _, s in rows})
    half = len(seeds) // 2
    sets = {"first": seeds[:half] or seeds, "second": seeds[half:] or seeds, "pooled": seeds}
    return {"folder": str(out), "seed_sets": sets, "tables": {k: gate2_table(rows, v) for k, v in sets.items()}}


def format_gate2(rep: Dict[str, Any]) -> str:
    f = lambda ms, scale=1: "-" if not ms or ms.get("mean") is None else (
        f"{scale * ms['mean']:+.1f}" + (f"±{scale * ms['se']:.1f}" if ms.get("se") is not None else ""))
    lines = [f"== Gate 2: {rep['folder']} =="]
    for name, t in rep["tables"].items():
        lines.append(f"\nseeds {rep['seed_sets'][name]} ({name})")
        for ego in EGOS:
            lines.append(f"  {ego:<11} score {f(t['means'][f'{ego}:score']):>12}   alignment "
                         f"{f(t['means'][f'{ego}:alignment'], 100):>12} %")
        for k, v in t["diffs"].items():
            scale = 100 if k.endswith("alignment") else 1
            per = ", ".join(f"{c} {f(x, scale)}" for c, x in sorted(v["cards"].items()))
            lines.append(f"  {k:<28} {f(v['all'], scale):>12}   [{per}]")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g1 = sub.add_parser("gate1")
    g1.add_argument("run_dir", type=Path)
    g1.add_argument("--twin", type=Path, help="the twin's gate-1 folder, shown next to the LLM partner")
    g1.add_argument("--calibrate", nargs="+", type=int, default=[0, 1])
    g1.add_argument("--judge", nargs="+", type=int, default=[2, 3])
    g1.add_argument("--bar", type=float, default=0.7)
    g1.add_argument("--out", type=Path)
    g2 = sub.add_parser("gate2")
    g2.add_argument("run_dir", type=Path)
    g2.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "gate1":
        rep = gate1(args.run_dir, args.calibrate, args.judge, args.bar)
        twin = gate1(args.twin, args.calibrate, args.judge, args.bar) if args.twin else None
        print(format_gate1(rep, twin))
        if twin:
            rep["twin"] = twin
    else:
        rep = gate2(args.run_dir)
        print(format_gate2(rep))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------- #
# Direct simulation with scripted players (no runner, no files): used by coop.voi --cards
# ---------------------------------------------------------------------- #
def play_rule_episode(menu, layout: str, horizon: int, seed: int, partner_rules, ego_player,
                      view_radius=None, open_orders: int = 1, ego_index: int = 0) -> Dict[str, Any]:
    """One episode: ``ego_player`` (a SimulatedHuman or RulePlayer in ego mode) with the
    rule-based twin of ``partner_rules``; returns the score and the rule summary."""
    import contextlib
    import io
    from .env import ContinuousOrderEnv
    from .kitchen import Kitchen
    from .rule_eval import RuleEvaluator, intent_target
    from .rule_partner import RulePlayer
    from .rules import load_library
    env = ContinuousOrderEnv(menu, layout=layout, horizon=horizon, open_orders=open_orders)
    with contextlib.redirect_stdout(io.StringIO()):
        start = env.reset(seed)
    other = 1 - ego_index
    partner = RulePlayer(partner_rules, list(menu), "partner", "twin")
    ego_player.reset(env.mdp, ego_index, seed, view_radius=view_radius)
    partner.reset(env.mdp, other, seed, view_radius=view_radius)
    k = Kitchen(env.mdp, ego_index, start)
    ev = RuleEvaluator(partner_rules, load_library(), k, list(menu), ego_index, other, open_orders, env.mdp)
    while not env.done:
        t, pre = env.t, env.state
        moves = [None, None]
        moves[ego_index] = ego_player.act(pre, t)
        moves[other] = partner.act(pre, t)
        intents = {"ego": intent_target(ego_player, k), "partner": intent_target(partner, k)}
        with contextlib.redirect_stdout(io.StringIO()):
            env.step([m[0] for m in moves], [m[1] for m in moves])
        ev.observe(t, pre, env.state, intents)
    return {"score": env.summary()["score"], "rules": ev.summary(), "steps": env.t}
