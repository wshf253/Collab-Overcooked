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

Why does the partner break a rule? (any folder with a rule-following partner)

    python -m coop.persona_check why outputs/gate1_llm [--card p3] [--rule serves] [--examples 4]

* per card and rule: adherent / judged decisions, the partner's non-adherent decisions
  grouped by what happened (numbers masked), how often a reminder for that rule was shown
  at the deciding step, and examples with the partner's stated goal and chosen action.

Re-score saved episodes with the current rule checks (no model calls)

    python -m coop.persona_check rescore outputs/gate1_llm [--workers 8]

replays each episode's recorded actions (``steps.jsonl``) on the same seed, checks that the
replay matches (score and positions), and writes ``rule_events.rescored.jsonl`` and
``rules.rescored.json`` next to the originals. ``on_contention`` keeps its original
judgments (it needs the players' intended targets, which are not saved). ``gate1``,
``gate2`` and ``why`` read the re-scored files with ``--rescored``.
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
def gate1(out: Path, calibrate: List[int], judge: List[int], bar: float, rescored: bool = False) -> Dict[str, Any]:
    eps = load_episodes(out)
    n_rescored = use_rescored(eps) if rescored else 0
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
    if rescored:
        rep["rescored"] = {"episodes": len(eps), "rescored": n_rescored}
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
    if rep.get("rescored"):
        r = rep["rescored"]
        lines.append(f"rule results re-scored with the current checks: {r['rescored']} of {r['episodes']} episodes")
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
EXTRA_EGOS = ("uninformed",)     # twin screen only (coop.voi --cards)


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
    egos = [e for e in EGOS + EXTRA_EGOS if any(g == e for g, _, _ in rows)]
    out["egos"] = egos
    for ego in egos:
        for m in ("score", "alignment"):
            vals = [r[m] for (g, c, s), r in rows.items() if g == ego and (seeds is None or s in seeds)
                    and r.get(m) is not None]
            out["means"][f"{ego}:{m}"] = _mean_se(vals)
    for other in [e for e in egos if e != "aligned"]:
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


def gate2(out: Path, rescored: bool = False) -> Dict[str, Any]:
    eps = load_episodes(out)
    if rescored:
        use_rescored(eps)
    rows = gate2_rows(eps)
    seeds = sorted({s for _, _, s in rows})
    half = len(seeds) // 2
    sets = {"first": seeds[:half] or seeds, "second": seeds[half:] or seeds, "pooled": seeds}
    return {"folder": str(out), "seed_sets": sets, "tables": {k: gate2_table(rows, v) for k, v in sets.items()}}


def format_gate2(rep: Dict[str, Any]) -> str:
    f = lambda ms, scale=1: "-" if not ms or ms.get("mean") is None else (
        f"{scale * ms['mean']:+.1f}" + (f"±{scale * ms['se']:.1f}" if ms.get("se") is not None else ""))
    lines = [f"== Gate 2: {rep['folder']} ==",
             "score is the primary outcome, alignment the secondary one (decided 2026-11-17)"]
    for name, t in rep["tables"].items():
        lines.append(f"\nseeds {rep['seed_sets'][name]} ({name})")
        for ego in t.get("egos", EGOS):
            lines.append(f"  {ego:<11} score {f(t['means'][f'{ego}:score']):>12}   alignment "
                         f"{f(t['means'][f'{ego}:alignment'], 100):>12} %")
        for m, label in (("score", "primary"), ("alignment", "secondary")):
            lines.append(f"  {m} ({label}), aligned minus:")
            for k, v in t["diffs"].items():
                if not k.endswith(":" + m):
                    continue
                scale = 100 if m == "alignment" else 1
                per = ", ".join(f"{c} {f(x, scale)}" for c, x in sorted(v["cards"].items()))
                lines.append(f"    {k.split(':')[0][len('aligned-'):]:<12} {f(v['all'], scale):>12}   [{per}]")
        for other in ("generic", "uninformed"):
            d = t["diffs"].get(f"aligned-{other}:score", {}).get("all") or {}
            if d.get("mean") is not None and d.get("se") is not None:
                verdict = "yes" if d["mean"] - 2 * d["se"] > 0 else "no"
                lines.append(f"  score headroom over {other} (mean - 2 SE > 0): {verdict}")
    return "\n".join(lines)


RESCORED_RULES, RESCORED_EVENTS = "rules.rescored.json", "rule_events.rescored.jsonl"


def replay_rules(ep_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Re-judge one saved episode with the current rule checks by replaying its actions."""
    import contextlib
    import io
    from dataclasses import asdict
    from .days import DaysConfig
    from .env import ContinuousOrderEnv
    from .kitchen import Kitchen
    from .orders import normalize_menu
    from .rule_eval import Judgment, RuleEvaluator
    from .rules import load_library
    known = set(DaysConfig.__dataclass_fields__)
    dc = DaysConfig(**{k: v for k, v in cfg.items() if k in known})
    summary = json.loads((ep_dir / "summary.json").read_text())
    persona, kind = summary["persona"], summary.get("kind", "interaction")
    menu = dc.menu if (kind == "eval" or not dc.schedule) else dc.menu_for(summary["day"])
    env = ContinuousOrderEnv(menu, layout=dc.layout, horizon=dc.horizon,
                             wrong_delivery_penalty=dc.wrong_delivery_penalty, no_repeat=dc.no_repeat,
                             open_orders=dc.open_orders, order_mode=dc.order_lists)
    with contextlib.redirect_stdout(io.StringIO()):
        start = env.reset(summary["seed"])
    ego, other = dc.ego_index, 1 - dc.ego_index
    rules = dc.cards()[persona].active(dc.partner.get("complexity"))
    ev = RuleEvaluator(rules, load_library(), Kitchen(env.mdp, ego, start), normalize_menu(menu), ego, other,
                       dc.open_orders, env.mdp, view_radius=dc.view_radius)
    import bisect
    plog_path = ep_dir / "partner_log.json"
    dec_ts = sorted(d.get("t", 0) for d in (json.loads(plog_path.read_text()).get("decisions") or [])) \
        if plog_path.exists() else []
    match = True
    for line in (ep_dir / "steps.jsonl").read_text().splitlines():
        rec = json.loads(line)
        acts = [a if isinstance(a, str) else tuple(a) for a in rec["actions"]]
        pre = env.state
        with contextlib.redirect_stdout(io.StringIO()):
            post, _, _, _ = env.step(acts, rec["params"])
        i = bisect.bisect_right(dec_ts, rec["t"]) - 1      # the LLM partner's decision behind this step
        ev.observe(rec["t"], pre, post, {"decided": {"partner": dec_ts[i] if i >= 0 else None}})
        if abs(env.score - rec["score"]) > 1e-9 or any(
                list(p.position) != r["pos"] for p, r in zip(post.players, rec["players"])):
            match = False
            break
    contention = {r.label() for r in rules if r.id == "on_contention"}
    orig_path = ep_dir / "rule_events.jsonl"
    kept = [Judgment(**json.loads(x)) for x in (orig_path.read_text().splitlines() if orig_path.exists() else [])
            if x.strip() and json.loads(x)["rule"] in contention]
    ev.judgments = [j for j in ev.judgments if j.rule not in contention] + kept
    return {"match": match, "rules": ev.summary(), "judgments": [asdict(j) for j in ev.judgments]}


def _rescore_one(arg):
    ep, cfg = arg
    try:
        r = replay_rules(Path(ep), cfg)
    except Exception as e:      # report, do not stop the folder
        return ep, False, repr(e)
    if r["match"]:
        (Path(ep) / RESCORED_RULES).write_text(json.dumps(r["rules"], indent=1))
        with open(Path(ep) / RESCORED_EVENTS, "w") as f:
            for j in r["judgments"]:
                f.write(json.dumps(j) + "\n")
    return ep, r["match"], None


def rescore(out: Path, workers: int = 1) -> Dict[str, Any]:
    cfg = json.loads((out / "config.json").read_text())
    eps = [(str(p.parent), cfg) for p in sorted(out.glob("runs/*/*/seed*/day*/*/steps.jsonl"))
           if (p.parent / "rule_events.jsonl").exists()]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_rescore_one, eps, chunksize=2)
    else:
        res = [_rescore_one(e) for e in eps]
    bad = [(ep, err) for ep, ok, err in res if not ok]
    return {"folder": str(out), "episodes": len(res), "rescored": len(res) - len(bad), "failed": bad}


def use_rescored(eps: List[Dict[str, Any]]) -> int:
    """Swap in re-scored rule results where they exist; returns how many episodes had them."""
    n = 0
    for e in eps:
        path = Path(e["dir"]) / RESCORED_RULES
        if path.exists():
            e["summary"] = dict(e["summary"], rules=json.loads(path.read_text()))
            n += 1
    return n


def gate0_folder(out: Path, calibrate: List[int], judge: List[int], guard_factor: float = 2.0) -> Dict[str, Any]:
    """Plan v3 Gate 0 / Gate 2 from a runner folder: conditions are scripted egos (proactive,
    mismatched, the rest reactive), personas are context cards. Per card: crossover table on
    the judge seeds, the best reactive ego chosen on the calibration seeds, proactive - best
    reactive and proactive - mismatched paired over (seed, episode)."""
    from .voi import paired_ci
    eps = load_episodes(out)
    val: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}
    for e in eps:
        sm = e["summary"]
        guards = sm.get("partner_guards", (sm.get("partner_llm") or {}).get("guards")) or 0
        val[(e["persona"], e["condition"], e["seed"], e["episode"])] = {
            "score": sm["score"], "completed": sm.get("orders_completed", 0),
            "expired": sm.get("orders_expired", 0), "guards": guards}
    cards = sorted({k[0] for k in val})
    egos = sorted({k[1] for k in val}, key=lambda c: (c != "proactive", c != "mismatched", c))
    reactive = [c for c in egos if c not in ("proactive", "mismatched")]
    keys = lambda c, e, seeds: sorted(k for k in val if k[0] == c and k[1] == e and k[2] in seeds)
    mean = lambda c, e, seeds, m: statistics.fmean(val[k][m] for k in keys(c, e, seeds)) if keys(c, e, seeds) else float("nan")
    res: Dict[str, Any] = {"layout": str(out), "menu": "-", "view_radius": "-", "deadlines": "see config.json",
                           "cards": cards, "egos": egos, "calibrate": calibrate, "judge": judge, "table": {},
                           "cards_result": {}}
    for c in cards:
        for e in egos:
            res["table"][f"{c}|{e}"] = {m: mean(c, e, judge, m) for m in ("score", "completed", "expired", "guards")}
        best = max(reactive, key=lambda e: mean(c, e, calibrate, "score"))

        def diff(other):
            out_ = []
            for k in keys(c, "proactive", judge):
                k2 = (c, other, k[2], k[3])
                if k2 in val:
                    out_.append(val[k]["score"] - val[k2]["score"])
            return paired_ci(out_)
        guards = mean(c, "proactive", judge, "guards")
        bar = max(0.5, guard_factor * mean(c, "proactive", calibrate, "guards"))
        d_best = diff(best)
        res["cards_result"][c] = {"best_reactive": best, "proactive_minus_best_reactive": d_best,
                                  "proactive_minus_mismatched": diff("mismatched"), "guard_bar": bar,
                                  "proactive_guards": guards,
                                  "pass": bool(d_best["lo"] is not None and d_best["lo"] > 0 and guards <= bar)}
    res["pass"] = all(r["pass"] for r in res["cards_result"].values())
    return res


_NUM = re.compile(r"(?<![A-Za-z_])-?\d+")


def why(out: Path, cards: Optional[List[str]] = None, rule_filter: Optional[str] = None,
        examples: int = 3, rescored: bool = False) -> Dict[str, Any]:
    """The partner's non-adherent decisions per card and rule, with what it said it was doing."""
    import bisect
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for orig in sorted(out.glob("runs/*/*/seed*/day*/ep*/rule_events.jsonl")):
        ev_path = orig.parent / RESCORED_EVENTS if rescored else orig
        if not ev_path.exists():
            continue
        d = ev_path.parent
        card, seed, day, ep = d.parents[2].name, d.parents[1].name, d.parents[0].name, d.name
        if card == NEUTRAL or (cards and card not in cards):
            continue
        judged = [json.loads(x) for x in ev_path.read_text().splitlines() if x.strip()]
        judged = [j for j in judged if j.get("side") == "partner"
                  and (not rule_filter or rule_filter in j["rule"])]
        if not judged:
            continue
        log_path = d / "partner_log.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        decisions = sorted(log.get("decisions") or [], key=lambda r: r.get("t", 0))
        times = [r.get("t", 0) for r in decisions]
        shown = {(r["t"], r["rule"]) for r in log.get("reminders") or []}
        for j in judged:
            row = rows.setdefault((card, j["rule"]), {"card": card, "rule": j["rule"], "n": 0, "adherent": 0,
                                                      "not": Counter(), "ok": Counter(), "reminded": 0,
                                                      "examples": [], "episodes": set()})
            row["n"] += 1
            pattern = _NUM.sub("N", str(j.get("note", "")))
            if j["value"]:
                row["adherent"] += 1
                row["ok"][pattern] += 1
                continue
            row["not"][pattern] += 1
            i = bisect.bisect_right(times, j["t"]) - 1          # the decision that chose this action
            dec = decisions[i] if i >= 0 else {}
            reminded = bool(dec) and (dec.get("t"), j["rule"]) in shown
            row["reminded"] += reminded
            where = f"{seed} {day} {ep}"
            if len(row["examples"]) < examples and where not in row["episodes"]:
                row["episodes"].add(where)
                it = dec.get("intention") or {}
                row["examples"].append({"where": where, "t": j["t"], "note": j.get("note"),
                                        "decided_t": dec.get("t"), "goal": it.get("goal"), "order": it.get("order"),
                                        "reconsider_when": it.get("reconsider_when"), "chosen": dec.get("chosen"),
                                        "reminder": reminded})
    out_rows = []
    for (card, rule), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[1]["adherent"] / kv[1]["n"])):
        r.pop("episodes")
        r["rate"] = r["adherent"] / r["n"]
        r["not"] = r["not"].most_common()
        r["ok"] = r["ok"].most_common()
        out_rows.append(r)
    return {"folder": str(out), "rows": out_rows, "rescored": rescored}


def format_why(rep: Dict[str, Any]) -> str:
    lines = [f"== why the partner breaks a rule: {rep['folder']} (all seeds"
             + ("; re-scored with the current checks" if rep.get("rescored") else "") + ") =="]
    for r in rep["rows"]:
        bad = r["n"] - r["adherent"]
        lines.append(f"\n{r['card']}  {r['rule']}: {r['adherent']} of {r['n']} adherent ({100 * r['rate']:.0f}%)")
        if not bad:
            continue
        lines.append(f"  not adherent ({bad}; reminder shown at the deciding step: {r['reminded']}):")
        lines += [f"    {c:>4} x {p}" for p, c in r["not"][:6]]
        lines.append("  adherent: " + "; ".join(f"{c} x {p}" for p, c in r["ok"][:3]))
        for e in r["examples"]:
            lines.append(f"  e.g. {e['where']} t={e['t']}: {e['note']} | decided t={e['decided_t']}: "
                         f"{e['chosen']} | goal: {e['goal']!s:.90} | order: {e['order']} | "
                         f"reconsider: {e['reconsider_when']!s:.60} | reminder: {'yes' if e['reminder'] else 'no'}")
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
    w = sub.add_parser("why", help="the partner's non-adherent decisions per card and rule")
    w.add_argument("run_dir", type=Path)
    w.add_argument("--card", nargs="+", help="only these cards")
    w.add_argument("--rule", help="only rules whose label contains this text")
    w.add_argument("--examples", type=int, default=3)
    w.add_argument("--out", type=Path)
    g0 = sub.add_parser("gate0", help="plan v3 Gate 0 / Gate 2 from a runner folder (scripted egos x context cards)")
    g0.add_argument("run_dir", type=Path)
    g0.add_argument("--calibrate", nargs="+", type=int, default=[0, 1, 2, 3])
    g0.add_argument("--judge", nargs="+", type=int, default=[4, 5, 6, 7])
    g0.add_argument("--out", type=Path)
    rs = sub.add_parser("rescore", help="re-judge saved episodes with the current rule checks (replay)")
    rs.add_argument("run_dir", type=Path)
    rs.add_argument("--workers", type=int, default=1)
    rs.add_argument("--out", type=Path)
    for sp in (g1, g2, w):
        sp.add_argument("--rescored", action="store_true",
                        help="use the files written by 'rescore' (the current rule checks)")
    args = ap.parse_args(argv)
    if args.cmd == "gate0":
        from .voi import format_gate0
        rep = gate0_folder(args.run_dir, args.calibrate, args.judge)
        print(format_gate0(rep))
    elif args.cmd == "rescore":
        rep = rescore(args.run_dir, args.workers)
        print(f"{rep['rescored']} of {rep['episodes']} episodes re-scored in {rep['folder']}")
        for ep, err in rep["failed"]:
            print(f"  not re-scored: {ep}: {err or 'replay did not match the saved steps'}")
    elif args.cmd == "why":
        rep = why(args.run_dir, args.card, args.rule, args.examples, args.rescored)
        print(format_why(rep))
    elif args.cmd == "gate1":
        rep = gate1(args.run_dir, args.calibrate, args.judge, args.bar, args.rescored)
        twin = gate1(args.twin, args.calibrate, args.judge, args.bar, args.rescored) if args.twin else None
        print(format_gate1(rep, twin))
        if twin:
            rep["twin"] = twin
    else:
        rep = gate2(args.run_dir, args.rescored)
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
                      view_radius=None, open_orders: int = 1, ego_index: int = 0,
                      deadlines=None, order_mode: str = "random") -> Dict[str, Any]:
    """One episode: ``ego_player`` (a SimulatedHuman or RulePlayer in ego mode) with the
    rule-based twin of ``partner_rules``; returns the score and the rule summary."""
    import contextlib
    import io
    from .env import ContinuousOrderEnv
    from .kitchen import Kitchen
    from .rule_eval import RuleEvaluator, decided_at, intent_target
    from .rule_partner import RulePlayer
    from .rules import load_library
    env = ContinuousOrderEnv(menu, layout=layout, horizon=horizon, open_orders=open_orders,
                             deadlines=deadlines, order_mode=order_mode)
    with contextlib.redirect_stdout(io.StringIO()):
        start = env.reset(seed)
    other = 1 - ego_index
    from .rules import CONTEXT_RULE_IDS
    if any(r.id in CONTEXT_RULE_IDS for r in partner_rules):
        from .context_partner import ContextPlayer
        partner = ContextPlayer(partner_rules, list(menu), "partner", "card", name="twin")
    else:
        partner = RulePlayer(partner_rules, list(menu), "partner", "twin")
    ego_player.reset(env.mdp, ego_index, seed, view_radius=view_radius)
    partner.reset(env.mdp, other, seed, view_radius=view_radius)
    k = Kitchen(env.mdp, ego_index, start)
    ev = RuleEvaluator(partner_rules, load_library(), k, list(menu), ego_index, other, open_orders, env.mdp,
                       view_radius=view_radius)
    while not env.done:
        t, pre = env.t, env.state
        moves = [None, None]
        moves[ego_index] = ego_player.act(pre, t)
        moves[other] = partner.act(pre, t)
        intents = {"ego": intent_target(ego_player, k), "partner": intent_target(partner, k),
                   "decided": {"ego": decided_at(ego_player), "partner": decided_at(partner)}}
        with contextlib.redirect_stdout(io.StringIO()):
            env.step([m[0] for m in moves], [m[1] for m in moves])
        ev.observe(t, pre, env.state, intents)
    s = env.summary()
    return {"score": s["score"], "rules": ev.summary(), "steps": env.t, "orders_completed": s["orders_completed"],
            "orders_expired": s.get("orders_expired", 0), "guards": len(getattr(partner, "guards", [])),
            "judgments": ev.judgments}
