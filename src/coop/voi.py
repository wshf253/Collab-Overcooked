"""How much is it worth to know the partner? (value of information, no LLM)

For every partner persona, a set of scripted ego policies plays the same seeded
episodes:

* ``best fixed``: the ego policy with the highest mean score over all personas,
  i.e. the best you can do without knowing who the partner is;
* ``best response``: the ego policy with the highest score for this persona.

``value = best response - best fixed`` (points per episode) is what knowing the
partner could buy with these simple policies. Both are chosen on ``--seeds`` and
re-measured on ``--eval-seeds`` (picking the best of many policies on the same
episodes overstates the gain). A partner type whose value is
about 0 cannot reward ToM or memory, whatever the agent infers. It is a lower
bound: a cleverer policy could gain more. Same idea as the information-gap
screen of ICRL4AHT RQ3.

    PYTHONPATH=src:lib/overcooked_ai python -m coop.voi --layout open_env --out outputs/voi_open.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .partner import DEFAULT_PERSONAS, ORDER_PERSONAS, PERSONAS, Traits
from .screen import DEFAULT_MENU, simulate

# Ego policies: no preference or one preferred stage, with or without giving way on tasks,
# and (with several open orders) working on the oldest or the newest order first.
EGOS: Dict[str, Traits] = {}
for _pref in ((), ("prep",), ("cook",), ("serve",)):
    for _yield in (False, True):
        for _focus in ("first", "last"):
            _name = (("any" if not _pref else _pref[0] + "_first") + ("+yield" if _yield else "")
                     + ("+newest" if _focus == "last" else ""))
            EGOS[_name] = Traits(preference=_pref, yield_tasks=_yield, order_focus=_focus)


def egos_for(open_orders: int, menu: List[str] = None) -> Dict[str, Traits]:
    """Order focus only matters with several open orders. Then the egos also include
    order-habit responses: committing to the oldest or newest open order (with or
    without giving way), and leaving one dish of the menu to the partner
    (``any+leave_<dish>``: that dish's orders last)."""
    if open_orders <= 1:
        return {k: v for k, v in EGOS.items() if not k.endswith("+newest")}
    out = dict(EGOS)
    for _yield in (False, True):
        for focus, label in (("first", "oldest"), ("last", "newest")):
            out[f"commit_{label}" + ("+yield" if _yield else "")] = Traits(
                order_focus=focus, order_commit=True, yield_tasks=_yield)
    dishes = list(dict.fromkeys(menu or DEFAULT_MENU))
    for d in dishes:
        others = tuple(x for x in dishes if x != d)
        out[f"any+leave_{d}"] = Traits(favourite_dishes=others)
        out[f"commit_oldest+leave_{d}"] = Traits(favourite_dishes=others, order_commit=True)
    return out


EGO_SETS = ("all", "flexible", "commit")


def ego_set(name: str, open_orders: int, menu: List[str] = None) -> Dict[str, Traits]:
    """``flexible``: egos that take whatever job is needed next; ``commit``: egos that
    work on one order at a time (closer to an agent that plans its own order, such as
    an LLM choosing macro actions); ``all``: both."""
    egos = egos_for(open_orders, menu)
    if name == "all":
        return egos
    if name not in EGO_SETS:
        raise ValueError(f"ego set must be one of {EGO_SETS}")
    return {k: v for k, v in egos.items() if v.order_commit == (name == "commit")}


def screen(layout: str, personas: List[str], seeds: List[int], horizon: int, menu: List[str],
           egos: Dict[str, Traits] = None, ego_index: int = 0, view_radius=None, open_orders: int = 1,
           eval_seeds: List[int] = None) -> Dict:
    egos = egos or egos_for(open_orders, menu)
    partner_index = 1 - ego_index
    scores: Dict[str, Dict[str, float]] = {}       # persona -> ego -> mean score
    for persona in personas:
        scores[persona] = {}
        for ego, traits in egos.items():
            vals = []
            for seed in seeds:
                env, _ = simulate(menu, layout, horizon, seed, persona, partner_index,
                                  teammate=(f"ego_{ego}", traits), view_radius=view_radius,
                                  open_orders=open_orders)
                vals.append(env.summary()["score"])
            scores[persona][ego] = statistics.fmean(vals)
    overall = {ego: statistics.fmean(scores[p][ego] for p in personas) for ego in egos}
    best_fixed = max(overall, key=overall.get)
    rows = {}
    for persona in personas:
        br = max(scores[persona], key=scores[persona].get)
        rows[persona] = {"best_response": br, "best_response_score": scores[persona][br],
                         "best_fixed_score": scores[persona][best_fixed],
                         "value": scores[persona][br] - scores[persona][best_fixed],
                         "scores": scores[persona]}
    out = {"layout": layout, "seeds": seeds, "horizon": horizon, "view_radius": view_radius,
           "open_orders": open_orders, "best_fixed": best_fixed,
           "best_fixed_overall": overall[best_fixed], "overall": overall, "personas": rows}
    if eval_seeds:
        # Picking the best of many policies on the same episodes overstates the value
        # (winner's curse): measure the chosen policies again on new seeds.
        for persona in personas:
            r = rows[persona]
            per = {}
            for ego in {r["best_response"], best_fixed}:
                per[ego] = statistics.fmean(
                    simulate(menu, layout, horizon, sd, persona, partner_index, teammate=(f"ego_{ego}", egos[ego]),
                             view_radius=view_radius, open_orders=open_orders)[0].summary()["score"]
                    for sd in eval_seeds)
            r["eval_value"] = per[r["best_response"]] - per[best_fixed]
            r["eval_scores"] = per
        out["eval_seeds"] = eval_seeds
    return out


def card_screen(cards_path: Path, layout: str, seeds: List[int], horizon: int, menu: List[str],
                complexity=None, view_radius=None, open_orders: int = 1, eval_seeds: List[int] = None) -> Dict:
    """Gate 2 with the rule-based twin (docs/PERSONA_PARTNER_PLAN_v2.md, section 10): for each
    card, the twin partner with scripted egos aligned (does what this card wants), generic
    (steady), uninformed (the aligned ego's machinery with no rule) and misaligned (does what
    the next card wants). Reports aligned minus each for score (primary) and alignment, on
    ``seeds`` and re-measured on ``eval_seeds``."""
    from .partner import SimulatedHuman
    from .persona_check import composite, gate2_table, play_rule_episode
    from .rule_partner import RulePlayer
    from .rules import NEUTRAL, load_cards
    cards = {n: c for n, c in load_cards(cards_path).items() if n != NEUTRAL}
    names = list(cards)
    rows = {}
    for i, name in enumerate(names):
        rules = cards[name].active(complexity)
        other = cards[names[(i + 1) % len(names)]].active(complexity)
        for seed in list(seeds) + list(eval_seeds or []):
            egos = {"aligned": RulePlayer(rules, menu, "ego", "aligned"),
                    "generic": SimulatedHuman(PERSONAS["steady"], "steady"),
                    # the aligned ego's own machinery without any rule: separates knowing the
                    # partner from the difference between the two scripted implementations
                    "uninformed": RulePlayer([], menu, "ego", "uninformed"),
                    "misaligned": RulePlayer(other, menu, "ego", "misaligned")}
            for ego_name, player in egos.items():
                r = play_rule_episode(menu, layout, horizon, seed, rules, player, view_radius, open_orders)
                al, _ = composite(r, "ego")
                rows[(ego_name, name, seed)] = {"score": r["score"], "alignment": al}
    sets = {"selection": list(seeds)}
    if eval_seeds:
        sets["new seeds"] = list(eval_seeds)
    return {"cards_file": str(cards_path), "layout": layout, "seed_sets": sets,
            "tables": {k: gate2_table(rows, v) for k, v in sets.items()}, "folder": f"twin on {layout}"}


def _rule_job(arg):
    from .partner import SimulatedHuman
    from .persona_check import composite, play_rule_episode
    from .rule_partner import RulePlayer
    from .rules import load_library, parse_rule
    spec, opposite, ego, seed, menu, layout, horizon, view_radius, open_orders = arg
    lib = load_library()
    rules = [parse_rule(spec, lib)]
    player = {"generic": lambda: SimulatedHuman(PERSONAS["steady"], "steady"),
              "uninformed": lambda: RulePlayer([], menu, "ego", "uninformed"),
              "aligned": lambda: RulePlayer(rules, menu, "ego", "aligned"),
              "opposite": lambda: RulePlayer([parse_rule(opposite, lib)], menu, "ego", "opposite")}[ego]()
    r = play_rule_episode(menu, layout, horizon, seed, rules, player, view_radius, open_orders)
    al, _ = composite(r, "ego")
    opps = sum(v["partner"]["opportunities"] for v in r["rules"].values())
    return {"score": r["score"], "alignment": al, "opportunities": opps}


def rule_variants(layout: str, menu: List[str]) -> List[Tuple[object, object]]:
    """Every library rule with every parameter value that exists in this kitchen and menu,
    with its opposite (another station of the same kind, the other order / server / mode,
    another counter group), or None when the rule has no parameter to flip."""
    import contextlib
    import io
    from itertools import product
    from .env import ContinuousOrderEnv
    from .kitchen import Kitchen
    from .rules import load_library
    env = ContinuousOrderEnv(menu, layout=layout, horizon=10)
    with contextlib.redirect_stdout(io.StringIO()):
        start = env.reset(0)
    k = Kitchen(env.mdp, 0, start)
    kinds: Dict[str, List[str]] = {}
    for u in k.utensils:
        kinds.setdefault(k.kind_of[u], []).append(u)
    out = []
    for rid, spec in load_library().items():
        params = dict(spec.get("params") or {})
        options = {n: [str(x) for x in v] for n, v in (spec.get("options") or {}).items()}
        if not params and not options:
            out.append((rid, None))
            continue
        allowed = {}
        for name, values in params.items():
            if name == "station":
                values = [v for v in values if v in k.utensils and len(kinds[k.kind_of[v]]) >= 2]
            elif name == "dish":
                values = [v for v in values if v in menu]
            allowed[name] = list(values)
        names = list(allowed) + list(options)
        for combo in product(*allowed.values(), *options.values()):
            p = dict(zip(names, combo))
            if rid == "serves" and p.get("who") == "self" and p.get("patience", options.get("patience", ["20"])[0]) \
                    != options["patience"][0]:
                continue            # patience only matters for serves(who=ego)
            opp = None
            if len(allowed) == 1:
                (name, v), = [(n, p[n]) for n in allowed]
                if name == "station":
                    others = [u for u in kinds[k.kind_of[v]] if u != v]
                else:
                    others = [] if name == "dish" else [x for x in allowed[name] if x != v]
                if others:
                    opp = {rid: dict(p, **{name: others[0]})}
            out.append(({rid: p} if p else rid, opp))
    return out


def rule_screen(layout: str, seeds: List[int], horizon: int, menu: List[str], view_radius=None,
                open_orders: int = 1, workers: int = 1) -> Dict:
    """Score value of knowing one rule: the twin with a single rule, and scripted egos that
    know it (aligned), know nothing (uninformed: the same machinery with no rule; generic:
    steady) or believe its opposite. Paired over seeds (mean and SE)."""
    from multiprocessing import Pool
    from .analyze import _mean_se
    from .rules import load_library, parse_rule
    lib = load_library()
    variants = rule_variants(layout, list(menu))
    jobs, keys = [], []
    for i, (spec, opp) in enumerate(variants):
        for ego in ("generic", "uninformed", "aligned") + (("opposite",) if opp else ()):
            for sd in seeds:
                jobs.append((spec, opp, ego, sd, list(menu), layout, horizon, view_radius, open_orders))
                keys.append((i, ego, sd))
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(_rule_job, jobs, chunksize=4)
    else:
        results = [_rule_job(j) for j in jobs]
    got = dict(zip(keys, results))
    rows = []
    for i, (spec, opp) in enumerate(variants):
        rule = parse_rule(spec, lib)
        score = {e: [got[(i, e, sd)]["score"] for sd in seeds] for e in ("generic", "uninformed", "aligned", "opposite")
                 if (i, e, seeds[0]) in got}
        row = {"rule": rule.label(), "uninformed_score": statistics.fmean(score["uninformed"]),
               "opportunities": statistics.fmean(got[(i, "uninformed", sd)]["opportunities"] for sd in seeds)}
        for other in ("uninformed", "generic", "opposite"):
            if other in score:
                row[f"aligned-{other}"] = _mean_se([a - b for a, b in zip(score["aligned"], score[other])])
        rows.append(row)
    return {"layout": layout, "seeds": list(seeds), "horizon": horizon, "view_radius": view_radius,
            "open_orders": open_orders, "rows": rows}


def format_rule_screen(res: Dict) -> str:
    f = lambda ms: "-" if not ms or ms.get("mean") is None else f"{ms['mean']:+.0f}±{ms['se']:.0f}"
    lines = [f"score value of knowing one rule: {res['layout']}, view radius {res['view_radius']}, "
             f"{res['open_orders']} open orders, horizon {res['horizon']}, seeds {res['seeds']} (paired mean±SE)",
             f"{'rule':<52}{'uninformed':>11}{'aligned-uninformed':>20}{'aligned-generic':>17}"
             f"{'aligned-opposite':>18}{'opps':>6}"]
    for r in res["rows"]:
        lines.append(f"{r['rule']:<52}{r['uninformed_score']:>11.0f}{f(r.get('aligned-uninformed')):>20}"
                     f"{f(r.get('aligned-generic')):>17}{f(r.get('aligned-opposite')):>18}{r['opportunities']:>6.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Gate 0 (docs/CONTEXT_PARTNER_PLAN_v3.md, section 7): does acting on a partner model pay?
# ---------------------------------------------------------------------- #
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20,
        12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09, 20: 2.09, 25: 2.06, 30: 2.04}
REACTIVE = ["steady", "always_take_urgent", "never_take_urgent", "wait_and_see:3", "wait_and_see:6", "wait_and_see:10"]


def paired_ci(xs: List[float]) -> Dict:
    n = len(xs)
    m = statistics.fmean(xs) if xs else None
    if n < 2:
        return {"mean": m, "se": None, "lo": None, "hi": None, "n": n}
    se = statistics.stdev(xs) / n ** 0.5
    t = T975.get(n - 1) or T975[max(k for k in T975 if k <= n - 1)] if n - 1 <= 30 else 1.96
    return {"mean": m, "se": se, "lo": m - t * se, "hi": m + t * se, "n": n}


def _ctx_ego(name: str, cards, card: str, menu):
    from .context_partner import ContextPlayer
    from .partner import SimulatedHuman
    if name == "steady":
        return SimulatedHuman(PERSONAS["steady"], "steady")
    if name in ("proactive", "mismatched"):
        names = sorted(cards)
        other = next((c for c in names if c != card), card) if name == "mismatched" else card
        if isinstance(cards[other], list):
            return ContextPlayer(cards[other], menu, "ego", "proactive", name=name)
        return ContextPlayer(cards[other].rules, menu, "ego", "proactive", name=name)
    if name.startswith("wait_and_see:"):
        return ContextPlayer([], menu, "ego", "wait_and_see", k=int(name.split(":")[1]), name=name)
    return ContextPlayer([], menu, "ego", name, name=name)


def _ctx_job(arg):
    from .persona_check import play_rule_episode
    from .rules import load_cards
    card_path, card, ego, seed, menu, layout, horizon, view_radius, open_orders, deadlines, opposite = arg
    if card_path is None:                    # single-rule screen: (rule, opposite rule) passed in
        cards = {"own": [card], "opposite": [opposite]}
        rules, key = [card], "own"
    else:
        cards = {n: c for n, c in load_cards(card_path).items() if n != "neutral"}
        rules, key = cards[card].rules, card
    player = _ctx_ego(ego, cards, key, list(menu))
    r = play_rule_episode(list(menu), layout, horizon, seed, rules, player, view_radius, open_orders,
                          deadlines=deadlines, order_mode="controlled")
    return {"score": r["score"], "completed": r["orders_completed"], "expired": r["orders_expired"],
            "guards": r["guards"]}


def gate0(cards_path: Path, layout: str, menu: List[str], deadlines: Dict, calibrate: List[int], judge: List[int],
          horizon: int = 300, view_radius=2, open_orders: int = 2, workers: int = 1, guard_factor: float = 2.0) -> Dict:
    """Crossover of scripted egos x context cards with the twin (no LLM)."""
    from multiprocessing import Pool
    from .rules import load_cards
    cards = [n for n in load_cards(cards_path) if n != "neutral"]
    egos = ["proactive", "mismatched"] + REACTIVE
    seeds = list(calibrate) + list(judge)
    jobs = [(str(cards_path), c, e, s, list(menu), layout, horizon, view_radius, open_orders, deadlines, None)
            for c in cards for e in egos for s in seeds]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(_ctx_job, jobs, chunksize=2)
    else:
        res = [_ctx_job(j) for j in jobs]
    got = {(j[1], j[2], j[3]): r for j, r in zip(jobs, res)}
    mean = lambda c, e, ss, k: statistics.fmean(got[(c, e, s)][k] for s in ss)
    out: Dict[str, Any] = {"cards": cards, "egos": egos, "calibrate": list(calibrate), "judge": list(judge),
                           "deadlines": deadlines, "view_radius": view_radius, "layout": layout, "menu": list(menu),
                           "table": {}, "cards_result": {}}
    for c in cards:
        for e in egos:
            out["table"][f"{c}|{e}"] = {k: mean(c, e, judge, k) for k in ("score", "completed", "expired", "guards")}
        best = max(REACTIVE, key=lambda e: mean(c, e, calibrate, "score"))
        bar = max(0.5, guard_factor * mean(c, "proactive", calibrate, "guards"))
        d_best = paired_ci([got[(c, "proactive", s)]["score"] - got[(c, best, s)]["score"] for s in judge])
        d_mis = paired_ci([got[(c, "proactive", s)]["score"] - got[(c, "mismatched", s)]["score"] for s in judge])
        guards = mean(c, "proactive", judge, "guards")
        out["cards_result"][c] = {"best_reactive": best, "proactive_minus_best_reactive": d_best,
                                  "proactive_minus_mismatched": d_mis, "guard_bar": bar, "proactive_guards": guards,
                                  "pass": bool(d_best["lo"] is not None and d_best["lo"] > 0 and guards <= bar)}
    out["pass"] = all(r["pass"] for r in out["cards_result"].values())
    return out


# ---------------------------------------------------------------------- #
# Plan v4 V0 (docs/STYLE_PARTNER_PLAN_v4.md, 5): two-axis cards x scripted egos, twin partner
# ---------------------------------------------------------------------- #
def style_egos(card) -> List[str]:
    """The egos played against one two-axis card (plan v4, 4): informed, partly informed
    (wrong adherence, wrong workflow, one context flipped) and reactive."""
    from .rules import WORKFLOWS
    ident = [c for c, v in card.contexts.items() if v["identifiable"]]
    out = ["best_response", "wrong_adherence"]
    out += [f"wrong_workflow:{wf}" for wf in WORKFLOWS if wf != card.latent["workflow"]]
    if len(ident) > 1:
        out += [f"flip:{c}" for c in ident]
    return out + list(REACTIVE)


def _style_ego(name: str, card, menu, soft: bool = False):
    from .context_partner import ContextPlayer
    from .rules import flip_adherence
    if name == "best_response":
        target = card
    elif name == "wrong_adherence":
        target = flip_adherence(card)
    elif name.startswith("wrong_workflow:"):
        target = flip_adherence(card, contexts=[], workflow=name.split(":", 1)[1])
    elif name.startswith("flip:"):
        target = flip_adherence(card, contexts=[name.split(":", 1)[1]])
    else:
        return _ctx_ego(name, {}, None, menu)
    return ContextPlayer(target.rules, menu, "ego", "proactive_soft" if soft else "proactive", name=name,
                         near_cooking=True)


def _style_job(arg):
    from .context_partner import ContextPlayer
    from .partner import SimulatedHuman
    from .persona_check import play_rule_episode
    from .rules import load_cards
    card_path, card, ego, seed, ep, menu, layout, horizon, view_radius, open_orders, deadlines = arg[:11]
    soft = arg[11] if len(arg) > 11 else False
    eseed = seed * 1000 + ep
    if card is None:                            # reference: a steady pair on the same episode
        from .deadlines import play_pair
        s = play_pair(list(menu), layout, eseed, horizon, open_orders, view_radius, deadlines)
        return {"score": s["score"]}
    c = load_cards(card_path)[card]
    twin = ContextPlayer(c.rules, list(menu), "partner", "card", name=card, near_cooking=True, latent=c.latent)
    player = _style_ego(ego, c, list(menu), soft)
    r = play_rule_episode(list(menu), layout, horizon, eseed, c.rules, player, view_radius, open_orders,
                          deadlines=deadlines, order_mode="controlled", partner_player=twin)
    adh = sum(row["partner"]["adherent"] for row in r["rules"].values())
    opp = sum(row["partner"]["opportunities"] for row in r["rules"].values())
    return {"score": r["score"], "completed": r["orders_completed"], "expired": r["orders_expired"],
            "guards": r["guards"], "adherent": adh, "opportunities": opp}


def v0(cards_path: Path, layout: str, menu: List[str], deadlines: Dict, calibrate: List[int], judge: List[int],
       episodes: int = 3, horizon: int = 300, view_radius=2, open_orders: int = 2, workers: int = 1,
       guard_factor: float = 2.0, main: Optional[List[str]] = None, soft: bool = False) -> Dict:
    """Plan v4 V0: does each axis have value? Crossover of egos x cards (score minus a steady
    pair on the same episode, expired orders, guard firings) and, per card, paired intervals over
    the judge seeds of best_response minus the best reactive ego (chosen on the calibration
    seeds), minus wrong_adherence, minus each wrong_workflow, and minus each one-context flip."""
    from multiprocessing import Pool
    from .rules import load_cards
    cards = {n: c for n, c in load_cards(cards_path).items() if c.latent is not None}
    main = main or [n for n in cards if n != "mixed"]
    egos = {n: style_egos(c) for n, c in cards.items()}
    seeds = list(calibrate) + list(judge)
    eps = range(episodes)
    jobs = [(str(cards_path), n, e, s, k, list(menu), layout, horizon, view_radius, open_orders, deadlines, soft)
            for n in cards for e in egos[n] for s in seeds for k in eps]
    jobs += [(None, None, "reference", s, k, list(menu), layout, horizon, view_radius, open_orders, deadlines)
             for s in seeds for k in eps]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(_style_job, jobs, chunksize=4)
    else:
        res = [_style_job(j) for j in jobs]
    got = {(j[1], j[2], j[3], j[4]): r for j, r in zip(jobs, res)}
    ref = {s: statistics.fmean(got[(None, "reference", s, k)]["score"] for k in eps) for s in seeds}

    def per_seed(c, e, s, key="score"):
        return statistics.fmean(got[(c, e, s, k)][key] for k in eps)

    def mean(c, e, ss, key="score"):
        return statistics.fmean(per_seed(c, e, s, key) for s in ss)

    def delta(c, a, b):
        return paired_ci([per_seed(c, a, s) - per_seed(c, b, s) for s in judge])

    out: Dict[str, Any] = {"cards": list(cards), "main": main, "egos": egos, "calibrate": list(calibrate),
                           "judge": list(judge), "episodes": episodes, "deadlines": deadlines, "soft": soft,
                           "view_radius": view_radius, "layout": layout, "menu": list(menu), "horizon": horizon,
                           "reference_judge": statistics.fmean(ref[s] for s in judge), "table": {}, "cards_result": {}}
    for c, card in cards.items():
        for e in egos[c]:
            out["table"][f"{c}|{e}"] = {"vs_ref": statistics.fmean(per_seed(c, e, s) - ref[s] for s in judge),
                                        "score": mean(c, e, judge), "expired": mean(c, e, judge, "expired"),
                                        "guards": mean(c, e, judge, "guards")}
        best = max(REACTIVE, key=lambda e: mean(c, e, calibrate))
        bar = max(0.5, guard_factor * mean(c, "best_response", calibrate, "guards"))
        adh = sum(got[(c, "best_response", s, k)]["adherent"] for s in seeds for k in eps)
        opp = sum(got[(c, "best_response", s, k)]["opportunities"] for s in seeds for k in eps)
        r = {"workflow": card.latent["workflow"], "adherence": card.latent["adherence"],
             "not_identifiable": [x for x, v in card.contexts.items() if not v["identifiable"]],
             "best_reactive": best, "d_reactive": delta(c, "best_response", best),
             "d_adherence": delta(c, "best_response", "wrong_adherence"),
             "d_workflow": {e.split(":", 1)[1]: delta(c, "best_response", e) for e in egos[c] if e.startswith("wrong_workflow:")},
             "d_context": {e.split(":", 1)[1]: delta(c, "best_response", e) for e in egos[c] if e.startswith("flip:")},
             "guards_best_response": mean(c, "best_response", judge, "guards"), "guard_bar": bar,
             "twin_adherence": adh / opp if opp else None,
             "best_ego_judge": max(egos[c], key=lambda e: mean(c, e, judge))}
        out["cards_result"][c] = r
    # pass (plan v4, 5)
    lo = lambda d: d["lo"] is not None and d["lo"] > 0
    cr = out["cards_result"]
    wf_of = {c: cr[c]["workflow"] for c in main}
    crossover = {}
    for wf in sorted(set(wf_of.values())):
        pair = [c for c in main if wf_of[c] == wf]
        crossover[wf] = len(pair) == 2 and all(cr[c]["d_adherence"]["mean"] > 0 for c in pair)
    adh_ok = [c for c in main if lo(cr[c]["d_adherence"])]
    checks = {
        "d_reactive>0 on every main card": all(lo(cr[c]["d_reactive"]) for c in main),
        "d_adherence>0 on >=4 of 6 main cards": len(adh_ok) >= 4,
        "d_adherence>0 on a card of each workflow": all(any(wf_of[c] == wf for c in adh_ok) for wf in set(wf_of.values())),
        "crossover within each workflow": all(crossover.values()),
        "guards rare with best_response": all(cr[c]["guards_best_response"] <= cr[c]["guard_bar"] for c in main),
    }
    out.update(crossover=crossover, checks=checks, pass_=all(checks.values()))
    return out


def format_v0(res: Dict) -> str:
    f = lambda d: "-" if d.get("mean") is None else (f"{d['mean']:+6.1f} [{d['lo']:+6.1f},{d['hi']:+6.1f}]"
                                                     if d.get("lo") is not None else f"{d['mean']:+6.1f}")
    L = [("informed egos: complement first, then anything useful (--soft)\n" if res.get("soft") else "")
         + f"V0 (plan v4): {res['layout']}, menu {res['menu']}, timers {res['deadlines']}, view radius "
         f"{res['view_radius']}, horizon {res['horizon']}; calibration seeds {res['calibrate']}, judge seeds "
         f"{res['judge']}, {res['episodes']} episodes each; intervals: 95% t over judge seeds",
         f"reference (steady pair, same episodes, judge seeds): {res['reference_judge']:.1f}", "",
         "== crossover (judge seeds): score minus reference / expired orders / guards per episode =="]
    for c in res["cards"]:
        L.append(f"  {c}")
        for e in res["egos"][c]:
            t = res["table"][f"{c}|{e}"]
            L.append(f"    {e:<28}{t['vs_ref']:+7.1f}{t['expired']:6.2f}{t['guards']:6.2f}")
    L += ["", "== per card: best_response minus ... (judge seeds) =="]
    for c, r in res["cards_result"].items():
        adh = r["adherence"] if isinstance(r["adherence"], str) else "mixed"
        L.append(f"  {c} ({r['workflow']} / {adh}; twin adherence "
                 f"{100 * r['twin_adherence']:.0f}%; not identifiable: {r['not_identifiable'] or 'none'})")
        L.append(f"    d_reactive  (vs {r['best_reactive']}): {f(r['d_reactive'])}")
        L.append(f"    d_adherence (vs wrong_adherence): {f(r['d_adherence'])}")
        for wf, d in r["d_workflow"].items():
            L.append(f"    d_workflow  (vs {wf}): {f(d)}")
        for cx, d in r["d_context"].items():
            L.append(f"    per context (vs {cx} flipped): {f(d)}")
        L.append(f"    guards with best_response {r['guards_best_response']:.2f} (bar {r['guard_bar']:.2f}); "
                 f"best ego on judge seeds: {r['best_ego_judge']}")
    L += ["", "crossover within workflow (best_response beats wrong_adherence on both cards): "
          + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in res["crossover"].items())]
    L += [f"  {'PASS' if v else 'fail'}  {k}" for k, v in res["checks"].items()]
    L.append(f"V0: {'PASS' if res['pass_'] else 'FAIL'}")
    return "\n".join(L)


def context_rule_screen(layout: str, menu: List[str], deadlines: Dict, seeds: List[int], horizon: int = 300,
                        view_radius=2, open_orders: int = 2, workers: int = 1) -> Dict:
    """Value of each context rule alone: twin with that rule only; proactive ego for it vs steady
    and vs the opposite mode."""
    from multiprocessing import Pool
    from .rules import load_library, parse_rule
    lib = load_library()
    opp = {"prep": "whole", "whole": "prep", "takes": "keeps", "keeps": "takes", "clears": "waits",
           "waits": "clears", "switches": "continues", "continues": "switches"}
    specs = []
    for rid in ("default_work", "on_urgent", "on_blocked", "on_prep_ready"):
        for mode in lib[rid]["params"]["mode"]:
            variants = [{"mode": mode}] + ([{"mode": mode, "unless_ego_on_it": "yes"}] if rid == "on_urgent" and mode == "takes" else [])
            for v in variants:
                specs.append((parse_rule({rid: v}, lib), parse_rule({rid: {"mode": opp[mode]}}, lib)))
    jobs = [(None, r, e, s, list(menu), layout, horizon, view_radius, open_orders, deadlines, o)
            for r, o in specs for e in ("proactive", "mismatched", "steady") for s in seeds]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(_ctx_job, jobs, chunksize=2)
    else:
        res = [_ctx_job(j) for j in jobs]
    got = {(j[1].label(), j[2], j[3]): r for j, r in zip(jobs, res)}
    rows = []
    for r, _ in specs:
        lab = r.label()
        sc = lambda e: [got[(lab, e, s)]["score"] for s in seeds]
        rows.append({"rule": lab, "steady": statistics.fmean(sc("steady")),
                     "proactive-steady": paired_ci([a - b for a, b in zip(sc("proactive"), sc("steady"))]),
                     "proactive-opposite": paired_ci([a - b for a, b in zip(sc("proactive"), sc("mismatched"))]),
                     "guards_proactive": statistics.fmean(got[(lab, "proactive", s)]["guards"] for s in seeds)})
    return {"rows": rows, "seeds": list(seeds), "deadlines": deadlines}


def format_gate0(res: Dict) -> str:
    f = lambda d: "-" if d.get("mean") is None else (f"{d['mean']:+.1f}" + (f" [{d['lo']:+.1f}, {d['hi']:+.1f}]" if d.get("lo") is not None else ""))
    lines = [f"== Gate 0: {res['layout']}, menu {res['menu']}, radius {res['view_radius']}, deadlines {res['deadlines']}",
             f"calibration seeds {res['calibrate']}, judge seeds {res['judge']} (table: judge seeds, mean per episode)",
             f"{'ego':<22}" + "".join(f"{c + ' score':>12}{'expired':>9}{'guards':>8}" for c in res["cards"])]
    for e in res["egos"]:
        row = f"{e:<22}"
        for c in res["cards"]:
            t = res["table"][f"{c}|{e}"]
            row += f"{t['score']:>12.1f}{t['expired']:>9.2f}{t['guards']:>8.2f}"
        lines.append(row)
    for c, r in res["cards_result"].items():
        lines.append(f"card {c}: best reactive (calibration) {r['best_reactive']}; proactive - best reactive "
                     f"{f(r['proactive_minus_best_reactive'])}; proactive - mismatched {f(r['proactive_minus_mismatched'])}; "
                     f"guards {r['proactive_guards']:.2f} (bar {r['guard_bar']:.2f}) -> {'PASS' if r['pass'] else 'fail'}")
    lines.append(f"Gate 0: {'PASS' if res['pass'] else 'FAIL'}")
    return "\n".join(lines)


def format_context_rules(res: Dict) -> str:
    f = lambda d: "-" if d.get("mean") is None else f"{d['mean']:+.1f}±{d['se']:.1f}"
    lines = [f"value of each context rule alone (twin with that rule; seeds {res['seeds']})",
             f"{'rule':<48}{'steady':>8}{'proactive-steady':>18}{'proactive-opposite':>20}{'guards':>8}"]
    for r in res["rows"]:
        lines.append(f"{r['rule']:<48}{r['steady']:>8.1f}{f(r['proactive-steady']):>18}{f(r['proactive-opposite']):>20}"
                     f"{r['guards_proactive']:>8.2f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", default="open_env")
    ap.add_argument("--personas", nargs="+", default=None,
                    help="default: every persona; 'order' = the order-habit set (ORDER_PERSONAS)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--menu", nargs="+", default=DEFAULT_MENU)
    ap.add_argument("--view-radius", type=int, default=None, help="partial view for both players")
    ap.add_argument("--open-orders", type=int, default=1, help="orders open at once")
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(8, 16)),
                    help="new seeds to re-measure the chosen policies on (empty: skip)")
    ap.add_argument("--ego-set", choices=EGO_SETS, default="all",
                    help="which ego policies compete (commit: one order at a time; needs --open-orders > 1)")
    ap.add_argument("--cards", type=Path,
                    help="Gate 2 with the rule-based twin of each persona card (aligned / generic / misaligned egos)")
    ap.add_argument("--complexity", type=int, default=None, help="--cards: active rules per card (default all)")
    ap.add_argument("--rules", action="store_true",
                    help="score value of knowing each library rule alone (twin with one rule; see rule_screen)")
    ap.add_argument("--workers", type=int, default=1, help="--rules / --gate0: parallel episodes")
    ap.add_argument("--gate0", type=Path, help="plan v3 Gate 0: context cards (e.g. configs/coop/personas/batch_ctx.yaml)")
    ap.add_argument("--context-rules", action="store_true", help="plan v3: value of each context rule alone")
    ap.add_argument("--v0", type=Path, help="plan v4 V0: two-axis cards (configs/coop/personas/batch_style.yaml)")
    ap.add_argument("--episodes", type=int, default=3, help="--v0: episodes per seed")
    ap.add_argument("--soft", action="store_true", help="--v0: informed egos do their complement first, then "
                    "anything useful (policy proactive_soft) instead of only their complement")
    ap.add_argument("--log", type=Path, help="--v0: append the setting and the pass checks to this JSON-lines file")
    ap.add_argument("--slack", type=float, default=1.5, help="--gate0: order timer slack")
    ap.add_argument("--reference", type=str, help="--gate0: JSON {dish: reference steps} (coop.deadlines)")
    ap.add_argument("--urgent-steps", type=int, default=15, help="--gate0: theta, steps left that make an order urgent")
    ap.add_argument("--expiry-penalty", type=float, default=20)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.v0:
        dl = {"slack": args.slack, "reference": json.loads(args.reference), "urgent_steps": args.urgent_steps,
              "expiry_penalty": args.expiry_penalty}
        res = v0(args.v0, args.layout, args.menu, dl, args.seeds, args.eval_seeds, args.episodes, args.horizon,
                 args.view_radius, args.open_orders, args.workers, soft=args.soft)
        print(format_v0(res))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, indent=1, default=str))
            print(f"wrote {args.out}")
        log = args.log or (args.out.parent / "v0_tuning_log.jsonl" if args.out else None)
        if log:
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "soft": args.soft, "deadlines": dl, "view_radius": args.view_radius,
                   "menu": args.menu, "layout": args.layout, "seeds": [args.seeds, args.eval_seeds],
                   "episodes": args.episodes, "checks": res["checks"], "pass": res["pass_"],
                   "out": str(args.out) if args.out else None}
            with open(log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        return
    if args.gate0 or args.context_rules:
        dl = {"slack": args.slack, "reference": json.loads(args.reference), "urgent_steps": args.urgent_steps,
              "expiry_penalty": args.expiry_penalty}
        if args.gate0:
            res = gate0(args.gate0, args.layout, args.menu, dl, args.seeds, args.eval_seeds, args.horizon,
                        args.view_radius, args.open_orders, args.workers)
            print(format_gate0(res))
        else:
            res = context_rule_screen(args.layout, args.menu, dl, args.seeds, args.horizon, args.view_radius,
                                      args.open_orders, args.workers)
            print(format_context_rules(res))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, indent=1, default=str))
            print(f"wrote {args.out}")
        return
    if args.rules:
        res = rule_screen(args.layout, args.seeds, args.horizon, args.menu, args.view_radius, args.open_orders,
                          args.workers)
        print(format_rule_screen(res))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, indent=1, default=str))
            print(f"wrote {args.out}")
        return
    if args.cards:
        from .persona_check import format_gate2
        res = card_screen(args.cards, args.layout, args.seeds, args.horizon, args.menu, args.complexity,
                          args.view_radius, args.open_orders, args.eval_seeds)
        print(format_gate2(res))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, indent=1, default=str))
            print(f"wrote {args.out}")
        return
    if args.personas is None:
        args.personas = DEFAULT_PERSONAS + [p for p in PERSONAS if p not in DEFAULT_PERSONAS]
    elif args.personas == ["order"]:
        args.personas = list(ORDER_PERSONAS)
    started = time.time()
    egos = ego_set(args.ego_set, args.open_orders, args.menu)
    if not egos:
        ap.error("no ego policies in this set (commit needs --open-orders > 1)")
    res = screen(args.layout, args.personas, args.seeds, args.horizon, args.menu, egos=egos,
                 view_radius=args.view_radius, open_orders=args.open_orders, eval_seeds=args.eval_seeds)
    print(f"layout {args.layout}, horizon {args.horizon}, seeds {args.seeds}, view radius {args.view_radius}, "
          f"open orders {args.open_orders}; ego set {args.ego_set} ({len(egos)} policies); ego in slot 0")
    print(f"best fixed ego (without knowing the partner): {res['best_fixed']} "
          f"(mean {res['best_fixed_overall']:.1f} over personas)")
    print(f"{'persona':<18}{'best fixed':>11}{'best response':>36}{'value':>8}{'on new seeds':>14}")
    for p, r in res["personas"].items():
        ev = f"{r['eval_value']:>+14.1f}" if "eval_value" in r else ""
        print(f"{p:<18}{r['best_fixed_score']:>11.1f}{r['best_response'] + ' ' + format(r['best_response_score'], '.1f'):>36}"
              f"{r['value']:>+8.1f}{ev}")
    if res.get("eval_seeds"):
        print("value: on the seeds the policies were chosen with (optimistic); "
              "on new seeds: the same chosen policies re-measured (the number to trust)")
    print(f"({time.time() - started:.0f}s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
