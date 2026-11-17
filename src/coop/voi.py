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
from typing import Dict, List, Tuple

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
    ap.add_argument("--workers", type=int, default=1, help="--rules: parallel episodes")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
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
