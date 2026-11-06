"""Partner-type screen: does each simulated human behave measurably differently?

Runs every persona as one player with a neutral "steady" simulated human as the
other player (no LLM), on the same seeded order sequences, and reports team
outcomes plus the partner's ground-truth behaviour.

    PYTHONPATH=src:lib/overcooked_ai python -m coop.screen --out outputs/coop_screen.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

from .env import ContinuousOrderEnv
from .partner import PERSONAS, SimulatedHuman

DEFAULT_MENU = ["boiled_egg", "boiled_mushroom", "boiled_corn_slices",
                "baked_carrot_soup", "sliced_eggplant_and_chickpea_stew"]


def simulate(menu, layout, horizon, seed, persona, partner_index, penalty=20,
             teammate="steady") -> Tuple[ContinuousOrderEnv, List[SimulatedHuman]]:
    """One episode: ``persona`` in slot ``partner_index``, ``teammate`` in the other slot."""
    env = ContinuousOrderEnv(menu, layout=layout, horizon=horizon, wrong_delivery_penalty=penalty)
    with contextlib.redirect_stdout(io.StringIO()):   # the legacy MDP prints every step
        env.reset(seed)
    players: List[SimulatedHuman] = [None, None]
    players[partner_index] = SimulatedHuman.persona(persona)
    players[1 - partner_index] = SimulatedHuman.persona(teammate)
    for i, p in enumerate(players):
        p.reset(env.mdp, i, seed)
    while not env.done:
        moves = [p.act(env.state, env.t) for p in players]
        with contextlib.redirect_stdout(io.StringIO()):
            env.step([m[0] for m in moves], [m[1] for m in moves])
    return env, players


def run_episode(menu, layout, horizon, seed, persona, partner_index, penalty=20) -> Dict:
    env, players = simulate(menu, layout, horizon, seed, persona, partner_index, penalty)
    return {"env": env.summary(), "partner": players[partner_index].summary(),
            "ego": players[1 - partner_index].summary()}


def aggregate(runs: List[Dict]) -> Dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else None

    def share(key):
        names = sorted({k for r in runs for k in r["partner"][key]})
        return {n: mean([r["partner"][key].get(n, 0.0) for r in runs]) for n in names}

    return {
        "episodes": len(runs),
        "score": mean([r["env"]["score"] for r in runs]),
        "orders": mean([r["env"]["orders_completed"] for r in runs]),
        "wrong_deliveries": mean([r["env"]["wrong_deliveries"] for r in runs]),
        "rejected": mean([r["env"]["rejected_interactions"] for r in runs]),
        "steps_per_order": mean([r["env"]["mean_order_steps"] for r in runs]),
        "partner_slips": mean([r["partner"]["slips"] for r in runs]),
        "partner_abandons": mean([r["partner"]["abandons"] for r in runs]),
        "partner_status": share("status_share"),
        "partner_stage_when_acting": share("stage_share_when_acting"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--partner-index", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    ap.add_argument("--layout", default="new_env")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--menu", nargs="+", default=DEFAULT_MENU)
    ap.add_argument("--out", type=Path, default=None, help="JSON with every episode and the aggregates")
    args = ap.parse_args(argv)

    started = time.time()
    results, table = {}, []
    for index in args.partner_index:
        for persona in args.personas:
            runs = [run_episode(args.menu, args.layout, args.horizon, s, persona, index) for s in args.seeds]
            agg = aggregate(runs)
            results[f"{persona}@p{index}"] = {"aggregate": agg, "runs": runs}
            st = agg["partner_status"]
            sg = agg["partner_stage_when_acting"]
            table.append((index, persona, agg["score"], agg["orders"], agg["wrong_deliveries"],
                          agg["steps_per_order"], st.get("act", 0), sg.get("prep", 0), sg.get("cook", 0),
                          sg.get("serve", 0), agg["partner_slips"], agg["partner_abandons"], agg["rejected"]))

    print(f"layout {args.layout}, horizon {args.horizon}, seeds {args.seeds}, menu {args.menu}")
    print("p = player slot of the simulated partner (the other slot is 'steady'); means over seeds")
    head = ("p", "persona", "score", "orders", "wrong", "steps/ord", "act%", "prep%", "cook%", "serve%",
            "slips", "abandons", "rejected")
    print("{:>2} {:<12} {:>6} {:>6} {:>5} {:>9} {:>5} {:>5} {:>5} {:>6} {:>5} {:>8} {:>8}".format(*head))
    for r in table:
        print("{:>2} {:<12} {:>6.1f} {:>6.2f} {:>5.2f} {:>9.1f} {:>5.0%} {:>5.0%} {:>5.0%} {:>6.0%} {:>5.2f} {:>8.2f} {:>8.2f}"
              .format(*[x if x is not None else float("nan") for x in r]))
    print(f"({time.time() - started:.0f}s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"config": dict(vars(args), out=str(args.out)), "results": results},
                                       indent=1, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
