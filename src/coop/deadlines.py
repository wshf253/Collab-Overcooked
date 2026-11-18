"""Reference times for order timers (docs/CONTEXT_PARTNER_PLAN_v3.md, 3.1).

    PYTHONPATH=src:lib/overcooked_ai python -m coop.deadlines --layout open_ing2_env \
        --menu boiled_potato_slices baked_carrot_soup sliced_carrot_and_potato_stew \
        --open-orders 2 --view-radius 2 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

A pair of scripted ``steady`` cooks plays without timers; the reference time of a dish is
the median number of steps its orders stayed open (issue to delivery, counting the delivery
step). An order's limit is then ``slack x reference``. The table goes into the config's
``deadlines.reference`` and into docs/coop-setting.md.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
from typing import Dict, List, Optional

from .env import ContinuousOrderEnv
from .partner import PERSONAS, SimulatedHuman


def play_pair(menu: List[str], layout: str, seed: int, horizon: int = 300, open_orders: int = 2,
              view_radius: Optional[int] = None, deadlines: Optional[Dict] = None,
              order_mode: str = "controlled") -> Dict:
    env = ContinuousOrderEnv(menu, layout=layout, horizon=horizon, open_orders=open_orders,
                             order_mode=order_mode, deadlines=deadlines)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed)
    players = [SimulatedHuman(PERSONAS["steady"], "steady_a"), SimulatedHuman(PERSONAS["steady"], "steady_b")]
    for i, p in enumerate(players):
        p.reset(env.mdp, i, seed, view_radius=view_radius)
    while not env.done:
        t, state = env.t, env.state
        moves = [p.act(state, t) for p in players]
        with contextlib.redirect_stdout(io.StringIO()):
            env.step([m[0] for m in moves], [m[1] for m in moves])
    return env.summary()


def reference_times(menu: List[str], layout: str, seeds: List[int], horizon: int = 300, open_orders: int = 2,
                    view_radius: Optional[int] = None) -> Dict[str, Dict]:
    durations: Dict[str, List[int]] = {d: [] for d in menu}
    for seed in seeds:
        s = play_pair(menu, layout, seed, horizon, open_orders, view_radius)
        for o in s["orders"]:
            if o.get("duration") is not None:
                durations[o["name"]].append(o["duration"])
    return {d: {"median": statistics.median(v) if v else None, "n": len(v),
                "p25": statistics.quantiles(v, n=4)[0] if len(v) >= 2 else None,
                "p75": statistics.quantiles(v, n=4)[2] if len(v) >= 2 else None}
            for d, v in durations.items()}


def from_folder(out, conditions: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Per dish in a finished runner folder: orders completed / expired, and the steps the
    completed ones took (with timers these are only the orders that made it, so the median
    is a lower bound on what the players need)."""
    from pathlib import Path
    acc: Dict[str, Dict[str, List]] = {}
    for f in sorted(Path(out).glob("runs/*/*/seed*/day*/ep*/summary.json")):
        cond = f.parents[4].name
        if cond.startswith("_") or (conditions and cond not in conditions):
            continue
        s = json.loads(f.read_text())
        for o in s.get("orders", []):
            a = acc.setdefault(o["name"], {"done": [], "expired": 0, "limit": o.get("limit")})
            if o.get("completed_t") is not None:
                a["done"].append(o["duration"])
            elif o.get("expired_t") is not None:
                a["expired"] += 1
    return {d: {"completed": len(a["done"]), "expired": a["expired"], "limit": a["limit"],
                "median_steps": statistics.median(a["done"]) if a["done"] else None,
                "p75_steps": statistics.quantiles(a["done"], n=4)[2] if len(a["done"]) >= 2 else None}
            for d, a in sorted(acc.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", default="open_ing2_env")
    ap.add_argument("--menu", nargs="+")
    ap.add_argument("--from-folder", help="a finished runner folder: per dish, completed / expired and steps taken")
    ap.add_argument("--conditions", nargs="+", help="--from-folder: only these conditions")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(16)))
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--open-orders", type=int, default=2)
    ap.add_argument("--view-radius", type=int, default=None)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.from_folder:
        res = from_folder(args.from_folder, args.conditions)
        print(f"orders per dish in {args.from_folder} (conditions {args.conditions or 'all'})")
        print(f"{'dish':<36}{'limit':>7}{'done':>6}{'expired':>9}{'median steps':>14}{'p75':>6}")
        for d, r in res.items():
            f = lambda x: "-" if x is None else f"{x:.0f}"
            print(f"{d:<36}{f(r['limit']):>7}{r['completed']:>6}{r['expired']:>9}{f(r['median_steps']):>14}{f(r['p75_steps']):>6}")
        return
    if not args.menu:
        ap.error("--menu is required (or --from-folder)")
    res = reference_times(args.menu, args.layout, args.seeds, args.horizon, args.open_orders, args.view_radius)
    print(f"reference times (steady pair, {args.layout}, {args.open_orders} open orders, view radius "
          f"{args.view_radius}, seeds {args.seeds[0]}..{args.seeds[-1]})")
    print(f"{'dish':<36}{'median':>8}{'p25':>7}{'p75':>7}{'n':>5}")
    for d, r in res.items():
        f = lambda x: "-" if x is None else f"{x:.0f}"
        print(f"{d:<36}{f(r['median']):>8}{f(r['p25']):>7}{f(r['p75']):>7}{r['n']:>5}")
    print("deadlines.reference: " + json.dumps({d: r["median"] for d, r in res.items()}))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
