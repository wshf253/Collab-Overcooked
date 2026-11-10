"""How much is it worth to know the partner? (value of information, no LLM)

For every partner persona, a set of scripted ego policies plays the same seeded
episodes:

* ``best fixed``: the ego policy with the highest mean score over all personas,
  i.e. the best you can do without knowing who the partner is;
* ``best response``: the ego policy with the highest score for this persona.

``value = best response - best fixed`` (points per episode) is what knowing the
partner could buy with these simple policies. A partner type whose value is
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

from .partner import DEFAULT_PERSONAS, PERSONAS, Traits
from .screen import DEFAULT_MENU, simulate

# Ego policies: no preference or one preferred stage, with or without giving way on tasks.
EGOS: Dict[str, Traits] = {}
for _pref in ((), ("prep",), ("cook",), ("serve",)):
    for _yield in (False, True):
        _name = ("any" if not _pref else _pref[0] + "_first") + ("+yield" if _yield else "")
        EGOS[_name] = Traits(preference=_pref, yield_tasks=_yield)


def screen(layout: str, personas: List[str], seeds: List[int], horizon: int, menu: List[str],
           egos: Dict[str, Traits] = EGOS, ego_index: int = 0) -> Dict:
    partner_index = 1 - ego_index
    scores: Dict[str, Dict[str, float]] = {}       # persona -> ego -> mean score
    for persona in personas:
        scores[persona] = {}
        for ego, traits in egos.items():
            vals = []
            for seed in seeds:
                env, _ = simulate(menu, layout, horizon, seed, persona, partner_index,
                                  teammate=(f"ego_{ego}", traits))
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
    return {"layout": layout, "seeds": seeds, "horizon": horizon, "best_fixed": best_fixed,
            "best_fixed_overall": overall[best_fixed], "overall": overall, "personas": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", default="open_env")
    ap.add_argument("--personas", nargs="+", default=DEFAULT_PERSONAS + [p for p in PERSONAS
                                                                        if p not in DEFAULT_PERSONAS])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--menu", nargs="+", default=DEFAULT_MENU)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    started = time.time()
    res = screen(args.layout, args.personas, args.seeds, args.horizon, args.menu)
    print(f"layout {args.layout}, horizon {args.horizon}, seeds {args.seeds}; ego in slot 0")
    print(f"best fixed ego (without knowing the partner): {res['best_fixed']} "
          f"(mean {res['best_fixed_overall']:.1f} over personas)")
    print(f"{'persona':<18}{'best fixed':>11}{'best response':>28}{'value':>8}")
    for p, r in res["personas"].items():
        print(f"{p:<18}{r['best_fixed_score']:>11.1f}{r['best_response'] + ' ' + format(r['best_response_score'], '.1f'):>28}"
              f"{r['value']:>+8.1f}")
    print(f"({time.time() - started:.0f}s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
