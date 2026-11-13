"""Compare two coop.days output folders that differ in one setting (e.g. partial vs full view).

    PYTHONPATH=src:lib/overcooked_ai python -m coop.compare outputs/days_orders outputs/days_orders_full \
        [--out outputs/compare_view.json]

Both folders play the same episodes (the orders depend only on run seed, day and
episode), so a run (persona x run seed) of folder A has a twin in folder B. For each
contrast, e.g. ``tom_memory - direct``:

* within each folder, each run's mean paired difference over identical episodes;
* **the change**: per run, (contrast in A) - (contrast in B), mean ± SE over runs.
  A positive change for ``tom_memory - direct`` means tom_memory helps more in A.

Also: each condition's score minus the folder's own scripted reference, and the
probe accuracy per field. The folders must come from the same code (code_version.json)
and the same config except for the settings that differ; both are checked and
reported.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analyze import REFERENCE, _mean_se, load_episodes, rescore_probe

CONTRASTS = [("tom", "direct"), ("memory", "direct"), ("tom_memory", "direct"), ("tom_memory", "tom"),
             ("tom_memory_swap", "tom_memory"), ("tom_memory_placebo", "tom_memory"),
             ("tom_struct", "direct"), ("tom_model", "tom_memory"), ("tom_struct_model", "tom_model"),
             ("tom_struct_model", "tom_memory"), ("tom_struct_model", "direct"),
             ("tom_struct_model_swap", "tom_struct_model"), ("tom_struct_model_placebo", "tom_struct_model")]
IGNORED_KEYS = {"name"}


def _scores(eps) -> Dict[Tuple, float]:
    """(condition, persona, seed, day, episode) -> score."""
    return {(e["condition"], e["persona"], e["seed"], e["day"], e["episode"]): e["summary"]["score"] for e in eps}


def run_contrast(scores: Dict[Tuple, float], a: str, b: str) -> Dict[Tuple[str, int], float]:
    """(persona, seed) -> mean over identical episodes of score(a) - score(b)."""
    per: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for k, v in scores.items():
        if k[0] == a and (b,) + k[1:] in scores:
            per[k[1:3]].append(v - scores[(b,) + k[1:]])
    return {k: statistics.fmean(v) for k, v in per.items()}


def _summary(per_run: Dict[Tuple[str, int], float]) -> Dict[str, Any]:
    by_persona: Dict[str, List[float]] = defaultdict(list)
    for (persona, _), v in per_run.items():
        by_persona[persona].append(v)
    return {"all": _mean_se(list(per_run.values())),
            "personas": {p: _mean_se(v) for p, v in sorted(by_persona.items())}}


def _config_diff(a: Path, b: Path) -> Dict[str, Any]:
    try:
        ca, cb = (json.loads((d / "config.json").read_text()) for d in (a, b))
    except OSError:
        return {}
    return {k: [ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb))
            if k not in IGNORED_KEYS and ca.get(k) != cb.get(k)}


def _version(d: Path) -> Optional[str]:
    try:
        return json.loads((d / "code_version.json").read_text()).get("hash")
    except (OSError, ValueError, AttributeError):
        return None


def _probe_accuracy(eps) -> Dict[str, Dict[str, float]]:
    """condition -> field -> mean accuracy (current probe rules)."""
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for e in eps:
        p = rescore_probe(e)
        if p:
            for f, ok in p["correct"].items():
                acc[e["condition"]][f].append(float(ok))
    return {c: {f: statistics.fmean(v) for f, v in fs.items()} for c, fs in acc.items()}


def compare(a: Path, b: Path) -> Dict[str, Any]:
    eps_a, eps_b = load_episodes(a), load_episodes(b)
    sa, sb = _scores(eps_a), _scores(eps_b)
    out: Dict[str, Any] = {"a": str(a), "b": str(b), "config_differences": _config_diff(a, b),
                           "code_version": [_version(a), _version(b)], "contrasts": {}, "vs_reference": {}}
    conds = sorted({k[0] for k in sa} & {k[0] for k in sb})
    for x, y in CONTRASTS:
        if x not in conds or y not in conds:
            continue
        ca, cb = run_contrast(sa, x, y), run_contrast(sb, x, y)
        change = {k: ca[k] - cb[k] for k in ca if k in cb}
        out["contrasts"][f"{x} - {y}"] = {"a": _summary(ca), "b": _summary(cb), "change": _summary(change)}
    for c in conds:
        if c == REFERENCE:
            continue
        out["vs_reference"][c] = {"a": _summary(run_contrast(sa, c, REFERENCE)),
                                  "b": _summary(run_contrast(sb, c, REFERENCE))}
    out["probe_accuracy"] = {"a": _probe_accuracy(eps_a), "b": _probe_accuracy(eps_b)}
    return out


def _f(ms) -> str:
    if not ms or ms["mean"] is None:
        return "-"
    return f"{ms['mean']:+.1f}" + (f"±{ms['se']:.1f}" if ms.get("se") is not None else "")


def format_compare(rep: Dict[str, Any]) -> str:
    lines = [f"A = {rep['a']}", f"B = {rep['b']}"]
    va, vb = rep["code_version"]
    if va != vb:
        lines.append(f"WARNING: different code versions ({va} vs {vb}): the difference may come from code changes")
    diff = rep["config_differences"]
    lines.append("config differences: " + (", ".join(f"{k}: {v[0]!r} vs {v[1]!r}" for k, v in diff.items())
                                         or "none (are these really different settings?)"))
    lines.append("\n== contrasts: mean±SE over runs (persona x seed); change = A - B, paired by run ==")
    lines.append(f"  {'contrast':<34}{'in A':>14}{'in B':>14}{'change (A-B)':>16}")
    for name, r in rep["contrasts"].items():
        lines.append(f"  {name:<34}{_f(r['a']['all']):>14}{_f(r['b']['all']):>14}{_f(r['change']['all']):>16}")
    lines.append("\n  change per persona:")
    for name, r in rep["contrasts"].items():
        per = ", ".join(f"{p} {_f(v)}" for p, v in r["change"]["personas"].items())
        lines.append(f"    {name}: {per}")
    lines.append("\n== score minus the folder's own scripted reference (mean±SE over runs) ==")
    for c, r in rep["vs_reference"].items():
        lines.append(f"  {c:<22} A {_f(r['a']['all']):>12}   B {_f(r['b']['all']):>12}")
    pa = rep["probe_accuracy"]
    if pa["a"] or pa["b"]:
        lines.append("\n== probe accuracy per field (A / B) ==")
        for c in sorted(set(pa["a"]) | set(pa["b"])):
            fields = sorted(set(pa["a"].get(c, {})) | set(pa["b"].get(c, {})))
            cells = []
            for f in fields:
                x, y = pa["a"].get(c, {}).get(f), pa["b"].get(c, {}).get(f)
                cells.append(f"{f} {'-' if x is None else f'{x:.2f}'}/{'-' if y is None else f'{y:.2f}'}")
            lines.append(f"  {c:<22} " + " | ".join(cells))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path, help="first output folder (e.g. partial view)")
    ap.add_argument("b", type=Path, help="second output folder (e.g. full view)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rep = compare(args.a, args.b)
    print(format_compare(rep))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
