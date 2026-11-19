"""One-decision test: does the partner change the right action, and does the ego act on it?

A start is a fixed kitchen: two open orders, ``other`` (the older one, a long timer) and
``urgent`` (a dish with a short timer, urgent from step 0). The partner is the scripted twin
of a two-rule card: it does one order start to finish (``default_work: whole``) and, when an
order becomes urgent, either ``takes`` it or ``keeps`` its own (``on_urgent``). The ego's
choice is which order it works on first.

Stages (``--stage``):

``dishes``  per dish in a finished runner folder (``--from-folder``): how often the ego
            delivered it, expiries and the ego's wrong deliveries, to pick an urgent dish the
            ego completes reliably.
``screen``  scripted replay (CPU): both ego choices (a scripted cook committed to that order
            first) against both partners, on every start. A start is kept only if the better
            choice switches with the partner.
``llm``     the LLM ego on the kept starts, against both partners, with the correct partner
            description, the swapped one, or none. Records its first committed action (the
            first action that belongs to one of the two orders), whether the urgent order was
            delivered in time and by whom, and its recipe errors after that (loads and starts
            that make nothing an open order needs, wrong deliveries).
``report``  the screen and the LLM table, and which of the three cases holds:
            no switching starts -> change the partner behaviours; the correct description does
            not move the first choice -> a stronger ego; the choice is right but the dish is not
            finished in time or is spoiled -> cooking competence.

    PYTHONPATH=src:lib/overcooked_ai python -m coop.decision configs/coop/decision/urgent.yaml \\
        --stage screen --out outputs/decision_urgent
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .context import tree
from .context_partner import ContextPlayer
from .env import ContinuousOrderEnv
from .executor import START_VERBS
from .kitchen import Kitchen
from .orders import normalize_menu
from .rules import load_library, parse_rule, render

BEHAVIOURS = ("takes", "keeps")
DESCRIPTIONS = ("correct", "swapped", "none")
CHOICES = ("urgent", "other")


# ---------------------------------------------------------------------- #
# Config and starts
# ---------------------------------------------------------------------- #
def load(path) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("layout", "open_ing2_env")
    cfg.setdefault("horizon", 150)
    cfg.setdefault("view_radius", 2)
    cfg.setdefault("wrong_delivery_penalty", 20)
    cfg.setdefault("expiry_penalty", 20)
    cfg.setdefault("other_limit", 250)
    cfg.setdefault("seeds", [0, 1, 2, 3])
    cfg.setdefault("ego_slots", [0, 1])
    cfg.setdefault("min_gap", 1)
    return cfg


def starts(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for urgent, other in cfg["pairs"]:
        for limit in cfg["urgent_limits"]:
            for slot in cfg["ego_slots"]:
                for seed in cfg["seeds"]:
                    out.append({"id": f"{urgent}__{other}__L{limit}__slot{slot}__s{seed}", "urgent": urgent,
                                "other": other, "limit": int(limit), "slot": int(slot), "seed": int(seed)})
    return out


def tail_dishes(cfg, start) -> List[str]:
    """Orders after the first two: the menu's other dishes, so neither order comes back."""
    rest = [d for d in cfg["menu"] if d not in (start["urgent"], start["other"])]
    return (rest * 10)[:10] if rest else []


def make_env(cfg, start) -> Tuple[ContinuousOrderEnv, Any]:
    menu = [start["other"], start["urgent"]] + [d for d in cfg["menu"] if d not in (start["other"], start["urgent"])]
    ref = {d: (start["limit"] if d == start["urgent"] else cfg["other_limit"]) for d in menu}
    theta = int(cfg.get("urgent_steps") or max(cfg["urgent_limits"]))
    if theta >= cfg["other_limit"]:
        raise ValueError("urgent_steps must be below other_limit (only the urgent order may be urgent)")
    deadlines = {"slack": 1.0, "reference": ref, "urgent_steps": theta, "expiry_penalty": cfg["expiry_penalty"]}
    env = ContinuousOrderEnv(menu, layout=cfg["layout"], horizon=cfg["horizon"],
                             wrong_delivery_penalty=cfg["wrong_delivery_penalty"], open_orders=2,
                             order_mode="controlled", deadlines=deadlines)
    with contextlib.redirect_stdout(io.StringIO()):
        state = env.reset(start["seed"], sequence=[start["other"], start["urgent"]] + tail_dishes(cfg, start))
    return env, state


def card_rules(behaviour: str):
    lib = load_library()
    return [parse_rule({"default_work": {"mode": "whole"}}, lib), parse_rule({"on_urgent": {"mode": behaviour}}, lib)]


def description(behaviour: str, kitchen) -> List[str]:
    """The partner's card as the oracle shows it: its wants lines."""
    lib = load_library()
    return [render(r, "wants", lib, kitchen) for r in card_rules(behaviour)]


class CommitEgo(ContextPlayer):
    """Scripted cook committed to ``first`` until that order is closed, then the newest open
    order the partner is not on; does its order start to finish."""

    def __init__(self, first: str, menu: List[str]):
        super().__init__(card_rules("takes")[:1], menu, "ego", "proactive", name=f"commit_{first}")
        self.first = first

    def _pick_commit(self, ctx, newest):
        if self.first is not None:
            if self.first in ctx["open"]:
                self.commit = self.first
                return self.first
            self.first = None
        return super()._pick_commit(ctx, newest)


# ---------------------------------------------------------------------- #
# One episode
# ---------------------------------------------------------------------- #
def classify(macro, state, kit: Kitchen, index: int, urgent: str, other: str) -> Optional[str]:
    """Which of the two orders an action belongs to (None: neither or both)."""
    name, args = macro.name, list(macro.args)
    held = state.players[index].held_object
    held_name = None if held is None else (held.state[0] if held.state is not None else held.name)
    items: List[str] = []
    if name == "pickup":
        item, place = args
        if item == "dish":
            need = [d for d in (urgent, other) if kit.need_dish(d)]
            return {urgent: "urgent", other: "other"}[need[0]] if len(need) == 1 else None
        items = [item]
    elif name in ("put_obj_in_utensil", "place_obj_on_counter", "deliver_soup"):
        items = [held_name] if held_name else []
    elif name in START_VERBS or name == "fill_dish_with_food":
        _, product, contents, _ = kit.utensil_status(state, args[0])
        items = list(contents) + ([product] if product else [])
    tu, to = tree(kit, urgent), tree(kit, other)
    in_u = any(i in tu for i in items)
    in_o = any(i in to for i in items)
    if in_u and not in_o:
        return "urgent"
    if in_o and not in_u:
        return "other"
    return None


def recipe_error(ex, macro, state, index: int) -> Optional[str]:
    """The recipe guard's reason (coop.executor) for this action on the true state, if any."""
    name = macro.name
    if name == "put_obj_in_utensil":
        held = state.players[index].held_object
        if held is None:
            return None
        return ex._useless_load(state, macro.args[0], held.state[0] if held.state is not None else held.name)
    if name in START_VERBS:
        return ex._useless_start(state, macro.args[0])
    return None


def play(cfg, start, behaviour: str, ego, llm: bool) -> Dict[str, Any]:
    env, state0 = make_env(cfg, start)
    slot, other_i = start["slot"], 1 - start["slot"]
    menu = list(env.menu) if hasattr(env, "menu") else [start["other"], start["urgent"]]
    partner = ContextPlayer(card_rules(behaviour), list(normalize_menu(menu)), "partner", "card", name=behaviour)
    partner.reset(env.mdp, other_i, start["seed"], view_radius=cfg["view_radius"])
    kit = Kitchen(env.mdp, slot, state0)
    if llm:
        ego.begin_episode(env.mdp, slot, {"day": 1, "episode": 1, "episode_seed": start["seed"],
                                          "layout": cfg["layout"], "horizon": cfg["horizon"],
                                          "menu": normalize_menu(menu), "index": slot, "partner_index": other_i,
                                          "wrong_delivery_penalty": cfg["wrong_delivery_penalty"],
                                          "open_orders": 2, "view_radius": cfg["view_radius"]})
    else:
        ego.reset(env.mdp, slot, start["seed"], view_radius=cfg["view_radius"])
    actions: List[Dict[str, Any]] = []
    first: Optional[Dict[str, Any]] = None
    last_macro = None
    while not env.done:
        t, state = env.t, env.state
        moves = [None, None]
        moves[slot] = ego.act(state, t)
        moves[other_i] = partner.act(state, t)
        if llm and ego.executor is not None and ego.executor.macro is not None and ego.executor.macro is not last_macro:
            m = last_macro = ego.executor.macro
            if m.name != "wait":
                ex = ego.executor
                row = {"t": t, "action": str(m), "order": classify(m, state, kit, slot, start["urgent"], start["other"]),
                       "recipe_error": recipe_error(ex, m, state, slot)}
                actions.append(row)
                if first is None and row["order"] is not None:
                    first = row
        with contextlib.redirect_stdout(io.StringIO()):
            state, reward, _, info = env.step([m[0] for m in moves], [m[1] for m in moves])
        if llm:
            ego.observe(t, reward, info)
        if hasattr(partner, "observe"):
            partner.observe(t, reward, info)
    s = env.summary()
    u = next(o for o in s["orders"] if o["name"] == start["urgent"])
    o = next(o for o in s["orders"] if o["name"] == start["other"])
    who = lambda rec: None if rec.get("delivered_by") is None else ("ego" if rec["delivered_by"] == slot else "partner")
    res = {"start": start, "partner": behaviour, "score": s["score"],
           "urgent_on_time": u.get("completed_t") is not None, "urgent_by": who(u),
           "urgent_expired": u.get("expired_t") is not None, "other_done": o.get("completed_t") is not None,
           "other_by": who(o), "deliveries": s.get("deliveries", []),
           "ego_wrong_deliveries": sum(1 for d in s.get("deliveries", []) if d["agent"] == slot and not d["correct"]),
           "partner_wrong_deliveries": sum(1 for d in s.get("deliveries", []) if d["agent"] != slot and not d["correct"]),
           "partner_decisions": partner.decision_log[:12]}
    if llm:
        after = [a for a in actions if first is not None and a["t"] >= first["t"]]
        res.update(first=first, ego_actions=actions,
                   recipe_errors=[a for a in after if a["recipe_error"]],
                   invalid=getattr(ego, "invalid", 0), fallbacks=getattr(ego, "fallbacks", 0))
    return res


# ---------------------------------------------------------------------- #
# Stages
# ---------------------------------------------------------------------- #
def _check_out(cfg, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    f = out / "config.json"
    if f.exists():
        old = json.loads(f.read_text())
        if {k: v for k, v in old.items() if k != "agent"} != {k: v for k, v in cfg.items() if k != "agent"}:
            raise SystemExit(f"{f} holds a different config: use a new --out")
    f.write_text(json.dumps(cfg, indent=1))


def screen(cfg, out: Path) -> Dict[str, Any]:
    _check_out(cfg, out)
    rows = []
    for st in starts(cfg):
        cell = {}
        for b in BEHAVIOURS:
            for c in CHOICES:
                ego = CommitEgo(st[c], list(normalize_menu(cfg["menu"])))
                r = play(cfg, st, b, ego, llm=False)
                cell[f"{b}/{c}"] = {k: r[k] for k in ("score", "urgent_on_time", "urgent_by", "other_done")}
        best = {}
        for b in BEHAVIOURS:
            su, so = cell[f"{b}/urgent"]["score"], cell[f"{b}/other"]["score"]
            best[b] = "urgent" if su - so >= cfg["min_gap"] else "other" if so - su >= cfg["min_gap"] else None
        kept = best["takes"] is not None and best["keeps"] is not None and best["takes"] != best["keeps"]
        rows.append({"start": st, "cells": cell, "best": best, "kept": kept})
    res = {"rows": rows, "kept": [r["start"]["id"] for r in rows if r["kept"]]}
    (out / "screen.json").write_text(json.dumps(res, indent=1))
    return res


def _llm_job(cfg, out: str, start, behaviour: str, desc: str) -> Dict[str, Any]:
    from .agents import make_agent
    from . import llm_agent  # noqa: F401  (registers the "llm" agent type)
    d = Path(out) / "llm" / start["id"] / behaviour / desc
    if (d / "result.json").exists():
        return json.loads((d / "result.json").read_text())
    spec = dict(cfg["agent"])
    spec.update(memory=False, probe=False)
    env, state0 = make_env(cfg, start)
    kit = Kitchen(env.mdp, start["slot"], state0)
    told = {"correct": behaviour, "swapped": [b for b in BEHAVIOURS if b != behaviour][0], "none": None}[desc]
    if told:
        spec["oracle_wants"] = description(told, kit)
    agent = make_agent(spec)
    r = play(cfg, start, behaviour, agent, llm=True)
    r.update(description=desc, told=told)
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(r, indent=1, default=str))
    (d / "agent_log.json").write_text(json.dumps(agent.episode_log(), default=str))
    return r


def llm_stage(cfg, out: Path, workers: int = 1, max_starts: Optional[int] = None) -> List[Dict[str, Any]]:
    _check_out(cfg, out)
    if not (out / "screen.json").exists():
        screen(cfg, out)
    scr = json.loads((out / "screen.json").read_text())
    kept = [r["start"] for r in scr["rows"] if r["kept"]][:max_starts]
    if not kept:
        print("no kept starts: the better choice never switches with the partner; change the partner "
              "behaviours or the starts (urgent_limits, pairs) before running the LLM")
        return []
    jobs = [(st, b, d) for st in kept for b in BEHAVIOURS for d in DESCRIPTIONS]
    print(f"{len(kept)} kept starts x 2 partners x 3 descriptions = {len(jobs)} episodes")
    res = []
    if workers <= 1:
        res = [_llm_job(cfg, str(out), *j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_llm_job, cfg, str(out), *j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                res.append(f.result())
                if i % 10 == 0 or i == len(futs):
                    print(f"{i}/{len(futs)} episodes done", flush=True)
    return res


def _rate(xs) -> str:
    xs = list(xs)
    return "-" if not xs else f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})"


def report(cfg, out: Path) -> str:
    lines = []
    scr = json.loads((out / "screen.json").read_text()) if (out / "screen.json").exists() else None
    if scr is None:
        return "no screen.json: run --stage screen first"
    lines.append("== Screen (scripted ego committed to one order first, scripted twin partner) ==")
    groups = defaultdict(list)
    for r in scr["rows"]:
        st = r["start"]
        groups[(st["urgent"], st["other"], st["limit"])].append(r)
    lines.append(f"{'urgent / other / limit':<58}{'takes: U  O':>14}{'keeps: U  O':>14}{'best takes/keeps':>20}{'kept':>8}")
    for (u, o, lim), rs in sorted(groups.items()):
        m = lambda key: statistics.mean(r["cells"][key]["score"] for r in rs)
        bt = "/".join(f"{sum(r['best'][b] == 'urgent' for r in rs)}U{sum(r['best'][b] == 'other' for r in rs)}O"
                      for b in BEHAVIOURS)
        lines.append(f"{u + ' / ' + o + ' / ' + str(lim):<58}{m('takes/urgent'):>7.1f}{m('takes/other'):>7.1f}"
                     f"{m('keeps/urgent'):>7.1f}{m('keeps/other'):>7.1f}{bt:>20}{sum(r['kept'] for r in rs):>5}/{len(rs)}")
    lines.append(f"kept starts: {len(scr['kept'])} of {len(scr['rows'])} (the better choice switches with the partner)")
    best = {r["start"]["id"]: r["best"] for r in scr["rows"]}
    res = [json.loads(f.read_text()) for f in sorted((out / "llm").glob("*/*/*/result.json"))] if (out / "llm").exists() else []
    if not scr["kept"]:
        lines.append("\nVerdict: no start switches the better choice -> change the partner behaviours (or the starts).")
        return "\n".join(lines)
    if not res:
        lines.append("\nno LLM results yet: run --stage llm")
        return "\n".join(lines)
    lines.append("\n== LLM ego on the kept starts ==")
    lines.append("right = first committed action on the order that the screen found better for this partner")
    hdr = f"{'description':<12}{'partner':<8}{'n':>4}{'right first':>16}{'no choice':>11}{'urgent on time':>17}{'score':>8}{'recipe err':>12}{'wrong del':>11}"
    lines.append(hdr)
    by = defaultdict(list)
    for r in res:
        r["right"] = r["first"] is not None and r["first"]["order"] == best[r["start"]["id"]][r["partner"]]
        by[(r["description"], r["partner"])].append(r)
        by[(r["description"], "all")].append(r)
    for d in DESCRIPTIONS:
        for b in BEHAVIOURS + ("all",):
            rs = by.get((d, b), [])
            if not rs:
                continue
            lines.append(f"{d:<12}{b:<8}{len(rs):>4}{_rate(r['right'] for r in rs):>16}"
                         f"{sum(r['first'] is None for r in rs):>11}{_rate(r['urgent_on_time'] for r in rs):>17}"
                         f"{statistics.mean(r['score'] for r in rs):>8.1f}"
                         f"{statistics.mean(len(r['recipe_errors']) for r in rs):>12.2f}"
                         f"{statistics.mean(r['ego_wrong_deliveries'] for r in rs):>11.2f}")
    # what it was told vs what it chose: the first decision comes before the partner acts, so
    # the partner's true behaviour does not enter it; pool over partners by the description told
    lines.append("\nFirst choice by what the ego was told (pooled over the true partner; ideal: told keeps ->"
                 " urgent 100%, told takes -> urgent 0%)")
    lines.append(f"{'told':<8}{'pair (urgent / other)':<52}{'n':>4}{'first = urgent':>17}{'at t=0':>13}")
    told_rows = defaultdict(list)
    for r in res:
        told_rows[(r.get("told") or "none", r["start"]["urgent"] + " / " + r["start"]["other"])].append(r)
        told_rows[(r.get("told") or "none", "all")].append(r)
    pairs = sorted({k[1] for k in told_rows if k[1] != "all"}) + ["all"]
    for tl in ("keeps", "takes", "none"):
        for pr in pairs:
            rs = told_rows.get((tl, pr), [])
            if rs:
                lines.append(f"{tl:<8}{pr:<52}{len(rs):>4}{_rate(r['first'] is not None and r['first']['order'] == 'urgent' for r in rs):>17}"
                             f"{_rate(r['first'] is not None and r['first']['t'] == 0 for r in rs):>13}")
    pu = lambda tl: (lambda rs: 100 * sum(r["first"] is not None and r["first"]["order"] == "urgent" for r in rs) / len(rs)
                     if rs else float("nan"))(told_rows.get((tl, "all"), []))
    effect = pu("keeps") - pu("takes")
    lines.append(f"description effect on the first choice: told keeps - told takes = {effect:+.0f} points "
                 f"(100 = follows it fully, 0 = ignores it); any description vs none: "
                 f"{(pu('keeps') + pu('takes')) / 2 - pu('none'):+.0f}")

    # outcome, paired per start and partner
    lines.append("\nTeam score, paired per start (mean +- SE over starts)")
    cell = {(r["start"]["id"], r["partner"], r["description"]): r for r in res}

    def paired(b, d1, d2):
        xs = [cell[(sid, b, d1)]["score"] - cell[(sid, b, d2)]["score"] for sid in {k[0] for k in cell}
              if (sid, b, d1) in cell and (sid, b, d2) in cell]
        if len(xs) < 2:
            return "-"
        return f"{statistics.mean(xs):+6.1f} +- {statistics.stdev(xs) / len(xs) ** 0.5:4.1f} (n={len(xs)})"
    for b in BEHAVIOURS:
        lines.append(f"  partner {b}: correct - none {paired(b, 'correct', 'none')}; correct - swapped "
                     f"{paired(b, 'correct', 'swapped')}")

    # knowing vs finishing: among right first choices, was the chosen order finished in time by the ego?
    lines.append("\nAfter a right first choice (all descriptions): did the ego finish what it chose?")
    right = [r for r in res if r["right"]]
    fin = []
    for r in right:
        if r["first"]["order"] == "urgent":
            fin.append(r["urgent_on_time"] and r["urgent_by"] == "ego")
        else:
            fin.append(r["other_done"] and r["other_by"] == "ego")
    lines.append(f"  chosen order delivered by the ego: {_rate(fin)}; with a recipe error after the choice: "
                 f"{_rate(bool(r['recipe_errors']) for r in right)}; wrong delivery: "
                 f"{_rate(r['ego_wrong_deliveries'] > 0 for r in right)}")
    lines.append("\nRecipe errors after the first choice, per episode, by kind and by the order the action belonged to")
    lines.append(f"{'description':<12}{'partner':<8}{'load':>7}{'start':>7}{'urgent':>8}{'other':>7}{'neither':>9}")
    for d in DESCRIPTIONS:
        for b in BEHAVIOURS:
            rs = by.get((d, b), [])
            if not rs:
                continue
            errs = [e for r in rs for e in r["recipe_errors"]]
            n = len(rs)
            cnt = lambda f: sum(1 for e in errs if f(e)) / n
            lines.append(f"{d:<12}{b:<8}{cnt(lambda e: e['action'].startswith('put_obj')):>7.2f}"
                         f"{cnt(lambda e: not e['action'].startswith('put_obj')):>7.2f}"
                         f"{cnt(lambda e: e['order'] == 'urgent'):>8.2f}{cnt(lambda e: e['order'] == 'other'):>7.2f}"
                         f"{cnt(lambda e: e['order'] is None):>9.2f}")
    # verdict
    lines.append(f"\nVerdict (description effect {effect:+.0f} points; chosen order finished {_rate(fin)}):")
    if effect < 50:
        lines.append("  the ego follows the description only partly -> benchmark a stronger ego on these starts "
                     "(same config with another agent backend, new --out)")
    if fin and 100 * sum(fin) / len(fin) < 70:
        lines.append("  after a right choice the ego often does not finish the chosen order -> cooking competence")
    if effect >= 50 and (not fin or 100 * sum(fin) / len(fin) >= 70):
        lines.append("  the ego uses the description and finishes the dish: knowing the partner pays here")
    return "\n".join(lines)


def dishes_table(folder) -> str:
    """Per dish in a runner folder: orders, ego / partner deliveries, expiries, ego wrong deliveries."""
    acc: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {"orders": 0, "ego": 0, "partner": 0,
                                                                      "expired": 0, "ego_wrong": 0, "steps": []})
    files = sorted(Path(folder).glob("runs/*/*/seed*/*/ep*/summary.json")) + \
        sorted(Path(folder).glob("runs/*/*/seed*/*/eval*/summary.json"))
    for f in files:
        if f.parents[4].name.startswith("_"):
            continue
        s = json.loads(f.read_text())
        ego = s.get("ego_index", 0)
        kind = "held-out" if f.parent.name.startswith("eval") else "interaction"
        for o in s.get("orders", []):
            a = acc[(kind, o["name"])]
            if o.get("completed_t") is not None:
                a["orders"] += 1
                a["ego" if o.get("delivered_by") == ego else "partner"] += 1
                a["steps"].append(o.get("duration"))
            elif o.get("expired_t") is not None:
                a["orders"] += 1
                a["expired"] += 1
        for d in s.get("deliveries", []):
            if d["agent"] == ego and not d["correct"]:
                acc[(kind, d["item"])]["ego_wrong"] += 1
    lines = [f"per dish in {folder} (closed orders only)",
             f"{'episodes':<13}{'dish':<34}{'orders':>7}{'by ego':>8}{'by partner':>11}{'expired':>9}{'ego wrong del':>15}{'median steps':>14}"]
    for (kind, d), a in sorted(acc.items(), key=lambda kv: (kv[0][0] != "interaction", -(kv[1]["ego"] / max(1, kv[1]["orders"])))):
        steps = [x for x in a["steps"] if x is not None]
        lines.append(f"{kind:<13}{d:<34}{a['orders']:>7}{a['ego']:>8}{a['partner']:>11}{a['expired']:>9}{a['ego_wrong']:>15}"
                     f"{(statistics.median(steps) if steps else float('nan')):>14.0f}")
    lines.append("ego wrong del counts the wrongly delivered item, not the order it was meant for.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--stage", choices=("dishes", "screen", "llm", "report"), required=True)
    ap.add_argument("--out")
    ap.add_argument("--from-folder", help="--stage dishes: a finished coop.days folder")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-starts", type=int, help="--stage llm: only the first N kept starts")
    ap.add_argument("--pairs", nargs="+", help="override the config's pairs: urgent:other ...")
    ap.add_argument("--urgent-limits", nargs="+", type=int, help="override the config's urgent_limits")
    ap.add_argument("--horizon", type=int)
    args = ap.parse_args(argv)
    if args.stage == "dishes":
        if not args.from_folder:
            ap.error("--stage dishes needs --from-folder")
        print(dishes_table(args.from_folder))
        return
    cfg = load(args.config)
    if args.pairs:
        cfg["pairs"] = [p.split(":") for p in args.pairs]
    if args.urgent_limits:
        cfg["urgent_limits"] = args.urgent_limits
    if args.horizon:
        cfg["horizon"] = args.horizon
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    for u, o in cfg["pairs"]:
        env, st = make_env(cfg, {"urgent": u, "other": o, "limit": cfg["urgent_limits"][0], "slot": 0, "seed": 0})
        kit = Kitchen(env.mdp, 0, st)
        shared = (tree(kit, u) & tree(kit, o)) - {"dish"}
        if shared:
            raise SystemExit(f"pair {u}:{o} shares {sorted(shared)}: pick dishes with disjoint ingredients "
                             "so the first action shows which order the ego chose")
    if args.stage == "screen":
        screen(cfg, out)
    elif args.stage == "llm":
        llm_stage(cfg, out, args.workers, args.max_starts)
    text = report(cfg, out)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
