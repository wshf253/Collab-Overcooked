"""Multi-day runs: the same partner for several days, the ego's memory carried over.

A *run* is one partner persona and one run seed, played for ``days`` days of
``episodes_per_day`` continuous-order episodes each. The ego's memory carries
over from episode to episode and from day to day (see ``coop.agents``). A
config lists several personas and run seeds; each (persona, run seed) is its
own run with its own memory.

Orders for (run seed, day, episode) are the same for every persona, so
different partners are compared on identical order sequences.

    PYTHONPATH=src:lib/overcooked_ai python -m coop.days configs/coop/days_smoke.yaml --out outputs/days_smoke

Output layout::

    OUT/config.json                          resolved config (a resume must match it)
    OUT/results.json                         every episode's metrics + persona x day table
    OUT/runs/<persona>/seed<k>/
        run.json                             this run's episodes
        error.log                            traceback, if the run failed
        day01/ep01/
            summary.json                     orders, deliveries, score (what happened)
            steps.jsonl                      per-step actions, positions, holdings, events
            partner_truth.json               partner persona, traits, true jobs, slips/abandons
            agent_log.json                   ego-side record
            memory_after.json                ego memory after this episode (and day)

``partner_truth.json`` is for evaluation only; nothing in it is passed to the ego.
A run resumes from the last episode with both ``summary.json`` and
``memory_after.json``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from .agents import Agent, make_agent
from .env import ContinuousOrderEnv
from .orders import normalize_menu
from .partner import PERSONAS, SimulatedHuman, Traits

CONFIG_KEYS = {
    "name", "layout", "menu", "horizon", "wrong_delivery_penalty", "no_repeat", "days",
    "episodes_per_day", "ego_index", "personas", "custom_personas", "run_seeds", "agent",
}


@dataclass
class DaysConfig:
    name: str = "days"
    layout: str = "new_env"
    menu: Union[List[str], Dict[str, float]] = field(default_factory=lambda: [
        "boiled_egg", "boiled_mushroom", "boiled_corn_slices", "baked_carrot_soup",
        "sliced_eggplant_and_chickpea_stew"])
    horizon: int = 400
    wrong_delivery_penalty: float = 20
    no_repeat: bool = False
    days: int = 5
    episodes_per_day: int = 3
    ego_index: int = 0                  # player slot of the ego; the partner takes the other
    personas: List[str] = field(default_factory=lambda: ["steady"])
    custom_personas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_seeds: List[int] = field(default_factory=lambda: [0])
    agent: Dict[str, Any] = field(default_factory=lambda: {"type": "scripted", "persona": "steady"})

    def __post_init__(self):
        if self.days < 1 or self.episodes_per_day < 1 or self.horizon < 1:
            raise ValueError("days, episodes_per_day and horizon must be >= 1")
        if self.ego_index not in (0, 1):
            raise ValueError("ego_index must be 0 or 1")
        if not self.personas or len(set(self.personas)) != len(self.personas):
            raise ValueError(f"personas must be a non-empty list of distinct names: {self.personas}")
        if not self.run_seeds or len(set(self.run_seeds)) != len(self.run_seeds):
            raise ValueError(f"run_seeds must be a non-empty list of distinct ints: {self.run_seeds}")
        clash = set(self.custom_personas) & set(PERSONAS)
        if clash:
            raise ValueError(f"custom_personas reuse built-in names: {sorted(clash)}")
        for p in self.personas:
            persona_traits(p, self.custom_personas)     # raises on unknown names or bad traits
        menu = normalize_menu(self.menu)
        with contextlib.redirect_stdout(io.StringIO()):
            mdp = OvercookedGridworld.from_layout_name(self.layout)
        unknown = sorted(set(menu) - set(mdp.Recipe.recipe_list))
        if unknown:
            raise ValueError(f"Orders not producible in layout '{self.layout}': {unknown}")
        if "type" not in self.agent:
            raise ValueError("agent needs a 'type'")


def persona_traits(name: str, custom: Dict[str, Dict[str, Any]]) -> Traits:
    if name in PERSONAS:
        return PERSONAS[name]
    if name in custom:
        spec = dict(custom[name])
        if "preference" in spec:
            spec["preference"] = tuple(spec["preference"])
        return Traits(**spec)
    raise ValueError(f"Unknown persona {name!r}; built-in: {sorted(PERSONAS)}; custom: {sorted(custom)}")


def load_config(path: Union[str, Path], **overrides) -> DaysConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(raw) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return DaysConfig(**raw)


def episode_seed(run_seed: int, day: int, episode: int) -> int:
    """Stable seed for (run seed, day, episode), the same for every persona."""
    digest = hashlib.sha256(f"coop-days:{run_seed}:{day}:{episode}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------- #
# One episode
# ---------------------------------------------------------------------- #
def _player_view(p) -> Dict[str, Any]:
    held = p.held_object
    return {"pos": list(p.position), "dir": list(p.orientation),
            "held": None if held is None else {"name": held.name, "category": held.catagory}}


def _jsonable_action(a):
    return a if isinstance(a, str) else list(a)


def play_episode(cfg: DaysConfig, agent: Agent, persona: str, run_seed: int, day: int,
                 episode: int, ep_dir: Path) -> Dict[str, Any]:
    seed = episode_seed(run_seed, day, episode)
    ego, other = cfg.ego_index, 1 - cfg.ego_index
    env = ContinuousOrderEnv(cfg.menu, layout=cfg.layout, horizon=cfg.horizon,
                             wrong_delivery_penalty=cfg.wrong_delivery_penalty, no_repeat=cfg.no_repeat)
    env.reset(seed)
    partner = SimulatedHuman(persona_traits(persona, cfg.custom_personas), persona)
    partner.reset(env.mdp, other, seed)
    context = {"day": day, "episode": episode, "episode_seed": seed, "layout": cfg.layout,
               "horizon": cfg.horizon, "menu": normalize_menu(cfg.menu), "index": ego,
               "partner_index": other, "wrong_delivery_penalty": cfg.wrong_delivery_penalty}
    agent.begin_episode(env.mdp, ego, context)
    ep_dir.mkdir(parents=True, exist_ok=True)
    with open(ep_dir / "steps.jsonl", "w") as steps:
        while not env.done:
            t, state = env.t, env.state
            moves = [None, None]
            moves[ego] = agent.act(state, t)
            moves[other] = partner.act(state, t)
            order = env.current_order
            state, reward, _, info = env.step([m[0] for m in moves], [m[1] for m in moves])
            steps.write(json.dumps({
                "t": t, "order": order,
                "actions": [_jsonable_action(m[0]) for m in moves], "params": [m[1] for m in moves],
                "completed": list(state.ml_actions), "reward": reward, "score": env.score,
                "players": [_player_view(p) for p in state.players],
                "deliveries": [asdict(d) for d in info["deliveries"]], "rejected": info["rejected"],
            }) + "\n")
    summary = dict(env.summary(), day=day, episode=episode, run_seed=run_seed, persona=persona,
                   ego_index=ego)
    # The ego learns from what happened in the kitchen, never from the partner's
    # persona, traits or true jobs.
    public = {k: v for k, v in summary.items() if k != "persona"}
    agent.end_episode(dict(public, context=context))
    (ep_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (ep_dir / "partner_truth.json").write_text(json.dumps({
        "persona": persona, "index": other, "summary": partner.summary(),
        "events": partner.events, "trace": partner.trace}))
    (ep_dir / "agent_log.json").write_text(json.dumps(agent.episode_log()))
    return summary


# ---------------------------------------------------------------------- #
# One run (persona x run seed)
# ---------------------------------------------------------------------- #
def run_dir_for(out: Path, persona: str, run_seed: int) -> Path:
    return out / "runs" / persona / f"seed{run_seed}"


def ep_dir_for(run_dir: Path, day: int, episode: int) -> Path:
    return run_dir / f"day{day:02d}" / f"ep{episode:02d}"


def _done(ep_dir: Path) -> bool:
    return (ep_dir / "summary.json").exists() and (ep_dir / "memory_after.json").exists()


def run_one(cfg_dict: Dict[str, Any], out: str, persona: str, run_seed: int, quiet: bool = True
            ) -> Dict[str, Any]:
    """Play (or resume) every episode of one run. Picklable entry point for workers."""
    cfg = DaysConfig(**cfg_dict)
    run_dir = run_dir_for(Path(out), persona, run_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    agent = make_agent(cfg.agent)
    memory: Dict[str, Any] = {}
    episodes: List[Dict[str, Any]] = []
    resumed = 0
    try:
        agent.set_memory(memory)
        for day in range(1, cfg.days + 1):
            day_results = []
            for ep in range(1, cfg.episodes_per_day + 1):
                ep_dir = ep_dir_for(run_dir, day, ep)
                if _done(ep_dir):
                    summary = json.loads((ep_dir / "summary.json").read_text())
                    agent.set_memory(json.loads((ep_dir / "memory_after.json").read_text()))
                    resumed += 1
                else:
                    if ep_dir.exists():
                        shutil.rmtree(ep_dir)        # partial episode from an interrupted run
                    sink = io.StringIO() if quiet else sys.stdout
                    with contextlib.redirect_stdout(sink):  # the legacy MDP prints every step
                        summary = play_episode(cfg, agent, persona, run_seed, day, ep, ep_dir)
                    if ep == cfg.episodes_per_day:
                        agent.end_day(day, day_results + [summary])
                    (ep_dir / "memory_after.json").write_text(json.dumps(agent.get_memory()))
                day_results.append(summary)
                episodes.append(_metrics(summary))
        status = "ok"
        (run_dir / "error.log").unlink(missing_ok=True)
    except Exception:
        status = "error"
        (run_dir / "error.log").write_text(traceback.format_exc())
    record = {"persona": persona, "run_seed": run_seed, "status": status,
              "episodes_resumed": resumed, "episodes": episodes}
    (run_dir / "run.json").write_text(json.dumps(record, indent=1))
    return record


def _metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("day", "episode", "score", "orders_completed", "wrong_deliveries",
            "rejected_interactions", "mean_order_steps", "seed")
    return {k: summary.get(k) for k in keys}


# ---------------------------------------------------------------------- #
# All runs of a config
# ---------------------------------------------------------------------- #
def run_all(cfg: DaysConfig, out: Union[str, Path], workers: int = 1, log=print) -> Dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != json.loads(json.dumps(cfg_dict)):
            raise ValueError(f"{cfg_path} holds a different config; use a new --out to start over")
    else:
        cfg_path.write_text(json.dumps(cfg_dict, indent=1))
    jobs = [(p, s) for p in cfg.personas for s in cfg.run_seeds]
    records = []
    started = time.time()
    if workers <= 1:
        for p, s in jobs:
            records.append(run_one(cfg_dict, str(out), p, s))
            log(_progress(records[-1], len(records), len(jobs), started))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, cfg_dict, str(out), p, s) for p, s in jobs]
            for f in as_completed(futures):
                records.append(f.result())
                log(_progress(records[-1], len(records), len(jobs), started))
    records.sort(key=lambda r: (cfg.personas.index(r["persona"]), r["run_seed"]))
    results = {"config": cfg_dict, "runs": records, "table": day_table(cfg, records)}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


def _progress(rec, done, total, started) -> str:
    return (f"[{done}/{total}] {rec['persona']} seed{rec['run_seed']}: {rec['status']}, "
            f"{len(rec['episodes'])} episodes ({rec['episodes_resumed']} resumed), "
            f"{time.time() - started:.0f}s")


def day_table(cfg: DaysConfig, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """persona -> day -> mean metrics over run seeds and the day's episodes."""
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for persona in cfg.personas:
        eps = [e for r in records if r["persona"] == persona and r["status"] == "ok" for e in r["episodes"]]
        rows = {}
        for day in range(1, cfg.days + 1):
            d = [e for e in eps if e["day"] == day]
            if d:
                rows[str(day)] = {
                    "episodes": len(d),
                    "score": statistics.fmean(e["score"] for e in d),
                    "orders": statistics.fmean(e["orders_completed"] for e in d),
                    "wrong": statistics.fmean(e["wrong_deliveries"] for e in d),
                }
        table[persona] = rows
    return table


def format_table(cfg: DaysConfig, table) -> str:
    lines = ["mean score per day (orders completed in brackets)",
             "persona       " + "".join(f"{'day ' + str(d):>16}" for d in range(1, cfg.days + 1))]
    for persona, rows in table.items():
        cells = []
        for d in range(1, cfg.days + 1):
            r = rows.get(str(d))
            cells.append(f"{r['score']:>8.1f} ({r['orders']:>4.1f})" if r else f"{'-':>16}")
        lines.append(f"{persona:<14}" + "".join(f"{c:>16}" for c in cells))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-day runs with a simulated human partner.")
    ap.add_argument("config", help="YAML or JSON config (see configs/coop/)")
    ap.add_argument("--out", required=True, help="output directory; rerun with the same one to resume")
    ap.add_argument("--workers", type=int, default=1, help="runs in parallel processes")
    ap.add_argument("--personas", nargs="+")
    ap.add_argument("--run-seeds", nargs="+", type=int)
    ap.add_argument("--days", type=int)
    ap.add_argument("--episodes-per-day", type=int)
    ap.add_argument("--horizon", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, personas=args.personas, run_seeds=args.run_seeds, days=args.days,
                      episodes_per_day=args.episodes_per_day, horizon=args.horizon)
    results = run_all(cfg, args.out, args.workers, log=lambda s: print(s, file=sys.stderr, flush=True))
    failed = [r for r in results["runs"] if r["status"] != "ok"]
    print(format_table(cfg, results["table"]))
    print(f"wrote {Path(args.out) / 'results.json'}")
    if failed:
        print(f"{len(failed)} run(s) failed; see runs/<persona>/seed<k>/error.log", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
