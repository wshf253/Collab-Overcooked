"""Single-decision Qwen experiment on expectations of task ownership."""
import argparse
import contextlib
import hashlib
import io
import json
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path

from overcooked_ai_py.mdp.overcooked_mdp import ObjectState

from .decision import CommitEgo, load
from .env import ContinuousOrderEnv
from .llm import build_backend, parse_json

PROFILES = {
    "owner": "This teammate usually leaves a prepared item for the cook who prepared it.",
    "relay": "This teammate usually picks up a prepared item left by the other cook and finishes that dish.",
}
CONDITIONS = ("none", "correct", "swapped", "observed")
CHOICES = ("prepared", "second")


class OrderCook(CommitEgo):
    """Same cooking policy for both players, changing only initial order priority."""

    def __init__(self, first, alternative, menu):
        super().__init__(first, menu)
        self.priority = (first, alternative)

    def _pick_commit(self, ctx, newest):
        self.commit = next((dish for dish in self.priority if dish in ctx["open"]), None)
        return self.commit


def rollout(cfg, scene, partner, choice):
    prepared, second = scene["pair"]
    menu = [prepared, second, cfg["tail"]]
    env = ContinuousOrderEnv(menu, layout=cfg["layout"], horizon=cfg["horizon"],
                             open_orders=2, order_mode="controlled")
    with contextlib.redirect_stdout(io.StringIO()):
        state = env.reset(0, sequence=scene["tickets"] + [cfg["tail"]] * 20)
    pos = tuple(scene["counter"])
    item = prepared.removeprefix("boiled_")
    state.objects[pos] = ObjectState(item, "ingredient", pos)
    partner_target = prepared if partner == "relay" else second
    ego_target = prepared if choice == "prepared" else second
    cooks = []
    for index in range(2):
        target = ego_target if index == scene["slot"] else partner_target
        cook = OrderCook(target, second if target == prepared else prepared, menu)
        cook.reset(env.mdp, index, 0, view_radius=None)
        cooks.append(cook)
    with contextlib.redirect_stdout(io.StringIO()):
        while not env.done:
            moves = [cook.act(env.state, env.t) for cook in cooks]
            env.step([m[0] for m in moves], [m[1] for m in moves])
            if all(d not in env.state.order_list for d in scene["pair"]):
                break
    records = env.summary()["orders"][:2]
    times = [o.get("completed_t") for o in records]
    return {"completed": sum(t is not None for t in times), "steps": env.t,
            "completion_times": times, "partner_first_job":
            next(iter(cooks[1-scene["slot"]].decision_log), None)}


def screen(cfg):
    rows = []
    for pair, slot, counter, reverse in product(cfg["pairs"], (0, 1), cfg["counters"], (False, True)):
        scene = dict(pair=pair, slot=slot, counter=counter, tickets=pair[::-1] if reverse else pair)
        trials = {p: {c: rollout(cfg, scene, p, c) for c in CHOICES} for p in PROFILES}
        winners = {}
        for p, choices in trials.items():
            ranks = {c: (r["completed"], -r["steps"]) for c, r in choices.items()}
            if len(set(ranks.values())) == 2:
                winners[p] = max(ranks, key=ranks.get)
        rows.append(dict(scene=scene, trials=trials, winners=winners,
                         kept=winners == {"owner": "prepared", "relay": "second"}))
    return rows


def messages(scene, partner, condition, question, flipped, style, history):
    prepared, second = scene["pair"]
    other = "relay" if partner == "owner" else "owner"
    info = "There is no history with this teammate."
    if condition in ("correct", "swapped"):
        info = PROFILES[partner if condition == "correct" else other]
    if condition == "observed":
        demo = history["scene"]
        job = history["trials"][partner]["prepared"]["partner_first_job"]["job"]
        info = (f"In an earlier round the tickets were {demo['tickets'][0]} and {demo['tickets'][1]}. "
                f"You left slices for {demo['pair'][0]} on the counter. "
                f"Your teammate's first job involved bringing {job['item']} for {job['dest']}.")
    dishes = list(scene["pair"])
    if flipped:
        dishes.reverse()
    positions = [(2, 2), (6, 2)]
    slot = scene["slot"]
    facts = [
        f"Tickets, oldest first: {', '.join(scene['tickets'])}. No deadlines.",
        f"You prepared slices for {prepared} and left them on the shared counter at {scene['counter']}. "
        f"Work on {second} has not started. Both cooks saw you put down the slices.",
        f"You are at {positions[slot]}. Your teammate is at {positions[1 - slot]}.",
        info,
    ]
    if style == "structured":
        facts = [f"{label}\n{fact}" for label, fact in
                 zip(("Orders", "Recent actions", "Positions", "Partner information"), facts)]
    prompt = "\n".join(facts + [question, f"Options: A = {dishes[0]}, B = {dishes[1]}.",
                                 'Return JSON {"choice":"A"} or {"choice":"B"}.'])
    return [{"role": "user", "content": prompt}], dishes


def dump(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def evaluate(cfg, rows, out):
    backend = build_backend(cfg["agent"]["backend"])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if not row["kept"]:
            continue
        history = next(r for r in rows if r["scene"]["pair"] != row["scene"]["pair"]
                       and r["scene"]["slot"] == row["scene"]["slot"])
        for p, cond, flip, style in product(PROFILES, CONDITIONS, (False, True), ("compact", "structured")):
            for task, question in (("predict", "Which ticket will your teammate work on first?"),
                                   ("act", "Which ticket will you work on first to finish both dishes soonest?")):
                msgs, dishes = messages(row["scene"], p, cond, question, flip, style, history)
                digest = hashlib.sha256(json.dumps([msgs, backend.model, backend.extra_body, backend.params],
                                                   sort_keys=True).encode()).hexdigest()
                path = out / f"reply-{digest}.json"
                if not path.exists():
                    result = backend.chat(msgs, max_tokens=100, seed=0)
                    if result.error:
                        raise RuntimeError(result.error)
                    dump(path, dict(messages=msgs, reply=result.text, model=backend.model))
                answer = (parse_json(json.loads(path.read_text())["reply"]) or {}).get("choice")
                selected = dishes[("A","B").index(answer)] if answer in ("A","B") else None
                pair = row["scene"]["pair"]
                target = (pair[0 if p == "relay" else 1] if task == "predict"
                          else pair[CHOICES.index(row["winners"][p])])
                groups[style, cond, task].append(dict(scene=i, pair=pair, partner=p, flipped=flip,
                                               selected=selected, correct=selected == target, reply=path.name))
    report = {}
    for (style, cond, task), records in groups.items():
        report[f"{style}/{cond}/{task}"] = dict(n=len(records), accuracy=statistics.mean(r["correct"] for r in records),
                                       invalid=sum(r["selected"] is None for r in records), records=records)
    paired = {}
    for style in ("compact", "structured"):
        for cond in CONDITIONS:
            action = groups[style, cond, "act"]
            prediction = groups[style, cond, "predict"]
            both = [a["correct"] and b["correct"] for a, b in zip(action, prediction)]
            correct_predictions = sum(r["correct"] for r in prediction)
            subgroups = defaultdict(dict)
            for r in action:
                subgroups[r["scene"], r["flipped"]][r["partner"]] = r["correct"]
            paired[f"{style}/{cond}"] = {
                "choices_correct_for_both_partners": statistics.mean(all(g.values()) for g in subgroups.values()),
                "prediction_and_action_correct": statistics.mean(both),
                "correct_predictions": correct_predictions,
                "action_accuracy_given_correct_prediction":
                    sum(both) / correct_predictions if correct_predictions else None,
            }
    dump(out / "paired.json", paired)
    dump(out / "report.json", report)
    for k, v in report.items():
        print(k, v["n"], round(v["accuracy"], 3), "invalid", v["invalid"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--stage", choices=("screen", "qwen"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = load(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "screen.json"
    identity = hashlib.sha256(Path(args.config).read_bytes() + Path(__file__).read_bytes()).hexdigest()
    if args.stage == "screen":
        rows = screen(cfg)
        dump(path, dict(version=identity, rows=rows))
        print(f"{sum(r['kept'] for r in rows)}/{len(rows)} switching scenes")
    else:
        data = json.loads(path.read_text())
        if data["version"] != identity or not any(r["kept"] for r in data["rows"]):
            ap.error("Run the CPU screen with this config and check for switching scenes.")
        evaluate(cfg, data["rows"], args.out)


if __name__ == "__main__":
    main()
