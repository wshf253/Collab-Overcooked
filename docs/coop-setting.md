# Coop research setting (`src/coop`)

This setting uses the Collab-Overcooked environment (utensils, multi-step
recipes, layouts). Its goal is to learn about a partner by watching it act,
then use that knowledge to work together well. Both agents will have the same
abilities, with no fixed chef/assistant roles. Episodes will be grouped into
days. It is built in steps; this page lists what exists so far.
The original benchmark (`src/main.py`, `evaluation.py`) does not use this package.

## Continuous orders (`coop.env.ContinuousOrderEnv`)

```python
from coop.env import ContinuousOrderEnv

env = ContinuousOrderEnv(
    menu=["boiled_egg", "baked_carrot_soup"],   # or {"boiled_egg": 2, "baked_carrot_soup": 1}
    layout="new_env", horizon=400, wrong_delivery_penalty=20)
state = env.reset(seed=0)
state, reward, done, info = env.step(joint_action, parm)   # same arguments as OvercookedEnv.step
env.current_order, env.upcoming_orders(3), env.summary()
```

- **Seeded sequence.** At `reset(seed)` the episode's whole order sequence is
  drawn from the menu with a private RNG. The same seed gives the same orders
  under every condition (paired design). `no_repeat=True` forbids the same
  order twice in a row.
- **One open order at a time.** A correct delivery earns the layout's
  `delivery_reward` (20) and opens the next order in the sequence. The episode
  ends at `horizon`.
- **Wrong deliveries** cost `wrong_delivery_penalty` points. This covers a
  wrong item, or a dish served plated when it should not be (and the reverse).
  The item is consumed and the order stays open.
- **Records.** `info["deliveries"]` lists each step's deliveries: who delivered
  what, against which order, and whether it was correct. `env.summary()` gives
  the score, orders completed, wrong deliveries, steps per order, and per-order
  records (issued, completed, delivered by, wrong attempts). After every step
  the wrapper checks its bookkeeping against the MDP's reward and order list.

- **Safety filter.** Interactions the legacy MDP cannot handle become "stay"
  and are listed in `info["rejected"]` (e.g. adding an ingredient to a utensil
  that is already running, which crashes the MDP). The rules are the same as in
  the human-play server.

The penalty is an optional `wrong_delivery_penalty` attribute of
`OvercookedGridworld`. Its default is 0, so the original benchmark keeps its
behaviour: a wrong delivery gives 0 points.

## Simulated human (`coop.partner.SimulatedHuman`)

A scripted partner that works on the open order through jobs. Each job belongs
to one stage:

| stage | jobs |
|---|---|
| prep | bring raw ingredients from the dispenser; load and run the chopping board or blender |
| cook | load and run the pot or oven; move products between utensils |
| serve | fetch dishes, plate food, deliver |

Behaviour is set by `Traits`:

| trait | parameters | effect |
|---|---|---|
| preference | `preference`, `reluctance` | picks jobs of its favourite stage first; waits up to `reluctance` steps for one before doing other work |
| speed | `step_period`, `decision_delay` | acts on every n-th step; pauses before each new job |
| reliability | `slip_rate` | a raw-ingredient job may take a wrong ingredient into the utensil; starting the spoiled batch makes the env discard it |
| commitment | `abandon_rate`, `abandon_cooldown` | may drop its current job, leaving any held item on a counter, and won't take that job again for a while |

Starting personas (`coop.partner.PERSONAS`), each changing one trait from
`steady`: `prep_first`, `serve_first`, `slow`, `sloppy`, `flaky`.

The partner logs its true job and status every step (`trace`), and slips and
abandoned jobs as events (`events`). This is ground truth for evaluation, never
input for the agent being tested. The partner is reproducible for a given
(episode seed, player slot, persona).

### Screen: are the personas different?

```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.screen --out outputs/coop_screen.json
```

The screen runs each persona against a neutral `steady` partner (no LLM), in
both player slots, on the same seeded orders.

On `new_env`, `slow`, `flaky` and the two preference types each lower the team
score. `sloppy` slips only in slot 1, because only that side takes raw
ingredients from the dispenser. On this map, preference shows up as **waiting**
more than as a different job mix: in a split kitchen with one open order, a
player rarely has jobs from two stages to choose between. Maps where both
players can do every stage will separate the preference types better.

## Days (`coop.days`)

A **run** is one partner persona and one run seed, played for `days` days of
`episodes_per_day` continuous-order episodes. The partner stays the same for the
whole run, and the ego's memory carries over between episodes and days. A
config lists personas and run seeds; each pair is a separate run with its own
memory.

```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.days configs/coop/days_smoke.yaml --out outputs/days_smoke
```

- **Configs** live in `configs/coop/`: `days_smoke.yaml` (2 days × 2 episodes)
  and `days_default.yaml` (6 personas × 3 run seeds × 5 days × 3 episodes).
  `custom_personas` defines new partner types from any `Traits` fields.
  Command-line overrides: `--personas`, `--run-seeds`, `--days`,
  `--episodes-per-day`, `--horizon`.
- **Paired orders.** The orders for (run seed, day, episode) are the same for
  every persona, so partners are compared on identical order sequences.
- **Ego agents** implement `coop.agents.Agent`: `begin_episode`, `act`,
  `end_episode(result)`, `end_day`, and `get_memory`/`set_memory`. Memory must
  be a JSON-serialisable dict and is the only thing kept between episodes.
  `result` holds what happened in the kitchen (orders, deliveries, score),
  never the partner's persona, traits or true jobs. The built-in
  `scripted` ego (a simulated human, no memory) is the no-learning reference:
  its day-to-day changes come only from the orders.
- **Resume.** Rerun the same command with the same `--out` and finished episodes
  are loaded, not replayed. A different config in the same `--out` is refused.
- **Parallel runs:** `--workers N` runs N runs in separate processes.

Output: `OUT/results.json` (every episode, plus a persona × day table), and per
episode `OUT/runs/<persona>/seed<k>/dayDD/epEE/`:

| file | contents |
|---|---|
| `summary.json` | orders, deliveries, score |
| `steps.jsonl` | per step: actions, completed actions, positions, held items, deliveries, rejected actions |
| `partner_truth.json` | partner persona, traits, true job per step, slips, abandoned jobs (evaluation only) |
| `agent_log.json` | ego-side record |
| `memory_after.json` | ego memory after the episode (after the day, for its last episode) |

Size: about 0.23 MB per 400-step episode. The default config (270 episodes)
writes about 63 MB and takes a few seconds with the scripted ego.

## Run on the lab server

First time only (assumes the repository lives in `~/work/collab-overcooked`):

```bash
git clone https://github.com/wshf253/collab-overcooked ~/work/collab-overcooked
```

Update to this branch:

```bash
cd ~/work/collab-overcooked
```
```bash
git fetch origin
```
```bash
git switch claude/lucid-babbage-96q0zi
```
```bash
git pull --ff-only origin claude/lucid-babbage-96q0zi
```

Python environment (once, and again when `requirements-coop.txt` changes):

```bash
uv venv -p 3.11 .venv
```
```bash
uv pip install -p .venv -r requirements-coop.txt
```

Tests, the persona screen and multi-day runs (CPU only):

```bash
lab-run --new --gpus 0 -- .venv/bin/python -m unittest discover -s tests -v
```
```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.screen --out outputs/coop_screen.json
```
```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.days configs/coop/days_default.yaml --out outputs/days_default --workers 4
```

## Not built yet

One shared prompt and action list for both agents, no messages (and no view of
the teammate's plan), the partner-observation layer, and the LLM agent with
memory across days.

## Other environment changes

- A utensil start now checks only the batch in that utensil. Before, the env
  also checked batches sharing its row or column, and could delete one of
  them as a "wrong recipe". `new_env` has no two utensils in a shared row or
  column, so its results do not change.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tested with Python 3.11 and the pinned `requirements-coop.txt` (`numpy<2`: the
legacy env uses `np.Inf`; `pyyaml` for configs).
