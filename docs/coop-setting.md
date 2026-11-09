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

## Layouts

| layout | map | what it tests |
|---|---|---|
| `new_env` | Collab-Overcooked's original 5×5 kitchen, split by a column of counters. Left side: ingredient dispenser, chopping board, dish dispenser, blender. Right side: pot, oven, serving location. Items pass over the 3 shared counters. | Fixed roles: the right player cooks and serves, the left player supplies. |
| `open_env` | 7×5 open kitchen with one of each station around the walls; both players can reach everything. | Roles are chosen, so who does which stage (preference) is visible. Players can also get in each other's way. |
| `open2_env` | 9×5 open kitchen with two of every utensil (pot, oven, chopping board, blender), one ingredient dispenser, dish dispenser and serving location. | The same, but both players can work in parallel instead of waiting for a shared utensil. |

```
new_env        open_env       open2_env
XXXPX          XXXPXXX        XXPXXXPXX
I X1X          I     O        I       O
C2X X          C 1 2 S        C 1   2 S
D X O          D     B        D       O
XBXSX          XXXXXXX        XXBXCXBXX
```

The open layouts are `new_env.layout` with a different grid; a test keeps the
recipes identical. Choose the layout in a config (`layout:`) or with
`--layout` on `coop.days` and `coop.screen`.

Movement rules that make open kitchens work (partner and LLM executor alike):
an idle player steps off cells in front of a station; a player whose way is
cut off steps aside instead of freezing; after a blocked move, a player gives
way with probability ½ (seeded). On `new_env` these change results by at most
a few points, because the two sides never meet.

Screen on `open_env` (8 seeds, partner vs `steady`): steady 295; `prep_first`
does 92% prep, `serve_first` 83% serving (steady about 39% and 35%); `slow`
is active 8% of the time (steady 66%); `sloppy` slips 1.5–3.5 times and
`flaky` abandons 9–12.5 jobs per episode.

On `open2_env`: steady 255 (longer walks, and with one open order at a time the
second set of utensils helps only within an order); `prep_first` 67–70% prep,
`serve_first` 66% serving, `slow` active 12–14%, `sloppy` 1.1–2.5 slips,
`flaky` 9–11.5 abandoned jobs. Orders keep being completed to the end of every
episode (no stalls).

## Days (`coop.days`)

A **run** is one condition (ego agent settings), one partner persona and one
run seed, played for `days` days of `episodes_per_day` continuous-order
episodes. The partner stays the same for the whole run, and the ego's memory
carries over between episodes and days. Every (condition, persona, run seed)
is a separate run with its own memory.

```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.days configs/coop/days_smoke.yaml --out outputs/days_smoke
```

- **Configs** live in `configs/coop/`:

  | config | what |
  |---|---|
  | `days_smoke.yaml` | scripted ego, 2 days × 2 episodes |
  | `days_default.yaml` | scripted ego, 6 personas × 3 run seeds × 5 days × 3 episodes |
  | `days_llm_mock.yaml` | LLM ego with the offline mock backend (pipeline check, meaningless scores) |
  | `days_llm.yaml` | LLM ego (Qwen via vLLM), 4 conditions × 6 personas × 2 run seeds × 5 days × 2 episodes |

  `agent` is the base ego. `conditions` maps names to overrides of it (nested
  dicts merge one level), e.g. `tom: {tom: true}`. `custom_personas` defines
  new partner types from any `Traits` fields. Command-line overrides:
  `--conditions`, `--personas`, `--run-seeds`, `--days`, `--episodes-per-day`,
  `--horizon`.
- **Paired orders.** The orders for (run seed, day, episode) are the same for
  every persona and condition.
- **Reference ego.** `reference_agent` (default: the scripted no-memory player)
  plays every (persona, run seed) on the same episodes, under the condition
  `_reference`. The table's `vs_reference` is the mean of (score − reference
  score) over identical episodes. This removes how hard a day's orders were, so
  a change in `vs_reference` across days means the ego changed, not the orders.
  `reference_agent: null` turns it off.
- **Ego agents** implement `coop.agents.Agent`: `begin_episode`, `act`,
  `observe` (reward and public events after each step), `end_episode(result)`,
  `end_day`, and `get_memory`/`set_memory`. Memory must be a JSON-serialisable
  dict and is the only thing kept between episodes. `result` holds what
  happened in the kitchen (orders, deliveries, score), never the partner's
  persona, traits or true jobs.
- **Resume.** Rerun the same command with the same `--out` and finished episodes
  are loaded, not replayed. A different config in the same `--out` is refused.
- **Parallel runs:** `--workers N` runs N runs in separate processes. With the
  LLM ego, match it to vLLM's `MAX_NUM_SEQS`.

Output: `OUT/results.json` (every episode, plus a condition × persona × day
table), and per episode `OUT/runs/<condition>/<persona>/seed<k>/dayDD/epEE/`:

| file | contents |
|---|---|
| `summary.json` | orders, deliveries, score; probe score and LLM usage for the LLM ego |
| `steps.jsonl` | per step: actions, completed actions, positions, held items, deliveries, rejected actions |
| `partner_truth.json` | partner persona, traits, true profile, true job per step, slips, abandoned jobs (evaluation only) |
| `agent_log.json` | ego-side record (LLM ego: system prompt, every call, decisions, probe answer) |
| `memory_after.json` | ego memory after the episode (after the day, for its last episode) |

## LLM ego (`coop.llm_agent.LLMAgent`)

The agent the research is about. Its partner is the simulated human. Both have
the same abilities and there are no messages.

- **What it sees** (every decision): the map with what each player can reach,
  the recipes of the menu, the open order, utensils, counters, both players'
  positions and held items, the partner's completed actions with time steps,
  deliveries and discarded batches, its own recent actions and their results,
  and a `VALID_ACTIONS` list of actions that can start now. It never sees the
  partner's persona, traits or true jobs, or the partner's planned actions. A
  test scans every prompt for them.
- **Rules about reaching stations** are part of the system prompt for every
  condition: you cannot use a station you cannot reach; something only your
  partner can reach has to be put on a counter by them (you cannot ask);
  hand items over on counters both can reach; food served on a dish cannot be
  picked up by hand.
- **Actions** use Collab-Overcooked's names, with no role split:
  `pickup(item, place)`, `put_obj_in_utensil(u)`, `cut/cook/bake/stir(u)`,
  `fill_dish_with_food(u)`, `deliver_soup()`, `place_obj_on_counter()`,
  `wait(n)`. An executor (`coop.executor`) walks to the station and interacts.
  The LLM is asked again when the action finishes or fails (with the reason),
  or when a `wait` ends. A `wait` ends early when the partner completes an
  action. An impossible action gets one retry with the reason. After that the
  agent waits one step (counted as a fallback).
- **Conditions** (agent keys):

  | key | effect |
  |---|---|
  | `tom` | before each action, writes what the partner is doing and tends to do (`"partner"` field). Only the reply format differs from `tom: false`. |
  | `memory` | after each episode, reflects on the partner. The profile and notes (last `max_memory_notes`) appear in every later episode's system prompt. |
  | `probe` | after each episode, rates the partner on four traits: `prefers` (prep/cook/serve/no clear preference), `speed` (normal/slow), `reliability` (reliable/makes mistakes), `commitment` (finishes/abandons tasks). Scored against the persona's true profile (`coop.profile`). Without `memory`, the answer is logged but never shown to the agent again. |

- **Backends** (`coop.llm`, from ICRL4AHT): `openai_compat` (vLLM or a hosted
  API; `*_env` keys read URL, model, key and `chat_template_kwargs` from
  environment variables) and `mock` (offline: picks from `VALID_ACTIONS`).
  `coop.check` verifies that thinking is off for a served model.
- **Probe caveat on `new_env`:** the ego is on the right, so the partner works on
  the left, which has the dispenser, chopping board, blender and dishes. The
  true `prefers` label comes from the persona's traits, but a `steady` partner
  still does mostly prep there. So `prefers` answers are hard to get right on
  this map: compare the four fields separately (`probe_field_accuracy` in
  `results.json`). Maps where both players can do every stage will fix this.

## Analysis (`coop.analyze`)

```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.analyze outputs/days_llm --out outputs/days_llm/analysis.json
```

For an LLM run, prints:
- score minus the reference by day and condition (mean ± SE over episodes),
  each condition minus `direct` on identical episodes, and first day vs last
  day per persona;
- probe accuracy per field, with the most common answer and the truth;
- how decisions were spent: wait share, chosen wait length, waits that ran
  their full length vs ended by the partner, and full-length waits that ended
  with finished food sitting in a utensil the ego can reach;
- LLM health: calls, invalid replies, fallbacks, thinking/truncated replies.

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
```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.days configs/coop/days_llm_mock.yaml --out outputs/days_llm_mock --workers 4
```

LLM ego with Qwen3.8-27B (1 GPU). `scripts/run_qwen38_days.sh` serves the
model with vLLM inside the job, checks that thinking is off, runs `coop.days`,
and stops the server. It uses the same variables as ICRL4AHT's
`run_qwen38_coop.sh` (`LLM_MODEL_PATH`, `LLM_SERVE_ENV`, `MAX_NUM_SEQS`,
`GPU_MEM`, ...; with `ICRL_LLM_BASE_URL` set, it uses that server instead).

Small first run (one condition, one persona, 2 days × 1 short episode):

```bash
lab-run --new --gpus 1 -- bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --out outputs/days_llm_smoke --conditions tom_memory --personas slow --run-seeds 0 --days 2 --episodes-per-day 1 --horizon 150
```

The three full designs below can run at the same time, one GPU each. When more than one GPU job runs at once, start each with `lab-run --new --spare --gpus 1`.

Full design, split map (`new_env`):

```bash
lab-run --new --spare --gpus 1 -- bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --out outputs/days_llm_v2 --workers 16
```

Full design, open kitchen (`open_env`):

```bash
lab-run --new --spare --gpus 1 -- bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --layout open_env --out outputs/days_llm_open --workers 16
```

Full design, open kitchen with two of every utensil (`open2_env`):

```bash
lab-run --new --spare --gpus 1 -- bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --layout open2_env --out outputs/days_llm_open2 --workers 16
```

Partner screens on the open kitchens:

```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.screen --layout open_env --out outputs/coop_screen_open.json
```
```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.screen --layout open2_env --out outputs/coop_screen_open2.json
```

## Not built yet

Partial observability (view radius), the order-display tile, and an
LLM-simulated human partner.

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
