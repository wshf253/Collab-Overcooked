# Coop research setting (`src/coop`)

Current research status and results: [`RESEARCH_STATUS.md`](../RESEARCH_STATUS.md).

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
| order habit | `order_focus` (`first` / `last` / `same_dish`), `favourite_dishes`, `order_commit` | with several open orders: which order's jobs come first (oldest, newest, the dish it delivered last; favourite dishes before all). With `order_commit` it sticks to one open order, chosen by that habit, until it is delivered; after 8 idle steps on it (`COMMIT_PATIENCE`) it helps elsewhere until its order has work again |

Personas (`coop.partner.PERSONAS`), each changing one trait from `steady`, all
stable (the same traits every day):

| persona | trait |
|---|---|
| `steady` | none |
| `prep_first`, `serve_first` | prefers a stage; waits up to 8 steps for one before doing other work |
| `slow` | acts every second step, pauses 2 steps before each job |
| `yielding` | gives way on tasks: if the other player is heading for the same job (facing it, or closer and getting closer), it picks another |
| `avoids_board` | never uses the chopping board when the other player can reach one (on `new_env` only the partner can, so it still cuts there) |
| `leaves_unfinished` | half the time walks away from starting a loaded utensil or from taking finished food out, and leaves it for 15 steps |
| `sloppy`, `flaky` | slips / abandons jobs; kept for later tests, not in the default lists (their traits were hard to see or looked like hand-overs) |
| `oldest_first` | commits to the oldest open order until it is delivered |
| `newest_first` | commits to the newest open order |
| `favourite_dish` | commits to `baked_carrot_soup` (the one dish that needs the oven) whenever it is open, else the oldest |
| `repeats_dish` | after delivering a dish, commits to an open order for the same dish if there is one, else the oldest |

The order-habit types differ from `steady` only with several open orders.
`coop.partner.ORDER_PERSONAS` is `steady` plus these four. Every other type does
whatever is needed next, oldest order first. Measured with a `steady` teammate,
2 open orders, 3–6 seeds: `favourite_dish` spends 87–88% of its work on the carrot
soup when it is open (`steady` 51%), and `oldest_first` 10–15% on the newest
order (`steady` 28–41%).

`coop.partner.DEFAULT_PERSONAS` (used by the configs) is the first seven. The
probe asks about six fields: `prefers`, `avoids`, `gives_way`, `speed`,
`reliability`, `commitment`. `yielding` and `avoids_board` only differ from
`steady` on the open kitchens.

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

## Several open orders and partial view

Both are settings of `coop.days` (config keys `open_orders`, `view_radius`, or
`--open-orders`, `--view-radius`), of `coop.screen` and of `coop.voi`. They
apply to every player alike: the LLM ego, the simulated partner and the
scripted egos.

- **`open_orders: k`** (default 1): k orders are open at once; delivering any of
  them closes it and opens the next order of the sequence. The MDP gets an
  optional `num_open_orders` (default 1, the original rule).
- **`view_radius: r`** (default: full view): a player sees cells within r steps
  (Chebyshev, as OvercookedV2's `agent_view_size`). For other cells it remembers
  what it saw and when (`coop.view.Belief`). A batch seen cooking is assumed to
  finish by the known cook time (public rules). The partner is seen only in
  view (otherwise: last place and item seen). Its actions are known only when
  seen, and so is who made a delivery (the score is public). Orders and the map
  stay public. The LLM ego is told this, and its prompt marks out-of-view
  utensils and counters "last seen at t=...". Scripted players with nothing to
  do walk to the station they saw longest ago.

Movement and planning fixes that several orders and partial views needed
(partner and LLM executor alike):
- when cut off, step aside with probability ½;
- on A→B→A→B shuttling, stay with probability ½;
- set down a held item after waiting 5 steps with nowhere to put it;
- keep preparing inputs on free stations while an order's final station is
  busy;
- never take out food that another open order needs.

These change open-map results of every persona slightly (e.g. steady 260 → 280
on the pinned `open_env` seed); `new_env` is unchanged.

Scripted pairs (300 steps, 4 seeds) have no stalls on `open_env` / `open2_env`
with 1–3 open orders, or with radius 3. Radius 2 still stalls a few
`open_env` episodes; `open2_env` is fine.

### Is knowing the partner worth anything? (`coop.voi`)

```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.voi --layout open_env --out outputs/voi_open.json
```

Plays every persona with 8 scripted ego policies (no preference or prep / cook /
serve first, each with or without giving way). `value` = score of the best ego
for this persona − score of the ego that is best on average. That is what
knowing the partner buys with these policies (a lower bound).

The best policies are chosen on `--seeds` (0–7) and re-measured on
`--eval-seeds` (8–15). Picking the best of 16 policies on the same episodes
overstates the gain, so trust the re-measured value.

**Ego sets** (`--ego-set`, with several open orders): `flexible` egos take
whatever job is needed next: the 16 above, plus `any+leave_<dish>` (that dish's
orders last). `commit` egos work on one order at a time: `commit_oldest`,
`commit_newest` (each with or without giving way), and `commit_oldest+leave_<dish>`.
`all` (the default) uses both. The commit set is closer to an agent that plans its
own order, such as an LLM choosing macro actions.

**Order habits** (`--personas order`, 2 open orders, 8 + 8 seeds):
- **Flexible egos:** knowing the partner is worth about 0. Doing whatever is needed
  next for either order adapts to every habit.
- **Committed egos:** the right order depends on the partner. On `open2_env`, full
  view: `commit_newest` scores 180 with `newest_first` but 217.5 with `oldest_first`,
  and `commit_oldest` 217.5 vs 187.5. Re-measured values (commit set, new seeds):
  `open2_env` radius 2: `newest_first` +35, `oldest_first` +25; `open2_env` full
  view: `newest_first` +17.5, `favourite_dish` +7.5; `open_env`: about 0 or negative
  (one utensil of each kind makes the two orders compete for the same stations).
  Run 16+ seeds before quoting these.

Result, one open order, full view (8 seeds, 300 steps): `value` is at most +7.5
points (less than half an order) for every persona on all three maps, and 0 for
most.

With several open orders and/or a partial view (7 default personas, re-measured
on new seeds): almost every gain seen on the selection seeds disappears or
reverses. The one consistent case is a `slow` partner on `open2_env` with 2 open
orders and radius 2 (+22.5 on the selection seeds, +40 on new seeds). Everything
else is within about ±10, with occasional large negatives (noise of 8 seeds). With one order open
at a time and a full view of the kitchen, doing whatever the order needs next is
already close to the best response to every partner. A slow partner just means
doing more; an avoided station means doing that step yourself. So, as the
setting stands, ToM or memory cannot raise the score by knowing the partner,
only by fixing the agent's own play. Settings where knowing the partner should
pay: hidden information (partial observability, where the agent must predict what
it cannot see), several orders open at once (choosing complementary work early),
and costly commitments (long walks, contested stations).

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
  | `days_llm_orders.yaml` | order habits: `open2_env`, 2 open orders, radius 2, `ORDER_PERSONAS` × 4 run seeds, all 6 conditions; swap donors with the opposite habit |
  | `days_llm_model.yaml` | stronger partner modelling (structured ToM, partner-model file) in the `days_llm_orders` setting: direct, tom_memory, tom_struct, tom_model, tom_struct_model + swap / placebo |
  | `days_llm_orders_full.yaml` | the same with a full view (a test keeps the two identical otherwise): the comparison arm for partial view |

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
  | `probe` | after each episode, rates the partner: `prefers` (prep/cook/serve/no clear preference), `avoids` (a utensil, or nothing), `gives_way`, `speed` (normal/slow), `reliability` (reliable/makes mistakes), `commitment` (finishes tasks/leaves tasks unfinished), plus `stage_shares`. Scored against the persona's true profile (`coop.profile`). Without `memory`, the answer is logged but never shown to the agent again. |

- **Probe details.** The probe defines each answer the same way in every
  condition (`coop.profile.PROFILE_GUIDE`). It tells the agent how many actions it
  and its partner completed, and asks for `stage_shares` (the partner's prep /
  cook / serve split in %), which are scored against the partner's real split
  (`share_error`, percentage points). Events about a batch that matched no recipe
  say who put each item in and who started the utensil, from the public action log.
- **Probe rules** (since 2026-11-12; `coop.profile`):
  - `avoids` is asked only over utensils that this episode's orders needed and that
    both players can reach (`avoid_options`; e.g. never the blender with the default
    menu, and not at all on `new_env`). If none qualifies, the field is not asked or
    scored. The truth is the persona's avoided utensil if it is among them, else "nothing".
  - `speed`: the probe shows an **observed pace** line, the share of the partner's
    actions (move, turn, pick up, put down) that were followed by another action on the
    very next step, counted only while the partner was in view. The agent sees the
    kitchen only at its decision points, so it cannot see pace otherwise. `slow` scores
    0% and every other type 57–82% (scripted calibration). The guide says that waiting
    for preferred work is a preference, not slowness. Speed accuracy therefore measures
    reading and using this cue, not inferring pace unaided.
  - `commitment` is about visibly unfinished work (`leaves_unfinished`). Abandoning a
    carried item (`flaky`) puts it on a counter, which looks exactly like a hand-over,
    so `flaky`'s commitment is not scored (truth `None`).
  - `orders` (the order habit) is asked only with several open orders. Its options
    are "its oldest open order", "the newest open order", "the dish it served last",
    "favourite dish: <dish>" (one per dish ordered in the episode) and "no fixed
    habit". A partner without `order_commit` has no fixed habit. A favourite dish
    that was never ordered that episode can't be seen, so the truth is then
    "its oldest open order". The order list in every observation is marked "oldest first".
  - Fields with truth `None` are left out of `accuracy`. `coop.analyze` rescores every
    probe with the current rules (from `partner_truth.json` and the episode's orders), so
    older output folders are read the same way. Old "abandons tasks" answers count as
    "leaves tasks unfinished", and an old `avoids` answer naming a utensil outside the
    current options counts as "nothing". Old runs had no pace line, so their speed
    answers are not comparable to new ones.
- **Partner modelling options** (LLM agent; the defaults keep the original conditions):

  | option | values | what the agent gets |
  |---|---|---|
  | `tom_mode` (with `tom`) | `free` (default) | one or two sentences about the partner before each action, never shown again |
  | | `structured` | fields before each action: `doing`, `order` (the open order the partner works on, or none / unknown), `next` (the action it will complete next), `habit`. The last read is shown at the next decision ("Your last read of your partner (t=..)"), so it can be checked and corrected within the episode |
  | `memory_mode` (with `memory`) | `notes` (default) | the last `max_memory_notes` reflections (profile + up to 80 words) as a list |
  | | `model` | one markdown file about the partner, at most `model_words` words, with four sections: How they choose what to do (with evidence) / How they react to me / What I should do with them / Open questions. After every episode the agent rewrites the whole file in its own call (`model_max_tokens`): keep what holds, correct what was wrong, add evidence. The call is separate from the probe, so the file never sees the probe's trait list or pace line. The file is shown in every later episode and is all the agent remembers about the partner. `coop.analyze` reports how often the file was actually written |

  Swap and placebo work on the file too: swap gives the donor's file, placebo a file
  with the same headings and length holding neutral kitchen facts (`placebo_model`).
  `coop.analyze` scores structured reads against the truth ("structured ToM" table).
  `order` is compared with the dish of the order the partner really works on: its job's
  order, or the order it is committed to, or "none" when idle. The baseline always
  names the oldest open order. `next` is compared with the partner's next completed
  action (exact, and same verb). The baseline repeats its last action. Both are given
  per persona and per day. The partner trace now logs each step's `dish` and `committed`.
- **Memory controls** (conditions with `memory_source`, run after their source
  condition, from its finished runs):

  | control | memory shown before each episode |
  |---|---|
  | `swap` | the notes the source condition (e.g. `tom_memory`) built with a **different** partner after the same number of episodes. Donors default to the next persona in the list; set `donors: {persona: donor}` to choose. |
  | `placebo` | the source condition's notes for **this** partner, with the same number and length of notes, but neutral kitchen facts instead of the text and no profile |

  If `swap` does as well as `tom_memory`, the gain is a general habit rather
  than knowledge of this partner. If `placebo` does as well, the notes' content
  does not matter. The agent still reflects after each episode (for the probe),
  but in a control its own notes are replaced before the next episode.
  `coop.analyze` reports each condition against `tom_memory` on identical
  episodes, with standard errors over runs.
- **Backends** (`coop.llm`, from ICRL4AHT): `openai_compat` (vLLM or a hosted
  API; `*_env` keys read URL, model, key and `chat_template_kwargs` from
  environment variables) and `mock` (offline: picks from `VALID_ACTIONS`).
  `coop.check` verifies that thinking is off for a served model.
- **Probe caveat on `new_env`** (`avoids` is not asked there): the ego is on the right, so the partner works on
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
- probe accuracy per field (rescored with the current rules), with the most common
  answer and the truth, the mean observed pace (new runs), and the stage-share error;
- how decisions were spent: wait share, chosen wait length, waits that ran
  their full length vs ended by the partner, and full-length waits that ended
  with finished food sitting in a utensil the ego can reach;
- division of labour: each player's completed work split into prep / cook / serve
  (prep = raw ingredients, chopping board, blender; cook = pot, oven; serve =
  dishes, plating, delivering; the probe's definitions), from `steps.jsonl`, with
  an overlap index (sum over stages of min(ego %, partner %): 100 = same mix, 0 =
  fully split) and its difference vs `direct` / `tom_memory` per persona (SE
  over runs). Hand-overs (`place_obj_on_counter`) are counted separately;
- order habits (several open orders, from `summary.json`): for each player, the
  share of its deliveries that skipped an older open order ("newer"), the ego's
  share of all deliveries, and, against `favourite_dish`, the ego's share of the
  favourite dish's deliveries vs the other dishes. `ego_newer` and the favourite
  share are compared with `direct` / `tom_memory` (SE over runs). An ego that adapts
  should deliver older orders more often with `newest_first`, newer ones with
  `oldest_first`, and leave the carrot soup to `favourite_dish`;
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

**API models** (any OpenAI-compatible endpoint): skip the serving script and set the
backend variables yourself (the key stays in the environment, never in a config):

```bash
export ICRL_LLM_BASE_URL=https://<provider>/v1 ICRL_LLM_MODEL=<model name> ICRL_LLM_API_KEY=<key>
```
```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.days configs/coop/days_llm_model.yaml --out outputs/days_model_<model> --workers 8
```

Leave `COOP_CHAT_TEMPLATE_KWARGS` unset (it is a vLLM option). Use few workers to stay
under the provider's rate limits. The requests send `temperature` and a per-call
`seed`. Not yet tried against a hosted API; if a provider rejects `seed`, tell me.

Small first run (one condition, one persona, 2 days × 1 short episode):

```bash
lab-run --new --gpus 1 -- bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --out outputs/days_llm_smoke --conditions tom_memory --personas slow --run-seeds 0 --days 2 --episodes-per-day 1 --horizon 150
```

**Why a run takes long.** The agent asks the model whenever its action ends.
It mostly chooses `wait(1)`, so that is nearly every step: about 200–330 calls per
300-step episode, plus retries after invalid replies. A run's 10 episodes are
played one after another (memory carries over), and within an episode each call
waits for the previous one. A run therefore takes about (calls per episode ×
seconds per call × 10). `coop.analyze` prints both numbers ("time spent in the
model"). Running more runs in parallel shortens the whole config, but not one run.

**GPUs.** Qwen3.8-27B in bf16 needs about 54 GB, so one RTX PRO 6000 (96 GB) holds
one server with about 30 GB left for request memory, not three. Either run each
map as its own job on its own GPU (`--spare`, three GPUs), or run all three maps
against one server on one GPU:

```bash
lab-run --new --gpus 1 --cpus 24 --mem 110G -- env MAX_NUM_SEQS=48 LAYOUTS="new_env:outputs/days_llm_v2 open_env:outputs/days_llm_open open2_env:outputs/days_llm_open2" bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --workers 16
```

With `LAYOUTS`, the script starts one server and runs each `layout:out_dir` as its
own `coop.days` process (logs `run_<layout>.log` next to `vllm.log`). The
`--workers` value applies to each map, so 3 × 16 = 48 requests share `MAX_NUM_SEQS=48`.

Three jobs on three GPUs at the same time (`--spare`). With the spare pool's 84
CPUs and 341 GB, three jobs fit at `--cpus 24 --mem 110G` each (3 × 120G would be
360 GB). Each config has 48 LLM runs (4 conditions × 6 personas × 2 run seeds), so
`--workers 48` with `MAX_NUM_SEQS=48` plays them all at once.

**Resuming.** Stopping a job and restarting it with the same `--out` keeps every
finished episode (`--workers`, `MAX_NUM_SEQS` and `LAYOUTS` are not part of the
config). If the code that plays episodes changed since the folder was started,
`coop.days` refuses to continue (`code_version.json`), so one result folder never
mixes two agent versions. Use a new `--out`, or `--allow-code-change` when the
change cannot affect results.

Full design, split map (`new_env`):

```bash
lab-run --new --spare --gpus 1 --cpus 24 --mem 110G -- env MAX_NUM_SEQS=48 bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --out outputs/days_llm_v2 --workers 48
```

Full design, open kitchen (`open_env`):

```bash
lab-run --new --spare --gpus 1 --cpus 24 --mem 110G -- env MAX_NUM_SEQS=48 bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --layout open_env --out outputs/days_llm_open --workers 48
```

Full design, open kitchen with two of every utensil (`open2_env`):

```bash
lab-run --new --spare --gpus 1 --cpus 24 --mem 110G -- env MAX_NUM_SEQS=48 bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --layout open2_env --out outputs/days_llm_open2 --workers 48
```

Partner screens on the open kitchens:

```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.screen --layout open_env --out outputs/coop_screen_open.json
```
```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.screen --layout open2_env --out outputs/coop_screen_open2.json
```

Order habits, value of knowing the partner (16 selection + 16 new seeds; CPU):

```bash
lab-run --new --gpus 0 -- env PYTHONPATH=src:lib/overcooked_ai .venv/bin/python -m coop.voi --layout open2_env --open-orders 2 --view-radius 2 --personas order --ego-set commit --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --eval-seeds 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --out outputs/voi_orders_open2_r2_commit.json
```

Order habits with the LLM ego (all six conditions):

```bash
lab-run --new --spare --gpus 1 --cpus 24 --mem 110G -- env MAX_NUM_SEQS=48 bash scripts/run_qwen38_days.sh configs/coop/days_llm_orders.yaml --out outputs/days_orders --workers 48
```

## Comparing two settings (`coop.compare`)

Does a setting (e.g. partial view) change how much a condition helps? Run the
same config twice, differing only in that setting, then:

```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.compare outputs/days_orders outputs/days_orders_full --out outputs/compare_view_orders.json
```

The orders depend only on (run seed, day, episode), so each run (persona × run seed)
in A has a twin in B. For each contrast (`tom_memory - direct`, `tom_memory_swap -
tom_memory`, ...), it reports the contrast in A, in B, and the change (A − B) per
run, with mean ± SE over runs, overall and per persona. A positive change for
`tom_memory - direct` means `tom_memory` helps more in A. It also gives each
condition's score minus that folder's own scripted reference, and probe accuracy per
field. It warns if the two folders come from different code versions, and lists the
config differences.

Partial view vs full view with the LLM:

```bash
lab-run --new --spare --gpus 1 --cpus 24 --mem 110G -- env MAX_NUM_SEQS=48 bash scripts/run_qwen38_days.sh configs/coop/days_llm_orders_full.yaml --out outputs/days_orders_full --workers 48
```

One open order, original partner types (partial view without order habits): run
`days_llm.yaml --layout open_env --run-seeds 0 1 2 3` once with `--view-radius 2`
and once without, into two new folders.

## Not built yet

An order-display tile (orders are public everywhere), and an LLM-simulated
human partner.

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
