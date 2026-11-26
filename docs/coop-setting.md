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
| `open_ing2_env` | 9×5 open kitchen with two of every utensil and **one dispenser per ingredient**: carrot on the left wall (0, 1), potato on the right wall (8, 1). Its own 8-dish menu from those two (below). | Who fetches which ingredient, and the blender and two-ingredient dishes, give the partner's habits more to act on. The map stays open: either cook can do any job. |

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

**Several GPUs for one run:** `GPUS=2` (with `lab-run --gpus 2`) serves one model copy per GPU
behind one endpoint (vLLM `--data-parallel-size`), so twice as many requests run at once. Give
about `GPUS × MAX_NUM_SEQS` workers. `PARALLEL=tp` splits one copy over the GPUs instead, for a vLLM
without data parallelism. Only the server changes, so results are comparable and a run can
resume with more GPUs.

```bash
lab-run --new --spare --gpus 2 --cpus 24 --mem 110G -- env GPUS=2 MAX_NUM_SEQS=48 bash scripts/run_qwen38_days.sh configs/coop/days_llm_model.yaml --out outputs/days_model_v2 --workers 96
```

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

## LLM persona partner (plan v2: `docs/PERSONA_PARTNER_PLAN_v2.md`)

The experimental partner. The scripted personas above remain as a test harness.

**Rules and cards.**
- **The library** (`configs/coop/rules/library.yaml`, `coop.rules`) holds 10 rules. Each has a
  category, parameters, the partner's text (second person), a fallback, what it *wants* from
  the other cook, and a reminder line.
- **A persona card** (`configs/coop/personas/batch1.yaml`) is an ordered list of rules plus a
  backstory for voice. At complexity K the first K rules are active (nested).
- **`neutral`** is built in: a card with no rules, used as the competence baseline.
- **Contradictions are rejected on load:** `solo_dish` + `split_dish` on one dish,
  `serves(ego)` + `solo_dish`, `serves(self)` + `split_dish`, `wants_supplies` + `solo_dish`,
  a single-use rule twice, two `own_station` rules on one station kind.
- **Pilot cards:** p1–p4 have 6 rules each and no contradictions. Any two differ in ≥ 3 rules,
  and every category appears in ≥ 2 cards with different parameters (tested). The plan's
  example card combined `solo_dish` with `serves(ego)`, a contradiction under its own rule, so
  p1 does not.
- **Optional parameters** (`options` in the library, defaults in `coop.rules.OPTIONS`). A
  default is not stored in the rule, so cards that do not set an option keep their labels and
  texts.
  - `own_station(when_taken=wait)`: while the other cook's batch is in its station, the
    partner waits for it, up to 20 steps, instead of moving its work to the other station. The
    check counts using the other station during that wait as non-adherent.
  - `split_dish(patience=20|40|60)` and `serves(who=ego, patience=20|40|60)`: how long the
    steps it leaves to the other cook wait before it does them itself.
  - `waits_for_commit(patience=5|10)`: how long it waits for the other cook to commit.
- **Revised cards** (`configs/coop/personas/batch2.yaml`, 2026-11-17; "waits for its things",
  for score as the primary outcome):
  - dropped `handover_spot` (the put-down action cannot choose a counter) and `wants_supplies`
    (too rare to measure);
  - `own_station(when_taken=wait)` on every card;
  - `split_dish` / `serves(ego)` with patience 40;
  - p1, p2 and p4 have 5–6 rules, p3 has 4;
  - no hand-over category any more.

  Configs: `persona_gate1_batch2.yaml`, `persona_gate1_twin_batch2.yaml`, `persona_value.yaml`,
  and `days_persona.yaml` (now batch2).

**Shared priority** (in the partner's prompt, used by the twin, applied by the predicates):
1. a valid action and no deadlock;
2. finishing what it holds or started;
3. its rules in card order;
4. otherwise what is needed next, oldest order first.

So when two rules conflict, the earlier one wins, and the later one's predicate does not
count that decision. Examples: a `solo_dish` pick is not an `order_priority` violation, and an
earlier `order_priority` overrides `on_takeover` on the committed order.

**Predicates** (`coop.rule_eval`). Each step, every completed action becomes an *event*: verb,
target, item, the dishes it belongs to, and whether it was a *choice* (taken with empty hands).
A shared `Tracker` keeps the kitchen's memory: who loaded or started each utensil, who put each
counter item down, how long a complete batch or finished food has waited, each player's
current order, and new orders. The evaluator, the twin and the LLM partner's reminders all use
it. "Had work on X" means a job the partner's own rules allow, computed by the base job
generator.

| rule | adherent (partner) | aligned (ego) |
|---|---|---|
| `own_station(s)` | loads of that kind go into `s`, unless `s` cannot take the item or was busy when the partner decided | the ego's interactions with that kind avoid `s` (a load is not counted if no other station could take it) |
| `solo_dish(d)` | while `d` is open and the partner has `d` work, its choices are on `d` | the ego does not touch `d`; other choices while `d` is open count as aligned |
| `split_dish(d)` | each start or plating of `d` is adherent if the ego did it, or the partner after the batch or food waited ≥ 20 steps (what it works on otherwise is order choice, not this rule) | each start / plate / delivery of `d` is aligned if the ego did it |
| `order_priority(w)` | its next pick after its committed order is gone is the `w` order (any of the last 10 steps); leaving the committed order counts only if that order still had work for it | the ego's picks avoid the partner's committed order |
| `serves(self)` | if finished food was waiting when it decided (not only cooking, and not only ready by the time the action completed), its choices are serving | each plating / delivery is aligned if the partner did it |
| `serves(ego)` | when serving work exists, its choices are not serving (unless food waited ≥ 20 steps) | each plating / delivery is aligned if the ego did it |
| `wants_supplies` | it does not fetch a missing item of its incomplete batch from the dispenser for 15 steps (taking it from a counter is fine) | each missing item that reaches its batch is aligned if the ego supplied it (directly or through a counter) |
| `handover_spot(g)` | its counter placements are in group `g` (or `g` was full) | the ego's placements are in `g` (when `g` had room) |
| `on_takeover` | after the ego takes over its things, its next choice within 30 steps is not on that dish (only if it had other work) | the ego's interactions with existing items do not take over the partner's (allowed when another rule asks for it) |
| `on_contention(m)` | in a race to load the same utensil (both carry an item for it, both within 3 cells), resolved by who uses it first: `yields`: the ego got it, or the partner after 5 steps; `insists`: the partner got it, or still had it as its target | `yields`: the ego got it; `insists`: the partner got it |
| `waits_for_commit` | if it was free when a new order opened (nothing held, no open order of its own) and decided after the order appeared, its first choice comes ≥ 5 steps later or after the ego's first choice; carrying on with its own order is not judged | the ego (empty-handed at the opening) makes a choice within 5 steps |

A rule with no opportunity in an episode is undefined, not perfect. Counter groups are the
left / center / right thirds of the kitchen.

**Rule-based twin** (`coop.rule_partner.RulePlayer`, `mode="partner"`). It follows a card on
top of `SimulatedHuman`:
- its rule keys rank jobs in card order;
- order commitment is one of those keys;
- it drops its current job, with empty hands, when a higher rule's job appears;
- it waits by its station (5 steps) when yielding;
- its station, counter and fallback choices follow the rules.

With a full view (3 seeds, scripted steady ego) adherence per card is ≥ 95%; single rules
are 85–100%. With `mode="ego"` it does what a card *wants* (the aligned ego of Gate 2), as
agent type `rule_ego` with `align: own | other`.

**LLM partner** (`coop.llm_partner`, `partner.type: llm`). It uses the ego's stack
(observation, macros, executor, retry). Its system prompt has:
- the intro;
- the backstory;
- "Your habits" (the active rules with their fallbacks);
- the priority order.

Each reply is `{"goal", "order", "reconsider_when", "action"}`, and the previous goal is shown
at the next decision. **Reminders** (`remind: true`) add one line per rule whose opportunity
holds now, from the partner's own view, only at the decision the rule is about; every
reminder is logged. `enforce` is only a hook. There is no memory across episodes and no
probe. The partner's backend uses `ICRL_PARTNER_*` env vars; the serving script defaults them
to the ego's server.

**Runner.** `partner: {type: scripted | rule | llm, cards, complexity, remind, enforce, backend,
max_tokens}`; `personas` are card names. Per episode it writes:
- `summary.json` → `rules` (adherence and alignment counts per rule) and `partner_llm` (calls,
  invalid replies, fallbacks, reminders);
- `rule_events.jsonl` (every judgment with a note);
- `partner_truth.json` (card, trace);
- `partner_log.json` (the partner's prompts).

Per run folder, `run_meta.json` records models, the serving version, decoding settings and
every prompt template with its hash. The ego never sees the card, the rules, the partner's
intentions or reminders, or any rule score; `tests/test_coop_persona.py` searches every ego
prompt for them. Old configs give identical steps and prompts, checked against the previous
commit with the mock backend.

**Gates** (plan section 10; stop after them and report):

```bash
python -m coop.days configs/coop/persona_gate1_twin.yaml --out outputs/gate1_twin --workers 16
```
```bash
bash scripts/run_qwen38_days.sh configs/coop/persona_gate1.yaml --out outputs/gate1_llm --workers 48
```
```bash
python -m coop.persona_check gate1 outputs/gate1_llm --twin outputs/gate1_twin --out outputs/gate1_llm/gate1.json
```
```bash
python -m coop.voi --cards configs/coop/personas/batch1.yaml --layout open2_env --open-orders 2 --view-radius 2 --seeds 0 1 2 3 4 5 6 7 --eval-seeds 8 9 10 11 12 13 14 15 --out outputs/gate2_twin.json
```
```bash
bash scripts/run_qwen38_days.sh configs/coop/persona_gate2.yaml --out outputs/gate2_llm --workers 48
```
```bash
python -m coop.persona_check gate2 outputs/gate2_llm --out outputs/gate2_llm/gate2.json
```

`persona_check gate1` reports, per card:
- score against the neutral partner;
- adherence per rule on the calibration seeds 0–1 and the judge seeds 2–3 (bar 70%, frozen
  before judging);
- adherence by episode;
- partner invalid replies and fallbacks;
- the distinctness classifier.

`persona_check why` shows why the partner breaks a rule. For each card and rule it lists:
- the partner's non-adherent decisions, grouped by what happened;
- how often a reminder for that rule was shown at the deciding step;
- examples with the partner's stated goal, order and chosen action.

```bash
python -m coop.persona_check why outputs/gate1_llm --examples 3 --out outputs/gate1_llm/why.json
```

`persona_check rescore` re-judges a finished folder with the current rule checks, without any
model calls:
- **How:** it replays each episode's recorded actions on its seed and checks the replay
  matches (score and positions). Replaying with unchanged checks reproduces the original
  judgments exactly.
- **Output:** it writes `rules.rescored.json` and `rule_events.rescored.jsonl` next to the
  originals.
- **Exception:** `on_contention` keeps its original judgments, because the players' intended
  targets are not saved.
- **Decision steps:** checks that depend on the situation a choice was made in use the step
  it was decided, not the step it completed. That is an LLM player's macro decision (from its
  log when re-scoring) or a scripted player's job choice.
- **Reading the results:** `gate1`, `gate2` and `why` read the re-scored files with
  `--rescored`.

```bash
python -m coop.persona_check rescore outputs/gate1_llm --workers 8
```
```bash
python -m coop.persona_check gate1 outputs/gate1_llm --twin outputs/gate1_twin --rescored
```

`persona_check gate2` and `voi --cards` report aligned − generic and aligned − misaligned for
score (the primary outcome) and alignment (secondary), on two seed sets and per card, with a
line saying whether there is score headroom (mean − 2 SE > 0).

`voi --cards` adds an `uninformed` ego: the aligned ego's own machinery (`RulePlayer`) with no
rule. Aligned − uninformed is the value of knowing the card with everything else equal; aligned −
generic also includes the difference between the two scripted implementations.

**Score value of one rule** (`voi --rules`). The twin gets a single library rule (every parameter
value that exists in the kitchen). The aligned ego knows it, the uninformed and generic egos know
nothing, and the opposite ego believes the flipped parameter. All differences are paired over seeds.

```bash
python -m coop.voi --rules --layout open2_env --open-orders 2 --view-radius 2 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 16 --out outputs/rule_value_twin.json
```

## Multi-day partner learning (plan v2, sections 6–11)

Config `configs/coop/days_persona.yaml`. New keys are all opt-in; old configs play exactly
as before (mock check: identical steps, prompts, summaries and memory).

| key | meaning |
|---|---|
| `schedule: {1: [...], 2: [...], 3: all}` | dishes of each day's interaction episodes (staggered exposure) |
| `order_lists: controlled` | blocks containing every dish once, shuffled per episode seed, repeated (`orders.make_controlled_sequence`) |
| `eval_episodes: 2` | held-out episodes after each day and before day 1 (day 0): full menu, the same seeds every day, played on a frozen copy of the ego's memory, no learning from them |
| `knowledge_probe: true` | after each day (and day 0), on the frozen memory: one multiple-choice question per library rule (`coop.probe2`) |
| condition `feedback: specific` | the LLM partner's feedback at the end of each day, given to the ego's `end_day` (`coop.feedback`) |
| condition `oracle: true` | the ego is told the card's "wants" lines up front |
| `memory_source: {type: swap, from: memory_fb, donor: card \| seed}` | daily controls: at the start of each day, the file is replaced by the donor's file after the previous day (another card, same seed; or the same card, next seed). Placebo: same headings and length, neutral text. The run keeps its own notes and gets its own partner's feedback |

**Ego memory** (`memory_mode`):
- **`model_daily`:** after each interaction episode, a short note of joint evidence (≤ 60
  words: what I did, what they did, what followed). The day's notes are shown in the rest of
  that day. At the end of the day, one call rewrites the partner file from the old file, the
  notes and (if given) the feedback. The file has four sections (Partner Preferences, Partner
  Reactions, Cooperation Strategy, Uncertainties), with conditions, evidence tallies
  (`+D1E2 / -D2E3`), sources, and "Follows from" for strategy lines. The word budgets are
  80 / 60 / 60 / 40 per section. Trimming happens within a section, never across, and is logged
  in `dayDD/consolidation.json`.
- **`latest_feedback`:** no file; only the previous day's feedback is shown.

**Conditions** (all with structured ToM):
- `tom_struct`: no memory.
- `memory`: daily file, no feedback.
- `memory_fb`: daily file plus feedback.
- `latest_fb`: only the last feedback.
- `oracle`: the "wants" lines.
- `swap_other`, `swap_same`, `placebo`: daily controls built from memory_fb.

Conditions without memory across days (`tom_struct`, `oracle`, the reference) are
evaluated once, on day 0, and the analysis uses that value for every day.

**Feedback** (`coop.feedback`, LLM partner only):
- **Input:** the partner model gets its card (backstory and the active rules' text), the day's
  scores, and its own log of the day: its actions, the ego actions it saw, and public events,
  up to 40 lines each per episode.
- **Output:** it answers in at most 60 words in its own voice; it may ask for a change and must
  not list its rules.
- **Checks:** word count, plus rule recitation (8 consecutive words of a rule text, or a rule id).
- **Storage:** `dayDD/feedback.json` holds the prompt and reply for a manual sample of
  unsupported claims. It is generated once and read back on resume.

**Knowledge probe** (`coop.probe2`): which station of each kind (when both are free), the
solo dish, the split dish, which order, who serves, waiting for supplies, the hand-over side,
the reaction to a takeover, the reaction to contention, and waiting for commitment. Rules the
persona lacks have a "no preference / none / no" answer. Answers are scored separately for
rules exposed so far (the partner had an opportunity in the ego's view), not yet exposed, and
absent. The probe is a forked call: nothing enters memory.

**Exposure**: `summary.json` → `rules[rule].exposure` counts the partner's opportunities
in the ego's view (Chebyshev distance ≤ view radius).

**Output per run:**
- `dayDD/epEE/` (interaction);
- `dayDD/evalEE/` (held-out; `summary.json` has `"kind": "eval"`);
- `dayDD/feedback.json`;
- `dayDD/consolidation.json`;
- `dayDD/knowledge_probe.json`;
- `day00/` (before day 1).

**Kitchen with one dispenser per ingredient** (`open_ing2_env`, 2026-11-17).

```
XXPXOXPXX     I (0,1) carrot, I (8,1) potato: each dispenser gives only its ingredient
I       I     P pot ×2, O oven ×2 (top middle, right), C chopping board ×2,
C 1   2 S     B blender ×2, D dishes, S serving
D       O
XXBXCXBXX
```

- **Menu** (8 dishes, from carrot and potato):

  | dish | made from | steps |
  |---|---|---|
  | `boiled_potato` | potato | pot |
  | `boiled_carrot_slices` | carrot | board, pot |
  | `boiled_potato_slices` | potato | board, pot |
  | `baked_carrot_soup` | carrot | board, oven, pot (dish) |
  | `baked_potato_soup` | potato | board, oven, pot (dish) |
  | `mashed_potato` | potato | board, pot, blender (dish) |
  | `mashed_potato_and_carrot_mixture` | potato + carrot | boiled potato slices and a carrot into the blender (dish) |
  | `sliced_carrot_and_potato_stew` | carrot + potato | carrot slices and a potato into the pot (dish) |

- **How it works:**
  - **Layout key** `"dispensers": {ingredient: [x, y]}`. `ContinuousOrderEnv` refuses an
    interaction that asks a dispenser for another ingredient. The executor and the scripted
    players walk to the right one, and the map in the prompt says what each dispenser gives.
    Layouts without the key are unchanged (mock check: identical episodes).
  - **Recipe texts** for the new dishes are in `src/prompts/recipe_coop/`, because the original
    benchmark lists `src/prompts/recipe/` in full.
- **Cards** `configs/coop/personas/batch_ing2.yaml`: batch2's personas with dishes from this menu.
  - p2 splits the stew.
  - p4 splits the blender mixture and owns `blender0`.
  - p3 keeps `wants_supplies`.
- **Configs:** `persona_gate1_ing2.yaml`, `persona_gate1_twin_ing2.yaml`, `persona_value_ing2.yaml`,
  `days_persona_ing2.yaml`.
- **For `coop.voi` / `coop.screen`:** pass `--layout open_ing2_env --menu` with the 8 dishes.

## Plan v3: order timers and context features (`docs/CONTEXT_PARTNER_PLAN_v3.md`)

**Order timers** (config block `deadlines`, off by default; old configs play identically,
mock check over 2,821 episode files):
- **Config:** `deadlines: {slack: s, reference: {dish: steps}, expiry_penalty: 20, urgent_steps: θ}`.
- **Limit:** each order may stay open `round(s × reference[dish])` steps.
- **Expiry:** the order closes unserved, `expiry_penalty` points are lost, and the next order opens.
  The order record has `limit` and `expired_t`, and the summary has `orders_expired`.
- **Visibility:** both players see the steps left. `state.order_timers` is passed through the
  partial view, and the LLM observation shows `[N steps left]` after each open order.
- **The 300-step horizon stays.** The scripted reference ego plays under the same timers.

**Reference times** (`python -m coop.deadlines`). Median steps from issue to delivery for a
steady scripted pair, `open_ing2_env`, 2 open orders, radius 2, seeds 0–15:

| dish | pilot menu (3 dishes) | full 8-dish menu |
|---|---|---|
| boiled_potato | - | 36 |
| boiled_carrot_slices | - | 45 |
| boiled_potato_slices | 54 | 59 |
| baked_carrot_soup | 64 | 66.5 |
| baked_potato_soup | - | 72 |
| mashed_potato | - | 61.5 |
| mashed_potato_and_carrot_mixture | - | 63 |
| sliced_carrot_and_potato_stew | 65 | 48.5 |

The times depend on the menu (which orders compete), so measure them for the menu you run.

**Context features** (`coop.context`). They are computed on the full state or on a player's own
view:
- `urgent`: open orders with at most θ steps left;
- `on_order(who)`: the dish its held item serves, or its last choice;
- `blocked`: kinds whose every utensil is running or holds finished food;
- `prep_ready`: open dishes with a processed item waiting unused.

`ContextTracker` turns them into events at the step a context starts.

**Duplication check** (plan 3.3). When both players bring the last missing ingredient to one
utensil, the second item is accepted: the executor checks only that the utensil is empty or
loaded. Starting the utensil then throws the whole batch away. So duplication is a strong
mismatch cost, a lost batch. The scripted players re-check and put the item down instead.
Executor behaviour is unchanged.

**Context rules and cards** (plan v3, sections 4–6):
- **Library entries** (category `context`): `default_work(prep|whole)`, `on_urgent(takes|keeps,
  unless_ego_on_it)`, `on_blocked(clears|waits)`, `on_prep_ready(switches|continues)`.
  - They have no fallbacks.
  - Every `wants` line is an action for the ego.
  - Cards `configs/coop/personas/batch_ctx.yaml`: pilot A and B, opposite on every rule.
- **"Prepared items waiting"** counts an item only after it has lain unused for 5 steps
  (`coop.context.PREP_WAIT`). Otherwise a cook coming back for what it just cut triggers it.
- **`coop.context_partner.ContextPlayer`**, deciding on its own view. Its table of how each
  mode acts on jobs is in the module docstring.
  - **Twin** (`policy: card`): follows its card. It has one global deadlock guard: after 30
    steps without any completed interaction it does the next needed step once, and logs it as
    `guard`.
  - **Scripted egos:** `proactive` (does the card's `wants`), `mismatched` (the other card's),
    `always_take_urgent`, `never_take_urgent`, and `wait_and_see(k)`.
  - A `whole` cook only uses its own items.
- **Checks** (`coop.context_eval.ContextRules`). One checker per card, because card order
  decides. Each choice is judged in the context its chooser saw, in its own view, at the step
  it decided. The evaluator keeps each player's view, context history and (for the partner) a
  view-based tracker.
  - **Twin adherence**, radius 2, 8 seeds: card A 96–97%, card B 98–100%.

**Gate 0** (`python -m coop.voi --gate0 CARDS ...`). The crossover table (ego × card: score,
expired orders, guard firings) on the judge seeds. Then, per card:
- the best reactive ego, chosen on the calibration seeds;
- proactive − best reactive and proactive − mismatched, with paired 95% intervals;
- the guard rate against a bar set on the calibration seeds.

`--context-rules` gives the value of each context rule alone.

```bash
python -m coop.voi --gate0 configs/coop/personas/batch_ctx.yaml --layout open_ing2_env --menu boiled_potato_slices baked_carrot_soup sliced_carrot_and_potato_stew --open-orders 2 --view-radius 2 --slack 1.5 --urgent-steps 15 --reference '{"boiled_potato_slices": 54, "baked_carrot_soup": 64, "sliced_carrot_and_potato_stew": 65}' --seeds 0 1 2 3 4 5 6 7 --eval-seeds 8 9 10 11 12 13 14 15 --workers 8 --out outputs/gate0_base.json
```

## Plan v3: LLM experiments with context partners

**LLM partner on context cards** (W4):
- **Prompt:** the card's context rules in card order, with no fallbacks; it sees the order
  timers.
- **Reminders:** one line per decision, for the first rule (card order) whose context holds in
  its own view. It keeps the context features every step, so "prepared items waiting" counts
  waiting time.
- **Deadlock guard:** after 30 steps without a completed interaction, one prompt line tells it
  to do the next needed step. It is logged as `guard` and is not part of the persona prompt.

**Runner additions:**
- With `partner.type: rule`, a context card is played by the twin (`ContextPlayer`).
- Ego `type: context_ego` (`policy: proactive | always_take_urgent | never_take_urgent |
  wait_and_see`, `k`, `align: own | other`) gives the scripted egos for Gate 2.
- Condition `oracle: other` is `oracle_mismatched`: the next card's wants lines.
- Config `regimes` / `regime_order` set per-day timer slack and order mix for interaction
  episodes, counterbalanced by run-seed parity. Held-out episodes keep the base timers and the
  full menu.
- Controlled order lists honour integer weights (copies per block).

**Analysis:**
- `persona_check gate0 OUT` reads a runner folder (Gate 2): the crossover table, the best
  reactive ego, proactive − best reactive and − mismatched.
- `coop.learning` adds expired orders and guard firings per episode, and the transfer table:
  held-out orders completed for dishes seen in interaction vs never seen.
- The knowledge probe asks one conditional question per context rule (`probe2.CONTEXT_QUESTIONS`).

| config | step | players | size |
|---|---|---|---|
| `persona_gate1_twin_ctx.yaml` | Gate 1, twin | scripted | CPU |
| `persona_gate1_ctx.yaml` | Gate 1 | LLM partner, steady ego | A, B, neutral × 4 seeds × 3 episodes |
| `persona_gate2_ctx.yaml` | Gate 2 | LLM partner, 8 scripted egos | 2 cards × 8 seeds × 3 episodes |
| `persona_value_ctx.yaml` | Gate 3 | LLM partner, LLM ego (`tom_struct`, `oracle`, `oracle_mismatched`) | 2 cards × 8 seeds × 4 episodes |
| `days_ctx.yaml` | learning pilot / main | LLM partner, LLM ego, 9 conditions | 2 cards × 4 seeds × 5 days |
| `days_ctx_guard.yaml` | test: `days_ctx` with the recipe guard | as `days_ctx` | as `days_ctx` |

`days_ctx.yaml`:
- **Menu:** interaction episodes use the 3 pilot dishes. The held-out set adds
  `baked_potato_soup` and `boiled_carrot_slices`, never seen in interaction (transfer).
- **Regimes:** urgency-heavy days (slack 1.75), blocking-heavy days (oven-heavy orders, slack
  3.0), and a mixed day 5 (slack 2.5).
- **Order:** even seeds see urgency first, odd seeds blocking first.

**Recipe guard** (`recipe_guard: true` under `agent:` and/or `partner:`; test option, off
everywhere else). The executor refuses three kinds of recipe mistakes, which dominated the
pilot's feedback (§3.24 of `RESEARCH_STATUS.md`):
- putting an item into a utensil unless the contents stay part of a recipe of that kind whose
  product an open order needs (e.g. a second potato into a pot);
- starting a batch that makes nothing an open order needs (as `useful_starts_only`);
- delivering what no open order takes, or a dish plated the wrong way.

A refused action comes back as invalid with the reason, and guarded actions are left out of the
listed options. `days_ctx_guard.yaml` is `days_ctx` with the guard on both players; it asks
whether knowing the partner shows in the score once the ego's own mistakes are removed.

**One-decision test** (`coop.decision`, `configs/coop/decision/urgent.yaml`). Does the
partner change the right action, and does the ego act on it?
- **Start:** two open orders, `other` (older, timer 250) and `urgent` (timer 50–85, urgent
  from step 0). The two dishes use disjoint ingredients, so the ego's first action shows which
  order it chose. Horizon 150, radius 2, both start slots, 4 seeds.
- **Partner:** the scripted twin of `default_work: whole` plus `on_urgent: takes` (switches
  to the urgent order) or `keeps` (stays on its own order).
- `--stage dishes --from-folder OUT`: per dish, orders delivered by the ego / partner,
  expired, ego wrong deliveries (to pick an urgent dish the ego completes reliably).
- `--stage screen` (CPU): a scripted cook committed to each order first, against both
  partners. A start is kept only if the better choice (team score, gap `min_gap`) switches
  with the partner. With horizon 100 or limits 35–45, few starts switched: the urgent order
  expired whatever the ego did, or the horizon cut off the payoff of choosing `other`.
- `--stage llm` (GPU, `COOP_MODULE=coop.decision scripts/run_qwen38_days.sh ...`): the LLM ego
  on the kept starts, against both partners, told the correct description (the card's wants
  lines, as `oracle`), the swapped one, or none. Per episode: the first committed action and
  its order, urgent order delivered in time and by whom, recipe errors after the choice (the
  recipe guard's checks, recorded, not enforced), wrong deliveries.
- `--stage report` (also printed after each stage): the screen, the LLM table and a verdict:
  no kept starts → change the partner behaviours; correct − max(swapped, none) below 15 points
  of right first choices → a stronger ego on the same starts (new `--out`); right choices but
  the chosen order delivered by the ego in under 70% → cooking competence.

**Value check** (`configs/coop/persona_value.yaml`). Is knowing the partner worth points to
the LLM ego? It compares `oracle` (told the card's "wants" lines) with `tom_struct`: one day
of 4 full-menu episodes, 4 cards × 8 seeds, batch2, LLM partner. Read the "interaction-episode
score, condition minus tom_struct" block of `coop.learning`.

**Learning curves** (`coop.learning OUT`). Score is the primary outcome and alignment the
secondary one (decided 2026-11-17); the report prints them in that order.
- held-out score minus the reference for days 0–5;
- each condition minus `tom_struct` and minus `memory_fb`, paired over runs (per day, the
  average over days 1–5, and per persona), for score and then for alignment;
- interaction-episode score, each condition minus `tom_struct` (the same episode seeds in every
  condition). `oracle` − `tom_struct` here is the score value of knowing the partner for this ego;
- alignment and partner adherence on held-out episodes;
- probe accuracy (exposed / not exposed / absent);
- cumulative exposure per rule;
- feedback checks, trimmed lines, and invalid replies and fallbacks for ego and partner.

**Where this deviates from the plan:**
- the day's notes are shown during that day;
- episode notes are written by the ego model;
- feedback exists only with the LLM partner;
- the knowledge probe also runs on day 0;
- the plan's example card p1 was contradictory and was changed.

```bash
bash scripts/run_qwen38_days.sh configs/coop/days_persona.yaml --conditions tom_struct memory memory_fb oracle swap_other --run-seeds 0 1 2 --out outputs/persona_pilot --workers 64
```
```bash
python -m coop.learning outputs/persona_pilot --out outputs/persona_pilot/learning.json
```

## Plan v4: two-axis partners (`docs/STYLE_PARTNER_PLAN_v4.md`)

**Cards** (`configs/coop/personas/batch_style.yaml`): each card is a workflow and a plan
adherence; the loader (`coop.rules.expand_style`) expands it into v3 context rules.

| workflow | the twin does | the best response (ego) |
|---|---|---|
| `prep` | fetch and cut for every order, raw items for pots / ovens on a counter by the cooking area | cook, plate, serve |
| `whole` | one order start to finish, then the next | the other order start to finish |
| `station` (new) | load pots / ovens with what was brought (held, on a counter, from an oven), start, take out, plate; plated food on a counter; looks around only at the cooking stations | fetch, cut, bring cooking items to a counter by the cooking area, serve |

| adherence | on_urgent | on_blocked | on_prep_ready |
|---|---|---|---|
| `rigid` | keeps | waits | continues |
| `flexible` | takes (unless the ego is on it) | clears | switches |

- `station` fixes `on_blocked` and `on_prep_ready` (it clears and cooks as its own work): those
  pairs carry no rule and are marked `not_identifiable` (`Card.contexts`).
- `mixed` (diagnostic): `prep`, flexible on urgent orders, rigid otherwise.
- The loader rejects contradictions: an unknown workflow or adherence, a per-context map that
  misses a context, a map on a workflow that fixes contexts, both `rules` and `workflow`.
- Ground truth: `Card.latent` (workflow, adherence) and `Card.contexts` (mode or
  not_identifiable), written into `partner_truth.json`.
- Backstories: `style_only` and `style_habits` per card. These are drafts, frozen after review
  before V2.
- **Twin** (`ContextPlayer(..., near_cooking=True, latent=card.latent)`): hands cooking items
  over on counters next to the cooking area (walkable cells next to a pot or oven, plus the
  dish dispenser since the station cook plates: `context_partner.cooking_area`). It lists every
  job before the rules choose (the v3 twin missed the cooking jobs a context rule should switch
  to; v3 cards keep the old behaviour). Every decision logs the latent labels, the mode
  applied, and whether the twin was inside the ego's view. Adherence 98–100% on every card
  (steady ego, 8 seeds).
- **Evaluator:** `station` choices (a station step vs a fetch, cut or delivery; loads and
  deliveries are judged too). For v4 cards only (`RuleEvaluator(style=True)`), taking food out
  of a pot or oven counts as acting on waiting prepared items.

**V0** (`coop.voi --v0 CARDS`): egos per card, judged with the twin partner:
- `best_response`: proactive on the card;
- `wrong_adherence`: the same workflow, the opposite adherence;
- `wrong_workflow:<wf>`: the same adherence, each other workflow;
- `flip:<context>`: one context's adherence flipped, for the per-context value. This
  approximates the plan's "Δs restricted to each context's events": it is a whole-episode
  score difference;
- the reactive set: `steady`, `always_take_urgent`, `never_take_urgent`, `wait_and_see:3/6/10`.
  The best one is chosen on the calibration seeds.

It reports the crossover table (score minus a steady pair on the same episodes, expired
orders, guards) and per card Δ_reactive, Δ_adherence, Δ_workflow and Δ per context, with 95% t
intervals over the judge seeds. Then the pass checks of plan §5. `--soft`: the informed egos
use policy `proactive_soft` (the complement first, then anything useful) instead of the
strict complement. `--log` appends the setting
and the checks to a JSON-lines tuning log.

V0 command (CPU, about 5 minutes on 4 cores):

    PYTHONPATH=src:lib/overcooked_ai python -m coop.voi --v0 configs/coop/personas/batch_style.yaml \
      --layout open_ing2_env --menu boiled_potato_slices baked_carrot_soup sliced_carrot_and_potato_stew \
      --open-orders 2 --view-radius 2 --slack 1.5 --urgent-steps 15 --expiry-penalty 20 \
      --reference '{"boiled_potato_slices": 48, "baked_carrot_soup": 68, "sliced_carrot_and_potato_stew": 57.5}' \
      --seeds 0 1 2 3 --eval-seeds 4 5 6 7 --episodes 3 --workers 4 --out outputs/v0/v0_s1.5_t15.json

## API backends (`docs/API_BACKEND_PLAN.md`, W1–W4)

Hosted models as the ego (or the LLM partner), next to the local vLLM server. Code:
`src/coop/llm.py` (backends), `src/coop/costs.py` (report lines), `src/coop/check.py --api`
(smoke test), `scripts/run_api.sh` (lab runs). Configs: `configs/coop/api/backends.yaml`
(registry) and `configs/coop/api/prices.yaml`.

**Backend types:**

| type | API | what it sends |
|---|---|---|
| `openai_compat` | local vLLM (unchanged) | as before |
| `openai` | OpenAI Chat Completions | `max_completion_tokens` = caller cap + `reasoning_allowance`; `reasoning_effort`; no `temperature` / `top_p` / `seed` (`send_sampling: false`); `response_format: json_object` (`json_mode`); `prompt_cache_key: auto` = a hash of the system prompt |
| `anthropic` | Anthropic Messages (`POST /v1/messages`) | headers `x-api-key`, `anthropic-version` (`2026-10-21`); the system message as a top-level `system` text block with `cache_control: ephemeral`; `max_tokens` = caller cap + allowance; `anthropic_params` merged into the body as given (registry: `output_config: {effort: low}`); no sampling parameters; no response-format parameter (the prompt and `parse_json` handle JSON) |
| `mock` | none | as before |

**Use:** `backend: {ref: opus}` in a config (other keys override the entry), or
`--ego-backend NAME` (and `--partner-backend NAME` on `coop.days`). The override replaces the ego
backend in every condition; the folder's `config.json` then holds the resolved spec with its
`ref`. Keys are never in a config or an output: `api_key_env` names the variable.

**Per call** (`agent_log.json` calls, `knowledge_probe.json`, `feedback.json`, only for API
backends, so old outputs keep their shape): `cached_tokens`, `cache_write_tokens`,
`reasoning_tokens`, `cost_usd`, `retries`, `rate_limited_s`, `served_model`, and
`missing_usage` when the response lacked a usage field. `llm_usage` in `summary.json` sums
them and adds `refusals`. `prompt_tokens` is always the whole prompt (cached and written
included). Reasoning models (`expects_reasoning: true`) do not count as "thinking" in the
health check; `coop.analyze` prints reasoning tokens per call instead.

**Field mapping:**
- OpenAI: `prompt_tokens_details.cached_tokens` → `cached_tokens`,
  `completion_tokens_details.reasoning_tokens` → `reasoning_tokens`, `message.refusal` →
  finish reason `refusal`.
- Anthropic: `input_tokens` (the uncached rest) + `cache_read_input_tokens` +
  `cache_creation_input_tokens` = `prompt_tokens`; `cache_read_input_tokens` → `cached_tokens`;
  `cache_creation_input_tokens` → `cache_write_tokens`; `output_tokens_details.thinking_tokens`
  → `reasoning_tokens`; `text` blocks → the reply; `thinking` blocks → `reasoning` (empty with
  the default display); `stop_reason` `max_tokens` → `length`, `refusal` → `refusal`.

**Retries:** 429, 500, 502, 503, 504 and 529, with `Retry-After` when sent, else exponential
backoff; `max_retries` 6 and `timeout_s` 300 by default for API types. A 429 for a spend cap
(`enforced_spend_limit_reached`) and every other 4xx are not retried; the error text is kept
with the key removed.

**Concurrency:** `max_concurrency` is a limit shared by every process on the machine (one lock
file per slot under `$COOP_LOCK_DIR` or the temp dir). The runners' `--workers` are processes,
so a per-process semaphore would not hold the limit.

**Cost and budget:** cost per call = uncached input × input + cached × cached input + written ×
cache write + output × output price (`prices.yaml`; reasoning is not added again when
`reasoning_in_output: true`). Every API call is appended to the folder's `cost_ledger.tsv`.
`--max-cost USD` (`coop.days`, `coop.decision`): once the ledger reaches it, no new call starts.
Runs stop with status `budget` (a `budget.log` instead of `error.log`), the interrupted episode
has no summary, and rerunning the same command with the same `--out` resumes. Calls already
in flight finish, so the total can overshoot by about `max_concurrency` calls. Reports
(`coop.learning`, `coop.analyze`, `coop.decision --stage report`) end with cost per model and
condition, the total in the logs and the ledger total.

**Smoke test** before any paid run: `python -m coop.check --api NAME --planned-calls N` sends the 4
decision prompts of `tests/fixtures/api_decision_prompts.json` (captured from a mock run of
`configs/coop/decision/urgent.yaml`), repeats the first (cache check), probes optional
parameters, and prints per call: JSON valid, action parsed and in VALID_ACTIONS, input / cached /
output / reasoning tokens, latency, finish reason, cost; then the rejected parameters, missing
usage fields, whether output counts include reasoning, and the projected cost. Written to
`outputs/api_check/<name>.json`. Exit code 1 if any call failed or gave an invalid action.

**Checked against the Claude API reference (2026-11), for `claude-opus-5-5`:** API version
`2026-10-21`; thinking is always on (no `thinking` field runs adaptive; `{type: disabled}` is a
400); depth via `output_config.effort` (`low` … `max`, default `medium`); `temperature`,
`top_p`, `top_k` are rejected; no assistant prefill; the minimum cacheable prefix is far below
the ~1,600-token decision system prompt; prices $4 / $0.20 cached / $5 cache write (5 min) /
$20 output per million; `stop_reason: refusal` exists. Refusal fallbacks (`fallbacks`) are
deliberately not sent: a fallback answers with another model, which would mix models within a
condition; refusals are counted instead.

**To verify with the smoke test (not assumed):** for Astra, whether `response_format`,
`prompt_cache_key` and `reasoning_effort: low` are accepted, the names of the cached and
reasoning token fields, the prices; for both, whether output counts include reasoning, whether
the repeated prompt is read from the cache, and the account's rate limits (set
`max_concurrency` from them). Record the results here after the smoke test.

**Lab runs:** `scripts/run_api.sh` loads the keys from `~/.coop_api_env` (lines such as
`export ANTHROPIC_API_KEY=...`; must be `chmod 600`, the script refuses it otherwise) unless they
are already exported. Without `QWEN=1` it runs the module (`COOP_MODULE`, default `coop.days`)
directly: an API ego with the scripted twin partner needs no GPU (`lab-run --new --gpus 0`).
With `QWEN=1` it serves Qwen through `scripts/run_qwen38_days.sh` for the LLM partner, and the
ego stays on the API. The server must reach `api.openai.com` / `api.anthropic.com` over HTTPS;
the smoke test shows it.

**Deviations from the plan:**
- the registry and price table are in `configs/coop/api/`, not `configs/coop/` (every top-level
  file there is loaded as a `coop.days` config by the tests);
- `max_concurrency` and the budget work across processes (lock files, a ledger file), since the
  runners use process pools;
- `anthropic_version` is `2026-10-21` (checked), not `2023-06-01`;
- `coop.decision` has `--ego-backend` but no `--partner-backend` (its partner is the scripted twin);
- W6 (Gemini) and W7 (batch pricing) are not built.

## Pooling seeds (`coop.analyze` with several folders)

Folders of the same config run on different run seeds (e.g. a replication) can be
analysed together. Every table is computed over all runs, and the paired differences
are also shown per folder, so you can see whether the folders agree:

```bash
PYTHONPATH=src:lib/overcooked_ai python -m coop.analyze outputs/days_model_v2 outputs/days_model_v2_rep --out outputs/days_model_pooled.json
```

It refuses folders whose configs differ in anything but the name and run seeds, folders
that share a run seed, and (unless `--allow-code-change`) folders run with different code.

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
