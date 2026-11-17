# Research status: partner modelling (ToM) and memory in Collab-Overcooked

Last updated: 2026-11-12 (branch `claude/lucid-babbage-96q0zi`; probe rules changed 2026-11-12, see §5).
How to run everything: [`docs/coop-setting.md`](docs/coop-setting.md).

## 1. Question

Can an LLM agent learn about a **stable human partner by watching them**, over several days of
play, and use that to cooperate better? Specifically: do **ToM prompting** (describe the partner
before acting) and **memory across episodes** (reflect after each episode, keep the notes)
improve teamwork, and is the gain about **this partner** or just general skill?

## 2. Setting

- **Environment:** Collab-Overcooked's env (multi-step recipes: cut / cook / bake / stir; 30 tasks)
  with OvercookedV2-style changes: both agents have the **same abilities**, there are **no messages**,
  and orders arrive continuously.
- **Episodes and days:** one episode = 300 steps with a seeded stream of orders. Delivering an order
  gives +20; a wrong delivery costs −20. A run = the same partner for 5 days × 2 episodes. The agent's
  memory carries over across episodes and days.
- **Maps:**
  - `new_env`: the original split kitchen, where roles are forced (agent: pot, oven, serving; partner:
    dispenser, chopping board, blender, dishes);
  - `open_env`: an open kitchen with one of each station;
  - `open2_env`: an open kitchen with two of every utensil.
- **Partner:** a scripted simulated human, `coop.partner.SimulatedHuman`, with stable traits.
  - **Original types:** `steady`, `prep_first`, `serve_first`, `slow`, `sloppy` (slips), `flaky` (abandons).
  - **Added 2026-11-10:** `yielding`, `avoids_board`, `leaves_unfinished`. These replaced
    `sloppy`/`flaky` in the default lists; both are still defined.
  - **Order habits, added 2026-11-12** (`ORDER_PERSONAS`; only visible with several open orders):
    `oldest_first`, `newest_first`, `favourite_dish` (baked_carrot_soup), `repeats_dish`. Each
    commits to one open order until it is delivered. Every other type does whatever is needed
    next, oldest order first.
- **Agent:** Qwen3.8-27B via vLLM (temperature 0, thinking off). It uses Collab-Overcooked's actions,
  which an executor turns into moves. It sees only public information: the map, the recipes, the
  orders, utensils, counters, both players' positions and held items, and the actions the partner
  completed. It never sees the partner's type or true jobs.
- **Conditions:**

  | condition | what the agent does |
  |---|---|
  | `direct` | chooses actions only |
  | `tom` | describes the partner before each action |
  | `memory` | reflects after each episode; its notes are shown in later episodes |
  | `tom_memory` | both |
  | `tom_memory_swap` | gets the notes `tom_memory` built with a **different** partner |
  | `tom_memory_placebo` | gets neutral notes of the same length, with no profile |

- **Measures:**
  - score minus a scripted reference ego on **identical orders** (removes how hard a day's orders are);
  - paired differences with standard errors over **runs** (persona × seed);
  - a partner probe after each episode: trait labels plus prep/cook/serve shares, scored against
    the true persona (rules in `docs/coop-setting.md`, "Probe rules");
  - division of labour: each player's stage mix, and an overlap index (100 = same mix, 0 = fully split).

## 3. Results so far

All LLM numbers: Qwen3.8-27B, 300-step episodes, 5 days × 2 episodes, ego in slot 0.

### 3.1 Split map, before and after fixes (`days_llm`, `days_llm_v2`; 6 personas × 2 seeds)
- **Before the fixes:** the agent kept trying the partner's stations; 30–40% of `direct`'s replies were invalid.
  - **Fix 1:** a reaching/hand-over rule was added to the prompt; invalid replies fell 50–60%.
  - **Fix 2:** movement fixes.
- **v2, mean over days (score minus reference):** `direct` −29, `memory` −23, `tom` −25,
  `tom_memory` −10 (days 2–5: −6 to −10).
  - **tom_memory vs direct:** +19.0 ± 3.4 (episode-level SE).
- **The probe was flat:** preference was always "prep", because the map forces it. "Slow" and
  "abandons" were never detected.

### 3.2 Open maps, no controls (`days_llm_open`, `days_llm_open2`)
- **Effects vs `direct` are smaller:**
  - `open_env`: `tom` +6.5, `memory` −0.5, `tom_memory` +7.8;
  - `open2_env`: `tom_memory` +10.3.
- **No learning trend across days.**
- **Preference becomes observable:** `tom_memory` identified `serve_first` 1.00 (direct 0.45).

### 3.3 Memory controls, split map (`days_ctrl_split`; 6 original personas × 2 seeds, run-level SE)

| comparison | result |
|---|---|
| tom_memory − direct | +22.8 ± 6.4 |
| swap − tom_memory | **+0.5 ± 2.1** (no loss) |
| placebo − tom_memory | −8.7 ± 3.3 |

**Conclusion:** with fixed roles, the memory gain is **general skill**. Notes about any partner work.
The agent learns to wait for hand-overs: longer waits, ended by partner actions, fewer calls.

### 3.4 Memory controls, open kitchen (`days_ctrl_open`, then `days_ctrl_open_rep` with 4 seeds and the current code)

| comparison | first run (2 seeds) | replication (4 seeds) |
|---|---|---|
| tom_memory − direct | +13.2 ± 3.1 | **+12.0 ± 2.9** |
| swap − tom_memory | −7.3 ± 2.9 | **−5.6 ± 1.9** |
| placebo − tom_memory | −5.3 ± 3.5 | **−4.5 ± 2.2** |

- **About half of the gain needs notes about this partner.** Swap and placebo end at about the same
  level as `tom` alone.
- **The loss sits on the preference partners.** swap − tom_memory: `prep_first` −15, `serve_first` −11,
  others −1 to −3.
- **The probe agrees:** `serve_first`'s preference was recognised 0.85 with its own memory, vs
  0.25–0.40 otherwise.

### 3.5 Division of labour (open replication; the `serve_first` part did not replicate, see §3.9)
- **`serve_first`:** the overlap fell from 75 (`direct`) to **55** (`tom_memory`), −19.4 ± 3.5.
  - swap: +8.9 ± 2.3 vs `tom_memory`; placebo: +8.5 ± 2.1.
  - The partner's serving share rose from 36% to 57%, and the agent's serving fell from 21% to 15%.
  - **The agent hands serving over to a partner who prefers it**, and this is partner-specific.
    Probe, behaviour and score agree.
- **`prep_first`:** the overlap fell 7.0 ± 2.4, but the swap gives the same, so this part is not
  partner-specific. The agent still competes for prep work (28–31% vs the scripted reference's 17%).
- **No-preference partners:** no change, as expected. With memory the agent acts less (e.g. 72 → 49
  actions per episode with `steady`) and scores better.

### 3.6 Value of knowing the partner (`coop.voi`, scripted egos, no LLM)
- **Method:** best scripted ego for this partner minus best average ego. Chosen on seeds 0–7,
  re-measured on 8–15.
- **One order, full view:** at most +7.5 on all maps. A reactive "do what's needed" policy is near-optimal
  against every partner type.
- **Several open orders and/or a partial view:** almost all gains vanish on the new seeds. The one
  consistent case is `slow` on `open2_env` with 2 orders and radius 2: +22.5 on the selection seeds,
  +40 on the new seeds.
- **This is a lower bound.** No scripted policy predicts the partner, and every partner type above
  serves the oldest order first.
- **Order habits** (2 open orders, 8 + 8 seeds; `--ego-set`):
  - **Flexible egos** ("do whatever is needed next"): value about 0 against every habit.
  - **Egos that commit to one order** (closer to an LLM planning its own order): the right order
    depends on the partner. On `open2_env`, full view: `commit_newest` 180 with `newest_first` vs
    217.5 with `oldest_first`, and `commit_oldest` the reverse.
  - **Re-measured value of the commit set:** `open2_env` radius 2: `newest_first` +35,
    `oldest_first` +25; full view: `newest_first` +17.5; `open_env` about 0 (one utensil of each
    kind: both orders compete for it).
  - Preliminary (8 seeds); the 16-seed screen is the next run.

### 3.7 Partial view vs full view, order habits (`days_orders` vs `days_orders_full`, 2026-11-13)
Setting: `open2_env`, 2 open orders, `ORDER_PERSONAS` × 4 seeds (20 runs), 5 days × 2 episodes,
all six conditions. A = radius 2, B = full view; otherwise identical. `coop.compare`, SE over runs.

| contrast | partial (A) | full (B) | change A − B |
|---|---|---|---|
| tom − direct | +2.9 ± 3.5 | −1.8 ± 1.9 | +4.7 ± 4.0 |
| memory − direct | +2.7 ± 3.2 | −1.2 ± 1.9 | +3.9 ± 3.7 |
| **tom_memory − direct** | **+10.6 ± 3.6** | −0.8 ± 2.5 | **+11.4 ± 4.3** |
| tom_memory − tom | +7.7 ± 3.8 | +1.0 ± 2.4 | +6.7 ± 4.9 |
| swap − tom_memory | −5.8 ± 4.4 | +6.4 ± 3.0 | −12.2 ± 5.4 |
| placebo − tom_memory | −7.1 ± 4.3 | −1.4 ± 3.2 | −5.7 ± 5.7 |

- **Partial view makes tom_memory useful.** With a full view, ToM + memory does nothing here
  (−0.8). With a partial view it gains +10.6, and neither ToM nor memory alone does (+2.9, +2.7).
- **Partial view costs the LLM more than the scripted reference.** Score minus reference falls
  from −21 to −40 for `direct`, but only from −22 to −29 for `tom_memory`. tom_memory recovers
  about 11 of `direct`'s 18-point loss.
- **Partner-specific? Not established.**
  - With a partial view, swap and placebo both lose about 6–7 vs tom_memory; neither loss is
    significant alone.
  - The −12.2 change is half driven by swap doing *better* than tom_memory with a full view
    (+6.4 ± 3.0).
  - The largest gains and swap losses are on `steady` (no fixed habit; +22.5 ± 4.3 tom_memory −
    direct change) and `favourite_dish` (+19.5 ± 9.3). So the gain is not specific to order habits.
- **The probe never identifies the order habit:** `orders` accuracy 0.17–0.21 in every condition
  and both views. That is the level of one fixed answer right for one of the five partners.
- **Behaviour** (`coop.analyze` order-habit table; "ego newer" = share of the ego's deliveries that
  skipped an older open order; differences in percentage points, SE over 4 runs per partner):
  - **Partial view, tom_memory − direct, ego newer:** `oldest_first` +5±3, `favourite_dish` +10±4,
    `repeats_dish` +14±0 (all three usually work on the oldest order), but `newest_first` −2±3 and
    `steady` −1±5. The direction fits each partner's habit: take the newer order when the partner
    takes the older one.
  - **Partial view, swap − tom_memory:** the change reverses, towards the donor's habit. Notes about
    a `newest_first` partner give `favourite_dish` −10±6 and `repeats_dish` −10±4; notes about an
    `oldest_first` partner give `newest_first` +5±4. This is the clearest partner-specific signature
    so far, but small and based on few runs. Placebo is mixed (−6 to +4).
  - **Full view:** ToM alone makes the ego take newer orders with *every* partner, including
    `newest_first` (+8±2) and `steady` (+14±2). tom_memory is the same (+11 to +26), and swap
    matches it. That is a general strategy, not a response to the partner, and it does not raise
    the score.
  - **Favourite dish, full view only:** tom_memory leaves the carrot soup to `favourite_dish` (ego
    share 12% vs 19% for `direct`, −9±3). Swap (+9±2) and placebo (+15±6) undo it, so it is
    partner-specific, but the score does not improve. With a partial view there is no such effect
    (−3±12): the ego rarely sees the partner working on the soup.
  - **The score gain is not explained by order choice.** The largest partial-view score change is
    with `steady` (+22.5), where order choice does not change.
- **The probe never names a habit:** the answer to `orders` is "no fixed habit" for every partner, in
  every condition and both views. Behaviour adapts a little; the explicit report does not.
- **Open questions:**
  - Where does the partial-view gain come from? Check the decision stats and the division of labour
    of the two folders.
  - Does the ToM text track the partner's order? Score the `tom` "partner" sentences against the
    dish the partner was really working on (offline, from `partner_truth.json`).

### 3.9 Open kitchen, third run with the fixed probe (`days_probe_open`, 2026-11-13)
Same setting as `days_ctrl_open_rep`: `open_env`, 1 order, full view, 7 default personas × 4 seeds.
The code differs: the new probe has a pace line and new options, which also changes the reflection.

| comparison (run SE) | ctrl_open (2 seeds) | ctrl_open_rep | **probe_open** |
|---|---|---|---|
| tom_memory − direct | +13.2 ± 3.1 | +12.0 ± 2.9 | **+13.9 ± 2.0** |
| tom − direct | | | +8.4 ± 2.5 |
| memory − direct | | | −3.1 ± 2.2 |
| swap − tom_memory | −7.3 ± 2.9 | −5.6 ± 1.9 | **−4.6 ± 2.2** |
| placebo − tom_memory | −5.3 ± 3.5 | −4.5 ± 2.2 | **−4.6 ± 1.9** |

- **The score result replicates a third time.** ToM gives about +8. Memory adds about +5 on top,
  but only with the agent's own notes: swap and placebo fall back to the `tom` level.
- **Which partners show the swap loss changes between runs** (per persona SE 4–7):
  - this run: `slow` −14.5, `leaves_unfinished` −12.5, `prep_first` −11;
  - before: `prep_first` −15, `serve_first` −11.

  `prep_first` is the only partner with a swap loss in both runs.
- **The `serve_first` hand-over did not replicate as partner-specific.**
  - The overlap still falls (direct 74.7 → tom_memory 61.5), but `tom` alone does the same
    (62.6), and so do swap (59.6) and placebo (60.8).
  - Here the hand-over comes from ToM, not from memory about this partner.
  - `prep_first` now shows a small partner-specific split instead: overlap swap +3.3 ± 0.9 and
    placebo +4.7 ± 1.0 vs tom_memory. It showed none before.
  - So partner-specific division of labour is not robust across runs.
- **Probe with the new rules:**
  - **speed works:** `slow` is recognised 0.88–0.97 (pace line 0%), and the preference partners are
    now "normal" (`prep_first` 1.00, `serve_first` 0.88–1.00);
  - **avoids:** `avoids_board` is recognised 1.00, even by `direct`. But `serve_first` and `slow` are
    often said to avoid the board, and they may really not use it (scored against the trait, not
    behaviour);
  - **commitment** (`leaves_unfinished` 0.00–0.07) and **gives_way** (`yielding` 0.00) are never
    detected;
  - **prefers:** `prep_first` 0.95–1.00, `serve_first` 0.38 (direct) → 0.68 (memory, tom_memory).
    Partners with no preference are often called "cook";
  - **swapped notes carry over into the probe:** with notes from the `slow` partner, `serve_first`
    is called slow (speed 0.17). The answers lean on memory, not only on the episode.
- **How the gain arises (decision stats):**
  - **ToM makes the agent act less and wait more.** Wait share: `direct` 12–20%, `tom` 16–41%,
    `tom_memory` 15–47%. Agent actions with `steady`: 71 (`direct`) → 50 (`tom_memory`), close to
    the scripted reference (46).
  - **Fewer invalid replies:** `direct` 760, `tom` 807, `tom_memory` 491. With `tom_memory`, more
    waits end early because the partner acted (6 per episode with `steady` / `prep_first` /
    `yielding`, vs 0 for `direct`).
  - **The cost:** waits that ran their full length while finished food was ready rise from 6–9
    (`direct`) to 11–15 (`tom_memory`).
  - **Orders finish faster with tom_memory** (steps per order): baked_carrot_soup 50.6 → 46.0,
    stew 50.0 → 48.2. The scripted reference makes the stew in 38.8.
  - So much of the gain is the agent getting in its partner's way less. Only the part that needs
    the agent's own notes (about +5) is a candidate for partner modelling.
  - **`slow` barely completes any actions** (3–5 per episode, the reference too), so it is almost
    an absent partner on this map. `tom_memory` gains +15 with it and swap loses that (−14.5). The
    likely lesson in the notes is "do everything yourself".

### 3.15 Score is the primary outcome; with the twin, the cards have no score headroom (2026-11-17)
**Decision (user, 2026-11-17):** score is the primary outcome, alignment secondary (plan §15).
`coop.learning`, `persona_check gate2` and `voi --cards` print score first.

**Twin Gate 2** (`voi --cards`, open2_env, radius 2, 2 orders). It now includes an `uninformed`
ego: the aligned ego's machinery with no rule.

| aligned minus | score, seeds 0–7 | score, seeds 8–15 | alignment (points), 0–7 / 8–15 |
|---|---|---|---|
| generic (steady) | −19 ± 6 | −6 ± 4 | +24 / +22 |
| uninformed | −15 ± 7 | −4 ± 4 | +25 / +22 |
| misaligned | +49 ± 16 | +52 ± 13 | +31 / +28 |

- **The negative score is not an implementation artefact:** uninformed ≈ generic.
- **The misaligned gain comes from p2 and p4.** p2 is +160 / +108: nobody serves. p4 is +70 / +105.
- **For p1 and p3, the misaligned ego does as well as the aligned one or better.**

**Score value of one rule** (`voi --rules`, 16 seeds; the twin has only that rule):
- **Knowing a rule rarely matters.** Against the uninformed ego it is worth between −16 and
  +12 points, SE 2–7, out of about 230.
- **Accommodating costs the most for `serves(self)` (−16) and `on_contention(insists)` (−10).**
  The ego gives up serving or gives way.
- **The one large value is a wrong belief, not missing knowledge.** `serves(ego)` vs the opposite
  belief is +159, but vs no belief only +12: a reactive ego sees the plated dish and serves it.

**Reading.** With the twin, the habits are cheap to ignore: its fallbacks absorb conflicts, and
a reactive ego fills the gaps it sees. In score, knowing the partner pays only when it replaces a
wrong belief.

Two things the twin cannot show decide whether the current cards can produce a score gain:
- whether the LLM partner makes the habits costly to ignore (Gate 2 with the LLM partner);
- whether the LLM ego, which reacts less well than the scripted ego, gains from knowing them
  (`oracle` − `tom_struct`).

If both are about zero, the rules need consequences for being ignored before the learning runs
(plan §14, first row). For example:
- a station the partner waits for instead of switching;
- a dish it leaves unstarted for the ego;
- `serves(ego)` on more cards;
- tighter order deadlines, so anticipation pays.

### 3.14 Multi-day learning setup: built, not yet run (plan sections 6–11, 2026-11-17)
Built:
- day schedule and controlled order lists;
- held-out evaluation on frozen memory (days 0–5);
- daily partner file (`model_daily`) and `latest_feedback`;
- the partner's daily feedback;
- the knowledge probe and exposure;
- `oracle`, and the daily swap / swap-same / placebo controls;
- `coop.learning` (curves and contrasts).

Config: `configs/coop/days_persona.yaml` (8 conditions). Plan dates and stop points are not
binding (user, 2026-11-17); deviations are listed in `docs/coop-setting.md`.

### 3.13 LLM persona partner: built, gates not yet run (plan v2, work packages 1–2, 2026-11-17)
Built: rule library (10 rules), 4 pilot cards + neutral, predicates, rule-based twin, LLM persona
partner (intention state, reminders), runner integration, rule egos, gate tools, tests
(`docs/coop-setting.md`, "LLM persona partner"). Old configs are unchanged (mock check).

**Twin check** (scripted steady ego, open2_env, 2 orders):
- **Full view:** adherence ≥ 95% per card; single rules 85–100%.
- **Radius 2 (the gate setting):** lower for `order_priority` (57–90%), because the twin acts
  on its own view while the predicates use the full state.
- **Rare rules** (opportunities per 8 episodes): `on_contention` 1–7 (real loading races are
  rare), `wants_supplies` 0–1 (only the stew has two inputs), `handover_spot` 2–10 (nobody needs
  to hand over in an open kitchen). They may be unmeasurable and unlearnable with this menu
  and layout; the gate report must say so.
- **Distinctness:** 88% (chance 25%).

**Preliminary Gate 2 with the twin** (radius 2; seeds 0–3, then re-measured on 4–7):

| | seeds 0–3 | seeds 4–7 |
|---|---|---|
| aligned − generic, alignment | +21 ± 6 | +28 ± 5 |
| aligned − generic, score | −25 ± 12 | −14 ± 5 |
| aligned − misaligned, score | +38 ± 23 | +61 ± 21 |

- Doing what this partner wants costs points against the reactive generic ego, worst for p3,
  whose serving bottlenecks.
- A wrong partner model is very costly: p2 +150 to +170, where nobody serves.

This was the plan's open point §15. Decided 2026-11-17: score is primary, alignment secondary (§3.15).

### 3.12 Replication of §3.11 (`days_model_v2_rep`, run seeds 4–7, 2026-11-17)

| comparison (paired, SE over runs) | v2 (seeds 0–3) | **replication (4–7)** |
|---|---|---|
| tom_memory − direct | +6.1 ± 2.5 | +10.3 ± 3.3 |
| tom_struct − direct | +16.7 ± 3.4 | +13.8 ± 2.9 |
| tom_struct_model − direct | +23.5 ± 3.6 | +16.9 ± 3.7 |
| tom_struct − tom_memory | +10.6 ± 3.3 | +3.5 ± 2.5 |
| tom_struct_model − tom_memory | +17.4 ± 3.0 | +6.6 ± 3.6 |
| swap − tom_struct_model | about −9 | **+0.7 ± 3.2** |
| placebo − tom_struct_model | about −11 | **−2.4 ± 2.2** |

- **Replicates:** structured ToM helps (+14 to +17 vs direct), and adding the file helps a little more.
  The ordering direct < tom_memory < tom_struct ≤ tom_struct_model holds in both runs.
- **Does not replicate: the partner-specific memory effect.** In the replication, another partner's
  file does as well as the own file. The pooled estimate is about −4 (swap) and −7 (placebo), but
  the two runs disagree.
- **This fits what the files contain (§3.11):** a division-of-labour plan that is nearly the same
  for every partner, plus a station quirk every persona shares. With these scripted partners there
  is little partner-specific knowledge to gain.
- **One behavioural sign:** the own file leaves the carrot soup to `favourite_dish` (ego share 13%)
  and the swapped file does not (29%), +16 ± 3. In v2 this difference was absent (+1 ± 12).
- **Reads and probe:** the same as v2. The probe names no habit.
- **Conclusion for the paper:** structured ToM (reading the partner each step and carrying the
  read forward) is the robust gain. Partner-specific learning across days is not shown with the
  scripted partners. This motivates the LLM-simulated personas, which can differ in more ways than
  order choice.

### 3.11 Partner-model file fixed (`days_model_v2`, 2026-11-16)
Same setting and conditions as §3.10, with the file now written after every episode (100%, about
210 words), in its own call framed around the partner's choices. Mean ± SE over 20 runs.

| condition | vs direct | vs tom_memory |
|---|---|---|
| tom_memory (free ToM + notes) | +6.1 ± 2.5 | - |
| tom_model (free ToM + file) | +4.8 ± 3.1 | −1.3 ± 3.3 |
| tom_struct (structured ToM, no memory) | +16.7 ± 3.4 | +10.6 ± 3.3 |
| **tom_struct_model** (structured ToM + file) | **+23.5 ± 3.6** | **+17.4 ± 3.0** |
| tom_struct_model_swap (another partner's file) | +14.7 ± 3.8 | +8.6 ± 3.5 |
| tom_struct_model_placebo (neutral file) | +12.3 ± 2.4 | +6.2 ± 2.8 |

- **Structured ToM + the agent's own partner file is the best condition so far,** about 17 points
  per episode above tom_memory.
- **The file only helps together with structured ToM.** With free ToM it does nothing (tom_model
  ≈ tom_memory), and tom_model again has many repeated invalid actions (123 fallbacks).
- **The own file beats other files.** Swap and placebo fall back to about the tom_struct level or
  slightly below: about −9 and −11 vs tom_struct_model (from the vs-direct numbers; the paired SE is
  printed by `coop.analyze` from 2026-11-16). This is the clearest partner-specific memory effect so
  far, but from one run.
- **Not visible in the reads.**
  - Order accuracy is about the same with or without the file (68–80%) and does not rise over days.
  - The swapped file lowers it by about 3–5 points for every partner; placebo does not.
  - Next-action accuracy is 2–4 points higher with the own file.
  - So the file helps mostly through "what I should do with them", not by tracking the partner's
    order better.
- **The largest gain is again with `steady`** (+37.5 vs direct), which has no order habit. The gain
  is not specific to the habit partners.
- **First sign of the probe naming a habit:** `favourite_dish` is identified 0.25 with
  tom_struct_model, vs 0.00–0.05 for the conditions without a file. Placebo gets 0.12.
- **No learning curve over days** in the score: tom_struct_model −14.5 / −21 / −21 / −25.5 / −10
  vs the reference, episode SE about 7.
- **What the files say** (seed 0, days 1/3/5):
  - **Real models of the partner, revised over days.** For example, `steady` day 5: "highly flexible…
    contradicting previous assumptions of strict station loyalty".
  - **Mostly a division-of-labour convention:** "I handle the simple boiled dishes on pot1; they do
    the complex dishes on pot0; don't touch pot0 / chopping_board0". It is nearly the same for
    every partner. That explains why `steady` gains most, and why another partner's file still
    keeps much of the gain.
  - **"Station loyalty" is a quirk of the scripted partner,** not a persona trait. When several
    stations of a kind are free it always takes the first in map order (`_select_station`), so
    pot0 / chopping_board0 / oven0. The agent found a real regularity, but one every persona
    shares, so a swapped file is still right about it.
  - **Partner-specific content is partial:**
    - `favourite_dish`: "exclusively handles complex dishes (Baked Carrot Soup, Stew)". The soup
      is right; it over-generalises to "complex".
    - `newest_first`: the habit is never named, only station anchoring and dish types.
  - **Self-fulfilling:** "they do the complex dishes" partly follows from the agent's own decision
    to take every simple dish.
  - **Word-limit trimming dropped the last sections** (e.g. `favourite_dish` day 5 has an empty
    "What I should do with them"). `bound_words` keeps lines in order until the limit, so the
    advice at the end is the first thing lost. `coop.analyze` now reports the trimmed share.
    Fixing this changes play, so do it after the replication.
- **To check:** the paired controls (rerun `coop.analyze`) and the replication with new seeds.

### 3.10 Structured ToM and partner-model file (`days_model`, 2026-11-14)

**The file results here are not valid.** In most episodes the file was never written: the
reflection reply was strict JSON with markdown inside, and raw line breaks made it fail to parse.
`newest_first` seed 0 had no file on any day, and `favourite_dish` only from day 5. When a file was
written, the probe's trait list and pace line leaked into it ("Speed: very fast; 87%…"). That is
bookkeeping, not a model of the partner.

**Fixed 2026-11-14:**
- parsing allows raw line breaks;
- the file is rewritten in its own call, framed around the partner's goals, choices and reactions,
  without the probe's labels;
- `coop.analyze` reports how often the file was written.

Rerun in a new folder. The structured-ToM results (`tom_struct`) do not depend on the file and
stand.
Setting as `days_orders` (`open2_env`, 2 open orders, radius 2, `ORDER_PERSONAS` × 4 seeds,
20 runs). Mean ± SE over runs.

| condition | vs direct | vs tom_memory |
|---|---|---|
| tom_memory (free ToM + notes) | +7.2 ± 3.9 | - |
| tom_model (free ToM + file) | +8.4 ± 3.4 | +1.2 ± 4.0 |
| **tom_struct** (structured ToM, no memory) | **+13.0 ± 3.2** | +5.8 ± 3.7 |
| tom_struct_model (both) | +10.3 ± 2.6 | +3.1 ± 4.2 |
| tom_struct_model_swap | +6.4 ± 2.6 | −0.8 ± 3.4 |
| tom_struct_model_placebo | +6.5 ± 2.9 | −0.7 ± 4.3 |

- **Structured ToM is the best condition,** with no memory at all. The difference from
  tom_memory (+5.8 ± 3.7) is not significant.
- **The partner-model file does not help.**
  - The file ≈ the notes (tom_model − tom_memory +1.2).
  - Structured ToM with the file is no better than without it (10.3 vs 13.0).
  - The agent's own file beats swap / placebo files by about 4 (not significant).
- **File conditions get worse over days:** tom_struct_model −14.5 on day 1 → −41.5 on day 5,
  and swap / placebo also end around −42. Episode-level SE is about 8, but `direct` and
  `tom_struct` stay flat.
- **The own file brings repeated invalid actions:** tom_model 122 fallbacks and tom_struct_model
  55, mostly `put_obj_in_utensil(pot)` tried twice while the pot cannot take items. That compares
  with 8–18 fallbacks for the other conditions, including swap / placebo. The files probably hold
  rules that push the agent to the pot. Not yet inspected.
- **Reads** (structured ToM, about 3,000 per condition and persona):
  - **order accuracy 70–81%:**
    - `newest_first`: 74–80% vs 53–54% for "always the oldest order". This is the one case
      where the agent clearly tracks the partner.
    - Partners that usually work on the oldest order: 73–81%, *below* the oldest baseline
      (82–89%).
  - **next action:** exact 40–49%, verb 67–74%. The repeat baseline (0%) is too weak to judge this.
  - **No learning:** reads improve over days only as much as the baseline does, and the file
    (own, swap or placebo) does not change accuracy. Memory does not make the reads better.
- **Within-episode adaptation:** tom_struct leaves the carrot soup to `favourite_dish` (ego share
  11% vs 24% for direct, −14 ± 7; vs tom_memory −23 ± 5). The file variants show it too, so it
  comes from watching during the episode, not from memory.
- **The order-taking pattern of §3.7 did not replicate:** the swap now moves `oldest_first` and
  `newest_first` the opposite way.
- **The probe's `orders` field is still "no fixed habit"** for every habit partner.

### 3.8 How ToM and memory were implemented so far (important for reading §3)
- **`tom`:** one or two free sentences about the partner before each action. They are never shown
  again, so nothing builds up within an episode.
- **`memory`:** the last 5 end-of-episode reflections (fixed profile options + 80 words) as a list.
  Nothing is revised or merged, and evidence older than 5 episodes drops out.
- **Added 2026-11-13** (`tom_mode: structured`, `memory_mode: model`; config `days_llm_model.yaml`,
  not run yet):
  - a structured read each decision (`doing` / `order` / `next` / `habit`), carried to the next
    decision and scored offline against the partner's true order and next action;
  - one partner-model file the agent rewrites after every episode.
- **Placebo text fixed 2026-11-13:** one placebo sentence said orders open "one at a time". That is
  false with 2 open orders, so the placebo in `days_orders` / `days_orders_full` included one wrong
  rule. With several open orders it now says "Orders are listed oldest first, and a correct
  delivery opens a new one." One-order runs keep the original (true) sentence.
- **One-order runs are unchanged by the 2026-11-12/13 work.** Every prompt is word for word the
  same as at `15eaeea`, checked by playing the same config on both commits with the mock backend.
  So `days_probe_open` (started at `15eaeea`) can be finished with `--allow-code-change`.

## 4. Current conclusion

ToM + memory consistently improves an LLM's teamwork with a stable simulated human:
- **With order habits and two open orders** (§3.7), it helps only under a partial view:
  +11.4 ± 4.3 more than with a full view.
  - Under a partial view, the order the ego takes shifts with the partner's habit, and swapped
    notes shift it the donor's way (small effects).
  - The score gain is largest with `steady`, though, so its source is not yet known.
  - The probe never names an order habit.
  - **Structured ToM is the robust gain** (+14 to +17 vs direct in two runs). The agent's own
    partner file adds a little (tom_struct_model +17 to +24).
  - **Partner-specific memory is not established:** swap / placebo lost 9–11 in the first run and
    nothing in the replication (§3.12). The files mostly hold a generic division of labour.
- **When the map fixes the roles,** the gain is general skill.
- **When roles are free** (open kitchen, one order; replicated three times, §3.4 and §3.9):
  - ToM gives about +8, and the agent's own notes add about +5.
  - Swapped or neutral notes lose that extra part, so it needs notes about *this* partner.
  - *Which* behaviour carries it is not stable. The `serve_first` hand-over was partner-specific
    in one run (§3.5) but came from ToM alone in the next (§3.9).

## 5. Caveats

- **One model (Qwen3.8-27B).** Effects are a few points per episode. 2–4 seeds. Many comparisons:
  only the `serve_first` effects are large enough to survive correction for multiple comparisons.
- **Probe labels (fixed 2026-11-12; the runs in §3 used the old probe):**
  - **Old problems:** preference partners were called "slow" (waiting for preferred work looks
    like slowness); `avoids` asked about utensils no order needed (e.g. the blender); abandoning
    (`flaky`) was never detected.
  - **New rules:** `avoids` only over needed utensils both players reach, so it is not asked on
    `new_env`. `speed` comes with an observed pace line, and waiting is defined as a preference.
    `flaky`'s commitment is not scored, and commitment means visibly unfinished work.
  - **Old folders:** `coop.analyze` rescores them with the new `avoids`/`commitment` rules. Their
    speed answers are still the old ones (no pace line), so new speed accuracy is not comparable.
  - Speed accuracy now measures using an observed cue, not inferring pace unaided.

  For the runs in §3, quote the preference label and the stage shares, not the other fields.
- **Residual agent errors:**
  - lifting dish-served food by hand: about 51% of invalid replies on the split map;
  - full hands / starting with full hands: most invalid replies on open maps.
- **Not comparable across code versions.** Runs are made with different code versions, and
  `code_version.json` guards resumes, so compare only within one output folder. The movement fixes
  of 2026-11-10 changed open-map scripted results slightly. Runs before `9a05823` used 4 probe
  fields; later runs use 6.

## 6. Next steps (in order)

1. ~~Probe label fixes~~ (done 2026-11-12). The pace line and the new options change the
   reflection prompt, so memory conditions can behave differently: new runs need a new `--out`.
2. **Order-habit partners**: implemented 2026-11-12 (`ORDER_PERSONAS`, the probe's `orders` field, and the
   "order habits" table in `coop.analyze`). Still to run: the value screen with 16+ seeds, then
   `configs/coop/days_llm_orders.yaml` (`open2_env`, 2 open orders, radius 2, all controls).
3. **A second model**, to check generality.
0. **Design reference: `docs/PERSONA_PARTNER_PLAN_v2.md`** (dates and stop points not binding).
   Built: the persona partner and gate tools (§3.13), and the multi-day learning setup (§3.14).
   Score is the primary outcome (§3.15). Next: Gate 1 (running), Gate 2 with the LLM partner,
   and `oracle` vs `tom_struct` with the LLM ego (score value of knowing the partner). Then
   either the pilot with `configs/coop/days_persona.yaml`, or first rules whose violation costs
   score (§3.15).
4. **The real partner (decided 2026-11-13):** the scripted `SimulatedHuman` and its probe are a
   test harness for the pipeline, not the partner for the actual experiments. Don't refine the
   scripted personas or probe labels further. The experimental partner will be an LLM-simulated
   human, designed from papers that use LLMs as simulated humans or partners. Then partners that
   change over days, and human partners as a realism check.

## 7. Where things are

| what | where |
|---|---|
| setting, all commands | `docs/coop-setting.md` |
| env, orders, partial view | `src/coop/env.py`, `orders.py`, `view.py` |
| simulated human, partner types | `src/coop/partner.py` (`PERSONAS`, `DEFAULT_PERSONAS`) |
| LLM agent, prompts, probe | `src/coop/llm_agent.py`, `executor.py`, `kitchen.py`, `profile.py` |
| multi-day runner, controls | `src/coop/days.py`; configs in `configs/coop/` (`days_llm.yaml`) |
| analysis | `src/coop/analyze.py`; two folders: `src/coop/compare.py` |
| scripted screens | `src/coop/screen.py` (partner types), `src/coop/voi.py` (value of knowing) |
| serving script | `scripts/run_qwen38_days.sh` |
| order-habit LLM config | `configs/coop/days_llm_orders.yaml` |
| server output folders | `~/work/collab-overcooked/outputs/days_llm_v2`, `days_llm_open`, `days_llm_open2`, `days_ctrl_split`, `days_ctrl_open`, `days_ctrl_open_rep` |
