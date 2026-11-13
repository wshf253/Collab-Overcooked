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

### 3.5 Division of labour (open replication)
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
  rule. It now says "Orders are listed oldest first, and a correct delivery opens a new one."

## 4. Current conclusion

ToM + memory consistently improves an LLM's teamwork with a stable simulated human:
- **With order habits and two open orders** (§3.7), it helps only under a partial view:
  +11.4 ± 4.3 more than with a full view.
  - Under a partial view, the order the ego takes shifts with the partner's habit, and swapped
    notes shift it the donor's way (small effects).
  - The score gain is largest with `steady`, though, so its source is not yet known.
  - The probe never names an order habit.
- **When the map fixes the roles,** the gain is general skill.
- **When roles are free,** part of the gain is partner-specific. The clearest case is a
  serve-preferring partner: the agent recognises the preference, gives the serving work to the
  partner, and scores better. These line up only with memory about *that* partner.

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
4. **Later:** partners that change over days; an LLM-simulated or human partner as a realism check.

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
