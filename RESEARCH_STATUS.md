# Research status: partner modelling (ToM) and memory in Collab-Overcooked

Last updated: 2026-11-12 (branch `claude/lucid-babbage-96q0zi`, code up to commit `5563d6b`).
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
    the true persona;
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
- **This is a lower bound.** No scripted policy predicts the partner, and every partner type serves
  the oldest order first.

## 4. Current conclusion

ToM + memory consistently improves an LLM's teamwork with a stable simulated human:
- **When the map fixes the roles,** the gain is general skill.
- **When roles are free,** part of the gain is partner-specific. The clearest case is a
  serve-preferring partner: the agent recognises the preference, gives the serving work to the
  partner, and scores better. These line up only with memory about *that* partner.

## 5. Caveats

- **One model (Qwen3.8-27B).** Effects are a few points per episode. 2–4 seeds. Many comparisons:
  only the `serve_first` effects are large enough to survive correction for multiple comparisons.
- **Probe label problems:**
  - preference partners get called "slow", because waiting for preferred work looks like slowness;
  - `avoids` is scored "nothing" even when a station isn't needed by the menu (e.g. the blender);
  - abandoning (`flaky`) is never detected.

  Quote the preference label and the stage shares; not the other fields.
- **Residual agent errors:**
  - lifting dish-served food by hand: about 51% of invalid replies on the split map;
  - full hands / starting with full hands: most invalid replies on open maps.
- **Not comparable across code versions.** Runs are made with different code versions, and
  `code_version.json` guards resumes, so compare only within one output folder. The movement fixes
  of 2026-11-10 changed open-map scripted results slightly. Runs before `9a05823` used 4 probe
  fields; later runs use 6.

## 6. Next steps (in order)

1. **Probe label fixes:** score `avoids` only over stations the menu needs; separate "waiting for
   preferred work" from "slow"; make abandoning visible, or drop it.
2. **Order-habit partners** (e.g. `newest_first`, `same_dish_first`), so the best response depends on
   the partner. Then the value screen with 16+ seeds, then LLM runs with several open orders and
   radius 2, with the controls.
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
| analysis | `src/coop/analyze.py` |
| scripted screens | `src/coop/screen.py` (partner types), `src/coop/voi.py` (value of knowing) |
| serving script | `scripts/run_qwen38_days.sh` |
| server output folders | `~/work/collab-overcooked/outputs/days_llm_v2`, `days_llm_open`, `days_llm_open2`, `days_ctrl_split`, `days_ctrl_open`, `days_ctrl_open_rep` |
