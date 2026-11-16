# Plan v2: LLM persona partner and multi-day partner learning

Project: self-evolving collaborative agent, ICML 2027 (abstract Jan 14, paper Jan 22, 2027).
Repo: `Collab-Overcooked`, branch `claude/lucid-babbage-96q0zi`, research code in `src/coop`.
Written 2026-11-17 for Claude Code. Read `CLAUDE.md` and `RESEARCH_STATUS.md` first.

This file replaces three earlier documents: `PERSONA_PARTNER_PLAN.md` (Nov 14),
`LLM_PERSONA_EXPERIMENT_PLAN.md` (Nov 16) and the project's `IMPLEMENTATION_PLAN.md`
(Nov 4, written for OvercookedV2). Where they disagree with this file, this file holds.

## 0. Why this design

The scripted-partner tests (`RESEARCH_STATUS.md` §3) gave four lessons that shape everything
below:

1. Structured ToM plus the ego's own partner file is the best condition (§3.11), so the
   method works mechanically.
2. What the ego learned was one convention shared by every scripted persona (station order),
   learned in a single episode. There was no curve over days.
3. A reactive ego is near-optimal against the scripted personas (`coop.voi`), so there was
   little to gain from knowing the partner.
4. The cross-partner swap cannot tell knowledge of the partner from the run's own convention.

So the main experiment needs partners that (a) have several conditional preferences that
cannot all be seen in one episode, (b) change what the ego should do, and (c) come with
ground truth that can be checked by rule.

## 1. Decisions (do not reopen)

| Item | Decision |
|---|---|
| Partner | LLM persona. Fixed across days: same card, prompt, model and decoding; no memory of the ego across episodes. Reactions within an episode are allowed. |
| Persona content | A set of conditional rules chosen from a fixed rule library, plus a backstory for voice. |
| Models | Ego Qwen3.8-27B first. Partner Qwen3.8-27B first, then Gemma 4 31B. Frontier API ego later. |
| Run shape | 5 days × 3 interaction episodes, plus 2 held-out evaluation episodes after each day. 300 steps, `open2_env`, 2 open orders, `view_radius: 2`. |
| Feedback | The partner gives feedback at the end of each day. A no-feedback condition is the baseline. |
| Ego memory | One partner file with four sections: Partner Preferences, Partner Reactions, Cooperation Strategy, Uncertainties. |
| Metrics | Task score, preference alignment (rule-computed), knowledge probe, partner-prediction accuracy. |
| Communication | None during an episode. |

## 2. Invariants

1. Existing configs behave exactly as before. New behaviour is opt-in through new config
   keys whose defaults reproduce the current code. Check with the mock backend that
   `days_llm_mock.yaml`, `days_llm_orders.yaml` and `days_llm_model.yaml` give identical
   steps and prompts before and after.
2. No leakage. Outside the oracle condition the ego's prompts never contain the persona
   card, rule ids, rule text, the partner's intention state, alignment counts or probe
   scores. Add a test that builds every condition's prompts and searches for these strings.
3. Paired design. Seeds, order lists and persona assignment are identical across conditions.
4. Infrastructure failures (parse failures, invalid actions, timeouts) are logged per
   player and per condition, and reported next to every result.
5. Persona adherence is a diagnostic, not a filter. Never drop runs after seeing outcomes.
6. Do not edit the test personas in `coop.partner` or the old probe. New partner code goes
   in new modules.
7. New runs use new `--out` folders. `code_version.json` must cover the new files.
8. Record model ids and revisions, serving version, decoding settings, and a hash plus the
   full text of every prompt template in the run folder.

## 3. What exists and what is new

| Need | Status in the repo |
|---|---|
| Continuous orders, 2 open orders, partial view | Exists (`env.py`, `view.py`) |
| LLM player stack: observation text, macros, executor | Exists (`llm_agent.py`, `executor.py`, `kitchen.py`) |
| Structured ToM, partner file, swap and placebo controls | Exists (`tom_mode: structured`, `memory_mode: model`) |
| Multi-day runner, paired seeds, analysis | Exists (`days.py`, `analyze.py`, `compare.py`) |
| Partner choice in the runner | **New.** `play_episode` builds `SimulatedHuman` directly; add a partner factory |
| Rule library, persona cards, rule-based twin | **New** |
| LLM partner | **New** |
| Day schedule, controlled order lists, held-out evaluation | **New** (the config has one `menu`; a day is only a label today) |
| Day-end feedback and consolidation | **New** (`Agent.end_day` is an empty hook) |
| Preference alignment, knowledge probe | **New** |

## 4. Personas

### 4.1 Rule schema (`src/coop/rules.py`, `configs/coop/rules/library.yaml`)

```yaml
own_station:
  category: station
  params: {station: [pot0, pot1, oven0, oven1, chopping_board0, chopping_board1]}
  text: "You always use {station} and think of it as yours. You don't like your partner
         loading it or taking food out of it."
  wants: "Use the other {kind}; leave {station} to me."
  fallback: "If {station} is busy with your own batch, use the other one."
  opportunity: partner_needs_kind_and_both_free      # predicate name
  adherent: partner_targets_station                  # check on the partner's decision
  aligned: ego_does_not_touch_station                # check on the ego's decision
```

Every rule has: a condition, a tendency, what it wants from the ego, a fallback, a
second-person text for the partner prompt, and three predicates over the full evaluator
state: `opportunity` (does the rule apply to this decision), `adherent` (did the partner
follow it), `aligned` (did the ego respect it). Predicates are plain functions over the
state, the step record and both players' current macros.

### 4.2 Library v1 (10 rules; implement in this order)

| Id | Category | Tendency of the partner | What it wants from the ego |
|---|---|---|---|
| `own_station(s)` | station | Always uses station `s` | Use the other one of that kind |
| `solo_dish(d)` | dish | Makes dish `d` alone, start to finish | Do not add to, plate or serve `d` |
| `split_dish(d)` | dish | For dish `d`, prepares and loads, then leaves it | Start, plate and serve `d` |
| `order_priority(oldest/newest)` | order | Takes that open order first | Take the other open order |
| `serves(self/ego)` | stage | Plates and delivers itself / never plates | Leave serving to me / do all the serving |
| `wants_supplies` | stage | Stays at its station while a batch is incomplete | Bring the missing raw ingredients to the nearest counter |
| `handover_spot(c)` | hand-over | Puts items to pass on counter group `c` | Put items for me on `c` |
| `on_takeover` | reaction | If the ego takes its unfinished item, abandons that order and switches | Do not take my unfinished items |
| `on_contention(yields/insists)` | reaction | When both head for one station, steps back / carries on | Take it / give way |
| `waits_for_commit` | initiative | At a new order, waits up to 5 steps to see what the ego starts | Commit quickly; don't wait for me |

Shared priority, stated in the partner prompt and used by the twin: valid action and no
deadlock, then finishing the current commitment, then the persona's rules in card order,
then "do what is needed next, oldest order first". Rules are tendencies with a fallback,
never prohibitions.

Contradictions must be rejected when a card is loaded (for example `solo_dish(d)` with
`split_dish(d)`, or `serves(ego)` with `solo_dish`).

### 4.3 Persona cards (`configs/coop/personas/batch1.yaml`)

```yaml
p1:
  backstory: "Ten years in a bakery. Methodical, dislikes being interrupted mid-task."
  rules:            # ordered; the first K are active at complexity K
    - own_station: {station: oven0}
    - solo_dish: {dish: baked_carrot_soup}
    - order_priority: {which: oldest}
    - serves: {who: ego}
    - handover_spot: {group: near_serving}
    - on_takeover: {}
```

- Pilot: 4 hand-assembled cards with 6 rules each. Any two cards differ in at least 3
  rules, and each rule category appears in at least two cards with different parameters,
  so a swapped file is wrong about something.
- Complexity K ∈ {2, 4, 6} uses the first K rules of the same card (nested), so K is
  varied without changing the persona.
- Backstories are 2–4 sentences for voice only. Draft them with an LLM once, review by
  hand, then freeze. They must add no behaviour that is not in the rules.
- Later (not in the pilot): expand to 8 cards by letting an LLM select rules from the
  library to fit a generated profile, under the same constraints.

### 4.4 Rule-based twin (`src/coop/rule_partner.py`)

A scripted partner that follows a card deterministically, built on the job selection of
`SimulatedHuman` (reuse by composition or subclass; do not edit its personas). It is needed
for three things: checking the predicates (adherence must be ≥ 95% by construction), the
headroom gate, and tests without a model.

## 5. LLM partner (`src/coop/llm_partner.py`)

- Shares the ego's stack. Extract the shared parts of `LLMAgent` into a base class
  instead of copying. The partner has no ToM block, no cross-episode memory and no probe.
- Interface used by the runner: `reset(mdp, index, seed, view_radius)`, `act(state, t)`,
  `summary()`, `events`, `trace`, and it receives rejected actions.
- System prompt: the shared rules, map, recipes and action grammar, then the backstory and
  the active rules in second person, then the priority order.
- **Intention state.** The reply carries `{"goal", "order", "reconsider_when", "action"}`.
  The previous goal is shown at the next decision so the partner keeps or revises it. Reset
  at each episode start.
- **Opportunity reminders** (`remind: true`, default on for LLM partners). At each
  decision, append one line naming the active rules whose `opportunity` predicate is true
  now. Log every reminder. Keep `remind: false` runnable so the effect can be reported.
- **Action filtering** (`enforce: false`). Leave the hook; implement only if Gate 1 fails.
  Results with filtering are a separate, labelled simulator variant.
- `trace` logs per decision: t, goal, order, macro, canonical stage, the rules whose
  opportunity fired, and adherence for each. The ego's structured reads are scored against
  this trace as they are now.
- Backend: `partner.backend`, with its own env vars (`ICRL_PARTNER_BASE_URL`,
  `ICRL_PARTNER_MODEL`, `ICRL_PARTNER_API_KEY`). Thinking off, temperature 0 for the pilot.
- A **neutral partner** is the same class with an empty card. It is the competence baseline.
- Usage is logged separately from the ego's.

Config:

```yaml
partner:
  type: llm            # scripted (default, current behaviour) | rule | llm
  cards: configs/coop/personas/batch1.yaml
  complexity: 6
  remind: true
  backend: {...}
```

`personas:` keeps listing names; they are resolved by `partner.type`.

## 6. Days, order lists and held-out evaluation (`days.py`, `orders.py`)

```yaml
days: 5
episodes_per_day: 3
eval_episodes: 2            # after each day, on a frozen copy of the memory
schedule:                   # menu per day for the interaction episodes
  1: [boiled_egg, boiled_mushroom, boiled_corn_slices]
  2: [boiled_egg, boiled_mushroom, boiled_corn_slices, baked_carrot_soup]
  3: all
  4: all
  5: all
```

- **Controlled order lists.** Replace the random draw by a seeded list that contains each
  scheduled dish a fixed number of times in shuffled order, repeated to the horizon.
  Random draws stay the default for old configs.
- **Staggered exposure.** Rules tied to a dish or a station kind first become observable
  on a known day. Log, per rule and per episode, how many opportunities occurred inside
  the ego's view (`exposure`).
- **Held-out evaluation.** After each day's consolidation, play the same `eval_episodes`
  (same seeds every day, full menu) with a frozen copy of the ego's memory. No memory
  update, and nothing from these episodes reaches later prompts. Also play them once
  before day 1 with empty memory (day 0). The learning curve is the held-out result on
  days 0–5. Because the episodes are the same each day, the curve is paired across days.
- Conditions with no cross-day memory give the same held-out result every day: run them
  once per seed.
- Resume: evaluation episodes, feedback and consolidation are each written once and read
  back on resume. A partly finished day end must not regenerate feedback or consolidate
  twice.

## 7. Ego memory (`llm_agent.py`, new `memory_mode: model_daily`)

Existing modes stay unchanged. The new mode has three timescales:

| When | What | Kept until |
|---|---|---|
| Each decision | Structured read of the partner (existing `tom_mode: structured`) | End of episode |
| End of episode | Short note of joint evidence: what I did, what they did, what followed | End of day |
| End of day | One call rewrites the partner file from the old file, the day's notes and the feedback | Persistent |

File format:

```
## Partner Preferences
- When <condition>, <claim>. Evidence: +D1E2, +D2E1, -D2E3. Source: observed | feedback

## Partner Reactions
- When I <action>, they <response>. Evidence: ...

## Cooperation Strategy
- <what I do>. Follows from: <entry>

## Uncertainties
- <question>; what observation would settle it
```

- A word budget per section (default 80 / 60 / 60 / 40). Trim within a section, never
  across sections, and log what was trimmed.
- The consolidation prompt asks for conditions and evidence tallies, and asks that every
  strategy line cite the entry it follows from.
- Placebo file: same headings and length, neutral content.

## 8. Daily feedback (`src/coop/feedback.py`)

- Called after the last interaction episode of a day, before consolidation.
- Input to the partner model: its card and backstory, the day's score, and its own log of
  the day (its actions, public events, the ego actions it saw).
- Output: at most 60 words in the persona's voice, about what happened that day.
- It may ask for a change in the ego's behaviour. It must not list its rules. Claims about
  events must match something in its log; log unsupported claims and rule recitation
  separately (string check for rule text, plus a manual sample).
- Levels: `none` (no call) and `specific`. `vague` is deferred.
- Passed on as `agent.end_day(day, results, feedback=text)`; extend the base signature
  with `feedback=None`.

## 9. Measurements

| Measure | Definition |
|---|---|
| Score | Score minus the scripted reference on identical orders (existing); completed and wrong orders |
| Preference alignment | Per rule: aligned ego decisions / opportunities, where an opportunity also requires that `Executor.options` on the full evaluator state offered an aligned action. A rule with no opportunity is undefined, not perfect. Composite = mean over evaluable rules; report coverage |
| Partner adherence | Per rule: adherent partner decisions / opportunities (simulator quality) |
| Knowledge probe | See below |
| Partner prediction | Structured reads scored against the partner trace (existing), with a stronger baseline: the most frequent next action given the partner's held item |
| Exposure | Opportunities per rule seen by the ego, cumulative by day |
| Activity | Ego actions per episode and wait share, so an idle ego cannot pass as aligned |
| Cost | Calls, tokens and latency per player |

**Knowledge probe** (`src/coop/probe2.py`). After each day's consolidation, on the frozen
memory, ask one multiple-choice question per library rule, including rules this persona
does not have ("When both pots are free, which does your partner use? pot0 / pot1 / no
preference"). Score against the card. Report separately: rules exposed so far, rules not
yet exposed, and rules the persona lacks (correct answer: no preference). The probe is a
forked call; its answers never enter memory or later prompts.

Statistics: the unit is the run (persona × seed). Paired differences with SE over runs
and bootstrap intervals; episodes inside a run are never treated as independent. Always
report per-persona results as well as the mean.

## 10. Gates before any learning run

Run with partner-only tooling (`src/coop/persona_check.py`), scripted `steady` ego.
4 cards × 4 seeds × 3 episodes, for the twin and for the LLM partner.

**Gate 1: the partner is usable.**
- Competence: score within a set margin of the neutral LLM partner.
- Adherence per rule (twin ≥ 95% as a predicate check; LLM partner: calibrate the bar on
  seeds 0–1, freeze it, then judge on seeds 2–3; starting bar 70%).
- Distinctness: leave-one-seed-out classifier over the cards from per-episode features
  (station use, order choice, stage shares, hand-over cells, wait share).
- Consistency: adherence does not fall across episodes.

**Gate 2: knowing the partner matters.** Extend `coop.voi` with scripted egos:
`aligned(card)`, `generic` (the current `steady`), `misaligned(other card)`.
- Report aligned − generic and aligned − misaligned for alignment and for score, against
  the twin and the LLM partner, selected on one seed set and re-measured on another.
- Alignment headroom is partly true by construction. Score headroom is the informative
  result: report it honestly even if it is zero.

**Stop after the gates and report.** Give the commands and the tables. Do not start
section 11 until SJ has seen the result. If a rule fails adherence, fix its wording, then
try `enforce`, then drop the rule. Keep a log of every revision.

## 11. Conditions

All use structured ToM and the same observation, executor and action settings.

| Condition | Cross-day memory | Feedback | Purpose |
|---|---|---|---|
| `tom_struct` | none | none | Within-episode inference only |
| `memory` | own file | none | Learning from behaviour alone |
| `memory_fb` | own file | specific | Learning with feedback |
| `latest_fb` | only the previous day's feedback, no file | specific | Is accumulation needed, or just the last advice? |
| `oracle` | the card's "wants" lines, no file | none | Can the ego use the knowledge at all? Not an upper bound |
| `swap_other` | file of another card, same day | specific | Is the content partner-specific? |
| `swap_same` | file of the same card from another seed | specific | Partner knowledge vs the run's own convention |
| `placebo` | neutral file, same length | specific | Is it only longer context? |

- Swapped and placebo files replace the ego's file at the start of each day, using the
  donor's file after the previous day. The recipient still receives its own partner's
  feedback. Validate that all donor checkpoints exist before starting dependent runs.
- `oracle` is the one exception to invariant 2.
- Pilot: `tom_struct`, `memory`, `memory_fb`, `oracle`, `swap_other`; 4 cards × 3 seeds,
  K = 6. About 25 episodes per run, 60 runs.
- Main: all eight conditions, 4 seeds, then the K sweep (2, 4, 6) on `tom_struct`,
  `memory_fb` and `oracle` only.

## 12. Models

| Step | Ego | Partner |
|---|---|---|
| Pilot and main | Qwen3.8-27B | Qwen3.8-27B |
| Partner generalization | Qwen3.8-27B | Gemma 4 31B (rerun Gate 1 first, same cards) |
| Ego generalization | Frontier API model | The partner with the better adherence |

Check Gemma's serving support and memory use beside the ego before scheduling. Use the
existing serving script pattern; the partner endpoint is a second server.

## 13. Order of work and dates

| Dates | Work | Output |
|---|---|---|
| Nov 17–22 | W1: rule schema, first 6 library rules with predicates, twin, partner factory, `LLMPartner` with intention state and reminders, mock tests, old-config equality check | One commit per item |
| Nov 23–29 | W2: remaining 4 rules, alignment metric, `persona_check`, `voi` egos; run Gates 1 and 2 on Qwen | **Nov 29: gate report, stop** |
| Nov 30–Dec 6 | W3: schedule and order lists, held-out evaluation, `model_daily`, feedback, knowledge probe, `days_persona.yaml`; run the pilot | Pilot curves |
| Dec 7–20 | W4: main run, all conditions; Gemma Gate 1 and the Gemma partner run | Main tables |
| Dec 21–Jan 3 | W5: K sweep, frontier ego, extra seeds where SE is large | |
| Jan 4–22 | Analysis and writing | Abstract Jan 14 |

After each work package: tests pass, `docs/coop-setting.md` and `RESEARCH_STATUS.md` are
updated, and the reply ends with the pull and `lab-run` commands, as `CLAUDE.md` asks.

Background, not blocking: the replication of `days_model_v2` with new seeds.

## 14. Decision rules for reading results

| Observation | Next step |
|---|---|
| Aligned ≈ misaligned scripted ego on score and alignment | The rules do not matter in this kitchen; change rules or layout before any learning run |
| `oracle` ≈ `tom_struct` | The ego cannot use the knowledge; move the ego to a stronger model before tuning memory |
| `memory_fb` ≈ `latest_fb` | The gain is recent advice, not accumulated knowledge |
| `swap_same` ≈ `memory_fb` and `swap_other` < `memory_fb` | Partner-specific knowledge (the result we want) |
| `swap_same` < `memory_fb` as much as `swap_other` | The file encodes the run's own convention, not the partner |
| Probe accuracy rises with exposure but alignment does not | Knowledge is acquired but not used; inspect the strategy section |
| Curve is flat after day 1 at K = 6 | Too easy; check exposure and consider more rules before more days |

## 15. Open points for SJ

- Whether score or preference alignment is the primary outcome if Gate 2 shows alignment
  headroom but no score headroom.
- Whether opportunity reminders are acceptable as the default simulator, or only as a
  fallback.
- The margin for partner competence in Gate 1 and the word budgets in section 7.
- The frontier model for the ego.

## 16. Out of scope

Partners that drift or adapt to the ego; messages during an episode; weight updates; the
human study (later, through the browser play mode on `main`); layouts other than
`open2_env`; LLM-selected persona cards beyond the pilot four.
