# Plan v3: context-dependent partners and proactive collaboration

Project: self-evolving collaborative agent (ICML 2027).
Repo: `Collab-Overcooked`, branch `claude/lucid-babbage-96q0zi`, research code in `src/coop`.
For Claude Code. Read `CLAUDE.md` and `RESEARCH_STATUS.md` §3.13–3.19 first.

This plan builds on `docs/PERSONA_PARTNER_PLAN_v2.md` and the code that implements it. It
replaces v2's persona design (§4.2–4.3), its gates (§10) and its pilot (§11). Everything else
in v2 still holds: the invariants (§2), the LLM partner (§5), days, order lists and held-out
evaluation (§6), the partner file (§7), feedback (§8), the measurements (§9) and the
controls (§11). Where this file and v2 disagree, this file holds.

## 0. Why the design changes

Results so far (`RESEARCH_STATUS.md` §3.15–3.19):

- **The v2 gates found no score headroom.** With the LLM partner on batch1, aligned − generic
  score is −20 ± 5. With the twin, a single rule is worth −16 to +12 points out of about 230.
  Neither the waiting cards (batch2) nor the new kitchen (`open_ing2_env`) changed this.

Four causes, each addressed below:

1. **The cards describe preferences, and the aligned ego accommodated them.** Most `wants`
   lines were restrictions (leave this station, let me serve, give way). Following them costs
   the ego jobs, so v2's Gate 2 measured accommodation, not proactive collaboration.
   → Section 4: rules describe what the partner *does* in a context, and every `wants` line
   is an action for the ego to take.
2. **The partner's fallbacks absorbed every mismatch.** When the ego ignored a preference, the
   partner used the other station or finished the dish itself, so a mismatch cost almost
   nothing and nothing about the ego was measured.
   → Section 4: no conflict-absorbing fallbacks, only a deadlock guard.
3. **There was no per-order time pressure.** The 300-step horizon is a throughput limit:
   orders never expired, so a coordination mistake only shifted a delivery a few steps later.
   → Section 3.1: a timer per order, with expiry.
4. **A reactive ego was as good as an informed one.** In an open kitchen with no urgency, the
   ego can look at the current state and do the next needed step. Observation replaces
   knowledge.
   → Sections 2 and 6: contexts where the ego must commit before it can observe, and a
   proactive ego compared against the best reactive one.

The new direction (`Context_Dependent_Partner_Collaboration_Scenarios.md`): the partner
changes behaviour depending on context, and the ego must learn under which contexts it
changes. That doc's main test is kept: before scaling, show that the best ego strategy
differs between partners in the same state (a crossover), and that acting on a model of the
partner beats every strategy that ignores the partner.

## 1. Research question

How can an agent progressively learn, maintain and use a partner's context-dependent
behaviour to collaborate proactively across repeated interactions?

**Proactive collaboration** here means acting on a prediction of the partner's next behaviour
before that behaviour is observed. For example: preparing what the partner will need, or
taking the work the partner will not take. The comparison that matters is a proactive agent
against the best **reactive** agent, which decides from the current state only. The reactive
agent is the baseline to beat, not the agent under study.

Sub-questions:

1. Does accumulated partner knowledge improve cooperation beyond within-episode inference
   (structured ToM)?
2. Is the knowledge partner-specific and context-specific (swap controls, per-context
   analysis)?
3. Does it transfer to dishes not seen during learning?
4. Extension: can the agent tell a one-off exception from a lasting change (belief revision)?

Name the ego's file **structured partner memory** in docs and reports. Inferring the
partner's private beliefs is not evaluated in this plan. The per-step structured read stays
as the within-episode component.

## 2. Requirements for a context

A context and its partner rules are kept only if all of these hold:

| | Requirement | Why |
|---|---|---|
| R1 | **Crossover.** In the same context, the best ego action differs between partners. | Otherwise one policy fits every partner. |
| R2 | **Commit before observing.** The context is a public event that both players see at the same step. The ego has to choose before the partner's response is visible to it (view radius 2, macro latency). | Otherwise a reactive ego copes without knowing the partner. |
| R3 | **Costly mismatch.** A wrong choice loses an order or a batch, not a few steps. | In v2, fallbacks made every mismatch cheap. |
| R4 | **Costly waiting.** Waiting to see what the partner does costs time the order timer does not allow. | Otherwise "wait and see" is a free strategy that needs no knowledge. |
| R5 | **Dish-agnostic conditions.** Conditions refer to urgency, stations, stages and the ego's activity, never to dish names. | So knowledge can transfer to held-out dishes. |

Order timers are the main way to get R2–R4: when an order becomes urgent, both players see it
at the same step, and neither can afford to wait.

## 3. Environment additions

### 3.1 Order timers (`env.py`, `orders.py`)

- New config block `deadlines`, off by default. Old configs must behave exactly as before
  (mock-backend equality check, as in v2 §2).
- **Limit per order:** `slack × reference time` of its dish. Measure the reference time from
  steady-pair scripted runs on the layout and store the table in `docs/coop-setting.md`.
- **Visibility:** both players' observations show the steps left next to each open order.
- **Expiry (the main semantics):** when the timer runs out, the order closes,
  `expiry_penalty` points are lost (default 20), and the next order opens. Log it as
  `expired` in the order records and the summary (completed / expired / wrong).
- **Decaying reward:** a later variant only (`deadlines.mode: decay`), where the delivery
  reward falls with lateness. Implement it after the main result, not before.
- The 300-step episode horizon stays. The timers are per order, in addition to it.
- The scripted reference ego plays under the same timers, so score minus reference stays
  paired.

### 3.2 Context features (`src/coop/context.py`)

Computed at every step, both on the full state and on each player's own view. Logged in
every decision record. Used by the rules, the reminders, the analysis, the probe and
exposure counts.

| Feature | Definition |
|---|---|
| `urgent(order)` | Unfinished open order with at most θ steps left |
| `ego_on(order)` | The ego's current or last completed action serves that order |
| `blocked(kind)` | Both utensils of that kind are running or hold finished food nobody has taken |
| `prep_ready(order)` | Processed items for an open order are waiting and unused |

Predicates for the partner's adherence use the partner's own view. The v2 gates showed that
judging on the full state while the partner acts on its view lowers adherence for no real
reason.

### 3.3 Duplication cost (check, then decide)

Check what happens when both players bring the last missing ingredient to the same utensil.
If the second item is accepted, the contents match no recipe and the batch is thrown away
when started; that is a strong R3 cost. If the executor rejects it, the cost is only the
wasted trip. Report which applies. Do not change executor behaviour just to create the cost.

### 3.4 Kitchen and menu

Use `open_ing2_env`, the open kitchen with one dispenser per ingredient. Pilot menu: 3 dishes
of different length, for example `boiled_potato_slices`, `baked_carrot_soup` and
`sliced_carrot_and_potato_stew`. Hold 2 other dishes back for the transfer test
(section 8). If the Gate 0 screen shows another set works better, change it and record why.

## 4. Context rules (library additions, `category: context`)

| Id | Context | Modes | Partner behaviour | What the ego should do |
|---|---|---|---|---|
| `default_work` | No other context applies | `prep` / `whole` | Prepares ingredients for open orders and does not cook or serve / takes one order and does it start to finish | Cook and serve what it prepared / take the other order start to finish |
| `on_urgent` | An order becomes urgent | `takes` / `keeps` | Drops its work and finishes the urgent order / carries on with its own order | Keep working, e.g. prepare the next order / take the urgent order yourself |
| `on_blocked` | The utensil kind it needs is blocked | `clears` / `waits` | Plates or serves the finished food to free the utensil / does other work that doesn't need that utensil, and never clears it | Do other work; it clears / clear it yourself |
| `on_prep_ready` | Prepared items wait for an open order | `switches` / `continues` | Switches to cooking and serving them / keeps preparing | Keep preparing / cook and serve them |

- **Interaction option:** `on_urgent(takes)` has `unless_ego_on_it: [yes, no]`. With `yes`,
  the partner keeps its own work when the ego is already on the urgent order. This is the
  kind of interaction between context variables the scenario doc asks for (its Scenario C).
- **Priority** between context rules follows the card order, then v2's shared priority
  (valid action, finishing the current commitment, ...).
- **No conflict-absorbing fallbacks.** The partner does what its mode says whatever the ego
  does. If the ego does not do the complementary part, that work stays undone: a `keeps`
  partner never takes the urgent order, a `waits` partner never clears the utensil, a `prep`
  partner never cooks outside `on_urgent(takes)`. Remove the `fallback` field from the
  context rules; the static rules keep theirs.
- **One deadlock guard, global, not per rule.** If both players have been idle (waiting or
  failing) for `guard_steps` steps (start at 30) while an order is open, the partner does the
  next needed step once. Every firing is logged as `guard`. It is a failure count reported
  per card × ego, not part of the persona, and it must not appear in the partner prompt as a
  habit.
- **`wants` lines are actions for the ego**, never restrictions, and they are conditional.
  For example: "When an order becomes urgent and I'm on my own order, take it yourself." /
  "When an order becomes urgent, I take it; you prepare the next one."
- **Fields per rule:** `text`, `wants`, `reminder`, and the three predicates (`opportunity`,
  `adherent`, `aligned`) in `rule_eval.py`. `aligned` judges the ego's decision at the step it
  was decided, in the context it was decided in.
- **The static rules** (`own_station`, `solo_dish`, ...) stay in the library unchanged and
  are not used in the new cards. batch1, batch2 and batch_ing2 are kept as they are.

## 5. Cards (`configs/coop/personas/batch_ctx.yaml`)

- **Pilot: two cards that are opposite on every context rule,** plus the built-in neutral
  partner:

  | Card | `default_work` | `on_urgent` | `on_blocked` | `on_prep_ready` |
  |---|---|---|---|---|
  | A | prep | takes, unless_ego_on_it: yes | clears | continues |
  | B | whole | keeps | waits | switches |

- **Main:** 4 cards. Every pair of cards differs on at least 2 context rules, and each mode
  appears on 2 cards. A swapped file is then right about some contexts and wrong about
  others, which allows per-context analysis of the swap control.
- **Backstories:** short, voice only, frozen after review (as in v2). They must not imply a
  fallback ("helps out when needed").

## 6. Twin and scripted egos

- **Twin:** extend `RulePlayer` with the context rules, deciding on its own view, with the
  deadlock guard and no other fallback. Adherence must be at least 95% by construction.
- **Scripted egos**, in two groups:

  | Group | Ego | Policy |
  |---|---|---|
  | Proactive | `proactive(card)` | At each context event, acts at once on the card's predicted response: does what the card's `wants` says for that context, without waiting to observe the partner (extend `RuleEgoAgent`) |
  | Proactive | `mismatched(card)` | `proactive` of the opposite card: a wrong prediction |
  | Reactive | `steady` | Existing: the next needed step from the current state, oldest order first |
  | Reactive | `always_take_urgent` / `never_take_urgent` | Fixed answer at every urgency event |
  | Reactive | `wait_and_see(k)` | After a context event, waits k steps to see what the partner does, then acts on what it saw |

- **Best reactive** is the best reactive ego per card, chosen on calibration seeds and judged
  on other seeds. `wait_and_see` is the strongest reactive baseline and must be included, with
  k chosen the same way.

## 7. Gates (in order)

**Gate 0: does acting on a partner model pay?** Twin partner, scripted egos, no LLM; extend
`coop.voi`.

- Report a crossover table (ego × card) of score, expired orders and guard firings.
- Report proactive − best reactive and proactive − mismatched per card.
- Report the value of each context rule alone (`voi --rules` pattern).
- **Pass:** for each pilot card, proactive − best reactive is above 0 in score, with a paired
  interval excluding 0 on the judge seeds, and the guard fires rarely with the proactive ego
  (set the bar on calibration seeds, then freeze it).
- **If it fails,** tune in this order: timer slack, threshold θ, expiry penalty, view radius,
  menu. Log every setting tried and report all of them, not only the one that passed.

**Gate 1: LLM partner adherence.** Existing `persona_check` on the new cards:
- adherence per context rule and mode;
- competence against the neutral partner;
- distinctness;
- guard rate.

Check that the LLM partner does not add fallbacks of its own (for example, a `keeps` partner
taking the urgent order anyway). Report this as non-adherence.

**Gate 2: Gate 0 with the LLM partner.**

**Gate 3: value of knowing for the LLM ego.** `persona_value` pattern on the pilot cards, one
day of 4 episodes:
- `oracle` − `tom_struct`;
- `oracle_mismatched` (the other card's `wants` lines) − `tom_struct`. This checks that the
  LLM ego relies on the content.

**Stop after Gate 3 and report** the commands and tables.

## 8. Learning experiment

Run shape, as built: 5 days × 3 interaction episodes, 2 held-out episodes after each day,
and a day-0 baseline. The held-out set is the same every day.

- **Exposure:** the timer slack per day sets how often `urgent` occurs, and the order mix sets
  how often `blocked` occurs (oven-heavy lists block more).
- **Counterbalancing:** half of the run seeds see urgency-heavy days first, the other half
  see blocking-heavy days first. The held-out set always contains the full mix.
- **Transfer to dishes:** the 2 held-back dishes never appear in interaction episodes and do
  appear in the held-out set. Report held-out results separately for seen and unseen dishes.
- **Combinations (a later step):** interaction days keep `urgent` and `blocked` apart, and
  the held-out set mixes them. Analyse held-out decisions by the context they were made in.

**Conditions:**

| Condition | Memory |
|---|---|
| `tom_struct` | none |
| `generic_memory` | free-form notes (`memory_mode: notes`) |
| `memory` | structured partner memory, no feedback |
| `memory_fb` | structured partner memory with feedback |
| `latest_fb` | only the previous day's feedback |
| `oracle` | the card's `wants` lines |
| `swap_other`, `swap_same`, `placebo` | as built |

`generic_memory` is the scenario doc's "generic memory" baseline. It tests whether the
four-section structure matters.

**Measures** (score is primary):

- score minus reference, expired orders, and guard firings;
- alignment per context rule and mode;
- **proactive decisions:** share of context events where the ego chose the complementary
  action before the partner's response was visible to it, and how often that choice was
  right;
- **knowledge probe** with conditional questions, one per context rule. Report three
  groups: exposed so far, not yet exposed, and combinations never seen;
- exposure per context rule;
- **prediction at context events:** the ego's structured read of what the partner will do,
  against what it did;
- **per-context swap analysis:** loss under `swap_other` on contexts where the cards differ,
  against contexts where they agree.

**Order:** pilot with the 2 pilot cards and the conditions `tom_struct`, `generic_memory`,
`memory_fb`, `oracle` and `swap_other`. Then the main run with 4 cards and all conditions.

## 9. Extensions (after the main result, in order)

1. **Belief revision** (the scenario doc's Scenario D). Same partner id, three regimes:
   - stable;
   - exception: the opposite mode of one context rule on one day only;
   - lasting switch: the opposite mode from day 3 on.

   This needs a per-day card schedule in the config. Measures: revision cost after a lasting
   switch, and unnecessary revisions after an exception.
2. Decaying reward instead of expiry (section 3.1).
3. Gemma 4 31B as partner (rerun Gate 1 first).
4. Frontier ego.
5. **Reactions to the ego's actions** (Scenario B without messages): add `on_takeover` and
   `on_contention` to the cards. Explicit help requests need a message channel and stay out
   of scope.

## 10. Order of work

| Step | Work | Output |
|---|---|---|
| W0 | Run the already-built `persona_value_ing2.yaml` (static cards, LLM ego) in the background | `oracle` − `tom_struct` on static cards |
| W1 | Order timers with expiry, context features, duplication check, tests, equality check for old configs | Commits, reference times |
| W2 | Context rules without fallbacks, the deadlock guard, predicates, `batch_ctx.yaml`, twin, proactive and reactive scripted egos | Commits |
| W3 | Gate 0, tuning if needed | **Report, then stop** |
| W4 | LLM partner prompts and reminders for the context rules; Gates 1 and 2 | Report |
| W5 | Gate 3 | **Report, then stop** |
| W6 | Learning pilot (2 cards) | Curves and per-context tables |
| W7 | Main run (4 cards, transfer dishes) | Main tables |
| W8 | Extensions in the order of section 9 | |

After each step: tests pass, `docs/coop-setting.md` and `RESEARCH_STATUS.md` are updated,
and the reply ends with the pull and `lab-run` commands, as `CLAUDE.md` asks.

## 11. Decision rules

| Observation | Next step |
|---|---|
| Gate 0 fails after tuning | Observation still replaces knowledge in this kitchen. Try radius 1, then report before redesigning. |
| The guard fires often with the proactive ego | A card's modes deadlock against their own best response. Revise the mode definitions, not the guard. |
| Gate 0 passes, Gate 3 fails (`oracle` ≈ `tom_struct`) | The ego cannot use conditional knowledge. Inspect the oracle prompt and the ego's decisions at context events; consider the frontier ego. |
| Gate 0 fails, Gate 3 passes | Knowledge only helps the LLM ego. The claim is limited to that; say so. |
| W0 shows `oracle` > `tom_struct` on static cards | Static habits already have value for the LLM ego. Keep context rules for the learning curve and transfer, and report both. |
| `generic_memory` ≈ `memory` | The structure does not matter. |
| Probe rises for exposed rules and stays at chance for unexposed ones | Learning comes from experience with this partner (experience dependency). |
| Proactive decisions rise over days with `memory_fb` but not with `tom_struct` | Memory makes the ego proactive, not only better informed. |
| `swap_other` loses only on contexts where the cards differ | Partner- and context-specific knowledge. |
| Unseen dishes ≈ seen dishes in the held-out set | Knowledge transfers. |

## 12. Open points for SJ

- Whether the pilot uses `open_ing2_env` (assumed here) or `open2_env`.
- The guard length (`guard_steps`, start 30) and the guard-rate bar.
- How far Gate 0 tuning may go before the context design itself is revised.

## 13. Out of scope

- messages during an episode;
- partners that adapt to the ego (except the scheduled switch in extension 1);
- weight updates;
- the human study.
