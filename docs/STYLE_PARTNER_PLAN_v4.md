# Plan v4: two-axis partners (workflow × plan adherence) and validation without API models

Project: self-evolving collaborative agent (ICML 2027).
Repo: `Collab-Overcooked`, branch `claude/lucid-babbage-96q0zi`, research code in `src/coop`.
For Claude Code. Read `CLAUDE.md`, `RESEARCH_STATUS.md`, `docs/PERSONA_PARTNER_PLAN_v2.md` and
`docs/CONTEXT_PARTNER_PLAN_v3.md` first.

This plan replaces v3's cards (§5), gates (§7) and order of work (§10). Everything else in v3
still holds: the context requirements R1–R5 (§2), the environment additions (§3), the context
rules without fallbacks and the deadlock guard (§4), the twin and scripted egos (§6), and the
learning experiment (§8, deferred until this plan is done). v2's invariants (§2) still hold.
Where this file disagrees with v2 or v3, this file holds.

**No API model is called anywhere in this plan.** Partners are the rule-based twin or local
open-weight models on the lab servers. Egos are scripted.

## 0. Why the design changes

1. **v3 cards are combinations of context modes with no common cause.** Card A is
   `prep / takes / clears / continues` and card B is the opposite on every rule. Learning such a
   partner means memorising a table, which leaves no room for inferring one context from
   another and does not look like a real person.
2. **A working style alone is identifiable in the first episode.** If one latent variable
   determined every context response and were visible in default behaviour, within-episode
   inference would suffice and there would be nothing to learn across days.
3. **Rejected axes.** Movement speed and skill (capability) do not need theory of mind and
   are already covered by Overcooked-AI work. Hidden objectives such as "deadline first" are
   too obvious. Expectations about the ego (shared mental model) and stated-vs-revealed
   feedback are kept as later difficulty layers on top of a base persona (§11).
4. **Problems seen so far split into two kinds.** Feedback dominated by the ego's own errors
   came from the Qwen ego and is not addressed here. Missing score headroom, mismatches
   absorbed by fallbacks and reactive ≈ informed come from the environment, so v3's
   environment changes are still required.

So each partner has two latent axes. One is visible at once, and the other is revealed only
by rare events that the ego's own actions can hide.

## 1. Purpose

Build and validate a partner population for the later frontier-ego study. That study asks
whether a frontier model, across days:

- accumulates partner knowledge beyond within-episode inference;
- stays calibrated on contexts it has not observed;
- acts to reveal partner behaviour it is unsure about, rather than settling into a
  self-confirming belief;
- acts on what it inferred (knowing–doing);
- avoids over-generalising one context's behaviour to other contexts.

This plan only establishes that the partners and the environment make those questions
measurable: V0 (value), V1 (identifiability and exposure), V2 (LLM partner fidelity).

## 2. Partner model

### 2.1 Workflow axis (base persona, visible in default behaviour)

| Workflow | Default behaviour | Complement for the ego |
|---|---|---|
| `prep` | Prepares ingredients (dispense, chop) for every open order and leaves them on counters near the cooking stations. Does not load, cook, plate or serve. | Load, cook, plate and serve what it prepared; do not prepare. |
| `whole` | Takes one open order and does it from start to finish, then the next. | Take the other open order from start to finish; do not touch its order. |
| `station` | Stays in the cooking area (cells next to pots and ovens): loads, starts, takes out and plates whatever is brought. Does not fetch raw ingredients and does not serve. | Fetch and prepare ingredients, bring them to its station, serve plated food. |

`prep` and `whole` are v3's `default_work` modes. `station` is new: restrict the twin's job
selection to jobs whose target is a pot, an oven or plating.

**Cooking area** is computed from any layout by one rule, not written per layout: the walkable
cells adjacent to any pot or oven, plus the cells adjacent to the plating counter and plate
dispenser if `station` plates (§12). Only open kitchens are used, where both players can reach
every station (§11), so the `station` partner moves freely between all cooking stations.

### 2.2 Plan-adherence axis (revealed only at context events)

| Context (v3 §4) | `rigid` | `flexible` |
|---|---|---|
| `on_urgent` | `keeps` | `takes`, `unless_ego_on_it: yes` |
| `on_blocked` | `waits` | `clears` |
| `on_prep_ready` | `continues` | `switches` |

A `rigid` partner keeps its plan when something happens. A `flexible` partner drops its plan
for the event. One latent value sets all three context responses, so behaviour seen in one
context predicts behaviour in the others, except on the diagnostic card (§2.3).

**Identifiability per card.** If a card's workflow already fixes its response in a context
(for example, `station` cooks prepared items whatever its adherence), adherence is not
identifiable there for that card. The loader marks such (card, context) pairs as
`not_identifiable`. They are excluded from the adherence measures and reported.

### 2.3 Cards (`configs/coop/personas/batch_style.yaml`)

- **Main:** 6 cards, 3 workflows × 2 adherence values.
- **Diagnostic card `mixed`:** `flexible` on `on_urgent`, `rigid` on `on_blocked` and
  `on_prep_ready`. Its workflow is the one with the largest adherence value in V0 (default
  `prep`). It exists so the frontier study can test over-generalisation from `on_urgent` to the
  other contexts.

```yaml
prep_rigid:
  workflow: prep
  adherence: rigid              # rigid | flexible | {on_urgent: ..., on_blocked: ..., on_prep_ready: ...}
  backstory: "..."              # two variants, see §2.4
mixed:
  workflow: prep
  adherence: {on_urgent: flexible, on_blocked: rigid, on_prep_ready: rigid}
  backstory: "..."
```

- The loader expands `workflow` and `adherence` into v3 rule modes, rejects contradictions, and
  writes the expanded card and both latent labels into the run folder.
- Ground truth is recorded at two levels: latent (`workflow`, `adherence`) and per context
  (mode, or `not_identifiable`).
- The static rule cards (batch1, batch2, batch_ing2) and v3's `batch_ctx.yaml` stay as they are.

### 2.4 Backstories (for the LLM partner in V2)

Two prompt variants per card, both frozen after review:

- `style_only`: 2–4 sentences describing how the person works and how they handle
  disruptions, in natural language. No rule ids, no rule text, no mode names.
- `style_habits`: `style_only` plus at most 3 concrete habit lines (for example, "When an order
  is about to run out, you finish what you're doing first.").

Neither may imply a fallback ("helps out when needed", "steps in if things stall").

## 3. Twin (`src/coop/rule_partner.py`)

- Extend `RulePlayer` with the `station` workflow and with cards expanded from the two axes.
- Decides on its own view, uses the global deadlock guard, and has no other fallback (v3 §4).
- Adherence of at least 95% by construction, checked by the predicates in `rule_eval.py`.
- Logs per decision: latent labels, active context, mode applied, and whether the response was
  inside the ego's view.

## 4. Scripted egos (extend `RuleEgoAgent`, `coop.voi`)

| Group | Ego | Policy |
|---|---|---|
| Informed | `best_response(card)` | Workflow complement, plus the complement for the card's mode at each context event, chosen at once without waiting to observe the partner (v3 `proactive`) |
| Partly informed | `wrong_adherence(card)` | `best_response` of the card with the same workflow and the opposite adherence |
| Partly informed | `wrong_workflow(card)` | `best_response` of a card with the same adherence and another workflow (report each) |
| Reactive | `steady` | Existing |
| Reactive | `always_take_urgent` / `never_take_urgent` | Fixed answer at every urgency event |
| Reactive | `wait_and_see(k)` | Waits k steps after a context event, then acts on what it saw |

**Best reactive** is the best reactive ego per card, chosen on calibration seeds and judged on
the other seeds. k for `wait_and_see` is chosen the same way.

## 5. V0: does each axis have value? (twin partner, no LLM)

Runs: 7 cards × all egos × 4 calibration seeds + 4 judge seeds × 3 episodes. Kitchen and menu as
in v3 §3.4, timers on.

Report:

- a crossover table (ego × card) of score minus reference, expired orders and guard firings;
- per card, with paired intervals over judge seeds:
  - `Δ_reactive = best_response − best reactive`
  - `Δ_adherence = best_response − wrong_adherence` (value of the axis that needs days)
  - `Δ_workflow = best_response − wrong_workflow`
- value per context: the same Δs restricted to decisions at each context's events.

Pass (bars set on calibration seeds, then frozen):

- `Δ_reactive > 0` with the interval excluding 0 on every main card;
- `Δ_adherence > 0` with the interval excluding 0 on at least 4 of the 6 main cards, and on at
  least one card of each workflow;
- crossover: within each workflow, the two adherence values have different best egos;
- the guard fires rarely with `best_response` (bar set on calibration seeds).

If V0 fails, tune in v3 §7's order (timer slack, threshold θ, expiry penalty, view radius,
menu). Log every setting tried and report all of them, not only the one that passed.

**Stop after V0 and report.**

## 6. V1: identifiability and exposure across days (twin partner, no LLM)

Runs: the v3 §8 shape (5 days × 3 interaction episodes) with the twin partner and these egos:
`steady`, `always_take_urgent`, `never_take_urgent`, `wait_and_see(k*)`, and
`best_response` of each other card (an ego acting on a wrong belief). 7 cards × 4 seeds.

**Revealing event:** a context event where the ego did not act first on the contested job and
the partner's response was inside the ego's view within W steps (W = macro latency + 2; record
the value).

Report:

- **Workflow identification:** a Bayesian classifier over the 3 workflows on per-step job
  features from the ego's view. Report the steps until the posterior on the true workflow
  reaches 0.9.
- **Adherence exposure:** revealing events per context per day, the day of the first reveal,
  and the share of runs with zero reveals by day 5. Report these per card × ego and per context.
- **Masking:** for each ego policy, the contexts it leaves unrevealed (for example,
  `always_take_urgent` with a `flexible` partner never reveals `takes`).
- **Cross-context:** for `mixed`, the days on which each context is first revealed. This shows
  whether the frontier study will have cases where only `on_urgent` was seen.

Desired properties (pass):

- workflow is identified within day 1 in at least 90% of runs;
- under `steady`, the median first reveal of adherence comes after the first episode, and at
  least one context is first revealed after day 1 in a majority of runs;
- at least one ego policy leaves at least one identifiable context unrevealed by day 5 in a
  majority of runs (masking exists).

**Report V1.**

## 7. V2: LLM partner fidelity (local models only)

Partners: Qwen3.8-27B first, then Gemma 4 31B (check serving and memory beside each other
first). Backend through `partner.backend` pointing at lab servers. Thinking off, temperature 0.
Egos: `steady` and `best_response(card)`. 7 cards × 4 seeds × 3 episodes, for each prompt
variant (`style_only`, `style_habits`) and each `remind` setting.

Report:

- workflow adherence: job-category shares (prepare / cook / plate / serve) and cooking-area
  occupancy against the twin;
- adherence per context and mode on the partner's own view, only on identifiable pairs;
- **self-added fallbacks**: a `rigid` partner taking an urgent order or clearing a blocked
  utensil, or a `prep` partner cooking. Counted as non-adherence and reported separately;
- distinctness: a leave-one-seed-out classifier over the 7 cards, and separately over adherence
  within each workflow;
- competence against the neutral LLM partner;
- guard rate;
- `Δ_adherence` and `Δ_reactive` from V0, rerun with the LLM partner (scripted egos only).

Pass (calibrate on seeds 0–1, freeze, judge on seeds 2–3; starting bar 70% as in v2):

- workflow and context adherence above the bar;
- the within-workflow adherence classifier above chance with its interval excluding chance;
- `Δ_adherence` keeps its sign from V0 on the cards that passed V0.

Pick the prompt variant: `style_only` if it passes, otherwise `style_habits`. If neither passes
for a context, revise that context's wording, then try `remind: true`, then drop the context.
Log every revision.

**Stop after V2 and report** the commands and tables.

## 8. After this plan (not to implement now)

1. **Frontier offline probing** (separate plan, few calls): V1 twin logs are given to the
   frontier ego, which writes structured partner memory. Then it is probed on calibration
   (unrevealed contexts, with confidence), exploration (a decision point at an event whose mode
   it does not know) and knowing–doing (revealed contexts).
2. **Learning experiment** per v3 §8, with `batch_style.yaml` replacing `batch_ctx.yaml`.

## 9. Order of work

| Step | Work | Output |
|---|---|---|
| W0 | Check which parts of v3 W1–W2 exist (timers with expiry, context features, duplication check, context rules without fallbacks, deadlock guard, twin context rules, proactive and reactive egos). Implement what is missing, with tests and the old-config equality check. | Status table, commits |
| W1 | Two-axis card schema and loader expansion, `station` workflow, identifiability marking, `batch_style.yaml` (7 cards, backstories drafted but not frozen), twin logging. Tests: twin adherence ≥ 95%, loader rejects contradictions, no leakage of card text into ego prompts. | Commits |
| W2 | `best_response`, `wrong_adherence`, `wrong_workflow`; `coop.voi` extension for the three Δs and per-context values | Commits |
| W3 | V0, tuning if needed | **Report, then stop** |
| W4 | V1 runs, revealing-event detector, workflow classifier, exposure and masking tables | Report |
| W5 | LLM partner prompts for both variants (frozen after SJ's review), V2 on Qwen, then Gemma | **Report, then stop** |

After each step: tests pass, `docs/coop-setting.md` and `RESEARCH_STATUS.md` are updated, and
the reply ends with the pull and `lab-run` commands, as `CLAUDE.md` asks.

## 10. Decision rules

| Observation | Next step |
|---|---|
| `Δ_reactive` > 0 but `Δ_adherence` ≈ 0 | Only the workflow matters, and it is learned in one episode. Raise event frequency or cost (timer slack, θ, expiry penalty) before anything else. |
| `Δ_adherence` > 0 only for some workflows | Keep those cards for the main study; report the others as a limitation. |
| Adherence is revealed within the first episode under `steady` | Events are too frequent for a day-scale claim. Lower urgency or blocking frequency, then rerun V0 to check that value remains. |
| No ego policy masks any context | The self-confirming question cannot be studied here; report it and keep calibration and over-generalisation as the main questions. |
| `station` has many `not_identifiable` pairs | Keep `station` as a workflow-only card or redefine it; report the choice. |
| The LLM partner adds fallbacks on `rigid` cards | Revise wording, then `remind`, then consider the twin as the main partner with the LLM partner as a robustness check. |
| `style_only` fails and `style_habits` passes | Use `style_habits`; check that feedback text in later plans does not recite the habit lines. |

## 11. Out of scope

- any API model call;
- split or forced-coordination layouts, where each player can reach only part of the kitchen:
  the layout then fixes most of each player's role and leaves too few choices for persona
  differences to show;
- difficulty layers: expectations about the ego (shared mental model), stated-vs-revealed
  feedback;
- capability axes (speed, skill, error rate);
- partners that change across days;
- messages during an episode;
- weight updates;
- the human study.

## 12. Open points for SJ

- Whether `station` plates. This sets whether plating cells belong to the cooking area and
  whether the ego's best response includes plating.
- The workflow of the `mixed` card if V0 does not single one out.
- Seed counts for V0 and V1 (4 + 4 and 4 assumed).
- Whether `on_prep_ready` stays, if V1 shows it rarely occurs under any workflow.
- The final wording of the backstories before W5.
