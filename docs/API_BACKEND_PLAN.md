# Plan: API model backends (GPT-6 Astra, Claude Opus 5.5, optionally Gemini 4 Argon)

Project: self-evolving collaborative agent (ICML 2027).
Repo: `Collab-Overcooked`, branch `claude/lucid-babbage-96q0zi`, research code in `src/coop`.
For Claude Code. Read `CLAUDE.md` first. The experiments that use these backends are in
`API_PILOT_PLAN.md` (first-choice-only decision tests); this plan covers only the backend
layer, configuration, cost control and the smoke test.

## 0. Current code

- `src/coop/llm.py`: `OpenAICompatBackend` (POST `/chat/completions`, stdlib only),
  `MockBackend`, `build_backend(spec)`. Usage per call: prompt and completion tokens,
  `thinking` (reasoning text present), `finish_reason`.
- Callers pass `max_tokens` sized for the visible JSON reply: decisions 400 (config),
  probes 500, consolidation 1000, feedback 200.
- The ego's system prompt is built once per episode (rules, map, recipes, oracle or memory
  block) and the changing state goes in the user message. That makes the system prompt the
  natural cache prefix.
- `analyze.py` reports `thinking` as an LLM-health problem; that was right for Qwen with
  thinking off, but API reasoning models always reason.
- `scripts/run_qwen38_days.sh` starts vLLM unless `ICRL_LLM_BASE_URL` is set.

## 1. Invariants

1. Old configs behave exactly as before. New behaviour only through new backend types and
   new keys. Check with the mock backend (as before: identical steps and prompts).
2. API keys are read from environment variables only. They never appear in configs, resolved
   config files, logs, error messages or outputs. Add a test that runs a fake-server episode
   with a key set and searches every written file for it.
3. Tests make no network calls. Use a local fake HTTP server that returns OpenAI- and
   Anthropic-shaped responses.
4. Stay stdlib-only (`urllib`), like the existing backend.
5. Unverified API details (parameter names, usage fields) are checked in the smoke test
   (section 6) and documented in `docs/coop-setting.md`. Do not guess silently: if a field is
   missing, log it as missing.

## 2. Common changes (`src/coop/llm.py`)

### 2.1 Result and usage fields

`LLMResult` and `Usage` gain:

| Field | Meaning |
|---|---|
| `cached_tokens` | Input tokens read from the prompt cache |
| `cache_write_tokens` | Input tokens written to the cache (Anthropic) |
| `reasoning_tokens` | Hidden reasoning tokens, billed as output |
| `cost_usd` | Cost of the call from the price table (section 4) |
| `retries`, `rate_limited_s` | Retries used and seconds waited on rate limits |

Write them into every per-call record in `agent_log.json` and `partner_log.json`, and into
`llm_usage` in `summary.json`. Missing fields are 0 for old backends, so old outputs keep
their meaning.

### 2.2 Reasoning models

- Backend spec key `expects_reasoning: true`. With it, reasoning text or reasoning tokens are
  normal, not a health failure. `analyze.py` reports them as `reasoning_tokens` per call
  instead of counting `thinking`.
- **Completion cap.** Callers' `max_tokens` are sized for the visible reply. Reasoning tokens
  count against the same cap on API models, so the backend sends
  `max_tokens + reasoning_allowance` (spec key, default 3000 for reasoning models, 0
  otherwise). A reply cut off by the cap is logged as `truncated`, as now.
- Timeout default 300 s for API backends.

### 2.3 Retries and rate limits

- Retry 429, 500, 502, 503, 504 and Anthropic's 529 (overloaded), with exponential backoff
  and the `Retry-After` header when present. Other 4xx: no retry; log the error body with any
  key removed.
- `max_concurrency` (spec key): a semaphore shared by all threads using the same backend
  name, so `--workers` cannot exceed the account's rate limit.
- `max_retries` default 6 for API backends.

## 3. Backend types

### 3.1 `type: openai` (GPT-6 Astra)

Subclass of `OpenAICompatBackend`, Chat Completions at `https://api.openai.com/v1`.

- `reasoning_effort: low` from the spec. Astra rejects `none` and `minimal` (HTTP 400); its
  floor is `low`.
- Send no `temperature`, `top_p` or `seed`; Astra does not accept custom sampling values. The
  spec option `send_sampling: false` removes them even if a shared config sets them.
- `max_completion_tokens` instead of `max_tokens`.
- `response_format: {"type": "json_object"}` if the smoke test shows it is accepted; otherwise
  rely on the prompt and `parse_json`, and record that.
- Usage: `prompt_tokens_details.cached_tokens` → `cached_tokens`,
  `completion_tokens_details.reasoning_tokens` → `reasoning_tokens`.
- Caching is automatic for long stable prefixes. Keep the system prompt first and unchanged
  within an episode. Optional `prompt_cache_key` per (run, condition) if supported.

### 3.2 `type: anthropic` (Claude Opus 5.5)

New class for the Messages API, `POST https://api.anthropic.com/v1/messages`.

- Headers: `x-api-key` from the env var, `anthropic-version` from the spec, content type JSON.
- Convert the chat messages: the system message becomes the top-level `system` field, as a
  text block with `cache_control: {"type": "ephemeral"}`; the rest stay as `messages`.
- `max_tokens` is required: send the caller's cap plus `reasoning_allowance`.
- Thinking and effort: pass through `anthropic_params` from the spec unchanged (for example
  the effort setting). **Look up the current parameter names for `claude-opus-5-5` in the
  Anthropic documentation; do not guess.** Default: low effort.
- Temperature: if the API rejects a custom temperature together with thinking, omit it and
  record that in the docs.
- Response: concatenate the `text` blocks for `text`; `thinking` blocks go to `reasoning`.
  `stop_reason: max_tokens` → `finish_reason: length`.
- Usage: `input_tokens`, `output_tokens`, `cache_read_input_tokens` → `cached_tokens`,
  `cache_creation_input_tokens` → `cache_write_tokens`.
- JSON: no response-format parameter; rely on the prompt and `parse_json`. Use a native
  structured-output option only if the docs confirm it for this model.

### 3.3 Gemini 4 Argon (optional, after 3.1 and 3.2)

Use `type: openai` against Google's OpenAI-compatible endpoint if the smoke test shows that
the parameters used work there. Otherwise skip it.

## 4. Configuration and cost

### 4.1 Backend registry (`configs/coop/backends.yaml`)

```yaml
astra:
  type: openai
  base_url: https://api.openai.com/v1
  model: gpt-6-astra
  api_key_env: OPENAI_API_KEY
  reasoning_effort: low
  send_sampling: false
  expects_reasoning: true
  reasoning_allowance: 3000
  max_concurrency: 8
  price: astra
opus:
  type: anthropic
  base_url: https://api.anthropic.com/v1
  model: claude-opus-5-5
  api_key_env: ANTHROPIC_API_KEY
  anthropic_version: "2023-06-01"   # check against the docs
  anthropic_params: {}              # effort / thinking, from the docs
  expects_reasoning: true
  reasoning_allowance: 3000
  max_concurrency: 8
  price: opus
```

- A config may say `backend: {ref: astra}` instead of a full spec.
- CLI overrides in `coop.days` and `coop.decision`: `--ego-backend NAME` and
  `--partner-backend NAME`. They replace the backend in every condition without copying the
  config. The resolved config records the backend name and spec, never a key.

### 4.2 Price table (`configs/coop/prices.yaml`)

USD per million tokens. Recheck the providers' price pages before every paid run.

| Name | Input | Cached input | Cache write | Output (incl. reasoning) |
|---|---|---|---|---|
| astra | 10.00 | 1.00 | — | 50.00 |
| opus | 4.00 | 0.20 | 5.00 | 20.00 |
| argon | 2.00 → 4.00 (introductory rate ends) | 0.10 → 0.20 | — | 10.00 → 20.00 |

Cost per call = uncached input × input price + cached × cached price + cache writes × write
price + (output + reasoning) × output price. If the provider already includes reasoning in
the output count, do not add it twice; check this in the smoke test.

### 4.3 Budget guard

- `--max-cost USD` on `coop.days` and `coop.decision`. A shared tracker sums `cost_usd` over
  all threads. When the limit is reached, no new calls start, running episodes are marked
  unfinished, and the run exits cleanly so `--resume` can continue later.
- Every report prints cost per model and condition, and the total.

## 5. Running on the lab server

- API-ego runs with the scripted twin need no GPU: `lab-run --new --gpus 0 -- python -m ...`.
- API ego with the Qwen LLM partner: the server is needed for the partner only. Add
  `scripts/run_api.sh` (or an option to the existing script) that serves Qwen only when a
  partner or condition backend points at it, and leaves the ego on the API.
- Keys: check that `lab-run` passes the key environment variables. If not, read them from a
  file outside the repo (for example `~/.coop_api_env`, mode 600) sourced by the script.
- Check that the lab server can reach the API endpoints (outbound HTTPS).

## 6. Smoke test (`coop.check --api NAME`)

Before any paid run, per backend:

1. Send 4 real decision prompts. Capture them once from a mock run of
   `configs/coop/decision/urgent.yaml` and store them under `tests/fixtures/`.
2. Send the same prompt twice in a row, and check that the second call reports cached tokens.
3. Print per call: JSON valid, action parsed and in `VALID_ACTIONS`, input / cached / output /
   reasoning tokens, latency, finish reason, cost.
4. Print the mean cost per call and the projected cost for `--planned-calls N`.
5. Report every parameter the API rejected, with the error message.
6. Write the result to `outputs/api_check/<name>.json`, and add the verified parameter names
   and usage fields to `docs/coop-setting.md`.

## 7. Tests

- Payload construction:
  - Astra: no temperature or seed, `max_completion_tokens`, `reasoning_effort`;
  - Anthropic: system block with `cache_control`, required `max_tokens`, message conversion;
  - the reasoning allowance is added.
- Response parsing for both shapes: text, reasoning, usage fields, truncation.
- Retries: 429 with `Retry-After`, 503, 529, and a 400 that must not be retried; the key never
  appears in an error.
- Concurrency limit, cost arithmetic, and the budget stop with resume.
- The key-leak test (section 1) and the mock equality check for old configs.

## 8. Optional: batch pricing for decision tests

Only after the pilot has run at standard pricing. The first-choice-only calls are independent,
so they can use the OpenAI Batch API and Anthropic Message Batches at half price:
`coop.decision --stage llm --first-only --batch` writes the requests, submits them, polls,
and reads the results into the same records. Closed-loop episodes cannot use batch.

## 9. Order of work

| Step | Work |
|---|---|
| W1 | Common changes (section 2) and `type: openai`, with fake-server tests |
| W2 | `type: anthropic`, with fake-server tests |
| W3 | Registry, CLI overrides, price table, budget guard, cost lines in reports |
| W4 | Smoke test tool; mock equality check; docs. **Report, then stop:** SJ runs the smoke tests on Astra and Opus |
| W5 | Fix whatever the smoke tests showed; then the first-choice-only pilot from `API_PILOT_PLAN.md` |
| W6 | Gemini 4 Argon (optional) |
| W7 | Batch mode for decision tests (optional) |

After each step: tests pass, old configs are unchanged, `docs/coop-setting.md` and
`RESEARCH_STATUS.md` are updated, and the reply ends with the pull and `lab-run` commands,
as `CLAUDE.md` asks.

## 10. To verify, not assume

- Opus 5.5: the current parameter names for effort and thinking; whether temperature is
  allowed with thinking; whether thinking can be turned off.
- Astra on Chat Completions: whether `response_format` is accepted; the names of the cached
  and reasoning token fields; whether `prompt_cache_key` exists.
- Whether output token counts already include reasoning, for each provider.
- The account's rate limits, which set `max_concurrency`.
