"""Check a served model before an experiment: is thinking really off, and does JSON come back?

Which `chat_template_kwargs` the served template honours is checked, not assumed. Each
candidate is sent with a tiny JSON task; it passes when no reasoning text comes back (neither
`reasoning_content` nor inline <think>) and the reply is the requested JSON. Candidates go
from the most minimal switch to broader ones; the first that passes is used.

Observed with Qwen3.8-27B on vLLM: `enable_thinking: false` turns thinking off;
`reasoning_effort` only accepts xhigh (default) / medium / low, and "none" is rejected (400).

  python -m coop.check --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b
  python -m coop.check ... --pick --report check.jsonl  # stdout: chosen JSON only

API smoke test (hosted backends of configs/coop/api/backends.yaml), before any paid run:

  python -m coop.check --api opus --planned-calls 20000

sends 4 real decision prompts (tests/fixtures/api_decision_prompts.json), repeats the first
to check the prompt cache, probes optional parameters, and prints per call: JSON valid,
action parsed and in VALID_ACTIONS, input / cached / output / reasoning tokens, latency,
finish reason and cost; then the mean cost per call and the projected cost. Every parameter
the API rejected is reported with its error. Result: outputs/api_check/<name>.json.

Ported from ICRL4AHT research/coop_tom/check.py.

Exit code 1 if no candidate passes.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from .llm import OpenAICompatBackend, parse_json

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "api_decision_prompts.json"

CANDIDATES = [
    {"enable_thinking": False},
    {"reasoning_effort": "none"},
    {"enable_thinking": False, "reasoning_effort": "none"},
]
THINKING_ON = {"enable_thinking": True}  # detector sanity check; uses no template-specific values
TASK = [{"role": "user", "content": 'Reply with JSON only: {"action": "wait 1"}'}]


def probe(backend: OpenAICompatBackend, kwargs: dict, max_tokens: int = 256) -> dict:
    backend.extra_body = {"chat_template_kwargs": kwargs}
    r = backend.chat(TASK, max_tokens=max_tokens, seed=0)
    js = parse_json(r.text)
    return {"chat_template_kwargs": kwargs, "error": r.error, "thinking": r.thinking,
            "reasoning_chars": len(r.reasoning), "completion_tokens": r.completion_tokens,
            "finish_reason": r.finish_reason, "json_ok": bool(js) and js.get("action") == "wait 1",
            "content": r.text[:120],
            "passes": not r.error and not r.thinking and bool(js) and js.get("action") == "wait 1"}


def _in_valid(action: str, valid) -> bool:
    import re
    if action in valid:
        return True
    m = re.fullmatch(r"wait\((\d+)\)", action or "")
    w = next((v for v in valid if v.startswith("wait(")), None)
    if m and w:
        lo, hi = re.findall(r"\d+", w)[:2]
        return int(lo) <= int(m.group(1)) <= int(hi)
    return False


def _call_row(label, r, valid) -> dict:
    from .executor import parse_macro
    js = parse_json(r.text)
    action = js.get("action") if isinstance(js, dict) else None
    parsed = isinstance(action, str) and parse_macro(action)[0] is not None
    return {"label": label, "error": r.error, "json_ok": isinstance(js, dict), "action": action,
            "action_parsed": parsed, "action_valid": bool(parsed and _in_valid(action, valid)),
            "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens,
            "cache_write_tokens": r.cache_write_tokens, "completion_tokens": r.completion_tokens,
            "reasoning_tokens": r.reasoning_tokens, "latency_s": round(r.latency_s, 2),
            "finish_reason": r.finish_reason, "cost_usd": round(r.cost_usd, 6), "retries": r.retries,
            "served_model": r.served_model, "missing_usage": r.missing_usage, "reply": r.text[:300]}


PROBES = {
    "openai": [("response_format json_object", {"response_format": {"type": "json_object"}}),
               ("prompt_cache_key", {"prompt_cache_key": "coop-smoke"}),
               ("reasoning_effort minimal (plan: expected rejected)", {"reasoning_effort": "minimal"}),
               ("temperature 0 (plan: expected rejected)", {"temperature": 0.0})],
    "anthropic": [("output_config.effort low", {"output_config": {"effort": "low"}}),
                  ("thinking disabled (docs: rejected on this model)", {"thinking": {"type": "disabled"}}),
                  ("temperature 0 (docs: rejected on API version 2026-10-21)", {"temperature": 0.0})],
}


def smoke(name: str, planned_calls: int = 0, probes: bool = True, out_dir: str = "outputs/api_check") -> dict:
    """The API smoke test (see the module doc). Returns the report and writes it as JSON."""
    from .llm import APIError, build_backend, resolve_spec
    spec = resolve_spec({"ref": name})
    backend = build_backend(spec, f"check:{name}")
    if not backend._key:
        sys.exit(f"{spec.get('api_key_env')} is not set: export it or put it in ~/.coop_api_env (chmod 600)")
    prompts = json.loads(FIXTURES.read_text())["prompts"]
    rows = []
    for p in prompts:
        rows.append(_call_row(p["label"], backend.chat(p["messages"], max_tokens=p["max_tokens"]), p["valid_actions"]))
    again = _call_row(prompts[0]["label"] + " (repeat: cache)",
                      backend.chat(prompts[0]["messages"], max_tokens=prompts[0]["max_tokens"]), prompts[0]["valid_actions"])
    rows.append(again)
    probe_rows = []
    if probes:
        kind = spec["type"]
        base = backend.payload(prompts[0]["messages"], max_tokens=prompts[0]["max_tokens"])
        for label, extra in PROBES.get(kind, []):
            body = dict(base)
            body.update(extra)
            try:
                backend.send(backend.endpoint, body)
                probe_rows.append({"param": label, "accepted": True})
            except APIError as e:
                probe_rows.append({"param": label, "accepted": False, "error": str(e)[:400]})
    ok = [r for r in rows if not r["error"]]
    mean_cost = statistics.fmean(r["cost_usd"] for r in ok) if ok else None
    rep = {"backend": name, "spec": spec, "calls": rows, "probes": probe_rows,
           "rejected": [r for r in rows if r["error"].startswith("HTTP 4")] + [p for p in probe_rows if not p["accepted"]],
           "cache_read_on_repeat": again["cached_tokens"],
           "reasoning_included_in_output": (all(r["completion_tokens"] >= r["reasoning_tokens"] for r in ok)
                                            if any(r["reasoning_tokens"] for r in ok) else None),
           "missing_usage": sorted({m for r in rows for m in r["missing_usage"]}),
           "mean_cost_per_call": mean_cost,
           "planned_calls": planned_calls,
           "projected_cost": mean_cost * planned_calls if (mean_cost is not None and planned_calls) else None}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{name}.json").write_text(json.dumps(rep, indent=1))
    return rep


def format_smoke(rep: dict) -> str:
    L = [f"API smoke test: {rep['backend']} ({rep['spec'].get('type')}, {rep['spec'].get('model')})",
         f"{'call':<34}{'json':>5}{'valid':>6}{'in':>7}{'cached':>8}{'out':>6}{'reason':>8}{'s':>6}"
         f"{'finish':>9}{'USD':>9}  action / error"]
    for r in rep["calls"]:
        tail = r["error"][:120] if r["error"] else str(r["action"])
        L.append(f"{r['label'][:33]:<34}{'y' if r['json_ok'] else 'n':>5}{'y' if r['action_valid'] else 'n':>6}"
                 f"{r['prompt_tokens']:>7}{r['cached_tokens']:>8}{r['completion_tokens']:>6}{r['reasoning_tokens']:>8}"
                 f"{r['latency_s']:>6.1f}{r['finish_reason'][:8]:>9}{r['cost_usd']:>9.4f}  {tail}")
    L.append(f"prompt cache: the repeated call read {rep['cache_read_on_repeat']} cached tokens"
             + ("" if rep["cache_read_on_repeat"] else "  <- NOT CACHED: check prompt_cache_key / minimum prefix"))
    L.append(f"reasoning counted inside output tokens: {rep['reasoning_included_in_output']}")
    L.append(f"usage fields missing: {rep['missing_usage'] or 'none'}")
    served = sorted({r['served_model'] for r in rep['calls'] if r['served_model']})
    L.append(f"served model(s): {served}")
    for p in rep["probes"]:
        L.append(f"probe {p['param']}: " + ("accepted" if p["accepted"] else f"REJECTED: {p['error'][:200]}"))
    if rep["mean_cost_per_call"] is not None:
        L.append(f"mean cost per call ${rep['mean_cost_per_call']:.4f}" + (
            f"; projected for {rep['planned_calls']} calls: ${rep['projected_cost']:.2f}" if rep["projected_cost"] else ""))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", help="smoke-test this backend of configs/coop/api/backends.yaml (no --base-url)")
    ap.add_argument("--planned-calls", type=int, default=0, help="--api: project the cost of this many calls")
    ap.add_argument("--no-probes", action="store_true", help="--api: skip the optional-parameter probes")
    ap.add_argument("--out-dir", default="outputs/api_check")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key-env", default="ICRL_LLM_API_KEY")
    ap.add_argument("--pick", action="store_true", help="print only the first passing kwargs as JSON")
    ap.add_argument("--report", help="also write every probe result as JSON lines to this file")
    a = ap.parse_args(argv)
    if a.api:
        rep = smoke(a.api, a.planned_calls, not a.no_probes, a.out_dir)
        print(format_smoke(rep))
        print(f"wrote {Path(a.out_dir) / (a.api + '.json')}")
        bad = [r for r in rep["calls"] if r["error"] or not r["action_valid"]]
        sys.exit(1 if bad else 0)
    if not (a.base_url and a.model):
        ap.error("--base-url and --model are required (or --api NAME)")
    backend = OpenAICompatBackend("check", a.base_url, a.model, api_key_env=a.api_key_env,
                                  params={"temperature": 0.0}, max_retries=1)
    served = backend.models()
    if a.model not in served:
        sys.exit(f"model {a.model!r} not served at {a.base_url}; available {served}")
    results = [probe(backend, kwargs) for kwargs in CANDIDATES]
    # Sanity check of the detector: asking for thinking should produce reasoning.
    on = dict(probe(backend, THINKING_ON, max_tokens=512), note="thinking requested ON (detector sanity check)")
    lines = [json.dumps(r) for r in results + [on]]
    if a.report:
        with open(a.report, "w") as f:
            f.write("\n".join(lines) + "\n")
    if not a.pick:
        print("\n".join(lines))
    passing = [r for r in results if r["passes"]]
    if not passing:
        print("No candidate turns thinking off with valid JSON; see the probe results.", file=sys.stderr)
        sys.exit(1)
    if on["error"]:
        print(f"warning: the thinking-ON sanity request failed ({on['error'][:200]}), so the detector "
              "itself is unverified.", file=sys.stderr)
    elif not on["thinking"]:
        print("warning: thinking requested ON produced no reasoning text; the server may lack a "
              "reasoning parser, or the template ignores enable_thinking.", file=sys.stderr)
    if a.pick:
        print(json.dumps(passing[0]["chat_template_kwargs"]))


if __name__ == "__main__":
    main()
