"""API backends (openai, anthropic): payloads, parsing, retries, slots, cost, budget, key leaks.
No network: every call goes to a local fake server (tests/fake_api_server.py)."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai"), str(ROOT / "tests")]

from coop import llm  # noqa: E402
from coop.llm import (AnthropicBackend, BudgetExceeded, CostLedger, OpenAIBackend, Usage,  # noqa: E402
                      build_backend, call_cost)
from fake_api_server import FakeAPI  # noqa: E402

KEY = "sk-test-SECRETKEY0123456789abcdef"
MSGS = [{"role": "system", "content": "You cook. " * 50},
        {"role": "user", "content": "state...\nVALID_ACTIONS: pickup(potato, ingredient_dispenser) | wait(1..20)"}]
PRICE = {"input": 4.0, "cached_input": 0.2, "cache_write": 5.0, "output": 20.0}


class APITestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = {k: os.environ.get(k) for k in ("COOP_TEST_KEY", "COOP_COST_LEDGER", "COOP_MAX_COST", "COOP_LOCK_DIR")}
        os.environ["COOP_TEST_KEY"] = KEY
        os.environ["COOP_LOCK_DIR"] = str(self.tmp / "slots")
        os.environ.pop("COOP_COST_LEDGER", None)
        os.environ.pop("COOP_MAX_COST", None)
        self.fake = FakeAPI()

    def tearDown(self):
        self.fake.close()
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def openai(self, **kw):
        spec = dict(type="openai", base_url=self.fake.url, model="gpt-x", api_key_env="COOP_TEST_KEY",
                    reasoning_effort="low", expects_reasoning=True, reasoning_allowance=3000, price=PRICE,
                    params={"temperature": 0.0}, backoff_s=0.01)
        spec.update(kw)
        return build_backend(spec, "ego")

    def anthropic(self, **kw):
        spec = dict(type="anthropic", base_url=self.fake.url, model="claude-x", api_key_env="COOP_TEST_KEY",
                    anthropic_params={"output_config": {"effort": "low"}}, expects_reasoning=True,
                    reasoning_allowance=3000, price=PRICE, params={"temperature": 0.0}, backoff_s=0.01)
        spec.update(kw)
        return build_backend(spec, "ego")


class PayloadTests(APITestCase):
    def test_openai_payload(self):
        b = self.openai()
        self.assertIsInstance(b, OpenAIBackend)
        body = b.payload(MSGS, max_tokens=400, seed=3)
        self.assertNotIn("temperature", body)
        self.assertNotIn("seed", body)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], 3400)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(self.openai(prompt_cache_key="auto").payload(MSGS, 400)["prompt_cache_key"][:5], "coop-")
        self.assertIn("seed", self.openai(send_sampling=True).payload(MSGS, 400, seed=3))
        self.assertNotIn(KEY, repr(b))

    def test_anthropic_payload(self):
        b = self.anthropic()
        self.assertIsInstance(b, AnthropicBackend)
        body = b.payload(MSGS + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "again"}], 400)
        self.assertEqual(body["system"], [{"type": "text", "text": MSGS[0]["content"],
                                           "cache_control": {"type": "ephemeral"}}])
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "user"])
        self.assertEqual(body["max_tokens"], 3400)
        self.assertEqual(body["output_config"], {"effort": "low"})
        self.assertNotIn("temperature", body)
        self.assertEqual(b.payload(MSGS, None)["max_tokens"], 1024 + 3000)
        h = b.headers()
        self.assertEqual((h["x-api-key"], h["anthropic-version"]), (KEY, "2026-10-21"))


class ParseTests(APITestCase):
    def test_openai_roundtrip(self):
        self.fake.cached, self.fake.reasoning = 100, 50
        r = self.openai().chat(MSGS, max_tokens=400)
        self.assertEqual(r.error, "")
        self.assertIn("pickup(potato", r.text)
        self.assertEqual((r.cached_tokens, r.reasoning_tokens, r.served_model), (100, 50, "gpt-x"))
        self.assertFalse(r.thinking)                  # expected reasoning is not a health failure
        self.assertAlmostEqual(r.cost_usd, call_cost(PRICE, r.prompt_tokens, 100, 0, r.completion_tokens, 50))
        self.assertEqual(r.missing_usage, [])
        f = r.api_fields()
        self.assertEqual(set(f) >= {"cached_tokens", "reasoning_tokens", "cost_usd", "retries"}, True)

    def test_anthropic_roundtrip_truncation_and_refusal(self):
        self.fake.cached = 50
        r = self.anthropic().chat(MSGS, max_tokens=400)
        self.assertEqual(r.error, "")
        self.assertIn("pickup(potato", r.text)
        self.assertEqual(r.cached_tokens, 50)
        self.assertGreater(r.prompt_tokens, 50)         # whole prompt: uncached + read + written
        self.assertEqual((r.reasoning_tokens, r.finish_reason), (7, "stop"))
        self.fake.stop = "max_tokens"
        self.assertEqual(self.anthropic().chat(MSGS, 400).finish_reason, "length")
        self.fake.stop = "refusal"
        u = Usage()
        u.add(self.anthropic().chat(MSGS, 400))
        self.assertEqual((u.as_dict()["refusals"], u.as_dict()["truncated"]), (1, 0))

    def test_missing_usage_is_logged(self):
        self.fake.script.append((200, {}, {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                           "usage": {"prompt_tokens": 5, "completion_tokens": 2}}))
        r = self.openai().chat(MSGS, 10)
        self.assertIn("prompt_tokens_details.cached_tokens", r.missing_usage)

    def test_old_usage_keeps_its_keys(self):
        u = Usage()
        u.add(llm.LLMResult("x", prompt_tokens=3))
        self.assertEqual(sorted(u.as_dict()), ["calls", "completion_tokens", "errors", "prompt_tokens", "thinking",
                                               "truncated"])
        self.assertEqual(llm.LLMResult("x").api_fields(), {})


class RetryTests(APITestCase):
    def test_429_retry_after_503_529(self):
        self.fake.script += [(429, {"retry-after": "0.05"}, {"error": {"type": "rate_limit_error"}}),
                             (503, {}, {"error": "unavailable"}), (529, {}, {"error": {"type": "overloaded_error"}})]
        r = self.anthropic().chat(MSGS, 400)
        self.assertEqual(r.error, "")
        self.assertEqual(r.retries, 3)
        self.assertAlmostEqual(r.rate_limited_s, 0.05, places=3)
        self.assertEqual(len(self.fake.requests), 4)

    def test_400_not_retried_and_key_redacted(self):
        self.fake.script.append((400, {}, {"error": {"message": f"bad temperature; your key {KEY} is fine"}}))
        r = self.openai().chat(MSGS, 400)
        self.assertEqual(len(self.fake.requests), 1)
        self.assertIn("HTTP 400", r.error)
        self.assertNotIn(KEY, r.error)
        self.assertIn("[redacted]", r.error)

    def test_spend_cap_429_not_retried(self):
        self.fake.script.append((429, {}, {"error": {"details": {"error_code": "enforced_spend_limit_reached"}}}))
        r = self.anthropic().chat(MSGS, 400)
        self.assertEqual(len(self.fake.requests), 1)
        self.assertIn("429", r.error)

    def test_gives_up_after_max_retries(self):
        self.fake.script += [(500, {}, {})] * 3
        r = self.openai(max_retries=2).chat(MSGS, 400)
        self.assertEqual(len(self.fake.requests), 3)
        self.assertIn("HTTP 500", r.error)


class ConcurrencyCostBudgetTests(APITestCase):
    def test_slot_limit_across_threads(self):
        self.fake.delay_s = 0.15
        b = self.openai(max_concurrency=2, slot_name="t_threads")
        ts = [threading.Thread(target=b.chat, args=(MSGS, 50)) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(len(self.fake.requests), 6)
        self.assertLessEqual(self.fake.max_active, 2)

    def test_slot_limit_across_processes(self):
        self.fake.delay_s = 0.15
        code = ("import sys; sys.path[:0]=[%r]; from coop.llm import build_backend; "
                "b=build_backend(dict(type='openai', base_url=%r, model='m', max_concurrency=1, slot_name='t_procs',"
                " backoff_s=0.01), 'x'); [b.chat([{'role':'user','content':'VALID_ACTIONS: wait(1..2)'}], 5) for _ in range(2)]"
                % (str(ROOT / "src"), self.fake.url))
        ps = [subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ)) for _ in range(3)]
        [p.wait(timeout=60) for p in ps]
        self.assertEqual(len(self.fake.requests), 6)
        self.assertEqual(self.fake.max_active, 1)

    def test_cost_arithmetic(self):
        # 1000 prompt (200 cached, 100 written) and 500 output at 4 / 0.2 / 5 / 20 per million
        self.assertAlmostEqual(call_cost(PRICE, 1000, 200, 100, 500, 300),
                               (700 * 4 + 200 * 0.2 + 100 * 5 + 500 * 20) / 1e6)
        self.assertAlmostEqual(call_cost(dict(PRICE, reasoning_in_output=False), 1000, 0, 0, 500, 300),
                               (1000 * 4 + 800 * 20) / 1e6)
        self.assertEqual(call_cost(None, 1000, 0, 0, 500, 0), 0.0)

    def test_budget_stops_new_calls(self):
        os.environ["COOP_COST_LEDGER"] = str(self.tmp / "ledger.tsv")
        b = self.openai()
        one = b.chat(MSGS, 50).cost_usd
        self.assertGreater(one, 0)
        os.environ["COOP_MAX_COST"] = str(one * 2.5)
        b.chat(MSGS, 50)
        b.chat(MSGS, 50)
        self.assertGreaterEqual(CostLedger.total(), one * 2.5)
        n = len(self.fake.requests)
        with self.assertRaises(BudgetExceeded):
            b.chat(MSGS, 50)
        self.assertEqual(len(self.fake.requests), n)       # no request was sent


class RunnerTests(APITestCase):
    """coop.days / coop.decision with an API ego on the fake server."""

    def ego_spec(self):
        return dict(type="anthropic", base_url=self.fake.url, model="claude-x", api_key_env="COOP_TEST_KEY",
                    anthropic_params={"output_config": {"effort": "low"}}, expects_reasoning=True,
                    reasoning_allowance=3000, price=PRICE, backoff_s=0.01)

    def days_cfg(self):
        import contextlib
        import io
        from coop.days import load_config
        text = (ROOT / "configs/coop/days_ctx.yaml").read_text().replace("type: openai_compat", "type: mock")
        (self.tmp / "c.yaml").write_text(text)
        cfg = load_config(self.tmp / "c.yaml", run_seeds=[0], days=1, episodes_per_day=1, horizon=60,
                          conditions=["tom_struct"], personas=["A"])
        cfg.agent = dict(cfg.agent, backend=self.ego_spec())
        cfg.eval_episodes = 1
        return cfg, contextlib, io

    def test_registry_ref_and_override(self):
        from coop.days import override_backends
        from coop.llm import resolve_spec
        s = resolve_spec({"ref": "opus", "max_concurrency": 2})
        self.assertEqual((s["type"], s["model"], s["max_concurrency"], s["ref"]), ("anthropic", "claude-opus-5-5", 2, "opus"))
        with self.assertRaises(ValueError):
            resolve_spec({"ref": "nope"})
        cfg, _, _ = self.days_cfg()
        override_backends(cfg, ego="opus")
        self.assertEqual(cfg.condition_agent("tom_struct")["backend"]["ref"], "opus")
        with self.assertRaises(ValueError):          # the scripted partner has no backend
            from coop.days import DaysConfig
            override_backends(DaysConfig(), partner="opus")

    def test_days_budget_stop_resume_costs_and_no_key_leak(self):
        from coop.costs import days_folder_costs
        from coop.days import run_all
        cfg, contextlib, io = self.days_cfg()
        out = self.tmp / "out"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = run_all(cfg, out, 1, log=lambda s: None, max_cost=0.03)
        self.assertEqual(res["runs"][0]["status"], "budget")
        self.assertTrue((out / "runs/tom_struct/A/seed0/budget.log").exists())
        spent = CostLedger.total()
        self.assertGreaterEqual(spent, 0.03)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = run_all(cfg, out, 1, log=lambda s: None, max_cost=1000)
        self.assertEqual(res["runs"][0]["status"], "ok")
        self.assertGreater(CostLedger.total(), spent)
        s = json.loads((out / "runs/tom_struct/A/seed0/day01/ep01/summary.json").read_text())
        self.assertGreater(s["llm_usage"]["cost_usd"], 0)
        self.assertIn("cached_tokens", s["llm_usage"])
        log = json.loads((out / "runs/tom_struct/A/seed0/day01/ep01/agent_log.json").read_text())
        self.assertIn("cost_usd", log["calls"][0])
        lines = days_folder_costs(out)
        self.assertTrue(any("tom_struct" in ln for ln in lines))
        self.assertIn("cost ledger", lines[-1])
        self.assert_no_key(out)
        sent = [r["headers"].get("X-Api-Key") or r["headers"].get("x-api-key") for r in self.fake.requests]
        self.assertTrue(all(k == KEY for k in sent))   # the key was used, only in the header

    def test_decision_llm_stage_with_api_ego(self):
        import contextlib
        import io
        from coop import decision
        cfg = decision.load(ROOT / "configs/coop/decision/urgent.yaml")
        cfg.update(pairs=[["boiled_carrot_slices", "baked_potato_soup"]], urgent_limits=[60], seeds=[0],
                   ego_slots=[0])
        cfg["agent"] = dict(cfg["agent"], backend=self.ego_spec())
        out = self.tmp / "dec"
        os.environ["COOP_COST_LEDGER"] = str(out / "cost_ledger.tsv")
        with contextlib.redirect_stdout(io.StringIO()):
            decision.screen(cfg, out)
            res = decision.llm_stage(cfg, out)
        self.assertEqual(len(res), 6)
        self.assertGreater(res[0]["usage"]["cost_usd"], 0)
        self.assertIn("API cost per model and condition", decision.report(cfg, out))
        self.assert_no_key(out)

    def assert_no_key(self, out):
        hits = [str(f) for f in Path(out).rglob("*") if f.is_file() and KEY.encode() in f.read_bytes()]
        self.assertEqual(hits, [], "API key written to disk")


class SmokeTests(APITestCase):
    def test_smoke_report(self):
        import contextlib
        import io
        import yaml
        from coop import check
        reg = {"fake_anth": {"type": "anthropic", "base_url": self.fake.url, "model": "claude-x",
                             "api_key_env": "COOP_TEST_KEY", "expects_reasoning": True, "backoff_s": 0.01,
                             "anthropic_params": {"output_config": {"effort": "low"}}, "price": PRICE}}
        (self.tmp / "reg.yaml").write_text(yaml.safe_dump(reg))
        old = os.environ.get("COOP_BACKENDS")
        os.environ["COOP_BACKENDS"] = str(self.tmp / "reg.yaml")
        try:
            self.fake.cached = 80
            self.fake.script = [None] * 6 + [(400, {}, {"error": {"message": "thinking.type disabled is not supported"}})]
            with contextlib.redirect_stdout(io.StringIO()):
                rep = check.smoke("fake_anth", planned_calls=1000, out_dir=str(self.tmp / "chk"))
        finally:
            if old is None:
                os.environ.pop("COOP_BACKENDS", None)
            else:
                os.environ["COOP_BACKENDS"] = old
        self.assertEqual(len(rep["calls"]), 5)
        self.assertTrue(all(r["json_ok"] and r["action_valid"] for r in rep["calls"]), rep["calls"])
        self.assertEqual(rep["cache_read_on_repeat"], 80)
        self.assertTrue(rep["reasoning_included_in_output"])
        self.assertEqual([p["accepted"] for p in rep["probes"]], [True, False, True])
        self.assertIn("thinking", rep["rejected"][0]["param"])
        self.assertAlmostEqual(rep["projected_cost"], rep["mean_cost_per_call"] * 1000)
        text = check.format_smoke(rep)
        self.assertIn("REJECTED", text)
        self.assertNotIn(KEY, text)
        self.assertNotIn(KEY, (self.tmp / "chk" / "fake_anth.json").read_text())
        self.assertTrue(check._in_valid("wait(5)", ["pickup(dish, dish_dispenser)", "wait(1..20)"]))
        self.assertFalse(check._in_valid("wait(25)", ["wait(1..20)"]))


if __name__ == "__main__":
    unittest.main()
