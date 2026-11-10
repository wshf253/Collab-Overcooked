"""LLM ego (src/coop/llm_agent.py): executor, prompts, memory, probe, backends."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from overcooked_ai_py.mdp.actions import Action
from coop.days import DaysConfig, play_episode, run_all
from coop.env import ContinuousOrderEnv
from coop.executor import Executor, parse_macro
from coop.kitchen import Kitchen
from coop.llm import LLMResult, build_backend
from coop.llm_agent import LLMAgent
from coop.partner import SimulatedHuman


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class ParseTests(unittest.TestCase):
    def test_valid_actions(self):
        for text, expect in [("pickup(carrot, ingredient_dispenser)", "pickup(carrot, ingredient_dispenser)"),
                             ("  Cook( pot0 ) ;", "cook(pot0)"), ("deliver_soup()", "deliver_soup()"),
                             ("wait(5)", "wait(5)"), ("'place_obj_on_counter()'", "place_obj_on_counter()")]:
            with self.subTest(text=text):
                m, reason = parse_macro(text)
                self.assertEqual((str(m), reason), (expect, ""))

    def test_invalid_actions(self):
        for text in ("", "fly()", "pickup(carrot)", "wait(0)", "wait(21)", "wait(x)", "cook pot0",
                     "deliver_soup(now)"):
            with self.subTest(text=text):
                m, reason = parse_macro(text)
                self.assertIsNone(m)
                self.assertTrue(reason)


class ExecutorTests(unittest.TestCase):
    """Two executors make a baked carrot soup on new_env through high-level actions only."""

    def setUp(self):
        self.env = ContinuousOrderEnv(["baked_carrot_soup"], horizon=400)
        quiet(self.env.reset, 0)
        self.ex = [Executor(Kitchen(self.env.mdp, i, self.env.state)) for i in (0, 1)]

    def macro(self, i, text, max_steps=80):
        ex, env = self.ex[i], self.env
        m, reason = parse_macro(text)
        self.assertIsNotNone(m, reason)
        self.assertIsNone(ex.check(m, env.state), text)
        ex.begin(m)
        for _ in range(max_steps):
            ended = ex.update(env.state)
            if ended:
                return ended
            a, p = ex.step(env.state)
            acts, parms = [Action.STAY, Action.STAY], [None, None]
            acts[i], parms[i] = a, p
            _, _, _, info = quiet(env.step, acts, parms)
            for r in info["rejected"]:
                if r["agent"] == i:
                    ex.note_rejected(r["reason"])
        self.fail(f"{text} did not end in {max_steps} steps")

    def done(self, i, text, expect=None):
        status, msg = self.macro(i, text)
        self.assertEqual(status, "done", msg)
        if expect:
            self.assertEqual(msg, expect)

    def test_full_recipe_with_handovers(self):
        L, R = 1, 0      # left: dispenser, board, blender, dishes; right: pot, oven, serving
        self.done(L, "pickup(carrot, ingredient_dispenser)", "pickup(carrot, ingredient_dispenser)")
        self.done(L, "put_obj_in_utensil(chopping_board0)", "put_obj_in_utensil(chopping_board0)")
        self.done(L, "cut(chopping_board0)", "cut(chopping_board0)")
        self.done(L, "wait(4)")
        self.done(L, "pickup(carrot_slices, chopping_board0)")
        self.done(L, "place_obj_on_counter()", "place_obj_on_counter()")
        self.done(R, "pickup(carrot_slices, counter)", "pickup(carrot_slices, counter)")
        self.done(R, "put_obj_in_utensil(oven0)")
        self.done(R, "bake(oven0)", "bake(oven0)")
        self.done(R, "wait(4)")
        self.done(R, "pickup(baked_carrot_slices, oven0)")
        self.done(R, "put_obj_in_utensil(pot0)")
        self.done(R, "cook(pot0)", "cook(pot0)")
        self.done(L, "pickup(dish, dish_dispenser)", "pickup(dish, dish_dispenser)")
        self.done(L, "place_obj_on_counter()")
        self.done(R, "pickup(dish, counter)")
        self.done(R, "fill_dish_with_food(pot0)", "fill_dish_with_food(pot0)")
        self.done(R, "deliver_soup()", "deliver_soup()")
        self.assertEqual(self.env.score, 20)
        self.assertEqual(self.env.summary()["orders_completed"], 1)
        self.assertEqual(self.env.summary()["rejected_interactions"], 0)

    def test_preconditions_explain_why(self):
        R, L = 0, 1
        cases = [(R, "pickup(carrot, ingredient_dispenser)", "cannot reach the ingredient dispenser"),
                 (L, "pickup(steak, ingredient_dispenser)", "not a raw ingredient"),
                 (L, "cook(chopping_board0)", "is started with cut(chopping_board0)"),
                 (L, "cut(chopping_board0)", "cannot be started"),
                 (R, "fill_dish_with_food(pot0)", "need to hold a dish"),
                 (R, "deliver_soup()", "hold nothing"),
                 (R, "pickup(carrot, counter)", "no counter you can reach holds carrot"),
                 (R, "put_obj_in_utensil(pan0)", "unknown utensil")]
        for i, text, phrase in cases:
            with self.subTest(text=text):
                m, _ = parse_macro(text)
                self.assertIn(phrase, self.ex[i].check(m, self.env.state))
        self.done(L, "pickup(carrot, ingredient_dispenser)")
        m, _ = parse_macro("pickup(dish, dish_dispenser)")
        self.assertIn("hands are full", self.ex[L].check(m, self.env.state))

    def test_options_list_only_possible_actions(self):
        opts = self.ex[1].options(self.env.state, ["carrot"])
        self.assertIn("pickup(carrot, ingredient_dispenser)", opts)
        self.assertIn("pickup(dish, dish_dispenser)", opts)
        self.assertNotIn("deliver_soup()", opts)
        self.assertEqual(opts[-1], "wait(1..20)")
        self.assertEqual(self.ex[0].options(self.env.state, ["carrot"]), ["wait(1..20)"])

    def test_wait_ends_early_when_partner_acts(self):
        partner = SimulatedHuman.persona("steady")
        partner.reset(self.env.mdp, 1, 0)
        ex = self.ex[0]
        m, _ = parse_macro("wait(20)")
        ex.begin(m)
        for _ in range(25):
            ended = ex.update(self.env.state)
            if ended:
                break
            a0, _ = ex.step(self.env.state)
            a1, p1 = partner.act(self.env.state, self.env.t)
            quiet(self.env.step, [a0, a1], [None, p1])
        self.assertEqual(ended[0], "done")
        self.assertIn("your partner just did pickup(carrot, ingredient_dispenser)", ended[1])


def mock_config(**kw):
    base = dict(name="t", menu=["boiled_egg", "baked_carrot_soup"], horizon=60, days=2,
                episodes_per_day=2, personas=["flaky"], run_seeds=[0],
                agent={"type": "llm", "backend": {"type": "mock"}, "probe": True},
                conditions={"direct": {}, "tom": {"tom": True}, "memory": {"memory": True}},
                reference_agent=None)
    base.update(kw)
    return DaysConfig(**base)


class LLMAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.res = run_all(mock_config(), cls.tmp, log=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def ep(self, cond, day, ep, name):
        p = self.tmp / "runs" / cond / "flaky" / "seed0" / f"day{day:02d}" / f"ep{ep:02d}" / name
        return json.loads(p.read_text())

    def test_runs_and_logs(self):
        self.assertTrue(all(r["status"] == "ok" for r in self.res["runs"]))
        log = self.ep("tom", 1, 1, "agent_log.json")
        self.assertGreater(log["usage"]["calls"], 5)
        self.assertEqual(log["fallbacks"], 0)
        self.assertTrue(any(c["kind"] == "probe" for c in log["calls"]))
        self.assertTrue(all(d["attempts"][0]["partner"] == "mock belief" for d in log["decisions"]))
        for cond in ("direct", "tom", "memory"):
            self.assertIn("you cannot ask for it", self.ep(cond, 1, 1, "agent_log.json")["system_prompt"])

    def test_prompts_never_reveal_the_partner(self):
        for cond in ("direct", "tom", "memory"):
            for day, ep in ((1, 1), (2, 2)):
                log = self.ep(cond, day, ep, "agent_log.json")
                text = log["system_prompt"] + "".join(c["user"] for c in log["calls"])
                for secret in ("flaky", "abandon_rate", "slip_rate", "step_period", "persona", "traits"):
                    self.assertNotIn(secret, text, (cond, day, ep, secret))

    def test_only_the_reply_format_differs_between_direct_and_tom(self):
        a = self.ep("direct", 1, 1, "agent_log.json")["system_prompt"].split("Reply with JSON only")[0]
        b = self.ep("tom", 1, 1, "agent_log.json")["system_prompt"].split("Reply with JSON only")[0]
        self.assertEqual(a, b)

    def test_memory_carries_notes_into_later_prompts(self):
        head = "What you noted after earlier episodes"
        self.assertNotIn(head, self.ep("memory", 1, 1, "agent_log.json")["system_prompt"])
        later = self.ep("memory", 2, 1, "agent_log.json")["system_prompt"]
        self.assertIn(head, later)
        self.assertIn("day 1, episode 2", later)
        mem = self.ep("memory", 2, 2, "memory_after.json")
        self.assertEqual([(n["day"], n["episode"]) for n in mem["notes"]], [(1, 1), (1, 2), (2, 1), (2, 2)])
        # Without memory, probes are answered but never fed back.
        for day, ep in ((1, 1), (2, 2)):
            self.assertNotIn(head, self.ep("direct", day, ep, "agent_log.json")["system_prompt"])
        self.assertEqual(self.ep("direct", 2, 2, "memory_after.json")["notes"], [])

    def test_analysis_report(self):
        from coop.analyze import analyse, decision_stats, format_report
        rep = analyse(self.tmp)
        self.assertEqual(set(rep["conditions"]), {"direct", "tom", "memory"})
        row = rep["conditions"]["tom"]["flaky"]
        self.assertEqual(row["episodes"], 4)
        self.assertEqual(row["probe"]["commitment"]["truth"], "abandons tasks")
        self.assertIsNotNone(rep["overall"]["tom"]["vs_direct"]["mean"])
        self.assertIn("partner probe", format_report(rep))
        self.assertIn("steps per completed order, by dish", format_report(rep))
        for cond in ("direct", "tom", "memory"):
            for dish, r in rep["dishes"].get(cond, {}).items():
                self.assertIn(dish, ("boiled_egg", "baked_carrot_soup"))
        system = self.ep("tom", 1, 1, "agent_log.json")["system_prompt"]
        fake = {"system_prompt": system, "decisions": [], "calls": [
            {"kind": "decide", "user": "Utensils: pot0: boiled_egg is ready to take out; blender0: x is ready "
             "to take out.\nResult of your last action: done: waited 10 step(s)"},
            {"kind": "decide", "user": "Utensils: blender0: x is ready to take out.\n"
             "Result of your last action: done: waited 10 step(s)"},
            {"kind": "decide", "user": "Utensils: pot0: boiled_egg is ready to take out.\n"
             "Result of your last action: done: waited 2 step(s); your partner just did y"}]}
        st = decision_stats(fake)     # ego is slot 0: it reaches pot0 and oven0, not blender0
        self.assertEqual((st["waits_full"], st["waits_ended_by_partner"], st["full_waits_with_ready_food"]),
                         (2, 1, 1))
        st = decision_stats({"decisions": [
            {"chosen": "wait(1)", "attempts": [{"action": "wait(1..20)", "error": "wait needs a whole number of steps"},
                                               {"action": "pickup(egg, ingredient_dispenser)",
                                                "error": "you cannot reach the ingredient dispenser"}]},
            {"chosen": "cook(pot0)", "attempts": [{"action": "cook(pot0)", "error": ""}]}]})
        self.assertEqual(st["invalid_reasons"], {"wait needs a whole number of steps": 1,
                                                 "you cannot reach the ingredient dispenser": 1})
        self.assertEqual(st["fallback_pairs"], {"wait(1..20) -> pickup(_, ingredient_dispenser)": 1})

    def test_probe_is_scored_against_the_true_persona(self):
        probe = self.ep("direct", 1, 1, "summary.json")["probe"]
        self.assertEqual(probe["truth"], {"prefers": "no clear preference", "avoids": "nothing",
                                          "gives_way": "does not give way", "speed": "normal",
                                          "reliability": "reliable", "commitment": "abandons tasks"})
        self.assertFalse(probe["correct"]["commitment"])     # the mock always says "finishes tasks"
        self.assertAlmostEqual(probe["accuracy"], 5 / 6)
        self.assertAlmostEqual(self.res["table"]["direct"]["flaky"]["1"]["probe_accuracy"], 5 / 6)


class FakeBackend:
    def __init__(self, replies):
        self.replies, self.i = replies, 0

    def chat(self, messages, max_tokens=None, seed=None, json_mode=None):
        text = self.replies[self.i % len(self.replies)]
        self.i += 1
        if "PARTNER_PROFILE_PROBE" in messages[-1]["content"]:
            text = '{"partner_profile": {"prefers": "Prep", "speed": "fast"}, "notes": 3}'
        return LLMResult(text, finish_reason="stop")


class RetryTests(unittest.TestCase):
    def play(self, replies):
        cfg = mock_config(personas=["steady"], days=1, episodes_per_day=1, horizon=30)
        agent = LLMAgent({"type": "mock"}, probe=True)
        agent.backend = FakeBackend(replies)
        agent.set_memory({})
        with tempfile.TemporaryDirectory() as d:
            summary = quiet(play_episode, cfg, agent, "steady", 0, 1, 1, Path(d))
        return agent, summary

    def test_one_retry_with_the_reason(self):
        agent, _ = self.play(['{"action": "fly()"}', '{"action": "wait(1)"}'])
        self.assertGreater(agent.invalid, 0)
        self.assertEqual(agent.fallbacks, 0)
        retry = [c for c in agent.calls if c["kind"] == "retry"][0]
        self.assertIn("unknown action 'fly'", retry["user"])

    def test_fallback_after_two_invalid_replies(self):
        agent, summary = self.play(["not json at all"])
        self.assertEqual(agent.fallbacks, len(agent.decisions))
        self.assertTrue(all(d["chosen"] == "wait(1)" for d in agent.decisions))
        # Invalid profile values are dropped instead of guessed.
        self.assertEqual(summary["probe"]["answer"], {"prefers": "prep", "avoids": None, "gives_way": None,
                                                      "speed": None, "reliability": None, "commitment": None})


class OpenAIBackendTests(unittest.TestCase):
    def test_request_shape_and_env_settings(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"] = self.path
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen["auth"] = self.headers.get("Authorization")
                out = json.dumps({"choices": [{"message": {"content": '{"action": "wait(2)"}'},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 11, "completion_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = {"T_URL": f"http://127.0.0.1:{server.server_port}/v1", "T_MODEL": "qwen-test",
               "T_KEY": "secret", "T_KW": '{"enable_thinking": false}'}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            b = build_backend({"type": "openai_compat", "base_url_env": "T_URL", "model_env": "T_MODEL",
                               "model": "unused", "api_key_env": "T_KEY", "chat_template_kwargs_env": "T_KW",
                               "params": {"temperature": 0.0}})
            r = b.chat([{"role": "user", "content": "hi"}], max_tokens=50, seed=7)
        finally:
            server.shutdown()
            for k, v in old.items():
                os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
        self.assertEqual(r.text, '{"action": "wait(2)"}')
        self.assertEqual((r.prompt_tokens, r.completion_tokens), (11, 3))
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer secret")
        body = seen["body"]
        self.assertEqual(body["model"], "qwen-test")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((body["max_tokens"], body["seed"], body["temperature"]), (50, 7, 0.0))
        self.assertEqual(body["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
