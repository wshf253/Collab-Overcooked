"""Probe rules: which fields are asked and scored, the observed pace line, rescoring."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.analyze import rescore_probe
from coop.days import play_episode
from coop.env import ContinuousOrderEnv
from coop.kitchen import Kitchen
from coop.llm_agent import LLMAgent
from coop.partner import PERSONAS, SimulatedHuman
from coop.profile import (PaceCounter, avoid_options, legacy_answer, probe_fields, score_profile,
                          truth_profile)
from coop.screen import DEFAULT_MENU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_coop_llm import mock_config  # noqa: E402


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def kitchen(layout):
    env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout=layout)
    return Kitchen(env.mdp, 0, quiet(env.reset, 0))


class AvoidTests(unittest.TestCase):
    def test_options_need_a_shared_needed_utensil(self):
        self.assertEqual(avoid_options(kitchen("new_env"), DEFAULT_MENU), [])     # split map
        k = kitchen("open_env")
        self.assertEqual(avoid_options(k, DEFAULT_MENU), ["chopping board", "pot", "oven"])  # no blender
        self.assertEqual(avoid_options(k, ["boiled_egg"]), ["pot"])
        self.assertNotIn("avoids", probe_fields([]))
        self.assertEqual(probe_fields(["pot"])["avoids"], ["pot", "nothing"])

    def test_truth_depends_on_the_episode(self):
        board = PERSONAS["avoids_board"]
        self.assertEqual(truth_profile(board, ["chopping board", "pot"])["avoids"], "chopping board")
        self.assertEqual(truth_profile(board, ["pot"])["avoids"], "nothing")   # board not needed
        self.assertIsNone(truth_profile(board, [])["avoids"])                   # not asked
        self.assertEqual(truth_profile(PERSONAS["steady"], ["pot"])["avoids"], "nothing")

    def test_abandoning_is_not_scored(self):
        t = truth_profile(PERSONAS["flaky"], ["pot"])
        self.assertIsNone(t["commitment"])
        self.assertEqual(truth_profile(PERSONAS["leaves_unfinished"])["commitment"], "leaves tasks unfinished")
        # Traits stored as a dict (partner_truth.json) give the same truth.
        from dataclasses import asdict
        self.assertEqual(truth_profile(asdict(PERSONAS["slow"]), ["pot"]), truth_profile(PERSONAS["slow"], ["pot"]))
        score = score_profile({"prefers": "no clear preference", "avoids": "nothing", "commitment": "finishes tasks",
                               "gives_way": "does not give way", "speed": "normal", "reliability": "reliable"}, t)
        self.assertNotIn("commitment", score["correct"])
        self.assertEqual(score["accuracy"], 1.0)

    def test_legacy_answers(self):
        old = {"avoids": "blender", "commitment": "abandons tasks"}
        new = legacy_answer(old, ["chopping board", "pot"])
        self.assertEqual(new, {"avoids": "nothing", "commitment": "leaves tasks unfinished"})
        self.assertEqual(legacy_answer({"avoids": "pot"}, ["pot"])["avoids"], "pot")


class PaceTests(unittest.TestCase):
    def pace(self, persona, layout="open_env"):
        env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout=layout, horizon=200)
        quiet(env.reset, 3)
        a, b = SimulatedHuman(PERSONAS["steady"], "steady"), SimulatedHuman(PERSONAS[persona], persona)
        a.reset(env.mdp, 0, 3)
        b.reset(env.mdp, 1, 4)
        pc = PaceCounter()
        while not env.done:
            s, t = env.state, env.t
            ma, mb = a.act(s, t), b.act(s, t)
            quiet(env.step, [ma[0], mb[0]], [ma[1], mb[1]])
            pc.update(env.state.players[1])
        return pc

    def test_slow_is_visible_and_waiting_is_not_slowness(self):
        self.assertEqual(self.pace("slow").share(), 0.0)
        for persona in ("steady", "prep_first", "serve_first"):
            self.assertGreater(self.pace(persona).share(), 0.5, persona)
        self.assertIn("% of the time", self.pace("steady").text())

    def test_out_of_view_steps_are_not_counted(self):
        class P:
            def __init__(self, x):
                self.position, self.orientation, self.held_object = (x, 0), (0, 1), None
        pc = PaceCounter()
        for x, vis in ((0, True), (1, True), (2, False), (3, True), (4, True), (5, True)):
            pc.update(P(x), vis)
        self.assertEqual((pc.pairs, pc.again), (1, 1))     # only 3 -> 4 -> 5 is a seen pair
        self.assertIn("not see enough", pc.text())


class OpenProbeTests(unittest.TestCase):
    def test_open_map_asks_avoids_over_needed_utensils(self):
        cfg = mock_config(layout="open_env", menu=DEFAULT_MENU, personas=["avoids_board"], days=1,
                          episodes_per_day=1, horizon=80)
        agent = LLMAgent({"type": "mock"}, probe=True)
        with tempfile.TemporaryDirectory() as d:
            summary = quiet(play_episode, cfg, agent, "avoids_board", 0, 1, 1, Path(d))
            truth = json.loads((Path(d) / "partner_truth.json").read_text())
        dishes = [o["name"] for o in summary["orders"]]
        asked = agent.probe_answer["asked"]["avoids"]
        self.assertEqual(asked, truth["avoid_options"])
        self.assertNotIn("blender", asked)
        user = [c for c in agent.calls if c["kind"] == "probe"][0]["user"]
        self.assertIn("Utensils asked about under avoids", user)
        expect = "chopping board" if "chopping board" in asked else "nothing"
        self.assertEqual(summary["probe"]["truth"]["avoids"], expect, dishes)
        self.assertIn("avoids", summary["probe"]["correct"])

    def test_rescore_old_episode(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            from dataclasses import asdict
            (d / "partner_truth.json").write_text(json.dumps({"traits": asdict(PERSONAS["flaky"])}))
            answer = {"prefers": "no clear preference", "avoids": "blender", "gives_way": "does not give way",
                      "speed": "normal", "reliability": "reliable", "commitment": "finishes tasks"}
            old = {"answer": answer, "truth": {}, "correct": {}, "accuracy": 4 / 6, "share_error": 7.0}
            summary = {"layout": "open_env", "ego_index": 0, "probe": old,
                       "orders": [{"name": "boiled_egg"}, {"name": "baked_carrot_soup"}]}
            new = quiet(rescore_probe, {"summary": summary, "dir": d})
        self.assertNotIn("commitment", new["correct"])
        self.assertTrue(new["correct"]["avoids"])          # "blender" was never needed -> "nothing"
        self.assertEqual(new["accuracy"], 1.0)
        self.assertEqual(new["share_error"], 7.0)


if __name__ == "__main__":
    unittest.main()
