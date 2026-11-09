"""Memory controls (swap, placebo), discard attribution and stage-share scoring."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState
from coop.days import PLACEBO_SENTENCES, DaysConfig, placebo_text, run_all
from coop.env import ContinuousOrderEnv
from coop.llm_agent import LLMAgent
from coop.profile import clean_shares, score_profile


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def controls_config(**kw):
    base = dict(name="c", menu=["boiled_egg", "baked_carrot_soup"], horizon=50, days=2, episodes_per_day=2,
                personas=["steady", "slow", "flaky"], run_seeds=[0],
                agent={"type": "llm", "backend": {"type": "mock"}, "probe": True},
                conditions={"tom_memory": {"tom": True, "memory": True},
                            "swap": {"tom": True, "memory": True,
                                     "memory_source": {"type": "swap", "from": "tom_memory"}},
                            "placebo": {"tom": True, "memory": True,
                                        "memory_source": {"type": "placebo", "from": "tom_memory"}}},
                reference_agent=None)
    base.update(kw)
    return DaysConfig(**base)


class ControlRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.res = run_all(controls_config(), cls.tmp, workers=2, log=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def mem(self, cond, persona, day, ep):
        p = self.tmp / "runs" / cond / persona / "seed0" / f"day{day:02d}" / f"ep{ep:02d}" / "memory_after.json"
        return json.loads(p.read_text())

    def prompt(self, cond, persona, day, ep):
        p = self.tmp / "runs" / cond / persona / "seed0" / f"day{day:02d}" / f"ep{ep:02d}" / "agent_log.json"
        return json.loads(p.read_text())["system_prompt"]

    def test_all_runs_ok(self):
        self.assertTrue(all(r["status"] == "ok" for r in self.res["runs"]))

    def test_swap_shows_the_donor_partners_notes(self):
        # Default donors: the next persona in the list (steady <- slow <- flaky <- steady).
        for persona, donor in (("steady", "slow"), ("slow", "flaky"), ("flaky", "steady")):
            given = self.mem("swap", persona, 2, 2)
            self.assertEqual(given["source"], {"condition": "tom_memory", "persona": donor, "after": [2, 1]})
            donor_notes = self.mem("tom_memory", donor, 2, 1)["notes"]
            self.assertEqual(given["notes"][:len(donor_notes)], donor_notes)
        self.assertNotIn("What you noted after earlier episodes", self.prompt("swap", "slow", 1, 1))
        self.assertIn("What you noted after earlier episodes", self.prompt("swap", "slow", 1, 2))

    def test_placebo_keeps_shape_but_no_partner_information(self):
        own = self.mem("tom_memory", "slow", 2, 1)["notes"]
        given = self.mem("placebo", "slow", 2, 2)["notes"][:len(own)]
        self.assertEqual([(n["day"], n["episode"]) for n in given], [(n["day"], n["episode"]) for n in own])
        for g, o in zip(given, own):
            self.assertTrue(all(v is None for v in g["profile"].values()))
            self.assertLessEqual(len(g["notes"]), len(o["notes"]))
            self.assertTrue(any(g["notes"] and sent.startswith(g["notes"].split(" ")[0])
                                for sent in PLACEBO_SENTENCES))
        self.assertIn("profile (not noted)", self.prompt("placebo", "slow", 2, 2))

    def test_analysis_compares_controls_to_tom_memory(self):
        from coop.analyze import analyse
        rep = analyse(self.tmp)
        self.assertIn("vs_tom_memory_runs", rep["overall"]["swap"])
        self.assertEqual(rep["overall"]["swap"]["vs_tom_memory_runs"]["n"], 3)
        self.assertIn("share_error", rep["conditions"]["tom_memory"]["slow"]["probe"])


class ControlConfigTests(unittest.TestCase):
    def test_invalid_controls(self):
        bad = [
            {"x": {"memory": True, "memory_source": {"type": "swap", "from": "nope"}}},
            {"x": {"memory": True, "memory_source": {"type": "swap", "from": "x"}}},
            {"x": {"memory": True, "memory_source": {"type": "copy", "from": "tom_memory"}}},
            {"x": {"memory": False, "memory_source": {"type": "swap", "from": "tom_memory"}}},
            {"x": {"memory": True, "memory_source": {"type": "swap", "from": "tom_memory",
                                                     "donors": {"steady": "steady", "slow": "steady",
                                                                "flaky": "steady"}}}},
        ]
        for extra in bad:
            conds = {"tom_memory": {"tom": True, "memory": True}, **extra}
            with self.subTest(extra=str(extra)), self.assertRaises(ValueError):
                controls_config(conditions=conds)
        with self.assertRaises(ValueError):      # the source keeps no memory
            controls_config(conditions={"direct": {}, "x": {"memory": True, "memory_source":
                                                              {"type": "placebo", "from": "direct"}}})
        with self.assertRaises(ValueError):      # a swap needs another persona to take notes from
            controls_config(personas=["steady"])

    def test_placebo_text(self):
        self.assertEqual(placebo_text(0), "")
        t = placebo_text(200)
        self.assertLessEqual(len(t), 200)
        self.assertGreater(len(t), 150)
        self.assertTrue(t.startswith(PLACEBO_SENTENCES[0]))


class AttributionTests(unittest.TestCase):
    def test_thrown_away_batch_names_who_loaded_and_started_it(self):
        env = ContinuousOrderEnv(["boiled_egg"], horizon=30)
        quiet(env.reset, 0)
        agent = LLMAgent({"type": "mock"}, probe=False)
        agent.begin_episode(env.mdp, 0, {"day": 1, "episode": 1, "episode_seed": 1, "layout": "new_env",
                                         "horizon": 30, "menu": {"boiled_egg": 1.0}, "index": 0,
                                         "partner_index": 1, "wrong_delivery_penalty": 20})
        partner = env.state.players[1]          # left side: the chopping board is at (0, 2)
        partner.update_pos_and_or((1, 2), Direction.WEST)
        partner.set_object(ObjectState("egg", "ingredient", (1, 2)))
        for parm in (None, "[START]", None):    # put the egg on the board, start it, then nothing
            a0, p0 = agent.act(env.state, env.t)
            _, r, _, info = quiet(env.step, [a0, Action.INTERACT if parm or env.t == 0 else Action.STAY],
                                  [p0, parm])
            agent.observe(env.t - 1, r, info)
        agent.act(env.state, env.t)
        events = [e for e in agent.events if "matched no recipe" in e]
        self.assertEqual(len(events), 1, agent.events)
        self.assertIn("your partner put in egg", events[0])
        self.assertIn("when your partner started it", events[0])


class ShareScoringTests(unittest.TestCase):
    def test_shares(self):
        self.assertEqual(clean_shares({"prep": 2, "cook": 1, "serve": 1}),
                         {"prep": 50.0, "cook": 25.0, "serve": 25.0})
        for bad in (None, {"prep": "x"}, {"prep": -1, "cook": 1, "serve": 0}, {"prep": 0, "cook": 0}):
            self.assertIsNone(clean_shares(bad))
        s = score_profile({"prefers": "prep"}, {"prefers": "prep"},
                          {"prep": 50.0, "cook": 25.0, "serve": 25.0}, {"prep": 0.8, "cook": 0.1, "serve": 0.1})
        self.assertAlmostEqual(s["share_error"], (30 + 15 + 15) / 3)
        self.assertNotIn("share_error", score_profile({}, {"prefers": "prep"}))


if __name__ == "__main__":
    unittest.main()
