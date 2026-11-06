"""Simulated human partner (src/coop/partner.py), safety filter and utensil fix."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from overcooked_ai_py.data.layouts import read_layout_dict
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld
from coop.env import ContinuousOrderEnv
from coop.partner import PERSONAS, SimulatedHuman, Traits
from coop.screen import DEFAULT_MENU, simulate

HORIZON = 300


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def run(persona, index, seed=0, horizon=HORIZON):
    return simulate(DEFAULT_MENU, "new_env", horizon, seed, persona, index)


class TraitValidationTests(unittest.TestCase):
    def test_invalid_traits(self):
        bad = [dict(preference=("prep", "prep")), dict(preference=("chop",)), dict(reluctance=3),
               dict(step_period=0), dict(decision_delay=-1), dict(slip_rate=1.5),
               dict(abandon_rate=-0.1)]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Traits(**kwargs)


class PartnerBehaviourTests(unittest.TestCase):
    def test_steady_team_serves_orders_in_both_slots(self):
        for index in (0, 1):
            with self.subTest(index=index):
                env, _ = run("steady", index)
                s = env.summary()
                self.assertGreaterEqual(s["orders_completed"], 8)
                self.assertEqual(s["wrong_deliveries"], 0)
                self.assertEqual(s["rejected_interactions"], 0)

    def test_reproducible_for_a_seed(self):
        a_env, a = run("flaky", 1, seed=3)
        b_env, b = run("flaky", 1, seed=3)
        self.assertEqual(a_env.summary(), b_env.summary())
        self.assertEqual(a[1].trace, b[1].trace)
        self.assertEqual(a[1].events, b[1].events)
        c_env, c = run("flaky", 1, seed=4)
        self.assertNotEqual(a[1].events, c[1].events)

    def test_slow_acts_on_every_other_step(self):
        _, players = run("slow", 1)
        trace = players[1].trace
        self.assertTrue(all(r["status"] == "off_beat" for r in trace if r["t"] % 2 == 1))
        self.assertFalse(any(r["status"] == "off_beat" for r in trace if r["t"] % 2 == 0))
        self.assertGreater(sum(r["status"] == "deciding" for r in trace), 0)

    def test_sloppy_slips_and_the_wrong_batch_is_cleared(self):
        slips = cleared = orders = 0
        for seed in range(4):
            env, players = run("sloppy", 1, seed=seed)
            slips += sum(e["type"] == "slip" for e in players[1].events)
            cleared += sum(1 for r in players[1].trace if r["job"] and r["job"]["kind"] == "clear")
            orders += env.summary()["orders_completed"]
            self.assertEqual(env.summary()["rejected_interactions"], 0)
        self.assertGreater(slips, 0)
        self.assertGreater(cleared, 0)     # a spoiled batch is started, so the env discards it
        self.assertGreater(orders, 0)      # and the team recovers

    def test_flaky_abandons_and_respects_cooldown(self):
        _, players = run("flaky", 1)
        p = players[1]
        abandons = [e for e in p.events if e["type"] == "abandon"]
        self.assertGreater(len(abandons), 0)
        cooldown = PERSONAS["flaky"].abandon_cooldown
        for e in abandons:
            later = [r for r in p.trace if e["t"] < r["t"] < e["t"] + cooldown and r["job"]]
            self.assertFalse(any(r["job"] == e["job"] for r in later), e)

    def test_preference_waits_at_most_reluctance_steps(self):
        _, players = run("serve_first", 1)
        trace = players[1].trace
        self.assertGreater(sum(r["status"] == "reluctant" for r in trace), 0)
        streak = longest = 0
        for r in trace:
            streak = streak + 1 if r["status"] == "reluctant" else 0
            longest = max(longest, streak)
        self.assertLessEqual(longest, PERSONAS["serve_first"].reluctance)

    def test_personas_change_team_outcome(self):
        base = run("steady", 1)[0].summary()["score"]
        for persona in ("slow", "flaky", "serve_first"):
            with self.subTest(persona=persona):
                self.assertLess(run(persona, 1)[0].summary()["score"], base)


class SafetyFilterTests(unittest.TestCase):
    def setUp(self):
        self.env = ContinuousOrderEnv(["boiled_egg"], horizon=20)
        quiet(self.env.reset, 0)
        self.chef = self.env.state.players[0]       # new_env: (3, 1), pot P at (3, 0) above
        self.chef.update_pos_and_or((3, 1), Direction.NORTH)

    def test_start_on_empty_pot_is_rejected(self):
        _, _, _, info = quiet(self.env.step, [Action.INTERACT, Action.STAY], ["[START]", None])
        self.assertEqual(len(info["rejected"]), 1)
        self.assertEqual(info["rejected"][0]["agent"], 0)

    def test_adding_to_a_running_pot_is_rejected_instead_of_crashing(self):
        self.env.state.objects[(3, 0)] = ObjectState("soup", "semi-finished", (3, 0), ("boiled_egg", 1, 1))
        self.chef.set_object(ObjectState("egg", "ingredient", (3, 1)))
        _, _, _, info = quiet(self.env.step, [Action.INTERACT, Action.STAY], [None, None])
        self.assertEqual(len(info["rejected"]), 1)
        self.assertEqual(self.chef.held_object.name, "egg")


class UtensilStartFixTests(unittest.TestCase):
    # Chopping board C (0, 2) and blender B (0, 3) share a column.
    GRID = ["XXXXXXX",
            "I  1  P",
            "C     S",
            "B  2  O",
            "XXXDXXX"]

    def test_starting_one_utensil_leaves_a_batch_in_the_same_column(self):
        base = read_layout_dict("new_env")
        base.pop("grid")
        base["layout_name"] = "column_test"
        mdp = OvercookedGridworld.from_grid(self.GRID, base)
        mdp.start_order_list, mdp.order_probability = ["carrot_slices"], {}
        env = quiet(OvercookedEnv, mdp, horizon=10)
        state = env.state
        state.objects[(0, 2)] = ObjectState("soup", "semi-finished", (0, 2), (["carrot"], 1, 0))
        state.objects[(0, 3)] = ObjectState("soup", "semi-finished", (0, 3), (["peanut"], 1, 0))
        state.players[0].update_pos_and_or((1, 2), Direction.WEST)
        quiet(env.step, [Action.INTERACT, Action.STAY], ["[START]", None])
        self.assertEqual(env.state.objects[(0, 2)].state[0], "carrot_slices")
        self.assertIn((0, 3), env.state.objects)     # before the fix this batch was deleted
        self.assertEqual(env.state.objects[(0, 3)].state[0], ["peanut"])


if __name__ == "__main__":
    unittest.main()
