"""Open kitchen layout (open_env) and the no-deadlock movement rules."""
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
from coop.days import load_config
from coop.env import ContinuousOrderEnv
from coop.executor import Executor, parse_macro
from coop.kitchen import Kitchen
from coop.nav import Unstick, access_cells, floor_cells, step_aside
from coop.screen import DEFAULT_MENU, simulate


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


OPEN = ("open_env", "open2_env")


class OpenLayoutTests(unittest.TestCase):
    def test_same_recipes_as_new_env(self):
        for layout in OPEN:
            a, b = read_layout_dict("new_env"), read_layout_dict(layout)
            a.pop("grid"), b.pop("grid")
            self.assertEqual(a, b, layout)

    def test_two_of_every_utensil(self):
        env = ContinuousOrderEnv(["boiled_egg"], layout="open2_env")
        quiet(env.reset, 0)
        self.assertEqual(sorted(env.mdp.utensil_list),
                         sorted(f"{k}{i}" for k in ("pot", "oven", "chopping_board", "blender") for i in (0, 1)))
        ex = Executor(Kitchen(env.mdp, 0, env.state), seed=0)
        opts = ex.options(env.state, ["egg"])
        self.assertIn("pickup(egg, ingredient_dispenser)", opts)
        m, _ = parse_macro("cook(pot1)")
        self.assertIn("cannot be started", ex.check(m, env.state))     # known utensil, just empty

    def test_both_players_reach_every_station(self):
        for layout in OPEN:
            env = ContinuousOrderEnv(["boiled_egg"], layout=layout)
            quiet(env.reset, 0)
            for i in (0, 1):
                k = Kitchen(env.mdp, i, env.state)
                for p in list(k.utensils.values()) + [p for t in "IDS" for p in k.cells[t]]:
                    self.assertEqual(k.side(p), "both of you", (layout, p))
                self.assertIn("Both of you can reach every station.", k.map_text())

    def test_steady_pair_never_deadlocks(self):
        for layout in OPEN:
            for seed in range(4):
                env, players = simulate(DEFAULT_MENU, layout, 400, seed, "steady", 1)
                s = env.summary()
                self.assertGreaterEqual(s["orders_completed"], 10, (layout, seed))
                self.assertEqual(s["rejected_interactions"], 0)
                # Orders keep being completed until the end (no stall in the second half).
                self.assertTrue(any(o["completed_t"] and o["completed_t"] > 300 for o in s["orders"]),
                                (layout, seed))

    def test_preference_shows_in_the_job_mix(self):
        for layout in OPEN:
            share = {}
            for persona in ("steady", "prep_first", "serve_first"):
                _, players = simulate(DEFAULT_MENU, layout, 400, 0, persona, 1)
                share[persona] = players[1].summary()["stage_share_when_acting"]
            self.assertGreater(share["prep_first"]["prep"], 0.6, layout)
            self.assertGreater(share["serve_first"]["serve"], 0.6, layout)
            self.assertLess(share["steady"]["prep"], 0.6, layout)
            self.assertLess(share["steady"]["serve"], 0.6, layout)

    def test_ego_on_open_map_can_use_every_station(self):
        env = ContinuousOrderEnv(["boiled_egg"], layout="open_env")
        quiet(env.reset, 0)
        ex = Executor(Kitchen(env.mdp, 0, env.state), seed=0)
        for text in ("pickup(egg, ingredient_dispenser)", "pickup(dish, dish_dispenser)"):
            m, _ = parse_macro(text)
            self.assertIsNone(ex.check(m, env.state), text)

    def test_layout_override(self):
        cfg = load_config(ROOT / "configs" / "coop" / "days_llm.yaml", layout="open_env")
        self.assertEqual(cfg.layout, "open_env")


class MovementRuleTests(unittest.TestCase):
    TERRAIN = ["XXXPXXX", "I     O", "C     S", "D     B", "XXXXXXX"]

    def test_step_aside_leaves_station_fronts(self):
        floor = floor_cells(self.TERRAIN)
        access = access_cells(self.TERRAIN, floor)
        self.assertIn((3, 1), access)          # in front of the pot
        self.assertNotIn((2, 2), access)
        self.assertEqual(step_aside(floor, access, (2, 2), None), Action.STAY)
        move = step_aside(floor, access, (3, 1), None)
        self.assertIn(move, (Direction.WEST, Direction.EAST, Direction.SOUTH))
        nxt = (3 + move[0], 1 + move[1])
        self.assertNotIn(nxt, access)

    def test_unstick_gives_way_only_after_a_blocked_move(self):
        import random
        floor = floor_cells(self.TERRAIN)
        u = Unstick(random.Random(0))
        self.assertEqual(u.filter(floor, (2, 2), Direction.EAST, (4, 2)), Direction.EAST)
        # The move failed (still at (2, 2)): some later attempt must give way.
        outs = [u.filter(floor, (2, 2), Direction.EAST, (4, 2)) for _ in range(20)]
        self.assertTrue(any(o != Direction.EAST for o in outs))
        self.assertGreater(u.give_ways, 0)
        # Turning towards a station (no floor ahead) is never treated as blocked.
        v = Unstick(random.Random(0))
        for _ in range(10):
            self.assertEqual(v.filter(floor, (1, 2), Direction.WEST, None), Direction.WEST)
        self.assertEqual(v.give_ways, 0)


if __name__ == "__main__":
    unittest.main()
