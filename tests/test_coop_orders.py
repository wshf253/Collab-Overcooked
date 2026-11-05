"""Continuous-order episodes (src/coop) on the real Collab-Overcooked MDP."""
import asyncio
import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld
from coop.env import ContinuousOrderEnv
from coop.orders import make_order_sequence, normalize_menu
from human_coop.policies import RecipePolicy
from human_coop.session import object_dict

MOVES = {"up": Direction.NORTH, "down": Direction.SOUTH, "left": Direction.WEST,
         "right": Direction.EAST, "stay": Action.STAY, "interact": Action.INTERACT}
SERVE_FROM = (3, 3)  # new_env: floor cell above the serving location S at (3, 4)


def quiet(fn, *args, **kwargs):
    # The legacy MDP prints on every transition.
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def observation(env, index):
    state, mdp = env.state, env.mdp
    players = [dict(p.to_dict(), held_object=object_dict(p.held_object)) for p in state.players]
    return copy.deepcopy({
        "agent_index": index,
        "state": {"players": players, "objects": [object_dict(o) for o in state.objects.values()],
                  "order_list": state.order_list, "timestep": env.t},
        "terrain": mdp.terrain_mtx, "ingredients": list(dict.fromkeys(mdp.all_ingredients)),
        "raw_ingredients": mdp.default_ingredients, "recipes": mdp.recipes,
        "utensils": mdp.utensils, "need_dish": mdp.need_dish, "history": []})


def hold_at_serving(env, item, plated=False):
    """Put player 0 in front of S holding a finished ``item``."""
    player = env.state.players[0]
    player.update_pos_and_or(SERVE_FROM, Direction.SOUTH)
    cook_time = env.mdp.Recipe.recipe_list[item]["cook_time"]
    player.set_object(ObjectState(item, "with_dish" if plated else "finished", SERVE_FROM,
                                  (item, 1, cook_time)))


def deliver(env):
    return quiet(env.step, [Action.INTERACT, Action.STAY], [None, None])


class OrderSequenceTests(unittest.TestCase):
    def test_same_seed_same_sequence(self):
        menu = ["boiled_egg", "boiled_mushroom", "baked_carrot_soup"]
        self.assertEqual(make_order_sequence(menu, 3, 50), make_order_sequence(menu, 3, 50))
        self.assertNotEqual(make_order_sequence(menu, 3, 50), make_order_sequence(menu, 4, 50))

    def test_menu_dict_order_does_not_change_sequence(self):
        a = make_order_sequence({"boiled_egg": 1, "boiled_mushroom": 2}, 0, 40)
        b = make_order_sequence({"boiled_mushroom": 2, "boiled_egg": 1}, 0, 40)
        self.assertEqual(a, b)

    def test_zero_weight_never_drawn_and_no_repeat(self):
        seq = make_order_sequence({"a": 1, "b": 1, "c": 0}, 1, 200, no_repeat=True)
        self.assertNotIn("c", seq)
        self.assertTrue(all(x != y for x, y in zip(seq, seq[1:])))

    def test_invalid_menus(self):
        for menu in ([], {"a": 0}, ["a", "a"], {"a": -1}):
            with self.subTest(menu=menu), self.assertRaises(ValueError):
                normalize_menu(menu)
        with self.assertRaises(ValueError):
            make_order_sequence(["a"], 0, 5, no_repeat=True)

    def test_unknown_order_rejected_by_env(self):
        with self.assertRaises(ValueError):
            ContinuousOrderEnv(["boiled_egg", "not_a_dish"])


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.env = ContinuousOrderEnv({"boiled_egg": 1, "boiled_mushroom": 1},
                                      horizon=50, wrong_delivery_penalty=20)
        quiet(self.env.reset, 0)
        self.first, self.second = self.env.log.order_sequence[:2]

    def test_wrong_delivery_costs_points_and_keeps_order(self):
        wrong = "boiled_mushroom" if self.first == "boiled_egg" else "boiled_egg"
        hold_at_serving(self.env, wrong)
        _, reward, _, info = deliver(self.env)
        self.assertEqual(reward, -20)
        self.assertEqual(self.env.current_order, self.first)
        [d] = info["deliveries"]
        self.assertEqual((d.agent, d.item, d.order, d.correct), (0, wrong, self.first, False))
        self.assertIsNone(self.env.state.players[0].held_object)  # the item is consumed
        self.assertEqual(self.env.log.orders[0].wrong_deliveries, 1)

    def test_plated_mismatch_is_wrong(self):
        # boiled_egg / boiled_mushroom are served without a dish.
        hold_at_serving(self.env, self.first, plated=True)
        _, reward, _, info = deliver(self.env)
        self.assertEqual(reward, -20)
        self.assertFalse(info["deliveries"][0].correct)

    def test_correct_delivery_opens_next_order(self):
        quiet(self.env.step, [Action.STAY, Action.STAY], [None, None])
        hold_at_serving(self.env, self.first)
        _, reward, _, info = deliver(self.env)
        self.assertEqual(reward, 20)
        self.assertEqual(self.env.current_order, self.second)
        first, nxt = self.env.log.orders
        self.assertEqual((first.completed_t, first.delivered_by, first.duration), (1, 0, 2))
        self.assertEqual((nxt.index, nxt.name, nxt.issued_t), (1, self.second, 2))
        self.assertEqual(self.env.score, 20)

    def test_reset_restores_sequence(self):
        hold_at_serving(self.env, self.first)
        deliver(self.env)
        quiet(self.env.reset, 0)
        self.assertEqual(self.env.current_order, self.first)
        self.assertEqual(self.env.score, 0)
        self.assertEqual(len(self.env.log.orders), 1)

    def test_original_mdp_has_no_penalty_by_default(self):
        mdp = OvercookedGridworld.from_layout_name("new_env")
        mdp.start_order_list, mdp.order_probability = ["boiled_mushroom"], {}
        env = quiet(OvercookedEnv, mdp, horizon=50)
        hold_at_serving(env, "boiled_egg")
        _, reward, _, _ = quiet(env.step, [Action.INTERACT, Action.STAY], [None, None])
        self.assertEqual(reward, 0)
        self.assertEqual(env.state.order_list, ["boiled_mushroom"])


class ScriptedTeamTests(unittest.TestCase):
    def test_recipe_team_serves_successive_orders(self):
        menu = ["boiled_egg", "boiled_mushroom", "baked_carrot_soup", "boiled_corn_slices"]
        env = ContinuousOrderEnv(menu, horizon=250)
        quiet(env.reset, 7)
        policies = [RecipePolicy(), RecipePolicy()]
        while not env.done:
            decisions = [asyncio.run(p.decide(observation(env, i))) for i, p in enumerate(policies)]
            quiet(env.step, [MOVES[d.action] for d in decisions], [d.parameter for d in decisions])
        s = env.summary()
        self.assertGreaterEqual(s["orders_completed"], 3)
        self.assertEqual(s["wrong_deliveries"], 0)
        self.assertEqual(s["score"], 20 * s["orders_completed"])
        served = [o["name"] for o in s["orders"] if o["completed_t"] is not None]
        self.assertEqual(served, env.log.order_sequence[:len(served)])
        # Orders follow each other without gaps.
        for a, b in zip(s["orders"], s["orders"][1:]):
            self.assertEqual(b["issued_t"], a["completed_t"] + 1)


if __name__ == "__main__":
    unittest.main()
