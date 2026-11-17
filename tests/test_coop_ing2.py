"""Open kitchen with one dispenser per ingredient (open_ing2_env: carrot on the left wall,
potato on the right wall) and its 8-dish menu."""
import contextlib
import io
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.env import ContinuousOrderEnv, unsafe_interact_reason  # noqa: E402
from coop.executor import Executor, parse_macro  # noqa: E402
from coop.kitchen import RECIPE_DIR, Kitchen  # noqa: E402
from coop.partner import PERSONAS, SimulatedHuman  # noqa: E402
from coop.rules import NEUTRAL, check_rules, load_cards, load_library  # noqa: E402
from coop.screen import DEFAULT_MENU  # noqa: E402

MENU = ["boiled_potato", "boiled_carrot_slices", "boiled_potato_slices", "baked_carrot_soup", "baked_potato_soup",
        "mashed_potato", "mashed_potato_and_carrot_mixture", "sliced_carrot_and_potato_stew"]


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def make_env(**kw):
    return quiet(ContinuousOrderEnv, MENU, layout="open_ing2_env", open_orders=2, **kw)


class DispenserTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env(horizon=50)
        self.state = quiet(self.env.reset, 0)

    def test_each_dispenser_gives_one_ingredient(self):
        self.assertEqual(self.env.mdp.dispensers, {(0, 1): "carrot", (8, 1): "potato"})
        k = Kitchen(self.env.mdp, 0, self.state)
        self.assertEqual(k.dispensers_for("carrot"), [(0, 1)])
        self.assertEqual(k.dispensers_for("potato"), [(8, 1)])
        p = self.state.players[0]
        p.position, p.orientation = (1, 1), (-1, 0)                 # facing the carrot dispenser
        self.assertIn("only gives carrot", unsafe_interact_reason(self.env.mdp, self.state, 0, "potato"))
        self.assertIsNone(unsafe_interact_reason(self.env.mdp, self.state, 0, "carrot"))
        state, _, _, info = quiet(self.env.step, ["interact", (0, 0)], ["potato", None])
        self.assertIsNone(state.players[0].held_object)              # refused
        self.assertEqual(len(info["rejected"]), 1)
        state, _, _, _ = quiet(self.env.step, ["interact", (0, 0)], ["carrot", None])
        self.assertEqual(state.players[0].held_object.name, "carrot")

    def test_executor_goes_to_the_dispenser_of_the_item(self):
        k = Kitchen(self.env.mdp, 0, self.state)
        ex = Executor(k)
        for item, cell in (("carrot", (0, 1)), ("potato", (8, 1))):
            m, _ = parse_macro(f"pickup({item}, ingredient_dispenser)")
            self.assertIsNone(ex.check(m, self.state))
            self.assertEqual(ex._targets(m, self.state)[0], [cell])
        text = k.map_text()
        self.assertIn("(gives carrot)", text)
        self.assertIn("(gives potato)", text)
        self.assertIn("each gives one raw ingredient", text)

    def test_other_layouts_are_unchanged(self):
        env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout="open2_env", open_orders=2)
        st = quiet(env.reset, 0)
        self.assertEqual(env.mdp.dispensers, {})
        k = Kitchen(env.mdp, 0, st)
        self.assertEqual(k.dispensers_for("egg"), k.cells["I"])
        self.assertNotIn("gives", k.map_text())
        self.assertIn("I ingredient dispenser (any raw ingredient)", k.map_text())


class MenuTests(unittest.TestCase):
    def test_recipe_texts_stay_out_of_the_original_folder(self):
        env = make_env(horizon=10)
        k = Kitchen(env.mdp, 0, quiet(env.reset, 0))
        for d in MENU:
            text = k.recipe_text(d)
            self.assertIn("INGREDIENTS", text, d)                       # a written recipe, not just a name
            self.assertIn("PROCESS:", text)
        for d in ("boiled_potato", "boiled_carrot_slices", "mashed_potato", "mashed_potato_and_carrot_mixture",
                  "sliced_carrot_and_potato_stew"):
            self.assertFalse(list(RECIPE_DIR.glob(f"*_{d}.txt")), d)    # src/main.py lists that folder

    def test_scripted_pair_makes_every_dish(self):
        done, rejected = Counter(), 0
        for seed in range(4):
            env = make_env(horizon=300, order_mode="controlled")
            quiet(env.reset, seed)
            ps = [SimulatedHuman(PERSONAS["steady"], "a"), SimulatedHuman(PERSONAS["steady"], "b")]
            for i, p in enumerate(ps):
                p.reset(env.mdp, i, seed)
            while not env.done:
                t, st = env.t, env.state
                moves = [p.act(st, t) for p in ps]
                _, _, _, info = quiet(env.step, [m[0] for m in moves], [m[1] for m in moves])
                rejected += len(info["rejected"])
            for o in env.summary()["orders"]:
                if o.get("completed_t") is not None:
                    done[o["name"]] += 1
        self.assertEqual(rejected, 0)
        self.assertEqual(set(done), set(MENU))

    def test_cards(self):
        lib = load_library()
        cards = {n: c for n, c in load_cards(ROOT / "configs/coop/personas/batch_ing2.yaml", lib).items()
                 if n != NEUTRAL}
        self.assertEqual(sorted(cards), ["p1", "p2", "p3", "p4"])
        for c in cards.values():
            self.assertEqual(check_rules(c.rules), [])
            for r in c.rules:
                if "dish" in r.p:
                    self.assertIn(r.p["dish"], MENU, (c.name, r.label()))


if __name__ == "__main__":
    unittest.main()
