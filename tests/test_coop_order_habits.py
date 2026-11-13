"""Order-habit partners: committing to one open order, the probe's orders field, VOI egos."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.days import play_episode
from coop.env import ContinuousOrderEnv
from coop.llm_agent import LLMAgent
from coop.partner import ORDER_PERSONAS, PERSONAS, SimulatedHuman, Traits
from coop.profile import NO_ORDER_HABIT, order_options, order_truth, probe_fields, truth_profile
from coop.screen import DEFAULT_MENU
from coop.voi import ego_set, egos_for

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_coop_llm import mock_config  # noqa: E402

SOUP = "baked_carrot_soup"


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def focus_shares(persona, seeds=(0, 1, 2), layout="open2_env"):
    """(share of acting steps on the newest open order, share on the carrot soup when it is open)."""
    newest = on_soup = acting = soup_acting = 0
    for seed in seeds:
        env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout=layout, horizon=300, open_orders=2)
        quiet(env.reset, seed)
        a, b = SimulatedHuman(PERSONAS["steady"], "steady"), SimulatedHuman(PERSONAS[persona], persona)
        a.reset(env.mdp, 0, seed)
        b.reset(env.mdp, 1, seed)
        while not env.done:
            s, t = env.state, env.t
            opened = list(s.order_list[:2])
            ma, mb = a.act(s, t), b.act(s, t)
            r = b.trace[-1]
            if r["status"] == "act" and r["job"]:
                acting += 1
                newest += r["job"]["order"] == 1
                if SOUP in opened:
                    soup_acting += 1
                    on_soup += r["job"]["order"] < len(opened) and opened[r["job"]["order"]] == SOUP
            quiet(env.step, [ma[0], mb[0]], [ma[1], mb[1]])
    return newest / acting, on_soup / soup_acting


class HabitBehaviourTests(unittest.TestCase):
    def test_personas_and_validation(self):
        self.assertEqual(ORDER_PERSONAS, ["steady", "oldest_first", "newest_first", "favourite_dish", "repeats_dish"])
        for name in ORDER_PERSONAS[1:]:
            self.assertTrue(PERSONAS[name].order_commit, name)
        with self.assertRaises(ValueError):
            Traits(order_focus="random")
        with self.assertRaises(ValueError):
            Traits(favourite_dishes=("boiled_egg", "boiled_egg"))

    def test_habits_change_which_order_is_worked_on(self):
        steady_newest, steady_soup = focus_shares("steady")
        oldest_newest, _ = focus_shares("oldest_first")
        _, fav_soup = focus_shares("favourite_dish")
        self.assertLess(oldest_newest, steady_newest - 0.1)
        self.assertGreater(fav_soup, 0.8)
        self.assertLess(steady_soup, 0.7)

    def test_commit_target(self):
        env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout="open2_env", open_orders=2)
        quiet(env.reset, 0)

        def target(persona, orders, last=None, committed=None):
            h = SimulatedHuman(PERSONAS[persona], persona)
            h.reset(env.mdp, 1, 0)
            h.state, h.last_dish, h.committed = SimpleNamespace(order_list=orders), last, committed
            return h._commit_target()
        orders = ["boiled_egg", SOUP, "boiled_mushroom"]          # only the first two are open
        self.assertEqual(target("oldest_first", orders), "boiled_egg")
        self.assertEqual(target("newest_first", orders), SOUP)
        self.assertEqual(target("favourite_dish", orders), SOUP)
        self.assertEqual(target("favourite_dish", ["boiled_egg", "boiled_mushroom"]), "boiled_egg")
        self.assertEqual(target("repeats_dish", orders, last=SOUP), SOUP)
        self.assertEqual(target("repeats_dish", orders, last="boiled_corn_slices"), "boiled_egg")
        # Committed: stays on its order while it is open, even if it is now the oldest.
        self.assertEqual(target("newest_first", orders, committed="boiled_egg"), "boiled_egg")

    def test_repeats_dish_remembers_its_delivery(self):
        env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout="open_env", horizon=300, open_orders=2)
        quiet(env.reset, 1)
        a, b = SimulatedHuman(PERSONAS["steady"], "steady"), SimulatedHuman(PERSONAS["repeats_dish"], "repeats_dish")
        a.reset(env.mdp, 0, 1)
        b.reset(env.mdp, 1, 1)
        while not env.done:
            s, t = env.state, env.t
            ma, mb = a.act(s, t), b.act(s, t)
            quiet(env.step, [ma[0], mb[0]], [ma[1], mb[1]])
        mine = [d.item for d in env.log.deliveries if d.agent == 1]
        self.assertTrue(mine)
        self.assertEqual(b.last_dish, mine[-1])


class OrderProbeTests(unittest.TestCase):
    def test_truth(self):
        dishes = ["boiled_egg", SOUP]
        self.assertEqual(order_truth(PERSONAS["steady"], dishes), NO_ORDER_HABIT)
        self.assertEqual(order_truth(PERSONAS["oldest_first"], dishes), "its oldest open order")
        self.assertEqual(order_truth(PERSONAS["newest_first"], dishes), "the newest open order")
        self.assertEqual(order_truth(PERSONAS["favourite_dish"], dishes), f"favourite dish: {SOUP}")
        self.assertEqual(order_truth(PERSONAS["favourite_dish"], ["boiled_egg"]), "its oldest open order")
        self.assertEqual(order_truth(PERSONAS["repeats_dish"], dishes), "the dish it served last")
        self.assertIsNone(truth_profile(PERSONAS["newest_first"])["orders"])     # one open order
        opts = order_options(dishes)
        for p in ORDER_PERSONAS:
            self.assertIn(order_truth(PERSONAS[p], dishes), opts)
        self.assertNotIn("orders", probe_fields([]))
        self.assertEqual(probe_fields([], opts)["orders"], opts)

    def test_llm_probe_asks_orders_with_two_open_orders(self):
        cfg = mock_config(layout="open2_env", menu=DEFAULT_MENU, personas=["favourite_dish"], days=1,
                          episodes_per_day=1, horizon=80, open_orders=2)
        agent = LLMAgent({"type": "mock"}, probe=True)
        with tempfile.TemporaryDirectory() as d:
            summary = quiet(play_episode, cfg, agent, "favourite_dish", 0, 1, 1, Path(d))
        dishes = [o["name"] for o in summary["orders"]]
        user = [c for c in agent.calls if c["kind"] == "probe"][0]["user"]
        self.assertIn('"orders": "its oldest open order"', user)
        self.assertEqual(summary["probe"]["truth"]["orders"], order_truth(PERSONAS["favourite_dish"], dishes))
        self.assertIn("orders", summary["probe"]["correct"])
        decide = [c for c in agent.calls if c["kind"] == "decide"][0]["user"]
        self.assertIn("Open orders (oldest first):", decide)


class VOIEgoTests(unittest.TestCase):
    def test_ego_sets(self):
        self.assertFalse(any(t.order_commit for t in egos_for(1).values()))
        many = egos_for(2, DEFAULT_MENU)
        for name in ("commit_oldest", "commit_newest+yield", f"any+leave_{SOUP}", f"commit_oldest+leave_{SOUP}"):
            self.assertIn(name, many)
        self.assertNotIn(SOUP, many[f"any+leave_{SOUP}"].favourite_dishes)
        commit = ego_set("commit", 2, DEFAULT_MENU)
        self.assertTrue(commit and all(t.order_commit for t in commit.values()))
        self.assertEqual(ego_set("commit", 1), {})
        self.assertEqual(len(ego_set("flexible", 2, DEFAULT_MENU)) + len(commit), len(many))


if __name__ == "__main__":
    unittest.main()
