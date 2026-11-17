"""Order timers with expiry and context features (docs/CONTEXT_PARTNER_PLAN_v3.md, 3.1-3.3)."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from overcooked_ai_py.mdp.overcooked_mdp import ObjectState  # noqa: E402

from coop.context import ContextTracker, blocked, prep_ready, urgent  # noqa: E402
from coop.deadlines import play_pair  # noqa: E402
from coop.env import START, ContinuousOrderEnv  # noqa: E402
from coop.kitchen import Kitchen  # noqa: E402
from coop.view import Belief  # noqa: E402

MENU = ["boiled_potato_slices", "baked_carrot_soup", "sliced_carrot_and_potato_stew"]
REF = {"boiled_potato_slices": 54, "baked_carrot_soup": 64, "sliced_carrot_and_potato_stew": 65}


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def env_with(slack=0.2, **kw):
    return quiet(ContinuousOrderEnv, MENU, layout="open_ing2_env", open_orders=2, horizon=60,
                 deadlines={"slack": slack, "reference": REF, "expiry_penalty": 20}, **kw)


class TimerTests(unittest.TestCase):
    def test_orders_expire_with_a_penalty_and_the_next_opens(self):
        env = env_with(slack=0.2)          # limits 11-13 steps
        st = quiet(env.reset, 0)
        first = [r.name for r in env.open_records]
        self.assertEqual(st.order_timers, [env.limit_of(d) for d in first])
        expired, total = [], 0.0
        while not env.done:
            st, r, _, info = quiet(env.step, [(0, 0), (0, 0)], [None, None])
            expired += info.get("expired", [])
            total += r
            self.assertEqual(len(st.order_timers), 2)
            self.assertTrue(all(x > 0 for x in st.order_timers))
            self.assertEqual([r.name for r in env.open_records], list(st.order_list[:2]))
        s = env.summary()
        self.assertGreater(s["orders_expired"], 2)
        self.assertEqual(len(expired), s["orders_expired"])
        self.assertAlmostEqual(s["score"], -20 * s["orders_expired"])
        self.assertAlmostEqual(total, s["score"])
        for o in s["orders"]:
            if o["expired_t"] is not None:
                self.assertEqual(o["expired_t"] - o["issued_t"], o["limit"])
        self.assertEqual([o["name"] for o in s["orders"]][:2], first)

    def test_without_deadlines_nothing_changes(self):
        env = quiet(ContinuousOrderEnv, MENU, layout="open_ing2_env", open_orders=2, horizon=30)
        st = quiet(env.reset, 0)
        self.assertFalse(hasattr(st, "order_timers"))
        s = env.summary()
        self.assertNotIn("orders_expired", s)
        self.assertNotIn("limit", s["orders"][0])

    def test_timers_are_public_in_a_partial_view(self):
        env = env_with(slack=1.0)
        st = quiet(env.reset, 0)
        view = Belief(env.mdp, 0, 2).update(st, 0)
        self.assertEqual(view.order_timers, st.order_timers)

    def test_steady_pair_scores_with_timers(self):
        s = play_pair(MENU, "open_ing2_env", 0, horizon=300, view_radius=2,
                      deadlines={"slack": 1.5, "reference": REF})
        self.assertIn("orders_expired", s)
        self.assertGreater(s["orders_completed"], 3)


class ContextTests(unittest.TestCase):
    def test_features(self):
        env = env_with(slack=1.0)
        st = quiet(env.reset, 0)
        k = Kitchen(env.mdp, 0, st)
        self.assertEqual(urgent(st, 2, 10), [])
        st.order_timers = [8, 30]
        self.assertEqual(urgent(st, 2, 10), [(0, st.order_list[0])])
        self.assertEqual(blocked(st, k), [])
        # both ovens running -> oven blocked
        for u in ("oven0", "oven1"):
            st.objects[k.utensils[u]] = ObjectState("baked_carrot_slices", "semi-finished", k.utensils[u],
                                                    ("baked_carrot_slices", 1, 1))
        self.assertEqual(blocked(st, k), ["oven"])
        # carrot slices waiting on a counter: the carrot soup and the stew have prepared items
        st.objects[(1, 0)] = ObjectState("carrot_slices", "ingredient", (1, 0))
        open_ = st.order_list[:2]
        self.assertEqual(sorted(prep_ready(st, k, 2)),
                         sorted({d for d in open_ if d in ("baked_carrot_soup", "sliced_carrot_and_potato_stew")}))
        tr = ContextTracker(k, 2, 10)
        ev = tr.observe(st, 5)
        self.assertIn("on_urgent", {e["context"] for e in ev})
        self.assertIn("on_blocked", {e["context"] for e in ev})
        self.assertEqual(tr.observe(st, 6), [])             # only rising edges

    def test_duplicate_last_ingredient_spoils_the_batch(self):
        """Plan v3, 3.3: a second copy of the last missing ingredient is accepted, and the
        batch is thrown away when started (a strong mismatch cost)."""
        env = env_with(slack=5.0)
        st = quiet(env.reset, 0)
        st.players[0].position, st.players[0].orientation = (2, 1), (0, -1)     # facing pot0
        for item in ("carrot_slices", "potato", "potato"):
            env.state.players[0].held_object = ObjectState(item, "ingredient", (2, 1))
            st, _, _, info = quiet(env.step, ["interact", (0, 0)], [None, None])
            self.assertEqual(info["rejected"], [])
        self.assertEqual(st.objects[(2, 0)].state[0], ["carrot_slices", "potato", "potato"])
        st, _, _, _ = quiet(env.step, ["interact", (0, 0)], [START, None])
        self.assertNotIn((2, 0), st.objects)
        self.assertTrue(st.error_message)


if __name__ == "__main__":
    unittest.main()
