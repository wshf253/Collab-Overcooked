"""Several open orders and partial observability."""
import contextlib
import io
import shutil
import sys
import tempfile
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState
from coop.days import DaysConfig, run_all
from coop.env import ContinuousOrderEnv
from coop.partner import DEFAULT_PERSONAS
from coop.screen import DEFAULT_MENU, simulate
from coop.view import Belief

SERVE_FROM = (3, 3)    # new_env: floor cell above S at (3, 4)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def deliver(env, item):
    p = env.state.players[0]
    p.update_pos_and_or(SERVE_FROM, Direction.SOUTH)
    p.set_object(ObjectState(item, "finished", SERVE_FROM, (item, 1, 3)))
    return quiet(env.step, [Action.INTERACT, Action.STAY], [None, None])


class OpenOrdersTests(unittest.TestCase):
    def setUp(self):
        self.env = ContinuousOrderEnv(["boiled_egg", "boiled_mushroom", "apple"], horizon=50, open_orders=2)
        quiet(self.env.reset, 3)
        self.seq = self.env.log.order_sequence

    def test_any_open_order_can_be_served(self):
        first, second = self.seq[:2]
        self.assertEqual(self.env.open_order_names, [first, second])
        target = second if second != "apple" else first
        _, r, _, info = deliver(self.env, target)
        self.assertEqual(r, 20)
        self.assertTrue(info["deliveries"][0].correct)
        expected = [o for o in self.seq[:2]]
        expected.remove(target)
        self.assertEqual(self.env.open_order_names, expected + [self.seq[2]])
        done = [o for o in self.env.log.orders if o.completed_t is not None]
        self.assertEqual([o.name for o in done], [target])

    def test_item_matching_no_open_order_is_wrong(self):
        candidates = [d for d in ("boiled_egg", "boiled_mushroom") if d not in self.seq[:2]]
        if not candidates:
            self.skipTest("both test dishes are open for this seed")
        _, r, _, info = deliver(self.env, candidates[0])
        self.assertEqual(r, -20)
        self.assertFalse(info["deliveries"][0].correct)
        self.assertEqual(self.env.open_order_names, self.seq[:2])

    def test_one_open_order_keeps_old_behaviour(self):
        env = ContinuousOrderEnv(["boiled_egg", "boiled_mushroom"], horizon=50)
        for seed in range(20):
            quiet(env.reset, seed)
            first, second = env.log.order_sequence[:2]
            if first != second:
                break
        _, r, _, _ = deliver(env, second)      # the second order is not open yet
        self.assertEqual(r, -20)

    def test_scripted_pairs_never_stall_with_several_orders(self):
        for layout in ("open_env", "open2_env"):
            for k in (2, 3):
                for persona in ("steady", "prep_first", "yielding"):
                    env, _ = simulate(DEFAULT_MENU, layout, 300, 1, persona, 1, open_orders=k)
                    done = sorted(o["completed_t"] for o in env.summary()["orders"] if o["completed_t"] is not None)
                    gaps = [b - a for a, b in zip([0] + done, done + [300])]
                    with self.subTest(layout=layout, k=k, persona=persona):
                        self.assertLessEqual(max(gaps), 120)
                        self.assertEqual(env.summary()["wrong_deliveries"], 0)


class BeliefTests(unittest.TestCase):
    def setUp(self):
        self.env = ContinuousOrderEnv(["boiled_egg"], layout="open2_env", horizon=50)
        quiet(self.env.reset, 0)
        self.b = Belief(self.env.mdp, 0, 2)    # player 0 starts at (2, 2), player 1 at (6, 2)

    def test_hidden_cells_are_remembered_not_seen(self):
        st = self.env.state
        far, near = (8, 1), (2, 0)            # oven1 far away; pot0 above-left of the start, in view
        st.objects[near] = ObjectState("soup", "semi-finished", near, (["egg"], 1, 0))
        st.objects[far] = ObjectState("soup", "semi-finished", far, (["carrot_slices"], 1, 0))
        v = self.b.update(st, 0)
        self.assertIn(near, v.objects)
        self.assertNotIn(far, v.objects)       # never seen
        self.assertIsNone(v.last_seen(far))
        self.assertEqual(v.last_seen(near), 0)
        del st.objects[near]                   # changes out of view are not noticed...
        st.players[0].update_pos_and_or((6, 2), Direction.EAST)
        v = self.b.update(st, 5)
        self.assertIn(near, v.objects)         # ...pot0 is remembered as it was
        self.assertEqual(v.last_seen(near), 0)
        self.assertIn(far, v.objects)          # and oven1 is now in view
        self.assertEqual(v.last_seen(far), 5)

    def test_cooking_out_of_view_finishes_by_the_known_cook_time(self):
        st = self.env.state
        near = (2, 0)
        st.objects[near] = ObjectState("soup", "semi-finished", near, ("boiled_egg", 1, 1))
        self.b.update(st, 0)
        st.players[0].update_pos_and_or((6, 2), Direction.EAST)
        v = self.b.update(st, 1)
        self.assertEqual(v.objects[near].catagory, "semi-finished")
        v = self.b.update(st, 5)
        self.assertEqual((v.objects[near].name, v.objects[near].catagory), ("boiled_egg", "finished"))
        self.assertEqual(st.objects[near].state, ("boiled_egg", 1, 1))   # the real object is untouched

    def test_partner_out_of_view(self):
        st = self.env.state
        start = tuple(st.players[1].position)
        v = self.b.update(st, 0)
        self.assertFalse(v.players[1].visible)
        self.assertEqual(v.players[1].position, start)   # never seen: only the public start position
        st.players[1].update_pos_and_or((7, 3), Direction.EAST)
        st.players[1].set_object(ObjectState("egg", "ingredient", (7, 3)))
        v = self.b.update(st, 1)
        self.assertEqual(v.players[1].position, start)   # it moved out of view: no leak
        self.assertIsNone(v.players[1].held_object)
        self.assertIsNone(v.ml_actions[1])


class POPipelineTests(unittest.TestCase):
    def test_llm_run_with_partial_view_and_two_orders(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = DaysConfig(name="po", layout="open2_env", menu=["boiled_egg", "baked_carrot_soup"], horizon=60,
                             days=1, episodes_per_day=2, personas=["steady"], run_seeds=[0], open_orders=2,
                             view_radius=2, agent={"type": "llm", "backend": {"type": "mock"}, "probe": True},
                             conditions={"tom_memory": {"tom": True, "memory": True}})
            res = run_all(cfg, tmp, log=lambda _: None)
            self.assertTrue(all(r["status"] == "ok" for r in res["runs"]))
            log = json.loads((tmp / "runs" / "tom_memory" / "steady" / "seed0" / "day01" / "ep01" /
                              "agent_log.json").read_text())
            self.assertIn("What you can see: only cells within 2 steps", log["system_prompt"])
            self.assertIn("Up to 2 orders are open at once", log["system_prompt"])
            users = "".join(c["user"] for c in log["calls"])
            self.assertIn("Open orders:", users)
            self.assertTrue("not seen yet" in users or "last seen at t=" in users)

        finally:
            shutil.rmtree(tmp)

    def test_delivery_out_of_view_does_not_name_the_partner(self):
        from coop.env import Delivery
        from coop.llm_agent import LLMAgent
        agent = LLMAgent({"type": "mock"}, probe=False)
        agent.begin_episode(None, 0, {"index": 0, "view_radius": None, "wrong_delivery_penalty": 20})
        d = Delivery(t=5, agent=1, item="boiled_egg", plated=False, order="boiled_egg", correct=True)
        agent._view_partner_visible = False
        agent.observe(5, 20, {"deliveries": [d]})
        agent._view_partner_visible = True
        agent.observe(6, 20, {"deliveries": [d]})
        self.assertIn("someone (out of your view) delivered boiled_egg", agent.events[0])
        self.assertIn("your partner delivered boiled_egg", agent.events[1])

    def test_invalid_settings(self):
        for kw in (dict(open_orders=0), dict(view_radius=-1)):
            with self.subTest(**kw), self.assertRaises(ValueError):
                DaysConfig(**kw)


if __name__ == "__main__":
    unittest.main()
