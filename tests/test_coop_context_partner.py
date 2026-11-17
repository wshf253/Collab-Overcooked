"""Context rules, twin, scripted egos and Gate 0 (docs/CONTEXT_PARTNER_PLAN_v3.md, 4-7)."""
import contextlib
import io
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.context_partner import ContextPlayer  # noqa: E402
from coop.partner import PERSONAS, SimulatedHuman  # noqa: E402
from coop.persona_check import play_rule_episode  # noqa: E402
from coop.rules import CONTEXT_RULE_IDS, check_rules, load_cards, load_library, parse_rule, render  # noqa: E402
from coop.voi import format_gate0, gate0  # noqa: E402

MENU = ["boiled_potato_slices", "baked_carrot_soup", "sliced_carrot_and_potato_stew"]
REF = {"boiled_potato_slices": 54, "baked_carrot_soup": 64, "sliced_carrot_and_potato_stew": 65}
DL = {"slack": 1.5, "reference": REF, "urgent_steps": 15}
CARDS = ROOT / "configs/coop/personas/batch_ctx.yaml"


class LibraryTests(unittest.TestCase):
    def test_context_rules_have_no_fallback_and_action_wants(self):
        lib = load_library()
        for rid in CONTEXT_RULE_IDS:
            spec = lib[rid]
            self.assertEqual(spec["category"], "context")
            self.assertFalse(any(k.startswith("fallback") for k in spec), rid)
            for mode in spec["params"]["mode"]:
                r = parse_rule({rid: {"mode": mode}}, lib)
                for field in ("text", "wants", "reminder"):
                    self.assertTrue(render(r, field, lib), (rid, mode, field))
                w = render(r, "wants", lib).lower()
                self.assertNotIn("leave", w.split(";")[0] if "i " not in w[:2] else "", (rid, mode))
        u = parse_rule({"on_urgent": {"mode": "takes", "unless_ego_on_it": "yes"}}, lib)
        self.assertIn("unless your partner is already working on it", render(u, "text", lib))

    def test_pilot_cards_are_opposite_on_every_rule(self):
        cards = load_cards(CARDS, load_library())
        a, b = cards["A"].rules, cards["B"].rules
        self.assertEqual(check_rules(a), [])
        self.assertEqual([r.id for r in a], [r.id for r in b])
        for ra, rb in zip(a, b):
            self.assertNotEqual(ra.p["mode"], rb.p["mode"])


class TwinTests(unittest.TestCase):
    def test_twin_follows_its_card(self):
        cards = load_cards(CARDS, load_library())
        for card in ("A", "B"):
            a = n = 0
            for seed in range(4):
                r = play_rule_episode(MENU, "open_ing2_env", 300, seed, cards[card].rules,
                                      SimulatedHuman(PERSONAS["steady"], "steady"), 2, 2, deadlines=DL,
                                      order_mode="controlled")
                for row in r["rules"].values():
                    a += row["partner"]["adherent"]
                    n += row["partner"]["opportunities"]
            self.assertGreater(n, 30, card)
            self.assertGreaterEqual(a / n, 0.93, (card, a, n))

    def test_guard_fires_on_deadlock_and_is_logged(self):
        """A prep partner with an ego that never moves: nobody cooks, so after guard_steps
        without any completed interaction the partner does the next needed step once."""
        lib = load_library()
        prep = [parse_rule({"default_work": {"mode": "prep"}}, lib)]

        class Idle:
            def reset(self, mdp, index, seed, view_radius=None):
                pass

            def act(self, state, t):
                return (0, 0), None

        r = play_rule_episode(MENU, "open_ing2_env", 300, 0, prep, Idle(), 2, 2, deadlines=DL, order_mode="controlled")
        self.assertGreater(r["guards"], 0)
        with_partner = ContextPlayer(prep, MENU, "partner", "card", guard_steps=5)
        self.assertEqual(with_partner.guard_steps, 5)


class EgoTests(unittest.TestCase):
    def test_policies_and_mismatch(self):
        cards = load_cards(CARDS, load_library())
        scores = defaultdict(list)
        for seed in range(3):
            for name, ego in (("proactive", ContextPlayer(cards["A"].rules, MENU, "ego", "proactive")),
                              ("mismatched", ContextPlayer(cards["B"].rules, MENU, "ego", "proactive")),
                              ("wait", ContextPlayer([], MENU, "ego", "wait_and_see", k=5))):
                r = play_rule_episode(MENU, "open_ing2_env", 300, seed, cards["A"].rules, ego, 2, 2, deadlines=DL,
                                      order_mode="controlled")
                scores[name].append(r["score"])
        self.assertGreater(sum(scores["proactive"]), sum(scores["mismatched"]))
        with self.assertRaises(ValueError):
            ContextPlayer([], MENU, "partner", "proactive")

    def test_gate0_tool(self):
        res = gate0(CARDS, "open_ing2_env", MENU, DL, [0], [1], horizon=120, view_radius=2, open_orders=2)
        self.assertEqual(sorted(res["cards_result"]), ["A", "B"])
        self.assertIn("proactive_minus_best_reactive", res["cards_result"]["A"])
        self.assertIn("Gate 0", format_gate0(res))


if __name__ == "__main__":
    unittest.main()
