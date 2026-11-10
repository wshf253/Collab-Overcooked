"""New stable partner types (yielding, avoids_board, leaves_unfinished) and the VOI screen."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from coop.days import load_config
from coop.partner import DEFAULT_PERSONAS, PERSONAS, Traits
from coop.profile import truth_profile
from coop.screen import DEFAULT_MENU, simulate
from coop.voi import EGOS, screen


def run(persona, layout="open_env", seed=0, horizon=300):
    return simulate(DEFAULT_MENU, layout, horizon, seed, persona, 1)


class OriginalTypesUnchangedTests(unittest.TestCase):
    # Scores and event counts recorded with the code before the new traits (commit 656126f).
    SNAPSHOT = {("open_env", "steady"): (260.0, 0), ("open_env", "prep_first"): (220.0, 0),
                ("open_env", "slow"): (220.0, 0), ("open_env", "sloppy"): (260.0, 1),
                ("open_env", "flaky"): (220.0, 12), ("new_env", "steady"): (260.0, 0),
                ("new_env", "prep_first"): (220.0, 0), ("new_env", "slow"): (160.0, 0),
                ("new_env", "sloppy"): (260.0, 0), ("new_env", "flaky"): (200.0, 5)}

    def test_snapshot(self):
        for (layout, persona), expected in self.SNAPSHOT.items():
            env, players = run(persona, layout)
            with self.subTest(layout=layout, persona=persona):
                self.assertEqual((env.summary()["score"], len(players[1].events)), expected)

    def test_weak_types_kept_but_not_default(self):
        for name in ("sloppy", "flaky"):
            self.assertIn(name, PERSONAS)
            self.assertNotIn(name, DEFAULT_PERSONAS)
        for name in ("yielding", "avoids_board", "leaves_unfinished"):
            self.assertIn(name, DEFAULT_PERSONAS)
        for cfg in ("days_llm.yaml", "days_default.yaml"):
            self.assertEqual(load_config(ROOT / "configs" / "coop" / cfg).personas, DEFAULT_PERSONAS)


class NewTypeTests(unittest.TestCase):
    def test_traits_validate(self):
        for bad in (dict(avoid_stations=("sink",)), dict(unfinished_rate=1.5)):
            with self.subTest(**{k: str(v) for k, v in bad.items()}), self.assertRaises(ValueError):
                Traits(**bad)

    def test_yielding_gives_way_on_open_maps_only(self):
        yields = sum(run("yielding", seed=s)[1][1].summary()["yields"] for s in range(3))
        self.assertGreater(yields, 10)
        a, b = run("yielding", "new_env"), run("steady", "new_env")
        self.assertEqual(a[0].summary()["score"], b[0].summary()["score"])   # no shared tasks to yield

    def test_avoids_board_never_uses_it_when_the_other_can(self):
        for seed in range(3):
            env, players = run("avoids_board", seed=seed)
            p = players[1]
            used = [r for r in p.trace if r["job"] and r["status"] == "act" and r["action"] == "interact"
                    and r["job"]["kind"] in ("start", "clear", "unload")
                    and "slices" in (r["job"]["item"] or "")]
            self.assertEqual(used, [], seed)
            self.assertGreater(env.summary()["orders_completed"], 5)
        # Split map: only the partner reaches the board, so it still cuts (soft avoidance).
        a, b = run("avoids_board", "new_env"), run("steady", "new_env")
        self.assertEqual(a[0].summary()["score"], b[0].summary()["score"])

    def test_leaves_unfinished_leaves_visible_work(self):
        env, players = run("leaves_unfinished")
        events = [e for e in players[1].events if e["type"] == "left_unfinished"]
        self.assertGreater(len(events), 5)
        self.assertTrue(all(e["job"]["kind"] in ("start", "bring") for e in events))

    def test_truth_profiles(self):
        self.assertEqual(truth_profile(PERSONAS["yielding"])["gives_way"], "gives way to you")
        self.assertEqual(truth_profile(PERSONAS["avoids_board"])["avoids"], "chopping board")
        self.assertEqual(truth_profile(PERSONAS["leaves_unfinished"])["commitment"], "abandons tasks")
        self.assertEqual(truth_profile(PERSONAS["steady"])["gives_way"], "does not give way")


class VOITests(unittest.TestCase):
    def test_screen_structure(self):
        egos = {k: EGOS[k] for k in ("any", "prep_first+yield")}
        res = screen("open_env", ["steady", "slow"], [0], 120, DEFAULT_MENU, egos)
        self.assertIn(res["best_fixed"], egos)
        for p, r in res["personas"].items():
            self.assertGreaterEqual(r["value"], 0)
            self.assertEqual(set(r["scores"]), set(egos))


if __name__ == "__main__":
    unittest.main()
