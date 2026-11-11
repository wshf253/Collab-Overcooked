"""Division-of-labour analysis (coop.analyze)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from coop.analyze import action_stage, analyse, format_report
from coop.days import DaysConfig, run_all

RAW = {"egg", "carrot", "chickpea", "eggplant"}
FINALS = {"boiled_egg", "baked_carrot_soup"}


class StageTests(unittest.TestCase):
    def test_action_stages(self):
        cases = {
            "pickup(carrot, ingredient_dispenser)": "prep", "put_obj_in_utensil(chopping_board0)": "prep",
            "cut(chopping_board0)": "prep", "stir(blender1)": "prep",
            "pickup(carrot_slices, chopping_board0)": "prep",
            "put_obj_in_utensil(pot1)": "cook", "cook(pot0)": "cook", "bake(oven0)": "cook",
            "pickup(baked_carrot_slices, oven0)": "cook",
            "pickup(dish, dish_dispenser)": "serve", "fill_dish_with_food(pot0)": "serve",
            "deliver_soup()": "serve", "pickup(dish, counter)": "serve",
            "pickup(egg, counter)": "prep", "pickup(carrot_slices, counter)": "cook",
            "pickup(boiled_egg, counter)": "serve", "place_obj_on_counter()": "handover",
        }
        for action, stage in cases.items():
            with self.subTest(action=action):
                self.assertEqual(action_stage(action, RAW, FINALS), stage)


class LabourReportTests(unittest.TestCase):
    def test_scripted_preferences_show_up(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = DaysConfig(name="lab", layout="open_env", horizon=200, days=1, episodes_per_day=2,
                             personas=["steady"], run_seeds=[0, 1],
                             agent={"type": "scripted", "persona": "steady"},
                             conditions={"direct": {}, "tom_memory": {"persona": "serve_first"}})
            run_all(cfg, tmp, log=lambda _: None)
            rep = analyse(tmp)
            rows = rep["labour"]["rows"]
            for cond in ("direct", "tom_memory", "_reference"):
                r = rows[cond]["steady"]
                self.assertAlmostEqual(sum(r["ego_share"].values()), 100.0, places=6)
                self.assertAlmostEqual(sum(r["partner_share"].values()), 100.0, places=6)
            # A serve-first ego next to a steady partner serves more and splits the work more.
            self.assertGreater(rows["tom_memory"]["steady"]["ego_share"]["serve"],
                               rows["direct"]["steady"]["ego_share"]["serve"] + 20)
            diff = rep["labour"]["overlap_vs"]["tom_memory"]["direct"]["all"]
            self.assertLess(diff["mean"], -10)
            self.assertIn("division of labour", format_report(rep))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
