"""One-decision test (coop.decision): screen, LLM stage on the mock backend, report."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop import decision  # noqa: E402
from coop.executor import parse_macro  # noqa: E402
from coop.kitchen import Kitchen  # noqa: E402


def small_cfg():
    cfg = decision.load(ROOT / "configs/coop/decision/urgent.yaml")
    cfg.update(pairs=[["boiled_carrot_slices", "baked_potato_soup"]], urgent_limits=[60], seeds=[0], ego_slots=[0])
    cfg["agent"] = dict(cfg["agent"], backend=dict(cfg["agent"]["backend"], type="mock"))
    return cfg


class DecisionTests(unittest.TestCase):
    def test_start_and_classify(self):
        cfg = small_cfg()
        st = decision.starts(cfg)[0]
        env, s0 = decision.make_env(cfg, st)
        self.assertEqual(s0.order_list[:2], ["baked_potato_soup", "boiled_carrot_slices"])
        self.assertEqual(s0.order_timers, [250, 60])
        kit = Kitchen(env.mdp, 0, s0)
        c = lambda a: decision.classify(parse_macro(a)[0], s0, kit, 0, st["urgent"], st["other"])
        self.assertEqual(c("pickup(carrot, ingredient_dispenser)"), "urgent")
        self.assertEqual(c("pickup(potato, ingredient_dispenser)"), "other")
        self.assertEqual(c("pickup(dish, dish_dispenser)"), "other")      # only the soup is plated

    def test_screen_switches_and_llm_stage_runs(self):
        cfg = small_cfg()
        out = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            scr = decision.screen(cfg, out)
        row = scr["rows"][0]
        self.assertEqual(row["best"], {"takes": "other", "keeps": "urgent"})
        self.assertTrue(row["kept"])
        with contextlib.redirect_stdout(io.StringIO()):
            res = decision.llm_stage(cfg, out, workers=1)
        self.assertEqual(len(res), 6)
        r = json.loads((out / "llm" / row["start"]["id"] / "takes" / "correct" / "result.json").read_text())
        self.assertEqual(r["told"], "takes")
        sp = json.loads((out / "llm" / row["start"]["id"] / "takes" / "correct" / "agent_log.json").read_text())["system_prompt"]
        self.assertIn("When an order becomes urgent, I take it", sp)
        sp = json.loads((out / "llm" / row["start"]["id"] / "takes" / "none" / "agent_log.json").read_text())["system_prompt"]
        self.assertNotIn("What your partner wants", sp)
        text = decision.report(cfg, out)
        self.assertIn("Verdict", text)


if __name__ == "__main__":
    unittest.main()
