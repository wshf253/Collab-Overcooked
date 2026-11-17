"""Multi-day partner learning (docs/PERSONA_PARTNER_PLAN_v2.md, sections 6-11): schedule,
controlled order lists, held-out evaluation, daily memory, feedback, knowledge probe,
the eight conditions, resume, and the learning-curve report (mock backends)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.days import load_config, run_all  # noqa: E402
from coop.learning import format_report, report  # noqa: E402
from coop.llm_agent import DAILY_SECTIONS, trim_sections  # noqa: E402
from coop.orders import make_controlled_sequence  # noqa: E402
from coop.probe2 import NO, questions, score, truth  # noqa: E402
from coop.rules import load_cards, load_library, render  # noqa: E402

CARDS = ROOT / "configs" / "coop" / "personas" / "batch1.yaml"


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


class PieceTests(unittest.TestCase):
    def test_controlled_order_lists_are_balanced(self):
        seq = make_controlled_sequence(["a", "b", "c", "d"], 7, 40)
        self.assertEqual(Counter(seq), {"a": 10, "b": 10, "c": 10, "d": 10})
        self.assertEqual(seq, make_controlled_sequence(["a", "b", "c", "d"], 7, 40))
        self.assertNotEqual(seq, make_controlled_sequence(["a", "b", "c", "d"], 8, 40))
        for i in range(0, 40, 4):
            self.assertEqual(sorted(seq[i:i + 4]), ["a", "b", "c", "d"])     # every block has every dish

    def test_trim_sections_stays_within_sections(self):
        text = ("## Partner Preferences\n" + "\n".join(f"- pref {i} " + "w " * 20 for i in range(6)) +
                "\n## Cooperation Strategy\n- short line\n")
        out, trimmed = trim_sections(text, DAILY_SECTIONS, (80, 60, 60, 40))
        for h in DAILY_SECTIONS:
            self.assertIn(h, out)
        self.assertIn("- short line", out)               # never moved or cut across sections
        self.assertTrue(trimmed and all(t.startswith("Partner Preferences") for t in trimmed))
        pref = out.split("## Partner Reactions")[0]
        self.assertLessEqual(len(pref.split()), 80 + 4)

    def test_probe_truth(self):
        from types import SimpleNamespace
        k = SimpleNamespace(utensils={"pot0": 0, "pot1": 1, "oven0": 2, "oven1": 3}, kind_of={
            "pot0": "pot", "pot1": "pot", "oven0": "oven", "oven1": "oven"})
        qs = questions(k, ["boiled_egg", "baked_carrot_soup"])
        p1 = load_cards(CARDS)["p1"].rules
        t = {q["id"]: truth(q, p1)["answer"] for q in qs}
        self.assertEqual(t["own_station:oven"], "oven0")
        self.assertEqual(t["own_station:pot"], NO)                      # p1 has no pot rule
        self.assertEqual(t["solo_dish"], "baked_carrot_soup")
        self.assertEqual(t["split_dish"], "none")
        self.assertEqual(t["order_priority"], "oldest")
        self.assertEqual(t["on_contention"], "it carries on")
        res = score(qs, {"own_station:oven": "oven0", "solo_dish": "baked_carrot_soup"}, p1,
                    {"own_station(station=oven0)": 3})
        rows = {r["id"]: r for r in res["rows"]}
        self.assertEqual(rows["own_station:oven"]["group"], "exposed")
        self.assertTrue(rows["own_station:oven"]["correct"])
        self.assertEqual(rows["solo_dish"]["group"], "not_exposed")
        self.assertEqual(rows["own_station:pot"]["group"], "absent")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        text = (ROOT / "configs/coop/days_persona.yaml").read_text().replace("type: openai_compat", "type: mock")
        (cls.tmp / "cfg.yaml").write_text(text)
        cls.cfg = load_config(cls.tmp / "cfg.yaml", personas=["p1", "p2"], run_seeds=[0, 1], days=2,
                              episodes_per_day=2, horizon=50)
        cls.res = quiet(run_all, cls.cfg, cls.tmp / "out", 4)
        cls.R = cls.tmp / "out" / "runs"

    def sp(self, cond, day, ep="ep01", persona="p1", seed=0):
        return json.loads((self.R / cond / persona / f"seed{seed}" / f"day{day:02d}" / ep /
                           "agent_log.json").read_text())["system_prompt"]

    def test_all_runs_ok(self):
        self.assertTrue(all(r["status"] == "ok" for r in self.res["runs"]), self.res["runs"])
        self.assertEqual({r["condition"] for r in self.res["runs"]},
                         {"tom_struct", "memory", "memory_fb", "latest_fb", "oracle", "swap_other", "swap_same",
                          "placebo", "_reference"})

    def test_schedule_and_held_out_episodes(self):
        d1 = json.loads((self.R / "memory_fb/p1/seed0/day01/ep01/summary.json").read_text())
        self.assertTrue({o["name"] for o in d1["orders"]} <= {"boiled_egg", "boiled_mushroom", "boiled_corn_slices"})
        e1 = json.loads((self.R / "memory_fb/p1/seed0/day01/eval01/summary.json").read_text())
        e2 = json.loads((self.R / "memory_fb/p1/seed0/day02/eval01/summary.json").read_text())
        e0 = json.loads((self.R / "memory_fb/p1/seed0/day00/eval01/summary.json").read_text())
        self.assertEqual(e1["kind"], "eval")
        self.assertEqual(e0["seed"], e1["seed"])
        self.assertEqual(e1["seed"], e2["seed"])                         # the same held-out episodes every day
        self.assertEqual([o["name"] for o in e1["orders"]], [o["name"] for o in e2["orders"]])
        # Memoryless conditions (and the reference) are evaluated once.
        for cond in ("tom_struct", "oracle", "_reference"):
            self.assertEqual(sorted(p.parent.name for p in (self.R / cond / "p1/seed0").glob("day*/eval01")), ["day00"])
        self.assertTrue((self.R / "memory/p1/seed0/day02/eval02/summary.json").exists())
        # Held-out episodes do not touch memory: no note from them.
        mem = json.loads((self.R / "memory_fb/p1/seed0/day02/ep01/memory_after.json").read_text())
        self.assertTrue(all(n["day"] == 2 for n in mem.get("day_notes", [])))

    def test_daily_memory_feedback_and_controls(self):
        self.assertNotIn("Your partner file", self.sp("memory_fb", 1))
        self.assertIn("Your partner file", self.sp("memory_fb", 2))
        self.assertIn("Your notes from earlier episodes today", self.sp("memory_fb", 2, "ep02"))
        c1 = json.loads((self.R / "memory_fb/p1/seed0/day01/consolidation.json").read_text())
        self.assertTrue(c1["ok"])
        self.assertTrue(c1["feedback"])
        self.assertIsNone(json.loads((self.R / "memory/p1/seed0/day01/consolidation.json").read_text())["feedback"])
        self.assertFalse((self.R / "memory/p1/seed0/day01/feedback.json").exists())
        fb = json.loads((self.R / "memory_fb/p1/seed0/day01/feedback.json").read_text())
        self.assertTrue(fb["ok"])
        self.assertIn("recitation", fb["checks"])
        self.assertIn("At the end of day 1, your partner told you", self.sp("latest_fb", 2))
        self.assertNotIn("Your partner file", self.sp("latest_fb", 2))
        self.assertIn("What your partner wants from you", self.sp("oracle", 1))
        self.assertIn("leave oven0 to me", self.sp("oracle", 1))
        so = json.loads((self.R / "swap_other/p1/seed0/day02/ep01/memory_after.json").read_text())["source"]
        self.assertEqual((so["persona"], so["seed"], so["after"]), ("p2", 0, [1, 2]))
        ss = json.loads((self.R / "swap_same/p1/seed0/day02/ep01/memory_after.json").read_text())["source"]
        self.assertEqual((ss["persona"], ss["seed"]), ("p1", 1))
        donor = json.loads((self.R / "memory_fb/p2/seed0/day01/ep02/memory_after.json").read_text())["partner_model"]
        self.assertIn(donor, self.sp("swap_other", 2))
        placebo = self.sp("placebo", 2)
        self.assertIn("## Partner Preferences", placebo)
        self.assertNotIn("pot0", placebo.split("Your partner file")[1].split("Reply with JSON")[0])
        # The recipient of a swapped file still gets its own partner's feedback.
        self.assertTrue((self.R / "swap_other/p1/seed0/day01/feedback.json").exists())

    def test_knowledge_probe_is_forked(self):
        kp = json.loads((self.R / "memory_fb/p1/seed0/day02/knowledge_probe.json").read_text())
        self.assertEqual(kp["day"], 2)
        self.assertIn("all", kp["accuracy"])
        self.assertTrue((self.R / "memory_fb/p1/seed0/day00/knowledge_probe.json").exists())
        for mem_path in self.R.glob("*/*/seed*/day*/ep*/memory_after.json"):
            self.assertNotIn("KNOWLEDGE_PROBE", mem_path.read_text())
        for log_path in self.R.glob("memory_fb/*/seed*/day*/ep*/agent_log.json"):
            self.assertNotIn("KNOWLEDGE_PROBE", "".join(c["user"] for c in json.loads(log_path.read_text())["calls"]))

    def test_no_leakage_outside_oracle(self):
        lib = load_library()
        cards = load_cards(CARDS, lib)
        secrets = {"Reminder:", "reconsider", "Your current goal", "Your habits", "How you decide", "Who you are",
                   "What your partner wants from you"}
        for c in cards.values():
            if c.backstory:
                secrets.add(c.backstory[:40])
            for r in c.rules:
                secrets.add(r.id)
        checked = 0
        for log_path in self.R.glob("*/*/seed*/day*/*/agent_log.json"):
            cond = log_path.parts[-6]
            if cond in ("oracle", "_reference"):
                continue
            log = json.loads(log_path.read_text())
            for text in [log.get("system_prompt") or ""] + [c["user"] for c in log.get("calls", [])]:
                for s in secrets:
                    self.assertNotIn(s, text, f"{s!r} leaked into {log_path}")
                checked += 1
        self.assertGreater(checked, 100)

    def test_resume_does_not_regenerate(self):
        fb = self.R / "memory_fb/p1/seed0/day01/feedback.json"
        kp = self.R / "memory_fb/p1/seed0/day01/knowledge_probe.json"
        before = (fb.stat().st_mtime_ns, kp.stat().st_mtime_ns)
        res = quiet(run_all, self.cfg, self.tmp / "out", 4)
        self.assertTrue(all(r["episodes_resumed"] == len(r["episodes"]) for r in res["runs"]))
        self.assertEqual((fb.stat().st_mtime_ns, kp.stat().st_mtime_ns), before)

    def test_learning_report(self):
        rep = report(self.tmp / "out")
        self.assertEqual(sorted(rep["held_out_vs_reference"]["tom_struct"]), [0, 1, 2])
        self.assertIn("memory_fb", rep["contrasts"]["tom_struct"])
        self.assertIn("memory_fb", rep["probe"])
        self.assertIn("oracle", rep["interaction_contrasts"]["score"])
        self.assertEqual(sorted(rep["interaction_contrasts"]["score"]["oracle"]["days"]), [1, 2])
        self.assertIn("memory_fb", rep["alignment_contrasts"]["tom_struct"])
        text = format_report(rep)
        for part in ("held-out score minus reference", "knowledge probe", "daily feedback",
                     "score value of knowing the partner", "secondary: held-out alignment"):
            self.assertIn(part, text)
        self.assertLess(text.index("== PRIMARY"), text.index("== secondary"))


if __name__ == "__main__":
    unittest.main()
