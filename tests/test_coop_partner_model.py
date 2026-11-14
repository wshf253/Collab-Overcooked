"""Structured ToM (tom_mode: structured) and the partner-model file (memory_mode: model)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.analyze import analyse, format_report  # noqa: E402
from coop.days import load_config, placebo_model, run_all  # noqa: E402
from coop.llm_agent import LLMAgent, bound_words, clean_read  # noqa: E402
from coop.profile import MODEL_SECTIONS, model_template  # noqa: E402


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class HelperTests(unittest.TestCase):
    def test_clean_read(self):
        self.assertEqual(clean_read("just text")["doing"], "just text")
        r = clean_read({"order": " boiled_egg ", "next": "cook(pot0)", "extra": 1})
        self.assertEqual((r["order"], r["next"], r["habit"]), ("boiled_egg", "cook(pot0)", ""))
        self.assertIsNone(clean_read({}))
        self.assertIsNone(clean_read(None))

    def test_bound_words_keeps_headings(self):
        text = "\n".join([MODEL_SECTIONS[0]] + [f"- word {i} " * 10 for i in range(20)] + [MODEL_SECTIONS[1], "- x"])
        out, cut = bound_words(text, 50)
        self.assertTrue(cut)
        self.assertIn(MODEL_SECTIONS[0], out)
        self.assertIn(MODEL_SECTIONS[1], out)
        self.assertLessEqual(len(out.split()), 60)
        self.assertEqual(bound_words("short", 50), ("short", False))

    def test_placebo_model(self):
        real = model_template().replace("(nothing yet)", "my partner always takes the newest order " * 4)
        fake = placebo_model(real)
        for h in MODEL_SECTIONS:
            self.assertIn(h, fake)
        self.assertNotIn("partner", fake.lower())
        self.assertLess(abs(len(fake) - len(real)), 0.5 * len(real))

    def test_json_with_raw_line_breaks(self):
        from coop.llm import parse_json
        text = '{"partner_model": "## How they choose what to do (with evidence)\n- takes the soup\n"}'
        raw = text.replace("\\n", "\n")        # real line breaks inside the string, as models write them
        self.assertIn("\n", raw)
        self.assertEqual(parse_json(raw)["partner_model"].splitlines()[1], "- takes the soup")
        self.assertEqual(parse_json("noise " + raw + " tail")["partner_model"].splitlines()[1], "- takes the soup")

    def test_modes_are_validated(self):
        with self.assertRaises(ValueError):
            LLMAgent({"type": "mock"}, tom=True, tom_mode="deep")
        with self.assertRaises(ValueError):
            LLMAgent({"type": "mock"}, memory=True, memory_mode="db")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        text = (ROOT / "configs/coop/days_llm_model.yaml").read_text().replace("type: openai_compat", "type: mock")
        cfg_path = cls.tmp / "cfg.yaml"
        cfg_path.write_text(text)
        cfg = load_config(cfg_path, personas=["oldest_first", "newest_first"], run_seeds=[0], days=2,
                          episodes_per_day=1, horizon=60)
        cls.res = quiet(run_all, cfg, cls.tmp / "out", 1)
        cls.out = cls.tmp / "out"

    def ep(self, cond, day, name, persona="oldest_first"):
        return json.loads((self.out / "runs" / cond / persona / "seed0" / f"day{day:02d}" / "ep01" / name).read_text())

    def test_all_runs_ok(self):
        self.assertTrue(all(r["status"] == "ok" for r in self.res["runs"]), self.res["runs"])

    def test_structured_reads_are_logged_and_carried(self):
        log = self.ep("tom_struct", 1, "agent_log.json")
        reads = [d for d in log["decisions"] if d.get("read")]
        self.assertTrue(reads)
        self.assertEqual(set(reads[0]["read"]), {"doing", "order", "next", "habit"})
        decides = [c["user"] for c in log["calls"] if c["kind"] == "decide"]
        self.assertNotIn("Your last read of your partner", decides[0])
        self.assertIn("Your last read of your partner", decides[1])
        self.assertIn('"doing"', log["system_prompt"])
        # Free ToM keeps its one-sentence format and shows no read.
        free = self.ep("tom_memory", 1, "agent_log.json")
        self.assertNotIn('"doing"', free["system_prompt"])
        self.assertFalse(any("Your last read" in c["user"] for c in free["calls"] if c["kind"] == "decide"))

    def test_partner_model_file(self):
        first = self.ep("tom_struct_model", 1, "agent_log.json")
        self.assertNotIn("Your partner model", first["system_prompt"])
        probe = [c for c in first["calls"] if c["kind"] == "probe"][0]["user"]
        self.assertNotIn("YOUR PARTNER MODEL SO FAR", probe)     # the probe no longer writes the file
        update = [c for c in first["calls"] if c["kind"] == "model"][0]["user"]
        self.assertIn("YOUR PARTNER MODEL SO FAR", update)
        self.assertIn('"partner_model"', update)
        self.assertNotIn("pace", update.lower())                 # no probe pace line or trait guide
        self.assertNotIn("What the answers mean", update)
        self.assertTrue(first["model_update"]["ok"])
        mem = self.ep("tom_struct_model", 1, "memory_after.json")
        self.assertTrue(mem["partner_model"].startswith(MODEL_SECTIONS[0]))
        second = self.ep("tom_struct_model", 2, "agent_log.json")
        self.assertIn("Your partner model", second["system_prompt"])
        self.assertIn(mem["partner_model"], second["system_prompt"])
        self.assertNotIn("What you noted after earlier episodes", second["system_prompt"])
        # Notes mode is unchanged.
        self.assertIn("What you noted after earlier episodes", self.ep("tom_memory", 2, "agent_log.json")["system_prompt"])

    def test_controls_get_the_file(self):
        donor = self.ep("tom_struct_model", 1, "memory_after.json", persona="newest_first")["partner_model"]
        swap = self.ep("tom_struct_model_swap", 2, "agent_log.json")["system_prompt"]
        self.assertIn(donor, swap)
        placebo = self.ep("tom_struct_model_placebo", 2, "agent_log.json")["system_prompt"]
        self.assertIn("Your partner model", placebo)
        self.assertIn("Orders are listed oldest first", placebo)        # 2 open orders
        self.assertNotIn("one at a time", placebo)
        self.assertNotIn("mock habit", placebo)

    def test_reads_are_scored(self):
        trace = self.ep("tom_struct", 1, "partner_truth.json")["trace"]
        self.assertIn("dish", trace[0])
        self.assertIn("committed", trace[0])
        rep = analyse(self.out)
        rows = rep["reads"]["personas"]["tom_struct"]
        for persona, r in rows.items():
            self.assertGreater(r["reads"], 0)
            # The mock always names the oldest open order, so it scores the oldest baseline.
            self.assertAlmostEqual(r["order_acc"], r["oldest_baseline"])
        self.assertNotIn("tom_memory", rep["reads"]["personas"])
        self.assertIn("structured ToM", format_report(rep))
        health = rep["model_health"]
        self.assertEqual(set(health), {"tom_model", "tom_struct_model", "tom_struct_model_swap",
                                       "tom_struct_model_placebo"})
        self.assertEqual(health["tom_struct_model"]["written"], 1.0)
        self.assertIn("partner-model file: was it written", format_report(rep))


if __name__ == "__main__":
    unittest.main()
