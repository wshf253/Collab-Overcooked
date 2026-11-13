"""coop.compare: contrasts between two output folders, paired by run."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

import yaml  # noqa: E402

from coop.compare import compare, format_compare  # noqa: E402


def write(out: Path, scores, view_radius, version="v1"):
    """scores: {(condition, persona, seed, day, episode): score}"""
    for (cond, persona, seed, day, ep), score in scores.items():
        d = out / "runs" / cond / persona / f"seed{seed}" / f"day{day:02d}" / f"ep{ep:02d}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"persona": persona, "run_seed": seed, "day": day,
                                                    "episode": ep, "score": score}))
    (out / "config.json").write_text(json.dumps({"name": out.name, "view_radius": view_radius}))
    (out / "code_version.json").write_text(json.dumps({"hash": version}))


class CompareTests(unittest.TestCase):
    def test_change_is_paired_by_run(self):
        a, b = {}, {}
        for p, gain_a, gain_b in (("steady", 30, 10), ("newest_first", 50, 10)):
            for seed in (0, 1):
                for ep in (1, 2):
                    base = 100 + 10 * seed + ep          # harder or easier episodes, same in both arms
                    a[("direct", p, seed, 1, ep)] = base
                    a[("tom_memory", p, seed, 1, ep)] = base + gain_a + seed
                    b[("direct", p, seed, 1, ep)] = base - 20
                    b[("tom_memory", p, seed, 1, ep)] = base - 20 + gain_b + seed
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d) / "po", Path(d) / "full"
            write(pa, a, 2)
            write(pb, b, None, version="v2")
            rep = compare(pa, pb)
            text = format_compare(rep)
        c = rep["contrasts"]["tom_memory - direct"]
        self.assertAlmostEqual(c["a"]["all"]["mean"], 40.5)
        self.assertAlmostEqual(c["b"]["all"]["mean"], 10.5)
        self.assertAlmostEqual(c["change"]["all"]["mean"], 30.0)
        self.assertAlmostEqual(c["change"]["personas"]["newest_first"]["mean"], 40.0)
        self.assertAlmostEqual(c["change"]["all"]["se"], 5.7735, places=3)    # 20, 20, 40, 40 over 4 runs
        self.assertNotIn("tom - direct", rep["contrasts"])
        self.assertEqual(rep["config_differences"], {"view_radius": [2, None]})
        self.assertIn("WARNING: different code versions", text)

    def test_full_view_config_is_a_copy(self):
        po = yaml.safe_load((ROOT / "configs/coop/days_llm_orders.yaml").read_text())
        full = yaml.safe_load((ROOT / "configs/coop/days_llm_orders_full.yaml").read_text())
        self.assertEqual(po.pop("view_radius"), 2)
        self.assertIsNone(full.pop("view_radius"))
        po.pop("name"), full.pop("name")
        self.assertEqual(po, full)


if __name__ == "__main__":
    unittest.main()
