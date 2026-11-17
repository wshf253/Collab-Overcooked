"""A folder started before config keys were added still resumes (same config, new keys at default)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.days import load_config, run_all  # noqa: E402

LATER_KEYS = ("schedule", "order_lists", "eval_episodes", "knowledge_probe", "deadlines")


class ConfigCompatTests(unittest.TestCase):
    def test_older_folder_with_fewer_keys_resumes(self):
        tmp = Path(tempfile.mkdtemp())
        cfg = load_config(ROOT / "configs/coop/persona_gate1_twin.yaml", run_seeds=[0], horizon=30,
                          personas=["p1"], episodes_per_day=1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_all(cfg, tmp, 1)
        path = tmp / "config.json"
        old = json.loads(path.read_text())
        for k in LATER_KEYS:
            old.pop(k, None)                 # as written by the code before these keys existed
        path.write_text(json.dumps(old))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = run_all(cfg, tmp, 1)
        self.assertTrue(all(r["episodes_resumed"] == len(r["episodes"]) for r in res["runs"]))

    def test_a_real_difference_is_still_refused(self):
        tmp = Path(tempfile.mkdtemp())
        cfg = load_config(ROOT / "configs/coop/persona_gate1_twin.yaml", run_seeds=[0], horizon=30,
                          personas=["p1"], episodes_per_day=1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_all(cfg, tmp, 1)
        other = load_config(ROOT / "configs/coop/persona_gate1_twin.yaml", run_seeds=[0], horizon=40,
                            personas=["p1"], episodes_per_day=1)
        with self.assertRaises(ValueError):
            run_all(other, tmp, 1)


if __name__ == "__main__":
    unittest.main()
