"""coop.analyze pooling several output folders of one config run on different seeds."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.analyze import analyse, check_poolable, format_report  # noqa: E402
from coop.days import run_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_coop_llm import mock_config  # noqa: E402


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        base = dict(days=1, episodes_per_day=1, horizon=40, personas=["steady", "slow"],
                    conditions={"direct": {}, "tom": {"tom": True}})
        cls.a, cls.b, cls.c = cls.tmp / "a", cls.tmp / "b", cls.tmp / "c"
        quiet(run_all, mock_config(run_seeds=[0], **base), cls.a, 1)
        quiet(run_all, mock_config(run_seeds=[1, 2], **base), cls.b, 1)
        quiet(run_all, mock_config(run_seeds=[3], **dict(base, horizon=50)), cls.c, 1)

    def test_pooled_runs(self):
        rep = analyse([self.a, self.b])
        self.assertEqual(rep["overall"]["tom"]["vs_direct_runs"]["n"], 6)        # 2 personas x 3 seeds
        self.assertEqual(rep["by_folder"]["pooled"]["seeds"], [0, 1, 2])
        self.assertEqual(rep["by_folder"][str(self.b)]["paired"]["tom"]["vs_direct"]["n"], 4)
        self.assertIn("per folder", format_report(rep))
        self.assertNotIn("by_folder", analyse(self.a))                            # one folder: unchanged

    def test_refuses_what_cannot_be_pooled(self):
        self.assertEqual(check_poolable([self.a, self.b]), [])
        with self.assertRaisesRegex(ValueError, "run seed 1"):
            check_poolable([self.b, self.b])
        with self.assertRaisesRegex(ValueError, "horizon"):
            check_poolable([self.a, self.c])


if __name__ == "__main__":
    unittest.main()
