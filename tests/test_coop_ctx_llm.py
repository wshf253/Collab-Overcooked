"""Plan v3 LLM experiments with mock backends: LLM partner on context cards (reminders,
guard), Gate 2 with scripted context egos, Gate 3 (oracle / oracle_mismatched), and the
learning pilot (regimes, held-back dishes, context probe, report)."""
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
from coop.learning import format_report, report  # noqa: E402
from coop.orders import make_controlled_sequence  # noqa: E402
from coop.persona_check import gate0_folder, rescore  # noqa: E402


def run(name, **kw):
    tmp = Path(tempfile.mkdtemp())
    text = (ROOT / f"configs/coop/{name}.yaml").read_text().replace("type: openai_compat", "type: mock")
    (tmp / "cfg.yaml").write_text(text)
    cfg = load_config(tmp / "cfg.yaml", **kw)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        res = run_all(cfg, tmp / "out", 4)
    assert all(r["status"] == "ok" for r in res["runs"]), [r.get("error") for r in res["runs"] if r["status"] != "ok"][:1]
    return tmp / "out" / "runs", tmp / "out"


class CtxLLMTests(unittest.TestCase):
    def test_weighted_controlled_lists(self):
        seq = make_controlled_sequence({"a": 3, "b": 1}, 0, 40)
        self.assertEqual(seq.count("a"), 30)
        self.assertEqual(make_controlled_sequence(["a", "b"], 3, 10), make_controlled_sequence({"a": 1, "b": 1}, 3, 10))

    def test_gate1_partner_on_context_cards(self):
        runs, _ = run("persona_gate1_ctx", run_seeds=[0], horizon=150)
        log = json.loads((runs / "steady_ego/A/seed0/day01/ep01/partner_log.json").read_text())
        self.assertIn("Your job is preparation", log["system_prompt"])
        self.assertNotIn("Nothing has been done in the kitchen", log["system_prompt"])   # the guard is no habit
        s = json.loads((runs / "steady_ego/A/seed0/day01/ep01/summary.json").read_text())
        self.assertIn("orders_expired", s)
        self.assertIn("guards", s["partner_llm"])
        self.assertIn("default_work(mode=prep)", s["rules"])
        user = " ".join(c["user"] for c in log["calls"])
        self.assertIn("steps left]", user)                                             # timers are visible
        rs = rescore(runs.parent)                                    # replay under the same timers
        self.assertEqual(rs["failed"], [])

    def test_gate2_scripted_context_egos(self):
        runs, out = run("persona_gate2_ctx", run_seeds=[0, 1], horizon=100, episodes_per_day=1)
        lg = json.loads((runs / "mismatched/A/seed0/day01/ep01/agent_log.json").read_text())
        self.assertEqual((lg["card"], lg["policy"]), ("B", "proactive"))
        rep = gate0_folder(out, [0], [1])
        self.assertEqual(sorted(rep["cards_result"]), ["A", "B"])
        self.assertIn("proactive", rep["egos"])

    def test_useless_start_is_refused(self):
        """useful_starts_only: a pot with only the potato of an open stew may not be started."""
        from overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        from coop.env import ContinuousOrderEnv
        from coop.executor import Executor, parse_macro
        from coop.kitchen import Kitchen
        menu = ["boiled_potato_slices", "baked_carrot_soup", "sliced_carrot_and_potato_stew"]
        env = ContinuousOrderEnv(menu, layout="open_ing2_env", open_orders=2, order_mode="controlled")
        with contextlib.redirect_stdout(io.StringIO()):
            st = env.reset(0)
        st.order_list[:2] = ["sliced_carrot_and_potato_stew", "baked_carrot_soup"]
        k = Kitchen(env.mdp, 0, st)
        pot = k.utensils["pot0"]
        st.objects[pot] = ObjectState("potato", "semi-finished", pot, (["potato"], 1, 0))
        ex = Executor(k)
        m, _ = parse_macro("cook(pot0)")
        self.assertIsNone(ex.check(m, st))                       # off by default: allowed
        ex.useful_starts_only, ex.n_open = True, 2
        self.assertIn("no open order needs", ex.check(m, st))
        st.objects[pot] = ObjectState("potato", "semi-finished", pot, (["carrot_slices", "potato"], 2, 0))
        self.assertIsNone(ex.check(m, st))                       # the stew batch is complete

    def test_gate3_oracle_and_mismatched(self):
        runs, _ = run("persona_value_ctx", run_seeds=[0], horizon=200)
        sp = lambda c: json.loads((runs / f"{c}/A/seed0/day01/ep01/agent_log.json").read_text())["system_prompt"]
        self.assertIn("Cook and serve what I prepare", sp("oracle"))                 # card A
        self.assertIn("Take the other open order", sp("oracle_mismatched"))          # card B
        self.assertNotIn("What your partner wants", sp("tom_struct"))
        self.assertIn("What your partner wants", sp("oracle_cued"))
        lg = json.loads((runs / "oracle_cued/A/seed0/day01/ep01/agent_log.json").read_text())
        self.assertIn("cues_shown", lg)
        cued = [c["user"] for c in lg["calls"] if "What your partner told you applies now" in c["user"]]
        self.assertEqual(len(cued), lg["cues_shown"])

    def test_learning_pilot(self):
        runs, out = run("days_ctx", run_seeds=[0, 1], days=2, episodes_per_day=1, horizon=130,
                        conditions=["tom_struct", "generic_memory", "memory_fb"])
        reg = lambda seed, d: json.loads((runs / f"tom_struct/A/seed{seed}/day{d:02d}/ep01/summary.json").read_text())["regime"]
        self.assertEqual((reg(0, 1), reg(1, 1)), ("urgent", "blocked"))
        inter = {o["name"] for d in (1, 2) for s in (0, 1)
                 for o in json.loads((runs / f"memory_fb/A/seed{s}/day{d:02d}/ep01/summary.json").read_text())["orders"]}
        self.assertFalse(inter & {"baked_potato_soup", "boiled_carrot_slices"})    # held back
        kp = json.loads((runs / "memory_fb/A/seed0/day01/knowledge_probe.json").read_text())
        self.assertEqual([q["id"] for q in kp["questions"]], ["default_work", "on_urgent", "on_blocked", "on_prep_ready"])
        self.assertEqual(rescore(out)["failed"], [])                    # regimes replay too
        text = format_report(report(out))
        self.assertIn("transfer", text)
        self.assertIn("orders_expired", text)


if __name__ == "__main__":
    unittest.main()
