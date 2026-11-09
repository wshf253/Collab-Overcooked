"""Multi-day runs (src/coop/days.py): seeds, memory carry-over, resume, outputs."""
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from coop.agents import ScriptedAgent, register_agent
from coop.days import DaysConfig, episode_seed, load_config, run_all


class CountingAgent(ScriptedAgent):
    """Scripted ego that remembers every episode it played and every day it closed."""
    name = "counting"
    seen_results = []

    def end_episode(self, result):
        CountingAgent.seen_results.append(result)
        self.memory.setdefault("episodes", []).append(
            [result["day"], result["episode"], result["score"]])

    def end_day(self, day, results):
        self.memory.setdefault("days_closed", []).append([day, len(results)])

    def begin_episode(self, mdp, index, context):
        super().begin_episode(mdp, index, context)
        self.memory.setdefault("memory_at_start", []).append(len(self.memory.get("episodes", [])))


class BrokenAgent(ScriptedAgent):
    def act(self, state, t):
        if t == 5:
            raise RuntimeError("agent broke at t=5")
        return super().act(state, t)


register_agent("counting", CountingAgent)
register_agent("broken", BrokenAgent)


def small_config(**kw):
    base = dict(name="t", menu=["boiled_egg", "boiled_mushroom", "baked_carrot_soup"], horizon=60,
                days=2, episodes_per_day=2, personas=["steady", "slow"], run_seeds=[0],
                agent={"type": "counting", "persona": "steady"})
    base.update(kw)
    return DaysConfig(**base)


def silent(_):
    pass


class SeedTests(unittest.TestCase):
    def test_episode_seeds_are_stable_and_distinct(self):
        seeds = {episode_seed(0, d, e) for d in range(1, 6) for e in range(1, 4)}
        self.assertEqual(len(seeds), 15)
        self.assertEqual(episode_seed(2, 3, 1), episode_seed(2, 3, 1))
        self.assertNotEqual(episode_seed(0, 1, 1), episode_seed(1, 1, 1))


class ConfigTests(unittest.TestCase):
    def test_invalid_configs(self):
        for kw in (dict(personas=["nobody"]), dict(personas=[]), dict(days=0), dict(ego_index=2),
                   dict(run_seeds=[1, 1]), dict(custom_personas={"slow": {}}),
                   dict(custom_personas={"x": {"step_period": 0}}, personas=["x"]),
                   dict(menu=["not_a_dish"]), dict(agent={"persona": "steady"})):
            with self.subTest(**{k: str(v) for k, v in kw.items()}), self.assertRaises(ValueError):
                small_config(**kw)

    def test_conditions_merge_onto_the_base_agent(self):
        cfg = small_config(agent={"type": "llm", "backend": {"type": "mock", "seed": 1}, "probe": True},
                           conditions={"a": {"tom": True}, "b": {"backend": {"seed": 2}}})
        self.assertEqual(cfg.condition_agent("a"),
                         {"type": "llm", "backend": {"type": "mock", "seed": 1}, "probe": True, "tom": True})
        self.assertEqual(cfg.condition_agent("b")["backend"], {"type": "mock", "seed": 2})
        for bad in ({"_x": {}}, {}, {"a/b": {}}):
            with self.subTest(conditions=bad), self.assertRaises(ValueError):
                small_config(conditions=bad)

    def test_yaml_config_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("days: 3\npersonas: [steady]\ncustom_personas:\n  tired: {step_period: 3}\n")
            cfg = load_config(p, personas=["tired"])
            self.assertEqual((cfg.days, cfg.personas), (3, ["tired"]))
            p.write_text("dayz: 3\n")
            with self.assertRaises(ValueError):
                load_config(p)

    def test_shipped_configs_load(self):
        for p in sorted((ROOT / "configs" / "coop").glob("*.yaml")):
            with self.subTest(config=p.name):
                load_config(p)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        CountingAgent.seen_results = []

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_cfg(self, cfg, out=None):
        return run_all(cfg, out or self.tmp / "out", log=silent)

    def episode_dir(self, persona, day, ep, out=None, cond="main"):
        return (out or self.tmp / "out") / "runs" / cond / persona / "seed0" / f"day{day:02d}" / f"ep{ep:02d}"

    def test_outputs_and_paired_orders(self):
        res = self.run_cfg(small_config())
        self.assertTrue(all(r["status"] == "ok" for r in res["runs"]))
        for persona in ("steady", "slow"):
            for day in (1, 2):
                for ep in (1, 2):
                    d = self.episode_dir(persona, day, ep)
                    for f in ("summary.json", "steps.jsonl", "partner_truth.json", "agent_log.json",
                              "memory_after.json"):
                        self.assertTrue((d / f).exists(), d / f)
                    steps = (d / "steps.jsonl").read_text().splitlines()
                    self.assertEqual(len(steps), 60)
        # Same run seed, day and episode -> the same orders for every persona.
        for day, ep in ((1, 1), (2, 2)):
            orders = [json.loads((self.episode_dir(p, day, ep) / "summary.json").read_text())["orders"][0]["name"]
                      for p in ("steady", "slow")]
            self.assertEqual(orders[0], orders[1])
        seeds = {json.loads((self.episode_dir("steady", d, e) / "summary.json").read_text())["seed"]
                 for d in (1, 2) for e in (1, 2)}
        self.assertEqual(len(seeds), 4)
        self.assertEqual(set(res["table"]), {"main", "_reference"})
        self.assertEqual(set(res["table"]["main"]), {"steady", "slow"})
        self.assertEqual(res["table"]["main"]["steady"]["1"]["episodes"], 2)
        # The reference ego plays the same episodes; vs_reference is the paired difference.
        for persona in ("steady", "slow"):
            for day in (1, 2):
                diffs = []
                for ep in (1, 2):
                    a = json.loads((self.episode_dir(persona, day, ep) / "summary.json").read_text())
                    b = json.loads((self.episode_dir(persona, day, ep, cond="_reference") / "summary.json").read_text())
                    self.assertEqual(a["seed"], b["seed"])
                    self.assertEqual(a["orders"][0]["name"], b["orders"][0]["name"])
                    diffs.append(a["score"] - b["score"])
                self.assertAlmostEqual(res["table"]["main"][persona][str(day)]["vs_reference"], sum(diffs) / 2)
                self.assertIsNone(res["table"]["_reference"][persona][str(day)]["vs_reference"])

    def test_memory_carries_over_and_closes_days(self):
        self.run_cfg(small_config(personas=["steady"]))
        mem = json.loads((self.episode_dir("steady", 2, 2) / "memory_after.json").read_text())
        self.assertEqual([e[:2] for e in mem["episodes"]], [[1, 1], [1, 2], [2, 1], [2, 2]])
        self.assertEqual(mem["days_closed"], [[1, 2], [2, 2]])
        self.assertEqual(mem["memory_at_start"], [0, 1, 2, 3])

    def test_ego_never_sees_partner_identity(self):
        self.run_cfg(small_config(personas=["slow"]))
        self.assertEqual(len(CountingAgent.seen_results), 4)
        for result in CountingAgent.seen_results:
            text = json.dumps(result)
            self.assertNotIn("persona", text)
            self.assertNotIn('"slow"', text)
            self.assertNotIn("step_period", text)

    def test_resume_gives_the_same_run(self):
        cfg = small_config(personas=["steady"])
        full = self.run_cfg(cfg, self.tmp / "full")
        part = self.tmp / "part"
        self.run_cfg(cfg, part)
        shutil.rmtree(self.episode_dir("steady", 2, 1, part))      # as if interrupted on day 2
        shutil.rmtree(self.episode_dir("steady", 2, 2, part))
        (self.episode_dir("steady", 1, 2, part) / "memory_after.json").unlink()  # crashed mid-save
        resumed = self.run_cfg(cfg, part)
        self.assertEqual(resumed["runs"][0]["episodes_resumed"], 1)
        self.assertEqual(resumed["runs"][0]["episodes"], full["runs"][0]["episodes"])
        a = json.loads((self.episode_dir("steady", 2, 2, self.tmp / "full") / "memory_after.json").read_text())
        b = json.loads((self.episode_dir("steady", 2, 2, part) / "memory_after.json").read_text())
        self.assertEqual(a["episodes"], b["episodes"])
        self.assertEqual(a["days_closed"], b["days_closed"])

    def test_resume_refuses_a_different_config(self):
        self.run_cfg(small_config(personas=["steady"]))
        with self.assertRaises(ValueError):
            self.run_cfg(small_config(personas=["steady"], horizon=61))

    def test_resume_refuses_changed_code(self):
        import coop.days as days
        cfg = small_config(personas=["steady"], days=1, episodes_per_day=1)
        self.run_cfg(cfg)
        real = days.code_version
        try:
            days.code_version = lambda layout: "something-else"
            with self.assertRaises(ValueError):
                self.run_cfg(cfg)
            res = run_all(cfg, self.tmp / "out", log=silent, allow_code_change=True)
            self.assertEqual(res["runs"][0]["episodes_resumed"], 1)
            rec = json.loads((self.tmp / "out" / "code_version.json").read_text())
            self.assertEqual(rec["hash"], "something-else")
        finally:
            days.code_version = real

    def test_failed_run_is_reported(self):
        res = self.run_cfg(small_config(personas=["steady"], agent={"type": "broken"}))
        self.assertEqual(res["runs"][0]["status"], "error")
        self.assertIn("agent broke at t=5",
                      (self.tmp / "out" / "runs" / "main" / "steady" / "seed0" / "error.log").read_text())
        self.assertEqual(res["table"]["main"]["steady"], {})


if __name__ == "__main__":
    unittest.main()
