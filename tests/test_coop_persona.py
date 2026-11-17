"""LLM persona partner (docs/PERSONA_PARTNER_PLAN_v2.md): rules, cards, twin, predicates,
LLM partner (mock), runner integration, no leakage into the ego's prompts, gate tools."""
import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.days import load_config, run_all  # noqa: E402
from coop.env import ContinuousOrderEnv  # noqa: E402
from coop.kitchen import Kitchen  # noqa: E402
from coop.llm_partner import LLMPartner, macro_stage  # noqa: E402
from coop.partner import PERSONAS, SimulatedHuman  # noqa: E402
from coop.persona_check import (format_gate1, format_gate2, format_why, gate1, gate2,  # noqa: E402
                                play_rule_episode, rescore, why)
from coop.rule_partner import RulePlayer  # noqa: E402
from coop.rules import (CATEGORIES, NEUTRAL, Rule, category, check_rules, load_cards, load_library,  # noqa: E402
                        parse_rule, render)
from coop.screen import DEFAULT_MENU  # noqa: E402

CARDS = ROOT / "configs" / "coop" / "personas" / "batch1.yaml"


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def kitchen():
    env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout="open2_env", open_orders=2)
    return Kitchen(env.mdp, 1, quiet(env.reset, 0))


class RuleLibraryTests(unittest.TestCase):
    def setUp(self):
        self.lib = load_library()
        self.cards = load_cards(CARDS, self.lib)

    def test_library_has_the_ten_rules(self):
        self.assertEqual(set(self.lib), {"own_station", "solo_dish", "split_dish", "order_priority", "serves",
                                         "wants_supplies", "handover_spot", "on_takeover", "on_contention",
                                         "waits_for_commit"})
        for rid, spec in self.lib.items():
            self.assertIn(spec["category"], CATEGORIES)

    def test_pilot_cards_meet_the_design_constraints(self):
        cards = {n: c for n, c in self.cards.items() if n != NEUTRAL}
        self.assertEqual(len(cards), 4)
        for c in cards.values():
            self.assertEqual(len(c.rules), 6)
            self.assertEqual(check_rules(c.rules), [])
            self.assertTrue(c.backstory)
        for a, b in itertools.combinations(cards.values(), 2):
            self.assertGreaterEqual(len(set(a.rules) ^ set(b.rules)) // 2 + len(set(a.rules) - set(b.rules)) // 2, 3,
                                    (a.name, b.name))
            self.assertGreaterEqual(len(set(a.rules) - set(b.rules)), 3, (a.name, b.name))
        for cat in CATEGORIES:
            variants = {frozenset(r for r in c.rules if category(r, self.lib) == cat) for c in cards.values()}
            present = [c for c in cards.values() if any(category(r, self.lib) == cat for r in c.rules)]
            self.assertGreaterEqual(len(present), 2, cat)
            self.assertGreaterEqual(len(variants), 2, cat)      # differs between cards (params or present/absent)
        self.assertEqual(self.cards[NEUTRAL].rules, [])

    def test_optional_parameters(self):
        r = lambda e: parse_rule(e, self.lib)
        a = r({"split_dish": {"dish": "boiled_egg"}})
        self.assertEqual(a, r({"split_dish": {"dish": "boiled_egg", "patience": 20}}))   # default not stored
        self.assertEqual(a.label(), "split_dish(dish=boiled_egg)")
        self.assertEqual(a.opt("patience"), "20")
        b = r({"split_dish": {"dish": "boiled_egg", "patience": 40}})
        self.assertEqual(b.label(), "split_dish(dish=boiled_egg, patience=40)")
        self.assertIn("40 steps", render(b, "fallback", self.lib))
        self.assertIn("20 steps", render(a, "fallback", self.lib))
        w = r({"own_station": {"station": "pot1", "when_taken": "wait"}})
        self.assertIn("wait until it is free", render(w, "text", self.lib))
        self.assertIn("I wait for it", render(w, "wants", self.lib))
        self.assertNotIn("wait", render(r({"own_station": {"station": "pot1"}}), "text", self.lib))
        self.assertTrue(check_rules([w, r({"own_station": {"station": "pot0"}})]))      # still one per kind
        with self.assertRaises(ValueError):
            r({"split_dish": {"dish": "boiled_egg", "patience": 30}})
        with self.assertRaises(ValueError):
            r({"split_dish": {"patience": 40}})                                          # required missing

    def test_revised_cards(self):
        cards = {n: c for n, c in load_cards(ROOT / "configs/coop/personas/batch2.yaml", self.lib).items()
                 if n != NEUTRAL}
        self.assertEqual(sorted(cards), ["p1", "p2", "p3", "p4"])
        for c in cards.values():
            self.assertEqual(check_rules(c.rules), [])
            self.assertTrue(c.backstory)
            ids = {r.id for r in c.rules}
            self.assertFalse(ids & {"handover_spot", "wants_supplies"}, c.name)
            self.assertTrue(all(r.opt("when_taken") == "wait" for r in c.rules if r.id == "own_station"))
        for a, b in itertools.combinations(cards.values(), 2):
            self.assertGreaterEqual(len(set(a.rules) - set(b.rules)), 3, (a.name, b.name))
        for cat in CATEGORIES:
            if cat == "hand-over":
                continue      # dropped with handover_spot (deviation from the plan, see the card file)
            self.assertGreaterEqual(sum(any(category(r, self.lib) == cat for r in c.rules)
                                        for c in cards.values()), 2, cat)

    def test_decision_step(self):
        """Checks about the situation a choice was made in use the step it was decided at."""
        from types import SimpleNamespace
        from coop.rule_eval import Ctx, Tracker, decided_at
        self.assertEqual(decided_at(SimpleNamespace(agent=SimpleNamespace(decisions=[{"t": 3}, {"t": 9}]))), 9)
        self.assertEqual(decided_at(SimpleNamespace(job="x", job_t=5)), 5)   # scripted: its current job
        self.assertIsNone(decided_at(SimpleNamespace(job=None, job_t=5)))
        env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout="open2_env", open_orders=2)
        state = quiet(env.reset, 0)
        tr = Tracker(Kitchen(env.mdp, 0, state), DEFAULT_MENU, {0: "ego", 1: "partner"}, 2)
        self.assertEqual(Ctx(10, state, state, [], tr, {}, {"decided": {"partner": 4}}).decided("partner"), 4)
        tr.prev_event_t["partner"] = 7
        self.assertEqual(Ctx(10, state, state, [], tr, {}, None).decided("partner"), 8)   # estimate

    def test_contradictions_are_rejected(self):
        r = lambda e: parse_rule(e, self.lib)
        self.assertTrue(check_rules([r({"solo_dish": {"dish": "boiled_egg"}}), r({"split_dish": {"dish": "boiled_egg"}})]))
        self.assertTrue(check_rules([r({"serves": {"who": "ego"}}), r({"solo_dish": {"dish": "boiled_egg"}})]))
        self.assertTrue(check_rules([r({"serves": {"who": "self"}}), r({"split_dish": {"dish": "boiled_egg"}})]))
        self.assertTrue(check_rules([r({"own_station": {"station": "pot0"}}), r({"own_station": {"station": "pot1"}})]))
        self.assertTrue(check_rules([r({"order_priority": {"which": "oldest"}}), r({"order_priority": {"which": "newest"}})]))
        self.assertEqual(check_rules([r({"own_station": {"station": "pot0"}}), r({"own_station": {"station": "oven1"}})]), [])
        with self.assertRaises(ValueError):
            r({"own_station": {"station": "pot9"}})
        with self.assertRaises(ValueError):
            r("no_such_rule")

    def test_texts_render_fully(self):
        k = kitchen()
        for card in self.cards.values():
            for rule in card.rules:
                for fld in ("text", "fallback", "wants", "reminder"):
                    text = render(rule, fld, self.lib, k, {"station": "pot0", "missing": "chickpea", "dish": "x"})
                    self.assertTrue(text, (rule, fld))
                    self.assertNotIn("{", text, (rule, fld, text))
        self.assertEqual(self.cards["p1"].active(2), self.cards["p1"].rules[:2])     # nested complexity


class TwinTests(unittest.TestCase):
    """The rule-based twin follows its card: the predicate check (full view)."""

    @classmethod
    def setUpClass(cls):
        lib = load_library()
        cards = load_cards(CARDS, lib)
        cls.totals = defaultdict(lambda: [0, 0])
        cls.scores = {}
        for name in ("p1", "p2", "p3", "p4"):
            sc = []
            for seed in range(6):       # single rules with ~10 opportunities are noisy over fewer seeds
                r = play_rule_episode(DEFAULT_MENU, "open2_env", 300, seed, cards[name].rules,
                                      SimulatedHuman(PERSONAS["steady"], "steady"), None, 2)
                sc.append(r["score"])
                for rule, row in r["rules"].items():
                    t = cls.totals[(name, rule)]
                    t[0] += row["partner"]["adherent"]
                    t[1] += row["partner"]["opportunities"]
            cls.scores[name] = sum(sc) / len(sc)

    def test_twin_adheres_to_its_card(self):
        per_card = defaultdict(lambda: [0, 0])
        for (name, rule), (a, n) in self.totals.items():
            per_card[name][0] += a
            per_card[name][1] += n
            if n >= 10:      # single rules with ~10 opportunities are noisy: a floor, not the bar
                self.assertGreaterEqual(a / n, 0.80, (name, rule, a, n))
        # Card level: 0.93. Since waits_for_commit and split_dish judge only the decisions their
        # texts cover, cards have fewer easy judgments; p2's twin sits at 94-96% because its
        # order_priority(newest) is ~90% even with a full view (8 seeds: 95.5%).
        for name, (a, n) in per_card.items():
            self.assertGreaterEqual(a / n, 0.93, (name, a, n))

    def test_twin_completes_orders(self):
        for name, s in self.scores.items():
            self.assertGreater(s, 100, name)


class LLMPartnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        text = (ROOT / "configs/coop/persona_gate1.yaml").read_text().replace("type: openai_compat", "type: mock")
        cfg_path = cls.tmp / "gate1.yaml"
        cfg_path.write_text(text)
        cls.cfg = load_config(cfg_path, run_seeds=[0, 1], horizon=100)
        cls.res = quiet(run_all, cls.cfg, cls.tmp / "out", 2)
        # An LLM ego (structured ToM + partner file) with the LLM partner, for the leakage check.
        text2 = (ROOT / "configs/coop/days_llm_model.yaml").read_text().replace("type: openai_compat", "type: mock")
        text2 += ("partner:\n  type: llm\n  cards: configs/coop/personas/batch1.yaml\n  complexity: 6\n"
                  "  backend: {type: mock}\n")
        text2 = text2.replace("personas: [steady, oldest_first, newest_first, favourite_dish, repeats_dish]",
                              "personas: [p1, p2, p3, p4]")
        text2 = text2.replace("donors: {steady: newest_first, oldest_first: newest_first, newest_first: oldest_first,\n"
                              "               favourite_dish: newest_first, repeats_dish: newest_first}",
                              "donors: {p1: p2, p2: p3, p3: p4, p4: p1}")
        (cls.tmp / "ego.yaml").write_text(text2)
        cls.cfg2 = load_config(cls.tmp / "ego.yaml", run_seeds=[0], days=2, episodes_per_day=1, horizon=60)
        cls.res2 = quiet(run_all, cls.cfg2, cls.tmp / "ego_out", 4)

    def ep(self, out, cond, persona, name, seed=0, day=1, ep=1):
        return json.loads((self.tmp / out / "runs" / cond / persona / f"seed{seed}" / f"day{day:02d}" /
                           f"ep{ep:02d}" / name).read_text())

    def test_runs_complete_and_write_rule_results(self):
        self.assertTrue(all(r["status"] == "ok" for r in self.res["runs"]), self.res["runs"])
        s = self.ep("out", "steady_ego", "p1", "summary.json")
        self.assertEqual(s["partner_type"], "llm")
        self.assertIn("own_station(station=oven0)", s["rules"])
        self.assertIn("usage", s["partner_llm"])
        truth = self.ep("out", "steady_ego", "p1", "partner_truth.json")
        self.assertEqual(truth["card"]["name"], "p1")
        self.assertNotIn("traits", truth)
        self.assertTrue((self.tmp / "out" / "run_meta.json").exists())
        meta = json.loads((self.tmp / "out" / "run_meta.json").read_text())
        self.assertIn("partner_intro", meta["prompt_templates"])
        self.assertIn("sha256", meta["prompt_templates"]["rule_library"])

    def test_partner_prompt_intention_and_reminders(self):
        log = self.ep("out", "steady_ego", "p1", "partner_log.json")
        sp = log["system_prompt"]
        self.assertIn("Ten years in a bakery", sp)
        self.assertIn("You always use oven0", sp)
        self.assertIn('"reconsider_when"', sp)
        users = [c["user"] for c in log["calls"]]
        self.assertNotIn("Your current goal", users[0])
        self.assertTrue(any("Your current goal (set at t=" in u for u in users[1:]))
        self.assertTrue(log["reminders"])
        self.assertTrue(any(u.count("Reminder: ") for u in users))
        neutral = self.ep("out", "steady_ego", "neutral", "partner_log.json")["system_prompt"]
        self.assertNotIn("Your habits", neutral)
        self.assertIn("How you decide", neutral)

    def test_enforce_is_only_a_hook(self):
        card = load_cards(CARDS)["p1"]
        with self.assertRaises(NotImplementedError):
            LLMPartner({"type": "mock"}, card, load_library(), enforce=True)
        self.assertEqual(macro_stage("cut(chopping_board0)"), "prep")
        self.assertEqual(macro_stage("fill_dish_with_food(pot1)"), "serve")
        self.assertEqual(macro_stage("wait(3)"), "wait")

    def test_no_leakage_into_the_ego(self):
        """Outside the oracle condition the ego never sees the card, rules, the partner's
        intentions or reminders, or any score of the partner's rules."""
        self.assertTrue(all(r["status"] == "ok" for r in self.res2["runs"]), self.res2["runs"])
        lib = load_library()
        cards = load_cards(CARDS, lib)
        k = kitchen()
        # Specific strings only (generic words such as "habits" are part of the ego's own intro).
        secrets = {"Reminder:", "reconsider", "Your current goal", "mock goal", "Your habits", "adheren",
                   "How you decide", "Who you are"}
        for c in cards.values():
            if c.backstory:
                secrets.add(c.backstory[:40])
            for r in c.rules:
                secrets.add(r.id)
                secrets.add(render(r, "text", lib, k)[:40])
                secrets.add(render(r, "wants", lib, k)[:30])
        checked = 0
        for log_path in (self.tmp / "ego_out" / "runs").glob("*/*/seed*/day*/ep*/agent_log.json"):
            if "_reference" in str(log_path):
                continue
            log = json.loads(log_path.read_text())
            texts = [log.get("system_prompt") or ""] + [c["user"] for c in log.get("calls", [])]
            for text in texts:
                for s in secrets:
                    self.assertNotIn(s, text, f"{s!r} leaked into {log_path}")
            checked += len(texts)
        self.assertGreater(checked, 50)

    def test_gate_tools(self):
        rep = gate1(self.tmp / "out", [0], [1], 0.7)
        text = format_gate1(rep)
        self.assertIn("distinctness", text)
        self.assertIn("p1", rep["cards"])
        self.assertIn("own_station(station=oven0)", rep["cards"]["p1"]["rules"])
        w = why(self.tmp / "out", examples=2)
        self.assertTrue(w["rows"])
        self.assertNotIn("neutral", {r["card"] for r in w["rows"]})
        for r in w["rows"]:
            self.assertEqual(r["n"], r["adherent"] + sum(c for _, c in r["not"]))
            self.assertLessEqual(len(r["examples"]), 2)
            for e in r["examples"]:
                self.assertIsNotNone(e["chosen"])            # matched to the partner decision that chose it
        self.assertIn("adherent", format_why(w))
        # Re-scoring replays the saved actions; rules whose checks did not change keep their judgments.
        rs = rescore(self.tmp / "out")
        self.assertEqual(rs["failed"], [])
        self.assertEqual(rs["rescored"], rs["episodes"])
        changed = ("split_dish", "serves", "waits_for_commit", "own_station")
        key = lambda j: (j["t"], j["rule"], j["side"], j["value"], j["note"])
        for orig in (self.tmp / "out").glob("runs/*/*/seed*/day*/ep*/rule_events.jsonl"):
            a = sorted(key(json.loads(x)) for x in orig.read_text().splitlines() if not x.startswith('{"t"') or
                       not any(c in json.loads(x)["rule"] for c in changed))
            b = sorted(key(json.loads(x)) for x in (orig.parent / "rule_events.rescored.jsonl").read_text().splitlines()
                       if not any(c in json.loads(x)["rule"] for c in changed))
            self.assertEqual(a, b, orig)
        r2 = gate1(self.tmp / "out", [0], [1], 0.7, rescored=True)
        self.assertEqual(r2["rescored"]["rescored"], r2["rescored"]["episodes"])
        self.assertIn("re-scored", format_gate1(r2))
        self.assertTrue(why(self.tmp / "out", rescored=True)["rescored"])
        only = why(self.tmp / "out", cards=["p2"], rule_filter="serves")
        self.assertTrue(all(r["card"] == "p2" and "serves" in r["rule"] for r in only["rows"]))
        g2 = self.tmp / "g2.yaml"
        g2.write_text((ROOT / "configs/coop/persona_gate2.yaml").read_text().replace("  type: llm\n", "  type: rule\n"))
        cfg = load_config(g2, run_seeds=[0, 1], horizon=80, episodes_per_day=1)
        quiet(run_all, cfg, self.tmp / "g2out", 4)
        rep2 = gate2(self.tmp / "g2out")
        self.assertIn("aligned-generic:score", rep2["tables"]["pooled"]["diffs"])
        self.assertNotIn("aligned-uninformed:score", rep2["tables"]["pooled"]["diffs"])   # twin screen only
        text2 = format_gate2(rep2)
        self.assertIn("Gate 2", text2)
        self.assertLess(text2.index("score (primary)"), text2.index("alignment (secondary)"))
        misaligned = json.loads(next((self.tmp / "g2out" / "runs" / "misaligned" / "p1").glob(
            "seed0/day01/ep01/agent_log.json")).read_text())
        self.assertEqual(misaligned["card"], "p2")


class RevisedCardTwinTests(unittest.TestCase):
    """batch2: the twin follows the waiting variants (radius 2, the gate setting)."""

    def test_twin_adheres_to_revised_cards(self):
        lib = load_library()
        cards = load_cards(ROOT / "configs/coop/personas/batch2.yaml", lib)
        for name in ("p1", "p2", "p3", "p4"):
            a = n = 0
            for seed in range(3):
                r = play_rule_episode(DEFAULT_MENU, "open2_env", 300, seed, cards[name].rules,
                                      SimulatedHuman(PERSONAS["steady"], "steady"), 2, 2)
                self.assertGreater(r["score"], 100, name)
                for row in r["rules"].values():
                    a += row["partner"]["adherent"]
                    n += row["partner"]["opportunities"]
            self.assertGreaterEqual(a / n, 0.9, (name, a, n))

    def test_twin_waits_for_its_station(self):
        """own_station(when_taken=wait): while the other cook's batch is in its station, the
        twin does not move that work to the other station."""
        from coop.rule_partner import RulePlayer
        lib = load_library()
        rules = [parse_rule({"own_station": {"station": "pot1", "when_taken": "wait"}}, lib)]
        env = quiet(ContinuousOrderEnv, DEFAULT_MENU, layout="open2_env", open_orders=2)
        quiet(env.reset, 0)
        twin = RulePlayer(rules, list(DEFAULT_MENU), "partner", "twin")
        twin.reset(env.mdp, 1, 0)
        twin.act(env.state, 0)
        twin.rt.loaders["pot1"] = [("ego", "egg")]
        twin.rt.busy_since["pot1"] = 0
        k = twin.kitchen
        orig = k.utensil_status
        k.utensil_status = lambda state, u: ("loaded", None, ["egg"], 0) if u == "pot1" else orig(state, u)
        try:
            self.assertTrue(twin._taken_by_other("pot1"))
            twin.rt.busy_since["pot1"] = -100                     # waited longer than the patience
            self.assertFalse(twin._taken_by_other("pot1"))
            twin.rt.busy_since["pot1"] = 0
            twin.rt.loaders["pot1"] = [("partner", "egg")]        # its own batch: not a reason to wait
            self.assertFalse(twin._taken_by_other("pot1"))
        finally:
            k.utensil_status = orig


class ScoreValueScreenTests(unittest.TestCase):
    def test_card_screen_has_the_uninformed_ego(self):
        from coop.voi import card_screen
        res = card_screen(CARDS, "open2_env", [0], 60, DEFAULT_MENU, view_radius=2, open_orders=2)
        t = res["tables"]["selection"]
        self.assertEqual(t["egos"], ["aligned", "generic", "misaligned", "uninformed"])
        self.assertIn("aligned-uninformed:score", t["diffs"])
        self.assertIn("score headroom over uninformed", format_gate2(res))

    def test_rule_screen(self):
        from coop.voi import format_rule_screen, rule_screen, rule_variants
        variants = rule_variants("open2_env", list(DEFAULT_MENU))
        labels = [str(v[0]) for v in variants]
        self.assertEqual(len(variants), len(set(labels)))
        own = [v for v in variants if isinstance(v[0], dict) and "own_station" in v[0]]
        self.assertTrue(own and all(v[1]["own_station"]["station"][:-1] == v[0]["own_station"]["station"][:-1]
                                    for v in own))                    # opposite: the other station of that kind
        self.assertIn(({"waits_for_commit": {"patience": "10"}}, None), variants)       # options included
        self.assertIn(({"own_station": {"station": "pot1", "when_taken": "wait"}},
                       {"own_station": {"station": "pot0", "when_taken": "wait"}}), variants)
        self.assertIn(("on_takeover", None), variants)
        res = rule_screen("open2_env", [0, 1], 40, DEFAULT_MENU, 2, 2)
        self.assertEqual(len(res["rows"]), len(variants))
        row = next(r for r in res["rows"] if r["rule"] == "serves(who=ego)")
        self.assertIn("aligned-opposite", row)
        self.assertIn("aligned-uninformed", format_rule_screen(res))


if __name__ == "__main__":
    unittest.main()
