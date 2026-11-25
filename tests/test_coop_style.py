"""Plan v4: two-axis cards (workflow x adherence), the station workflow, the twin and V0
(docs/STYLE_PARTNER_PLAN_v4.md)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lib" / "overcooked_ai")]

from coop.context_partner import ContextPlayer  # noqa: E402
from coop.partner import PERSONAS, SimulatedHuman  # noqa: E402
from coop.persona_check import play_rule_episode  # noqa: E402
from coop.rules import expand_style, flip_adherence, load_cards, load_library, render  # noqa: E402

CARDS = ROOT / "configs/coop/personas/batch_style.yaml"
MENU = ["boiled_potato_slices", "baked_carrot_soup", "sliced_carrot_and_potato_stew"]
REF = {"boiled_potato_slices": 48, "baked_carrot_soup": 68, "sliced_carrot_and_potato_stew": 57.5}
DL = {"slack": 1.5, "reference": REF, "urgent_steps": 15}


def twin(card):
    return ContextPlayer(card.rules, MENU, "partner", "card", name=card.name, near_cooking=True, latent=card.latent)


class LoaderTests(unittest.TestCase):
    def test_cards_expand_with_latent_truth(self):
        cards = {n: c for n, c in load_cards(CARDS).items() if c.latent}
        self.assertEqual(len(cards), 7)
        pr = cards["prep_rigid"]
        self.assertEqual([r.label() for r in pr.rules],
                         ["default_work(mode=prep)", "on_urgent(mode=keeps)", "on_blocked(mode=waits)",
                          "on_prep_ready(mode=continues)"])
        pf = cards["prep_flexible"]
        self.assertEqual(pf.rules[1].label(), "on_urgent(mode=takes, unless_ego_on_it=yes)")
        st = cards["station_rigid"]
        self.assertEqual([r.id for r in st.rules], ["default_work", "on_urgent"])
        self.assertEqual({c: v["identifiable"] for c, v in st.contexts.items()},
                         {"on_urgent": True, "on_blocked": False, "on_prep_ready": False})
        mixed = cards["mixed"]
        self.assertEqual([r.p["mode"] for r in mixed.rules], ["prep", "takes", "waits", "continues"])
        d = mixed.to_dict()
        self.assertEqual(d["latent"]["workflow"], "prep")
        self.assertIn("style_habits", d["backstories"])
        # rule cards keep their old shape
        self.assertNotIn("latent", load_cards(ROOT / "configs/coop/personas/batch_ctx.yaml")["A"].to_dict())

    def test_loader_rejects_contradictions(self):
        lib = load_library()
        bad = [{"workflow": "chef", "adherence": "rigid"},
               {"workflow": "prep", "adherence": "lazy"},
               {"workflow": "prep"},
               {"workflow": "prep", "adherence": {"on_urgent": "rigid"}},
               {"workflow": "station", "adherence": {"on_urgent": "rigid", "on_blocked": "rigid", "on_prep_ready": "rigid"}},
               {"workflow": "prep", "adherence": "rigid", "rules": ["default_work"]},
               {"workflow": "prep", "adherence": "rigid", "backstory": {"long": "x"}}]
        for spec in bad:
            with self.assertRaises(ValueError, msg=spec):
                expand_style("x", spec, lib)

    def test_flips_for_wrong_belief_egos(self):
        cards = load_cards(CARDS)
        self.assertEqual([r.p["mode"] for r in flip_adherence(cards["prep_rigid"]).rules],
                         ["prep", "takes", "clears", "switches"])
        one = flip_adherence(cards["prep_rigid"], contexts=["on_blocked"])
        self.assertEqual([r.p["mode"] for r in one.rules], ["prep", "keeps", "clears", "continues"])
        ww = flip_adherence(cards["whole_flexible"], contexts=[], workflow="station")
        self.assertEqual([r.label() for r in ww.rules],
                         ["default_work(mode=station)", "on_urgent(mode=takes, unless_ego_on_it=yes)"])
        self.assertEqual(flip_adherence(cards["mixed"], contexts=[], workflow="station").latent["adherence"], "flexible")

    def test_station_texts_have_no_fallback(self):
        lib = load_library()
        for field in ("text", "wants", "reminder"):
            t = render(load_cards(CARDS)["station_rigid"].rules[0], field, lib).lower()
            self.assertTrue(t)
            for phrase in ("help", "if needed", "steps in", "stall"):
                self.assertNotIn(phrase, t)
        for c in load_cards(CARDS).values():
            for text in (c.backstories or {}).values():
                for phrase in ("helps out", "when needed", "steps in", "if things stall"):
                    self.assertNotIn(phrase, text.lower(), c.name)


class TwinTests(unittest.TestCase):
    def test_twin_adherence_on_every_card(self):
        for name, c in load_cards(CARDS).items():
            if c.latent is None:
                continue
            a = n = 0
            for seed in range(3):
                r = play_rule_episode(MENU, "open_ing2_env", 300, seed, c.rules, SimulatedHuman(PERSONAS["steady"], "steady"),
                                      2, 2, deadlines=DL, order_mode="controlled", partner_player=twin(c))
                for row in r["rules"].values():
                    a += row["partner"]["adherent"]
                    n += row["partner"]["opportunities"]
            self.assertGreater(n, 15, name)
            self.assertGreaterEqual(a / n, 0.95, (name, a, n))

    def test_station_cook_stays_at_the_stations_and_logs_latent(self):
        c = load_cards(CARDS)["station_rigid"]
        tw = twin(c)
        ego = ContextPlayer(c.rules, MENU, "ego", "proactive", name="br", near_cooking=True)
        r = play_rule_episode(MENU, "open_ing2_env", 300, 0, c.rules, ego, 2, 2, deadlines=DL,
                              order_mode="controlled", partner_player=tw)
        wf = [j for j in r["judgments"] if j.side == "partner" and j.rule.startswith("default_work")]
        self.assertTrue(wf)
        self.assertTrue(all(j.value for j in wf), [j.note for j in wf if not j.value])   # no fetch, cut or serve
        row = tw.decision_log[0]
        self.assertEqual(row["latent"], c.latent)
        self.assertIn("mode", row)
        self.assertIn(row["in_other_view"], (True, False))
        self.assertGreater(r["orders_completed"], 0)


class LeakageTests(unittest.TestCase):
    def test_no_card_text_in_ego_prompts(self):
        from coop.days import load_config, run_all
        text = (ROOT / "configs/coop/days_ctx.yaml").read_text().replace("type: openai_compat", "type: mock")
        text = text.replace("  type: llm\n  cards: configs/coop/personas/batch_ctx.yaml",
                            f"  type: rule\n  cards: {CARDS}")
        self.assertIn("batch_style", text)
        tmp = Path(tempfile.mkdtemp())
        (tmp / "c.yaml").write_text(text)
        cfg = load_config(tmp / "c.yaml", run_seeds=[0], days=1, episodes_per_day=1, horizon=80,
                          conditions=["tom_struct"], personas=["prep_rigid", "station_flexible"],
                          eval_episodes=0, knowledge_probe=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = run_all(cfg, tmp / "out", 1, log=lambda s: None)
        self.assertTrue(all(r["status"] == "ok" for r in res["runs"]), res["runs"])
        lib = load_library()
        cards = load_cards(CARDS)
        for persona in ("prep_rigid", "station_flexible"):
            ep = tmp / "out" / "runs" / "tom_struct" / persona / "seed0" / "day01" / "ep01"
            truth = json.loads((ep / "partner_truth.json").read_text())
            self.assertEqual(truth["card"]["latent"], cards[persona].latent)
            log = (ep / "agent_log.json").read_text()
            for r in cards[persona].rules:
                self.assertNotIn(render(r, "text", lib)[:60], log)
            for story in cards[persona].backstories.values():
                self.assertNotIn(story[:60], log)
            for word in ("rigid", "flexible", persona):
                self.assertNotIn(word, log)


class V0Tests(unittest.TestCase):
    def test_v0_tool(self):
        from coop.voi import format_v0, style_egos, v0
        cards = load_cards(CARDS)
        self.assertIn("flip:on_blocked", style_egos(cards["prep_rigid"]))
        self.assertNotIn("flip:on_urgent", style_egos(cards["station_rigid"]))   # one identifiable context
        self.assertEqual(len([e for e in style_egos(cards["whole_rigid"]) if e.startswith("wrong_workflow")]), 2)
        res = v0(CARDS, "open_ing2_env", MENU, DL, [0], [1], episodes=1, horizon=60, view_radius=2, open_orders=2)
        self.assertEqual(set(res["cards_result"]), set(n for n, c in cards.items() if c.latent))
        self.assertIn("d_adherence", res["cards_result"]["prep_rigid"])
        self.assertIn("V0:", format_v0(res))
        self.assertEqual(len(res["checks"]), 5)


if __name__ == "__main__":
    unittest.main()
