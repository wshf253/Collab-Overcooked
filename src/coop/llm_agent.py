"""LLM ego agent: watches its partner, optionally reasons about it (ToM) and
remembers it across episodes and days (memory).

Conditions (agent config keys):

* ``tom``: before each action, the agent writes what it thinks its partner is
  doing and tends to do (``"partner"`` field), then the action. Off: action only.
  Across conditions, only this output block differs.
* ``memory``: after each episode, the agent reflects on its partner. The profile
  and notes are kept (the last ``max_memory_notes``) and shown in every later
  episode's system prompt.
* ``probe``: after each episode, the agent rates its partner on four traits
  (``partner_profile``). This is scored against the true persona for evaluation.
  With ``memory`` on, the probe is the same call as the reflection. With memory
  off, the answer is logged but never shown to the agent again.

The agent sees only public information: the map, recipes, the open order,
utensils, counters, both players' positions and held items, the actions each
player completed, deliveries and discarded batches.
"""
from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from .agents import Agent, register_agent
from .executor import MAX_WAIT, Executor, parse_macro
from .kitchen import Kitchen, held_text
from .llm import Usage, build_backend, parse_json
from .profile import PROFILE_FIELDS, clean_profile

GRAMMAR = f"""\
- pickup(<item>, ingredient_dispenser): take a raw ingredient (hands must be empty)
- pickup(dish, dish_dispenser): take a clean dish (hands must be empty)
- pickup(<item>, counter): take <item> from a counter you can reach
- pickup(<item>, <utensil>): take finished food out of a utensil (food that needs a dish: use fill_dish_with_food)
- put_obj_in_utensil(<utensil>): put what you hold into a utensil that is not running
- cut(<utensil>) / cook(<utensil>) / bake(<utensil>) / stir(<utensil>): start a loaded utensil with empty hands (the verb must match the utensil)
- fill_dish_with_food(<utensil>): holding a dish, collect the food in a utensil (you wait there if it is still cooking)
- deliver_soup(): serve the food you hold at the serving location
- place_obj_on_counter(): put what you hold on a free counter (counters both of you can reach first: this is how items are handed over)
- wait(<n>): do nothing for n steps (1-{MAX_WAIT}); it ends early when your partner completes an action"""


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, backend: Dict[str, Any], tom: bool = False, memory: bool = False,
                 probe: bool = True, max_memory_notes: int = 5, max_tokens: int = 300,
                 probe_max_tokens: int = 500, partner_log_lines: int = 15, retry_invalid: bool = True):
        self.backend_spec = dict(backend)
        self.backend = build_backend(self.backend_spec, "ego")
        self.tom, self.use_memory, self.probe = tom, memory, probe
        self.max_memory_notes, self.max_tokens, self.probe_max_tokens = max_memory_notes, max_tokens, probe_max_tokens
        self.partner_log_lines, self.retry_invalid = partner_log_lines, retry_invalid
        self.set_memory({})

    def set_memory(self, memory):
        self.memory = {"notes": [], "episodes_played": 0, **copy.deepcopy(memory)}

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def begin_episode(self, mdp, index, context):
        self.context = context
        self.mdp = mdp
        self.kitchen: Optional[Kitchen] = None      # built on the first act (needs the start state)
        self.executor: Optional[Executor] = None
        self.partner_log: List[Tuple[int, str]] = []
        self.own_log: List[Tuple[int, str, str]] = []      # (t, action, result)
        self.events: List[str] = []
        self.last_result = "this is the first decision of the episode"
        self.score = 0.0
        self.orders_done = 0
        self.wrong = 0
        self.calls: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.usage = Usage()
        self.invalid = self.fallbacks = 0
        self.probe_answer: Optional[Dict[str, Any]] = None
        self.call_index = 0
        self.system = None
        self._last_macro = None

    def observe(self, t, reward, info):
        self.score += reward
        for d in info.get("deliveries", []):
            who = "you" if d.agent == self.context["index"] else "your partner"
            if d.correct:
                self.orders_done += 1
                self.events.append(f"t={t}: {who} delivered {d.item}: correct (+20)")
            else:
                self.wrong += 1
                self.events.append(f"t={t}: {who} delivered {d.item} for order {d.order}: WRONG "
                                   f"(-{self.context['wrong_delivery_penalty']:g})")
        for r in info.get("rejected", []):
            if r["agent"] == self.context["index"] and self.executor is not None:
                self.executor.note_rejected(r["reason"])

    def act(self, state, t):
        if self.kitchen is None:
            self.kitchen = Kitchen(self.mdp, self.context["index"], state)
            self.executor = Executor(self.kitchen)
            self.system = self._system_prompt()
        self._watch(state, t)
        ended = self.executor.update(state)
        if ended:
            status, message = ended
            self.last_result = f"{status}: {message}"
            self.own_log.append((t, str(self._last_macro), self.last_result))
        if not self.executor.busy:
            self._decide(state, t)
        return self.executor.step(state)

    def end_episode(self, result):
        self.memory["episodes_played"] = self.memory.get("episodes_played", 0) + 1
        if not (self.probe or self.use_memory):
            return
        answer = self._probe(result)
        self.probe_answer = answer
        if self.use_memory and answer is not None:
            notes = self.memory.setdefault("notes", [])
            notes.append({"day": result["day"], "episode": result["episode"],
                          "profile": answer["partner_profile"], "notes": answer["notes"]})
            del notes[:-self.max_memory_notes]

    def episode_log(self):
        return {"condition": {"tom": self.tom, "memory": self.use_memory, "probe": self.probe},
                "usage": self.usage.as_dict(), "invalid_actions": self.invalid, "fallbacks": self.fallbacks,
                "decisions": self.decisions, "probe": self.probe_answer, "system_prompt": self.system,
                "calls": self.calls}

    # ------------------------------------------------------------------ #
    # Watching (public information only)
    # ------------------------------------------------------------------ #
    def _watch(self, state, t):
        k = self.kitchen
        if t > 0:
            done = state.ml_actions[k.other]
            if done is not None:
                self.partner_log.append((t - 1, done))
        for msg in state.error_message or []:
            note = re.sub(r"^TIMESTEP \d+:\s*", "", msg)
            self.events.append(f"t={t - 1}: {note}")

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        k, ctx = self.kitchen, self.context
        menu = list(ctx["menu"])
        recipes = "\n\n".join(k.recipe_text(d) for d in menu)
        if self.tom:
            reply = ('{"partner": "<one or two sentences: what your partner is doing now and what they tend '
                     'to do>", "action": "<one action>"}')
        else:
            reply = '{"action": "<one action>"}'
        parts = [
            "You are one of two cooks in a kitchen game (an adapted Overcooked). You and your partner "
            "have the same abilities: either of you can use any station you can reach. You cannot talk "
            "to your partner; there are no messages. Your partner is a person with their own habits. "
            "Watch what they do and work with them.",
            f"Goal: complete as many orders as possible in {ctx['horizon']} time steps. One order is open "
            f"at a time. A correct delivery gives +20 points and opens the next order; a wrong delivery "
            f"costs {ctx['wrong_delivery_penalty']:g} points and the order stays open. Items placed in a "
            "utensil that match no recipe are thrown away when the utensil is started.",
            f"Kitchen map (you are player {ctx['index']}):\n{k.map_text()}",
            f"Dishes that can be ordered:\n\n{recipes}",
            f"Actions (choose exactly one each time you are asked):\n{GRAMMAR}",
        ]
        notes = self.memory.get("notes") if self.use_memory else None
        if notes:
            lines = []
            for n in notes:
                prof = ", ".join(f"{a}: {b}" for a, b in n["profile"].items() if b)
                lines.append(f"- day {n['day']}, episode {n['episode']}: profile ({prof}); notes: {n['notes']}")
            parts.append("You have worked with this same partner before. What you noted after earlier "
                         "episodes:\n" + "\n".join(lines))
        parts.append(f"Reply with JSON only: {reply}")
        return "\n\n".join(parts)

    def _observation(self, state, t) -> str:
        k = self.kitchen
        me, other = state.players[k.index], state.players[k.other]
        order = state.order_list[0] if state.order_list else None
        lines = [f"Time step {t} of {self.context['horizon']}. Score {self.score:+g}. "
                 f"Orders completed: {self.orders_done}.",
                 f"Current order: {order}" + (" (served on a dish)" if order and k.need_dish(order) else ""),
                 f"You: at {tuple(me.position)}, holding {held_text(me.held_object)}.",
                 f"Partner: at {tuple(other.position)}, holding {held_text(other.held_object)}.",
                 "Utensils: " + "; ".join(k.utensil_text(state, u) for u in k.utensils) + "."]
        items = [f"{p} {held_text(state.objects[p])} (reachable by {k.side(p)})"
                 for p in k.counters() if p in state.objects]
        lines.append("Counters: " + ("; ".join(items) if items else "all empty") + ".")
        recent = self.partner_log[-self.partner_log_lines:]
        if recent:
            lines.append("Your partner's completed actions (most recent last):")
            lines += [f"  t={tt}: {a}" for tt, a in recent]
            lines.append(f"  (last one {t - 1 - recent[-1][0]} step(s) ago)")
        else:
            lines.append("Your partner has not completed any action yet.")
        if self.events:
            lines.append("Events: " + " | ".join(self.events[-5:]))
        mine = self.own_log[-5:]
        if mine:
            lines.append("Your recent actions: " + " | ".join(f"t={tt} {a} -> {r}" for tt, a, r in mine))
        lines.append(f"Result of your last action: {self.last_result}")
        return "\n".join(lines)

    def _ingredients_to_list(self) -> List[str]:
        out: List[str] = []
        for d in self.context["menu"]:
            out += [x for x in self.kitchen.leaves(d) if x not in out]
        return out

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #
    def _seed(self) -> int:
        self.call_index += 1
        h = hashlib.sha256(f"{self.context['episode_seed']}:{self.call_index}".encode()).digest()
        return int.from_bytes(h[:4], "big") & 0x7FFFFFFF

    def _chat(self, kind: str, t: int, messages, max_tokens) -> Tuple[Optional[dict], str]:
        r = self.backend.chat(messages, max_tokens=max_tokens, seed=self._seed())
        self.usage.add(r)
        parsed = parse_json(r.text)
        self.calls.append({"t": t, "kind": kind, "user": messages[-1]["content"], "response": r.text,
                           "error": r.error, "latency_s": round(r.latency_s, 3),
                           "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                           "thinking": r.thinking, "finish_reason": r.finish_reason})
        return parsed, r.text

    def _decide(self, state, t):
        options = self.executor.options(state, self._ingredients_to_list())
        user = self._observation(state, t) + "\nVALID_ACTIONS: " + " | ".join(options)
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        record: Dict[str, Any] = {"t": t, "attempts": []}
        macro = None
        for attempt in range(2 if self.retry_invalid else 1):
            parsed, text = self._chat("decide" if attempt == 0 else "retry", t, messages, self.max_tokens)
            action = parsed.get("action") if parsed else None
            m, reason = parse_macro(action) if isinstance(action, str) else (None, "no action in the reply")
            if m is not None:
                reason = self.executor.check(m, state) or ""
            record["attempts"].append({"action": action, "partner": (parsed or {}).get("partner"),
                                       "error": reason})
            if m is not None and not reason:
                macro = m
                break
            self.invalid += 1
            messages = messages + [{"role": "assistant", "content": text or ""},
                                   {"role": "user", "content": f"That action is not possible: {reason}. "
                                    "Choose one action from VALID_ACTIONS and reply with JSON only."}]
        if macro is None:
            self.fallbacks += 1
            macro, _ = parse_macro("wait(1)")
        record["chosen"] = str(macro)
        self.decisions.append(record)
        self._last_macro = macro
        self.executor.begin(macro)

    # ------------------------------------------------------------------ #
    # End-of-episode reflection / probe
    # ------------------------------------------------------------------ #
    def _probe(self, result) -> Optional[Dict[str, Any]]:
        log = "\n".join(f"t={t}: {a}" for t, a in self.partner_log) or "(none)"
        events = "\n".join(self.events) or "(none)"
        options = ", ".join(f'"{k}": ' + " | ".join(f'"{o}"' for o in v) for k, v in PROFILE_FIELDS.items())
        user = (
            "PARTNER_PROFILE_PROBE\n"
            f"The episode is over (day {result['day']}, episode {result['episode']}). Team score "
            f"{result['score']:+g}, orders completed {result['orders_completed']}, wrong deliveries "
            f"{result['wrong_deliveries']}, {self.context['horizon']} steps.\n"
            f"Every action your partner completed (time step: action):\n{log}\n"
            f"Events:\n{events}\n\n"
            "Describe your partner from what you saw. Stages: prep = getting raw ingredients and "
            "using the chopping board or blender; cook = using the pot or oven and moving products "
            "between utensils; serve = dishes, plating and delivering. Note that a player can only "
            "do what their side of the kitchen allows.\n"
            f'Reply with JSON only: {{"partner_profile": {{{options}}}, "notes": "<at most 80 words: '
            'what to do differently with this partner next time>"}')
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        parsed, _ = self._chat("probe", self.context["horizon"], messages, self.probe_max_tokens)
        if not parsed:
            return None
        notes = parsed.get("notes")
        return {"partner_profile": clean_profile(parsed.get("partner_profile")),
                "raw_profile": parsed.get("partner_profile"),
                "notes": notes if isinstance(notes, str) else ""}


register_agent("llm", LLMAgent)
