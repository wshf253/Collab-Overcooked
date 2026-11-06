"""Ego-agent interface for multi-day runs.

The runner (``coop.days``) calls, in order:

    set_memory(memory)                 # once, before the first episode (or when resuming)
    for each day:
        for each episode of the day:
            begin_episode(mdp, index, context)
            act(state, t) ...          # every step
            end_episode(result)        # what the agent may learn from this episode
            get_memory()               # saved after every episode
        end_day(day, results)

Memory is the only thing that carries over between episodes: an agent must
keep everything it wants to remember in the dict returned by ``get_memory``, so
a run can stop and resume from any saved episode. The dict must be
JSON-serialisable.

``context`` and ``result`` only contain what the agent may know. They never
contain the partner's persona, traits or ground-truth jobs.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

from .partner import SimulatedHuman


class Agent:
    name = "agent"

    def set_memory(self, memory: Dict[str, Any]) -> None:
        self.memory = copy.deepcopy(memory)

    def get_memory(self) -> Dict[str, Any]:
        return copy.deepcopy(getattr(self, "memory", {}))

    def begin_episode(self, mdp, index: int, context: Dict[str, Any]) -> None:
        pass

    def act(self, state, t: int) -> Tuple[Any, Any]:
        raise NotImplementedError

    def end_episode(self, result: Dict[str, Any]) -> None:
        pass

    def end_day(self, day: int, results: List[Dict[str, Any]]) -> None:
        pass

    def episode_log(self) -> Dict[str, Any]:
        """Agent-side record of the episode (saved for analysis, not fed back)."""
        return {}


class ScriptedAgent(Agent):
    """A simulated human as the ego: the no-memory reference.

    It acts the same way on every day, so any change across days in its
    results comes from the orders, not from learning.
    """
    name = "scripted"

    def __init__(self, persona: str = "steady"):
        self.persona = persona
        self.memory: Dict[str, Any] = {}

    def begin_episode(self, mdp, index, context):
        self.player = SimulatedHuman.persona(self.persona)
        self.player.reset(mdp, index, context["episode_seed"])

    def act(self, state, t):
        return self.player.act(state, t)

    def episode_log(self):
        return {"summary": self.player.summary(), "trace": self.player.trace}


AGENTS: Dict[str, Callable[..., Agent]] = {"scripted": ScriptedAgent}


def register_agent(type_name: str, factory: Callable[..., Agent]) -> None:
    AGENTS[type_name] = factory


def make_agent(spec: Dict[str, Any]) -> Agent:
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in AGENTS:
        raise ValueError(f"Unknown agent type {kind!r}; known: {sorted(AGENTS)}")
    return AGENTS[kind](**spec)
