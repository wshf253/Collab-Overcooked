"""Continuous-order episodes on the Collab-Overcooked MDP.

An episode keeps issuing orders until the horizon ends:

* The order sequence is drawn up front from the episode seed (``coop.orders``),
  so every condition with the same seed sees the same orders.
* Exactly one order is open at a time. A correct delivery earns the layout's
  ``delivery_reward`` and opens the next order in the sequence.
* A wrong delivery (wrong item, or a plated/unplated mismatch) costs
  ``wrong_delivery_penalty`` points; the item is consumed and the order stays open.
* Interactions the legacy MDP cannot handle safely (e.g. adding an ingredient to
  a utensil that is already processing, which raises an assertion) become
  "stay" and are reported in ``info['rejected']``. The rules are the same as in
  the human-play server (``human_coop.session.Session.safe_decision``).

The wrapper only configures the MDP and records what happened. Movement,
utensils, recipes and delivery rules are the original Collab-Overcooked ones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from .orders import Menu, make_order_sequence, normalize_menu

DELIVER_ACTION = "deliver_soup()"
START = "[START]"


def unsafe_interact_reason(mdp: OvercookedGridworld, state: OvercookedState, index: int,
                           parameter: Any) -> Optional[str]:
    """Why player ``index`` must not interact now, or None if it may."""
    player = state.players[index]
    pos = Action.move_in_direction(player.position, player.orientation)
    tile = mdp.get_terrain_type_at_pos(pos)
    obj, held = state.objects.get(pos), player.held_object
    if tile == "I":
        if parameter not in mdp.default_ingredients:
            return "the dispenser needs a raw ingredient name"
    elif tile in mdp.interactive_terrain:
        if parameter == START:
            if held or not obj or obj.catagory != "semi-finished" or obj.state[2] != 0:
                return "starting needs empty hands and a loaded utensil that is not running"
        elif held:
            if held.name == "dish" and (not obj or obj.catagory != "finished"):
                return "a dish can only collect finished food"
            if held.name != "dish" and obj and not isinstance(obj.state[0], list):
                return "the utensil is already processing or holds finished food"
        elif obj and obj.catagory == "semi-finished":
            return "the utensil is loaded or running; start it or wait until it is done"
    return None


@dataclass
class Delivery:
    t: int              # step index at which the delivery action was taken (0-based)
    agent: int
    item: str
    plated: bool
    order: str          # order that was open when the item was delivered
    correct: bool


@dataclass
class OrderRecord:
    index: int          # position in the episode's order sequence
    name: str
    issued_t: int       # first step index at which the order was open
    completed_t: Optional[int] = None   # step index of the correct delivery
    delivered_by: Optional[int] = None
    wrong_deliveries: int = 0

    @property
    def duration(self) -> Optional[int]:
        """Steps from issue to completion, counting the delivery step."""
        if self.completed_t is None:
            return None
        return self.completed_t - self.issued_t + 1


@dataclass
class EpisodeLog:
    seed: int
    order_sequence: List[str]
    orders: List[OrderRecord] = field(default_factory=list)
    deliveries: List[Delivery] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)


class ContinuousOrderEnv:
    """Collab-Overcooked environment with a seeded, continuous order stream.

    Args:
        menu: order names that can be issued, as a list (equal weights) or a
            {name: weight} mapping. Names must be products of the layout's recipes.
        layout: layout file name under ``overcooked_ai_py/data/layouts``.
        horizon: episode length in steps.
        wrong_delivery_penalty: points lost per wrong delivery (>= 0).
        max_orders: length of the pre-drawn sequence. Defaults to ``horizon``,
            which can never run out (at most one delivery per agent per step,
            and every order needs several steps of preparation).
        no_repeat: never issue the same order twice in a row.
        sanitize: turn unsafe interactions into "stay" (see module docstring).
    """

    def __init__(
        self,
        menu: Menu,
        layout: str = "new_env",
        horizon: int = 400,
        wrong_delivery_penalty: float = 20,
        max_orders: Optional[int] = None,
        no_repeat: bool = False,
        sanitize: bool = True,
    ):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if wrong_delivery_penalty < 0:
            raise ValueError("wrong_delivery_penalty is a cost; pass a value >= 0")
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.menu = normalize_menu(menu)
        unknown = sorted(set(self.menu) - set(self.mdp.Recipe.recipe_list))
        if unknown:
            raise ValueError(f"Orders not producible in layout '{layout}': {unknown}")
        self.layout = layout
        self.horizon = horizon
        self.max_orders = max_orders or horizon
        self.no_repeat = no_repeat
        self.sanitize = sanitize
        # Continuous mode: our sequence replaces the layout's random replenishment.
        self.mdp.one_task_mode = False
        self.mdp.order_probability = {}
        self.mdp.wrong_delivery_penalty = wrong_delivery_penalty
        self.delivery_reward = self.mdp.delivery_reward
        self.wrong_delivery_penalty = wrong_delivery_penalty
        self.env: Optional[OvercookedEnv] = None
        self.log: Optional[EpisodeLog] = None
        self.score = 0.0

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, seed: int) -> OvercookedState:
        sequence = make_order_sequence(self.menu, seed, self.max_orders, self.no_repeat)
        self.mdp.start_order_list = list(sequence)
        if self.env is None:
            self.env = OvercookedEnv(self.mdp, horizon=self.horizon)
        else:
            self.env.reset()
        # Each episode owns its order list; the MDP's template is never mutated.
        self.env.state.order_list = list(sequence)
        self.log = EpisodeLog(seed=seed, order_sequence=list(sequence))
        self.log.orders.append(OrderRecord(index=0, name=sequence[0], issued_t=0))
        self.score = 0.0
        return self.env.state

    def step(self, joint_action: Sequence[Any], parm: Optional[Sequence[Any]] = None
             ) -> Tuple[OvercookedState, float, bool, Dict[str, Any]]:
        """Step the MDP; ``info['deliveries']`` lists this step's deliveries."""
        if self.env is None or self.log is None:
            raise RuntimeError("Call reset(seed) before step()")
        if self.done:
            raise RuntimeError("Episode is over; call reset(seed)")
        t = self.env.t
        prev = self.env.state
        joint_action = list(joint_action)
        parm = list(parm) if parm is not None else [None] * len(joint_action)
        rejected = []
        if self.sanitize:
            for i, action in enumerate(joint_action):
                if action != Action.INTERACT:
                    continue
                reason = unsafe_interact_reason(self.mdp, prev, i, parm[i])
                if reason:
                    rejected.append({"t": t, "agent": i, "parameter": parm[i], "reason": reason})
                    joint_action[i], parm[i] = Action.STAY, None
        held = [p.held_object for p in prev.players]
        open_orders = list(prev.order_list)

        state, reward, done, info = self.env.step(joint_action, parm)
        deliveries = self._resolve_deliveries(t, state, held, open_orders)
        self._check_consistency(reward, deliveries, open_orders, state)
        self.score += reward
        self.log.rejected.extend(rejected)
        info = dict(info, deliveries=deliveries, rejected=rejected)
        return state, reward, done, info

    # ------------------------------------------------------------------ #
    # Delivery bookkeeping
    # ------------------------------------------------------------------ #
    def _is_correct(self, obj, order: str) -> bool:
        """Same rule as OvercookedGridworld.deliver_soup."""
        name = obj.state[0] if obj.state is not None else obj.name
        if order != "any" and name != order:
            return False
        plated = obj.catagory == "with_dish"
        return plated == (name in self.mdp.Recipe.recipe_need_dish)

    def _resolve_deliveries(self, t, state, held, open_orders) -> List[Delivery]:
        # The MDP resolves interacts in player-index order, so a correct delivery
        # by player 0 opens the next order before player 1's item is checked.
        out: List[Delivery] = []
        k = 0
        for i, action in enumerate(state.ml_actions):
            if action != DELIVER_ACTION:
                continue
            obj = held[i]
            if obj is None or k >= len(open_orders):
                raise RuntimeError(f"Inconsistent delivery by agent {i} at t={t}")
            order = open_orders[k]
            correct = self._is_correct(obj, order)
            item = obj.state[0] if obj.state is not None else obj.name
            d = Delivery(t=t, agent=i, item=str(item), plated=obj.catagory == "with_dish",
                         order=order, correct=correct)
            out.append(d)
            self.log.deliveries.append(d)
            current = self.log.orders[-1]
            if correct:
                current.completed_t, current.delivered_by = t, i
                k += 1
                if k < len(open_orders):
                    self.log.orders.append(OrderRecord(index=current.index + 1,
                                                       name=open_orders[k], issued_t=t + 1))
            else:
                current.wrong_deliveries += 1
        return out

    def _check_consistency(self, reward, deliveries, open_orders, state):
        n_ok = sum(d.correct for d in deliveries)
        n_bad = len(deliveries) - n_ok
        expected = n_ok * self.delivery_reward - n_bad * self.wrong_delivery_penalty
        popped = len(open_orders) - len(state.order_list)
        if popped != n_ok or abs(reward - expected) > 1e-9:
            raise RuntimeError(
                f"Delivery bookkeeping disagrees with the MDP: reward {reward} vs {expected}, "
                f"orders closed {popped} vs {n_ok}")

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> OvercookedState:
        return self.env.state

    @property
    def t(self) -> int:
        return self.env.t

    @property
    def done(self) -> bool:
        return self.env.is_done()

    @property
    def current_order(self) -> Optional[str]:
        orders = self.env.state.order_list
        return orders[0] if orders else None

    def upcoming_orders(self, k: int) -> List[str]:
        """The open order followed by the next ``k - 1`` orders of the sequence."""
        return list(self.env.state.order_list[:k])

    def summary(self) -> Dict[str, Any]:
        log = self.log
        completed = [o for o in log.orders if o.completed_t is not None]
        durations = [o.duration for o in completed]
        return {
            "seed": log.seed,
            "layout": self.layout,
            "horizon": self.horizon,
            "steps": self.env.t,
            "score": self.score,
            "orders_completed": len(completed),
            "wrong_deliveries": sum(not d.correct for d in log.deliveries),
            "rejected_interactions": len(log.rejected),
            "mean_order_steps": sum(durations) / len(durations) if durations else None,
            "orders": [dict(asdict(o), duration=o.duration) for o in log.orders],
            "deliveries": [asdict(d) for d in log.deliveries],
        }
