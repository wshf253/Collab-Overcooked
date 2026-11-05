# Coop research setting (`src/coop`)

This setting uses the Collab-Overcooked environment (utensils, multi-step
recipes, layouts). Its goal is to learn about a partner by watching it act,
then use that knowledge to work together well. Both agents will have the same
abilities, with no fixed chef/assistant roles. Episodes will be grouped into
days. It is built in steps; this page lists what exists so far.
The original benchmark (`src/main.py`, `evaluation.py`) does not use this package.

## Continuous orders (`coop.env.ContinuousOrderEnv`)

```python
from coop.env import ContinuousOrderEnv

env = ContinuousOrderEnv(
    menu=["boiled_egg", "baked_carrot_soup"],   # or {"boiled_egg": 2, "baked_carrot_soup": 1}
    layout="new_env", horizon=400, wrong_delivery_penalty=20)
state = env.reset(seed=0)
state, reward, done, info = env.step(joint_action, parm)   # same arguments as OvercookedEnv.step
env.current_order, env.upcoming_orders(3), env.summary()
```

- **Seeded sequence.** At `reset(seed)` the episode's whole order sequence is
  drawn from the menu with a private RNG. The same seed gives the same orders
  under every condition (paired design). `no_repeat=True` forbids the same
  order twice in a row.
- **One open order at a time.** A correct delivery earns the layout's
  `delivery_reward` (20) and opens the next order in the sequence. The episode
  ends at `horizon`.
- **Wrong deliveries** cost `wrong_delivery_penalty` points. This covers a
  wrong item, or a dish served plated when it should not be (and the reverse).
  The item is consumed and the order stays open.
- **Records.** `info["deliveries"]` lists each step's deliveries: who delivered
  what, against which order, and whether it was correct. `env.summary()` gives
  the score, orders completed, wrong deliveries, steps per order, and per-order
  records (issued, completed, delivered by, wrong attempts). After every step
  the wrapper checks its bookkeeping against the MDP's reward and order list.

The only change to the MDP is the optional `wrong_delivery_penalty`
attribute of `OvercookedGridworld`. Its default is 0, so the original benchmark
keeps its behaviour: a wrong delivery gives 0 points.

## Not built yet

Days (several episodes with the same partner), one shared prompt and action
list for both agents, no messages (and no view of the teammate's plan), the
partner-observation layer, and the LLM agent.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tested with Python 3.11 and `numpy<2`. The legacy env uses `np.Inf`, which
NumPy 2 removed.
