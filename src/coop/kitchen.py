"""Kitchen facts and text views for an agent in one player slot.

Everything here is public: the map, the recipes, what is on counters and in
utensils, where both players are and what they hold, and the actions each
player has completed. Nothing reads the partner's internal state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .nav import Pos, floor_cells, reach, touches

RECIPE_DIR = Path(__file__).resolve().parents[1] / "prompts" / "recipe"
TILE_NAMES = {"X": "counter", "I": "ingredient dispenser", "D": "dish dispenser",
              "S": "serving location", " ": "floor"}


class Kitchen:
    """Static facts about a layout, seen from player slot ``index``."""

    def __init__(self, mdp, index: int, start_state):
        self.mdp, self.index, self.other = mdp, index, 1 - index
        self.terrain = mdp.terrain_mtx
        self.floor = floor_cells(self.terrain)
        self.cells: Dict[str, List[Pos]] = {}
        for y, row in enumerate(self.terrain):
            for x, c in enumerate(row):
                self.cells.setdefault(c, []).append((x, y))
        self.utensils: Dict[str, Pos] = {u: tuple(mdp.from_utensil_name_to_pos(u)) for u in mdp.utensil_list}
        self.kind_of = {u: u.rstrip("0123456789") for u in self.utensils}
        self.verb_of = {u: mdp.utensils[k]["operation"] for u, k in self.kind_of.items()}
        self.utensil_at = {p: u for u, p in self.utensils.items()}
        self.recipes: Dict[str, Tuple[str, List[str]]] = {}
        for kind, products in mdp.recipes.items():
            if kind != "none":
                for product, spec in products.items():
                    self.recipes[product] = (kind, list(spec["recipe"]))
        self.raw = list(mdp.default_ingredients)
        # Reachability is fixed by the layout: players never cross the counters.
        players = start_state.players
        self.my_region = reach(self.floor, tuple(players[index].position))
        self.other_region = reach(self.floor, tuple(players[self.other].position))

    # ------------------------------------------------------------------ #
    def mine(self, pos: Pos) -> bool:
        return touches(pos, self.my_region)

    def theirs(self, pos: Pos) -> bool:
        return touches(pos, self.other_region)

    def counters(self) -> List[Pos]:
        return [p for p in self.cells.get("X", []) if self.mine(p) or self.theirs(p)]

    def need_dish(self, item: str) -> bool:
        return bool(self.mdp.need_dish.get(item, 0))

    def cook_time(self, product: str) -> int:
        return int(self.mdp.Recipe.recipe_list[product]["cook_time"])

    def leaves(self, item: str) -> List[str]:
        """Raw ingredients ``item`` is made from."""
        if item not in self.recipes:
            return [item]
        out: List[str] = []
        for child in self.recipes[item][1]:
            out += [x for x in self.leaves(child) if x not in out]
        return out

    def chain(self, item: str) -> str:
        """Compact process description, e.g. cook in pot(bake in oven(cut on chopping_board(carrot)))."""
        if item not in self.recipes:
            return item
        kind, inputs = self.recipes[item]
        verb = self.mdp.utensils[kind]["operation"]
        return f"{verb} in {kind}({' + '.join(self.chain(c) for c in inputs)})"

    def recipe_text(self, dish: str) -> str:
        files = sorted(RECIPE_DIR.glob(f"*_{dish}.txt")) if RECIPE_DIR.exists() else []
        files = [f for f in files if f.stem.split("_", 1)[-1] == dish]
        text = files[0].read_text().strip() if files else f"NAME: {dish}"
        dish_note = "served on a dish" if self.need_dish(dish) else "served without a dish"
        return f"{text}\nPROCESS: {self.chain(dish)} -> {dish} ({dish_note})"

    # ------------------------------------------------------------------ #
    def side(self, pos: Pos) -> str:
        a, b = self.mine(pos), self.theirs(pos)
        return "both of you" if a and b else "you" if a else "your partner" if b else "nobody"

    def map_text(self) -> str:
        rows = ["    x=" + "".join(str(x % 10) for x in range(len(self.terrain[0])))]
        for y, row in enumerate(self.terrain):
            rows.append(f"  y={y} " + "".join(row))
        legend = ["Legend: ' ' floor, X counter (holds one item), I ingredient dispenser (any raw "
                  "ingredient), D dish dispenser, S serving location"]
        for u, p in self.utensils.items():
            legend.append(f"  {u} at {p} ({self.mdp.utensils[self.kind_of[u]]['symbol']}): "
                          f"start it with {self.verb_of[u]}({u}); reachable by {self.side(p)}")
        for tile, name in (("I", "ingredient dispenser"), ("D", "dish dispenser"), ("S", "serving location")):
            for p in self.cells.get(tile, []):
                legend.append(f"  {name} at {p}: reachable by {self.side(p)}")
        shared = [p for p in self.cells.get("X", []) if self.mine(p) and self.theirs(p)]
        legend.append(f"  counters both of you can reach (for handing items over): {shared}")
        return "\n".join(rows + legend)

    def utensil_status(self, state, u: str) -> Tuple[str, Optional[str], List[str], int]:
        """(status, product, contents, steps_left); status is empty|loaded|processing|finished."""
        obj = state.objects.get(self.utensils[u])
        if obj is None:
            return "empty", None, [], 0
        name, _, t = obj.state
        if isinstance(name, list):
            return "loaded", None, list(name), 0
        if obj.catagory == "finished":
            return "finished", name, [], 0
        return "processing", name, [], max(0, self.cook_time(name) - t)

    def utensil_text(self, state, u: str) -> str:
        status, product, contents, left = self.utensil_status(state, u)
        if status == "empty":
            return f"{u}: empty"
        if status == "loaded":
            return f"{u}: holds {', '.join(contents)} (not started)"
        if status == "processing":
            return f"{u}: making {product}, ready in {left} step(s)"
        return f"{u}: {product} is ready to take out"


def held_text(obj) -> str:
    if obj is None:
        return "nothing"
    if obj.catagory == "with_dish":
        return f"a dish of {obj.name}"
    return obj.name
