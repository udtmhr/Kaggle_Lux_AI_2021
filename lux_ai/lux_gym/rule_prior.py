import numpy as np
import torch
import math
from typing import Dict, Tuple, Optional, List
from ..lux.game import Game
from ..lux.game_objects import Unit, CityTile, Player
from ..lux.game_map import Cell, Position
from .act_spaces import ACTION_MEANINGS

class RulePriorEngine:
    def __init__(self):
        self.w_actions = ACTION_MEANINGS["worker"]
        self.c_actions = ACTION_MEANINGS["cart"]
        self.ct_actions = ACTION_MEANINGS["city_tile"]
        
        self.w_noop = self.w_actions.index("NO-OP")
        self.c_noop = self.c_actions.index("NO-OP")
        self.ct_research = self.ct_actions.index("RESEARCH")
        self.ct_build_worker = self.ct_actions.index("BUILD_WORKER")
        self.ct_build_cart = self.ct_actions.index("BUILD_CART")
        
        self.w_build_city = self.w_actions.index("BUILD_CITY")
        self.w_move = {
            "n": self.w_actions.index("MOVE_n"),
            "e": self.w_actions.index("MOVE_e"),
            "s": self.w_actions.index("MOVE_s"),
            "w": self.w_actions.index("MOVE_w"),
        }
        self.dir_delta = {
            "n": (0, -1),
            "e": (1, 0),
            "s": (0, 1),
            "w": (-1, 0),
        }

    def _dist(self, pos1: Position, pos2: Position) -> int:
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _get_night_info(self, game_state: Game) -> Tuple[bool, int]:
        cycle_turn = game_state.turn % 40
        is_night = cycle_turn >= 30
        remaining_night = 40 - cycle_turn if is_night else 10
        return is_night, remaining_night

    def _get_resource_tiles(self, game_state: Game) -> List[Cell]:
        rt = []
        for y in range(game_state.map.height):
            for x in range(game_state.map.width):
                cell = game_state.map.get_cell(x, y)
                if cell.has_resource():
                    rt.append(cell)
        return rt

    def _get_mineable(self, player: Player, rt: List[Cell]) -> List[Cell]:
        mineable = []
        for cell in rt:
            rtype = cell.resource.type
            if rtype == "wood":
                mineable.append(cell)
            elif rtype == "coal" and player.researched_coal():
                mineable.append(cell)
            elif rtype == "uranium" and player.researched_uranium():
                mineable.append(cell)
        return mineable

    def _compute_worker_prior(self, unit: Unit, player: Player, game_state: Game, mineable: List[Cell], is_night: bool, remaining_night: int) -> np.ndarray:
        prior = np.zeros(len(self.w_actions), dtype=np.float32)
        
        # 1. Resource attraction (Max 0.5)
        cargo_space = unit.get_cargo_space_left()
        cargo_pct = (100 - cargo_space) / 100.0
        empty_pct = 1.0 - cargo_pct
        attraction_scale = 0.5 * (empty_pct ** 2)
        
        # 4. NO-OP penalty (Base)
        prior[self.w_noop] -= 0.1

        closest_res_dist = 9999
        best_res_score = -1.0
        closest_res = None
        for cell in mineable:
            d = self._dist(unit.pos, cell.pos)
            nearby_units = 0
            for u in player.units:
                if u.is_worker() and self._dist(u.pos, cell.pos) <= 1:
                    nearby_units += 1
            score = 1.0 / ((d + 1.0) * (1.0 + nearby_units))
            if score > best_res_score:
                best_res_score = score
                closest_res_dist = d
                closest_res = cell
                
        if closest_res is not None and attraction_scale > 0:
            if closest_res_dist == 0:
                prior[self.w_noop] += attraction_scale
            else:
                for d_str, d_idx in self.w_move.items():
                    dx, dy = self.dir_delta[d_str]
                    nx, ny = unit.pos.x + dx, unit.pos.y + dy
                    if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                        ndist = abs(nx - closest_res.pos.x) + abs(ny - closest_res.pos.y)
                        if ndist < closest_res_dist:
                            prior[d_idx] += attraction_scale
                        elif ndist > closest_res_dist:
                            prior[d_idx] -= attraction_scale
        
        # 2. Night survival (Max 0.8)
        worker_fuel = unit.cargo.wood * 1 + unit.cargo.coal * 10 + unit.cargo.uranium * 40
        upkeep = 4
        
        closest_city_dist = 9999
        closest_city_tile = None
        for ct in player.city_tiles:
            d = self._dist(unit.pos, ct.pos)
            if d < closest_city_dist:
                closest_city_dist = d
                closest_city_tile = ct
                
        survival_deficit = max(0, upkeep * remaining_night - worker_fuel)
        if survival_deficit > 0 and closest_city_tile is not None:
            urgency = 0.8
            if closest_city_dist == 0:
                prior[self.w_noop] += urgency
            else:
                for d_str, d_idx in self.w_move.items():
                    dx, dy = self.dir_delta[d_str]
                    nx, ny = unit.pos.x + dx, unit.pos.y + dy
                    if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                        ndist = abs(nx - closest_city_tile.pos.x) + abs(ny - closest_city_tile.pos.y)
                        if ndist < closest_city_dist:
                            prior[d_idx] += urgency
                        elif ndist > closest_city_dist:
                            prior[d_idx] -= urgency
                            
        # 3. Unsafe City rescue (Max 0.6)
        if survival_deficit <= 0 and worker_fuel > 0:
            target_city_tile = None
            max_rescue_score = -1.0
            
            for city in player.cities.values():
                city_upkeep = city.get_light_upkeep()
                city_deficit = max(0, city_upkeep * remaining_night - city.fuel)
                if city_deficit > 0:
                    for ct in city.citytiles:
                        d = self._dist(unit.pos, ct.pos)
                        score = city_deficit / (d + 1.0)
                        if score > max_rescue_score:
                            max_rescue_score = score
                            target_city_tile = ct
                            
            if target_city_tile is not None:
                rescue_bias = 0.6
                if self._dist(unit.pos, target_city_tile.pos) == 0:
                    prior[self.w_noop] += rescue_bias
                else:
                    target_dist = self._dist(unit.pos, target_city_tile.pos)
                    for d_str, d_idx in self.w_move.items():
                        dx, dy = self.dir_delta[d_str]
                        nx, ny = unit.pos.x + dx, unit.pos.y + dy
                        if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                            ndist = abs(nx - target_city_tile.pos.x) + abs(ny - target_city_tile.pos.y)
                            if ndist < target_dist:
                                prior[d_idx] += rescue_bias
                            elif ndist > target_dist:
                                prior[d_idx] -= rescue_bias
        # 5. Build City (Max 0.5)
        if cargo_space == 0:
            cell = game_state.map.get_cell(unit.pos.x, unit.pos.y)
            if not cell.has_resource():
                build_score = 0.0
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = unit.pos.x + dx, unit.pos.y + dy
                        if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                            c = game_state.map.get_cell(nx, ny)
                            if c.has_resource():
                                build_score += 0.1
                prior[self.w_build_city] += min(0.5, build_score)
        
        return prior

    def _compute_city_tile_prior(self, ct: CityTile, player: Player, game_state: Game, is_night: bool, remaining_night: int) -> np.ndarray:
        prior = np.zeros(len(self.ct_actions), dtype=np.float32)
        
        # 1. BUILD_WORKER (Max 0.5)
        budget = max(0, len(player.city_tiles) - len(player.units))
        
        spawn_quality = 0.0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = ct.pos.x + dx, ct.pos.y + dy
                if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                    cell = game_state.map.get_cell(nx, ny)
                    if cell.has_resource():
                        spawn_quality += 0.1
                        
        build_bias = min(0.5, spawn_quality) if budget > 0 else -0.5
        prior[self.ct_build_worker] += build_bias
        
        # 2. RESEARCH (Max 0.5)
        rp = player.research_points
        if rp < 50:
            rp_gap = 50 - rp
            research_urgency = 0.5
        elif rp < 200:
            rp_gap = 200 - rp
            research_urgency = 0.3
        else:
            rp_gap = 0
            research_urgency = 0.0
            
        research_bias = research_urgency * (1.0 - min(1.0, spawn_quality))
        if rp_gap > 0:
            prior[self.ct_research] += research_bias
            
        # 3. City fuel safety (Max 0.2)
        city = player.cities[ct.cityid]
        city_upkeep = city.get_light_upkeep()
        city_deficit = max(0, city_upkeep * remaining_night - city.fuel)
        if city_deficit > 0:
            prior[self.ct_build_worker] -= 0.2
        # 4. BUILD_CART (Max 0.2)
        worker_count = sum(1 for u in player.units if u.is_worker())
        cart_count = sum(1 for u in player.units if u.is_cart())
        if budget > 0 and cart_count < worker_count / 4:
            prior[self.ct_build_cart] += 0.2
            
        return prior

    def compute(
        self,
        game_state: Game,
        board_dims: Tuple[int, int],
        pos_to_unit_dict: Dict[Tuple[int, int], Optional[Unit]],
        pos_to_city_tile_dict: Dict[Tuple[int, int], Optional[CityTile]],
        available_actions_mask: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        priors = {}
        
        is_night, remaining_night = self._get_night_info(game_state)
        rt = self._get_resource_tiles(game_state)
        
        for entity, mask in available_actions_mask.items():
            prior = np.zeros_like(mask, dtype=np.float32)
            
            for p_idx in [0, 1]:
                player = game_state.players[p_idx]
                mineable = self._get_mineable(player, rt)
                
                if entity == "worker":
                    for unit in player.units:
                        if unit.is_worker():
                            x, y = unit.pos.x, unit.pos.y
                            if 0 <= x < prior.shape[2] and 0 <= y < prior.shape[3]:
                                p_bias = self._compute_worker_prior(unit, player, game_state, mineable, is_night, remaining_night)
                                prior[0, p_idx, x, y, :] += p_bias
                            
                elif entity == "cart":
                    for unit in player.units:
                        if unit.is_cart():
                            x, y = unit.pos.x, unit.pos.y
                            if 0 <= x < prior.shape[2] and 0 <= y < prior.shape[3]:
                                prior[0, p_idx, x, y, self.c_noop] -= 0.1

                elif entity == "city_tile":
                    for ct in player.city_tiles:
                        x, y = ct.pos.x, ct.pos.y
                        if 0 <= x < prior.shape[2] and 0 <= y < prior.shape[3]:
                            p_bias = self._compute_city_tile_prior(ct, player, game_state, is_night, remaining_night)
                            prior[0, p_idx, x, y, :] += p_bias

            # Zero-mean and clip
            valid_counts = mask.sum(axis=-1, keepdims=True)
            has_valid = valid_counts > 0
            
            valid_prior_sum = (prior * mask).sum(axis=-1, keepdims=True)
            valid_prior_mean = np.zeros_like(valid_prior_sum)
            np.divide(valid_prior_sum, valid_counts, out=valid_prior_mean, where=has_valid)
            
            prior = prior - valid_prior_mean
            prior = prior * mask
            prior = np.clip(prior, -1.0, 1.0)
            
            priors[entity] = prior

        return priors
