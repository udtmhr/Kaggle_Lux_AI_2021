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

    def _get_night_info(self, game_state: Game) -> Tuple[bool, int, int]:
        cycle_turn = game_state.turn % 40
        if cycle_turn < 30:
            is_night = False
            turns_until_night = 30 - cycle_turn
            remaining_night = 10
        else:
            is_night = True
            turns_until_night = 0
            remaining_night = 40 - cycle_turn
        return is_night, remaining_night, turns_until_night

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

    def _estimate_city_upkeep_after_addition(self, player: Player, city, new_pos: Position) -> int:
        tiles = list(city.citytiles)
        tiles.append(CityTile(player.team, city.cityid, new_pos.x, new_pos.y, 0))
        edges = 0
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if abs(tiles[i].pos.x - tiles[j].pos.x) + abs(tiles[i].pos.y - tiles[j].pos.y) == 1:
                    edges += 1
        return len(tiles) * 23 - edges * 10

    def _compute_clusters(self, game_state: Game, mineable: List[Cell], player: Player) -> List[Dict]:
        visited = set()
        mineable_pos = {(c.pos.x, c.pos.y) for c in mineable}
        clusters = []
        for cell in mineable:
            pos_tuple = (cell.pos.x, cell.pos.y)
            if pos_tuple in visited: continue
            
            cluster_cells = []
            queue = [cell]
            visited.add(pos_tuple)
            
            while queue:
                curr = queue.pop(0)
                cluster_cells.append(curr)
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nx, ny = curr.pos.x + dx, curr.pos.y + dy
                    if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                        n_tuple = (nx, ny)
                        if n_tuple in mineable_pos and n_tuple not in visited:
                            visited.add(n_tuple)
                            queue.append(game_state.map.get_cell(nx, ny))
            
            total_amount = sum(c.resource.amount for c in cluster_cells)
            clusters.append({
                "cells": cluster_cells,
                "amount": total_amount,
                "workers_assigned": 0,
                "incoming_workers": 0
            })
            
        # We no longer pre-assign all workers here to avoid double-counting.
        # Fixed workers will be assigned in _global_worker_scheduler.
                    
        return clusters

    def _global_city_scheduler(self, game_state: Game, player: Player, clusters: List[Dict], city_action_mask: np.ndarray) -> Dict[str, str]:
        unit_budget = max(0, len(player.city_tiles) - len(player.units))
        worker_count = sum(1 for u in player.units if u.is_worker())
        
        rp = player.research_points
        if rp < 50:
            research_needed = 50 - rp
        elif rp < 200:
            research_needed = 200 - rp
        else:
            research_needed = 0
            
        proposals = []
        
        for ct in player.city_tiles:
            if not ct.can_act():
                continue
            
            x, y = ct.pos.x, ct.pos.y
            if not (0 <= x < city_action_mask.shape[0] and 0 <= y < city_action_mask.shape[1]):
                continue
                
            can_build_worker = city_action_mask[x, y, self.ct_build_worker]
            can_research = city_action_mask[x, y, self.ct_research]
            can_build_cart = city_action_mask[x, y, self.ct_build_cart]
                
            # Worker Score
            if can_build_worker:
                spawn_quality = 0.0
                for cluster in clusters:
                    dist = min(self._dist(ct.pos, c.pos) for c in cluster["cells"])
                    if dist <= 3:
                        desired_workers = max(1, math.ceil(len(cluster["cells"]) * 0.5))
                        shortage = (desired_workers - cluster["workers_assigned"]) / desired_workers
                        shortage = np.clip(shortage, -1.0, 1.0)
                        if shortage > 0:
                            spawn_quality += min(0.3, shortage * 0.15)
                worker_shortage = max(0.0, 1.0 - worker_count / max(1, len(player.city_tiles)))
                worker_score = 0.25 * worker_shortage + 0.25 * spawn_quality
                proposals.append((f"{ct.pos.x},{ct.pos.y}", "worker", worker_score))
            else:
                spawn_quality = 0.0
            
            # Research Score
            if can_research:
                if rp < 50:
                    proximity = rp / 50.0
                elif rp < 200:
                    proximity = (rp - 50) / 150.0
                else:
                    proximity = 0.0
                research_score = 0.1 + 0.4 * proximity
                research_score = research_score * (1.0 - min(1.0, spawn_quality))
                proposals.append((f"{ct.pos.x},{ct.pos.y}", "research", research_score))
            
            # Cart Score
            if can_build_cart:
                cart_score = 0.2
                proposals.append((f"{ct.pos.x},{ct.pos.y}", "cart", cart_score))
            
        # Tie-break deterministic sort
        proposals.sort(key=lambda x: (x[2], x[0], x[1]), reverse=True)
        
        assigned = {}
        allocated_research = 0
        allocated_workers = 0
        allocated_carts = 0
        existing_cart_count = sum(1 for u in player.units if u.is_cart())
        
        for pos_str, action, score in proposals:
            if pos_str in assigned:
                continue
            if score <= 0:
                continue
                
            if action == "worker":
                if (allocated_workers + allocated_carts) < unit_budget:
                    assigned[pos_str] = action
                    allocated_workers += 1
            elif action == "cart":
                if (allocated_workers + allocated_carts) < unit_budget and (existing_cart_count + allocated_carts) < (worker_count + allocated_workers) / 4:
                    assigned[pos_str] = action
                    allocated_carts += 1
            elif action == "research":
                if allocated_research < research_needed:
                    assigned[pos_str] = action
                    allocated_research += 1
                    
        return assigned

    def _global_worker_scheduler(self, game_state: Game, player: Player, clusters: List[Dict], worker_action_mask: np.ndarray, is_night: bool, remaining_night: int, turns_until_night: int) -> Dict[str, Dict]:
        assigned = {}
        eligible_workers = []
        upkeep = 4
        
        for unit in player.units:
            if not unit.is_worker():
                continue
            
            is_emergency = False
            if not unit.can_act():
                is_emergency = True
            else:
                cargo_space = unit.get_cargo_space_left()
                if cargo_space == 0:
                    is_emergency = True
                else:
                    worker_fuel = unit.cargo.wood * 1 + unit.cargo.coal * 10 + unit.cargo.uranium * 40
                    survival_deficit = max(0, upkeep * remaining_night - worker_fuel)
                    
                    closest_city_dist = 9999
                    for ct in player.city_tiles:
                        d = self._dist(unit.pos, ct.pos)
                        if d < closest_city_dist:
                            closest_city_dist = d
                    
                    if survival_deficit > 0:
                        urgency = np.clip(1.0 - turns_until_night / max(1.0, closest_city_dist + 5.0), 0.0, 1.0)
                        if urgency > 0.3:
                            is_emergency = True
                    
                    if not is_emergency and survival_deficit <= 0 and worker_fuel > 0:
                        for city in player.cities.values():
                            city_upkeep = city.get_light_upkeep()
                            city_deficit = max(0, city_upkeep * remaining_night - city.fuel)
                            if city_deficit > 0:
                                dist_to_danger_city = min(self._dist(unit.pos, ct.pos) for ct in city.citytiles)
                                delivery_urgency = np.clip(1.0 - turns_until_night / max(1.0, dist_to_danger_city + 10.0), 0.0, 1.0)
                                if delivery_urgency > 0.3:
                                    is_emergency = True
                                    break
                                    
            if is_emergency:
                continue
            else:
                eligible_workers.append(unit)
            
        proposals = []
        for unit in eligible_workers:
            x, y = unit.pos.x, unit.pos.y
            if not (0 <= x < worker_action_mask.shape[0] and 0 <= y < worker_action_mask.shape[1]):
                continue
            mask = worker_action_mask[x, y]
            
            for cluster in clusters:
                desired_workers = max(1, math.ceil(len(cluster["cells"]) * 0.5))
                need = max(0.0, desired_workers - cluster["workers_assigned"])
                if need <= 0:
                    continue
                    
                closest_cell = min(cluster["cells"], key=lambda c: self._dist(unit.pos, c.pos))
                current_dist = self._dist(unit.pos, closest_cell.pos)
                
                can_approach = False
                if current_dist == 0:
                    if mask[self.w_noop]:
                        can_approach = True
                else:
                    for d_str, d_idx in self.w_move.items():
                        if not mask[d_idx]:
                            continue
                        dx, dy = self.dir_delta[d_str]
                        nx, ny = unit.pos.x + dx, unit.pos.y + dy
                        if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                            ndist = abs(nx - closest_cell.pos.x) + abs(ny - closest_cell.pos.y)
                            if ndist < current_dist:
                                can_approach = True
                                break
                
                if not can_approach:
                    continue
                    
                score = need / (current_dist + 1.0)
                proposals.append((unit.id, cluster, score, unit))
                
        proposals.sort(key=lambda x: (x[2], x[0]), reverse=True)
        
        for uid, cluster, score, unit in proposals:
            if uid in assigned:
                continue
                
            desired_workers = max(1, math.ceil(len(cluster["cells"]) * 0.5))
            need = max(0.0, desired_workers - cluster["workers_assigned"])
            if need > 0:
                assigned[uid] = cluster
                cluster["workers_assigned"] += 1
                
        return assigned

    def _compute_worker_prior(self, unit: Unit, player: Player, game_state: Game, mineable: List[Cell], target_cluster: Optional[Dict], is_night: bool, remaining_night: int, turns_until_night: int) -> np.ndarray:
        prior = np.zeros(len(self.w_actions), dtype=np.float32)
        
        # 4. NO-OP penalty (Base)
        prior[self.w_noop] -= 0.1
        
        survival_urgency = 0.0
        delivery_urgency = 0.0
        
        closest_city_tile = None
        closest_city_dist = 9999
        for ct in player.city_tiles:
            d = self._dist(unit.pos, ct.pos)
            if d < closest_city_dist:
                closest_city_dist = d
                closest_city_tile = ct
                
        worker_fuel = unit.cargo.wood * 1 + unit.cargo.coal * 10 + unit.cargo.uranium * 40
        upkeep = 4
        
        survival_deficit = max(0, upkeep * remaining_night - worker_fuel)
        if survival_deficit > 0 and closest_city_tile is not None:
            survival_urgency = np.clip(1.0 - turns_until_night / max(1.0, closest_city_dist + 5.0), 0.0, 1.0)
            survival_bias = 0.8 * survival_urgency
            if survival_bias > 0:
                if closest_city_dist == 0:
                    prior[self.w_noop] += survival_bias
                else:
                    for d_str, d_idx in self.w_move.items():
                        dx, dy = self.dir_delta[d_str]
                        nx, ny = unit.pos.x + dx, unit.pos.y + dy
                        if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                            ndist = abs(nx - closest_city_tile.pos.x) + abs(ny - closest_city_tile.pos.y)
                            if ndist < closest_city_dist:
                                prior[d_idx] += survival_bias
                            elif ndist > closest_city_dist:
                                prior[d_idx] -= survival_bias
                                
        elif survival_deficit <= 0 and worker_fuel > 0:
            target_city_tile = None
            best_rescue_score = -1.0
            for city in player.cities.values():
                city_upkeep = city.get_light_upkeep()
                city_deficit = max(0, city_upkeep * remaining_night - city.fuel)
                if city_deficit > 0:
                    for ct in city.citytiles:
                        d = self._dist(unit.pos, ct.pos)
                        score = city_deficit / (d + 1.0)
                        if score > best_rescue_score:
                            best_rescue_score = score
                            target_city_tile = ct
                            
            if target_city_tile is not None:
                delivery_urgency = np.clip(1.0 - turns_until_night / max(1.0, self._dist(unit.pos, target_city_tile.pos) + 10.0), 0.0, 1.0)
                rescue_bias = 0.6 * delivery_urgency
                if self._dist(unit.pos, target_city_tile.pos) == 0:
                    prior[self.w_noop] += rescue_bias
                else:
                    for d_str, d_idx in self.w_move.items():
                        dx, dy = self.dir_delta[d_str]
                        nx, ny = unit.pos.x + dx, unit.pos.y + dy
                        if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                            ndist = abs(nx - target_city_tile.pos.x) + abs(ny - target_city_tile.pos.y)
                            if ndist < self._dist(unit.pos, target_city_tile.pos):
                                prior[d_idx] += rescue_bias
                            elif ndist > self._dist(unit.pos, target_city_tile.pos):
                                prior[d_idx] -= rescue_bias
                                
        cargo_space = unit.get_cargo_space_left()
        
        # Determine effective mineable resources
        if target_cluster is not None:
            resource_candidates = target_cluster["cells"]
        else:
            resource_candidates = mineable

        # 1. Resource attraction (Max 0.5)
        # Scale down if emergency is active
        max_urgency = max(survival_urgency, delivery_urgency)
        attraction_scale = 0.5 * (1.0 - max_urgency)
        if cargo_space > 0 and attraction_scale > 0 and len(resource_candidates) > 0:
            closest_res = min(resource_candidates, key=lambda c: self._dist(unit.pos, c.pos))
            res_dist = self._dist(unit.pos, closest_res.pos)
            if res_dist == 0:
                prior[self.w_noop] += attraction_scale
            else:
                for d_str, d_idx in self.w_move.items():
                    dx, dy = self.dir_delta[d_str]
                    nx, ny = unit.pos.x + dx, unit.pos.y + dy
                    if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                        ndist = abs(nx - closest_res.pos.x) + abs(ny - closest_res.pos.y)
                        if ndist < res_dist:
                            prior[d_idx] += attraction_scale
                        elif ndist > res_dist:
                            prior[d_idx] -= attraction_scale
                            
        # 5. Build City (Max 0.5)
        if cargo_space == 0 and worker_fuel >= 100:
            # ... calculate r_score, c_score, f_score ...
            r_score = 0.0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = unit.pos.x + dx, unit.pos.y + dy
                    if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                        if game_state.map.get_cell(nx, ny).has_resource():
                            r_score += 0.04
                            
            c_score = 0.0
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = unit.pos.x + dx, unit.pos.y + dy
                if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                    adj = game_state.map.get_cell(nx, ny)
                    if adj.citytile is not None and adj.citytile.team == player.team:
                        c_score += 0.25
                        
            f_score = 0.0
            if c_score > 0:
                closest_city = None
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nx, ny = unit.pos.x + dx, unit.pos.y + dy
                    if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                        adj = game_state.map.get_cell(nx, ny)
                        if adj.citytile is not None and adj.citytile.team == player.team:
                            closest_city = player.cities[adj.citytile.cityid]
                            break
                if closest_city is not None:
                    new_upkeep = self._estimate_city_upkeep_after_addition(player, closest_city, unit.pos)
                    deficit = max(0, new_upkeep * remaining_night - closest_city.fuel)
                    if deficit > 0:
                        f_score = 1.0
            else:
                f_score = 0.2
                
            build_score = 0.25 * r_score + 0.10 * c_score - 0.15 * f_score
            prior[self.w_build_city] += max(0.0, min(0.5, build_score))
            
        # 6. Resource cluster dispersion (Max 0.25)
        # Only disperse if no emergency overrides
        if target_cluster is not None and max_urgency < 0.3:
            dispersion_bias = 0.25 * (1.0 - max_urgency)
            
            best_cell = min(target_cluster["cells"], key=lambda c: self._dist(unit.pos, c.pos))
            target_dist = self._dist(unit.pos, best_cell.pos)
            if target_dist == 0:
                prior[self.w_noop] += dispersion_bias
            else:
                for d_str, d_idx in self.w_move.items():
                    dx, dy = self.dir_delta[d_str]
                    nx, ny = unit.pos.x + dx, unit.pos.y + dy
                    if 0 <= nx < game_state.map.width and 0 <= ny < game_state.map.height:
                        ndist = abs(nx - best_cell.pos.x) + abs(ny - best_cell.pos.y)
                        if ndist < target_dist:
                            prior[d_idx] += dispersion_bias
                        elif ndist > target_dist:
                            prior[d_idx] -= dispersion_bias
        
        return prior

    def _compute_city_tile_prior(self, ct: CityTile, player: Player, game_state: Game, assigned_action: Optional[str], is_night: bool, remaining_night: int, turns_until_night: int) -> np.ndarray:
        prior = np.zeros(len(self.ct_actions), dtype=np.float32)
        
        if assigned_action == "worker":
            prior[self.ct_build_worker] += 0.4
        elif assigned_action == "cart":
            prior[self.ct_build_cart] += 0.4
        elif assigned_action == "research":
            prior[self.ct_research] += 0.4
            
        # City Tile NO-OP penalty (Base)
        prior[self.ct_actions.index("NO-OP")] -= 0.1
        
        # City fuel safety (Max 0.2 penalty on worker build)
        fuel_urgency = np.clip(1.0 - turns_until_night / 20.0, 0.0, 1.0)
        city = player.cities[ct.cityid]
        city_upkeep = city.get_light_upkeep()
        city_deficit = max(0, city_upkeep * remaining_night - city.fuel)
        if city_deficit > 0:
            prior[self.ct_build_worker] -= 0.2 * fuel_urgency
            
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
        
        is_night, remaining_night, turns_until_night = self._get_night_info(game_state)
        rt = self._get_resource_tiles(game_state)
        
        player_data = {}
        for p_idx in [0, 1]:
            player = game_state.players[p_idx]
            mineable = self._get_mineable(player, rt)
            clusters = self._compute_clusters(game_state, mineable, player)
            
            worker_action_mask = available_actions_mask.get("worker")
            if worker_action_mask is not None:
                p_worker_mask = worker_action_mask[0, p_idx]
            else:
                p_worker_mask = np.ones((game_state.map.width, game_state.map.height, len(self.w_actions)), dtype=bool)
                
            worker_assignments = self._global_worker_scheduler(game_state, player, clusters, p_worker_mask, is_night, remaining_night, turns_until_night)
            
            city_action_mask = available_actions_mask.get("city_tile")
            if city_action_mask is not None:
                p_city_mask = city_action_mask[0, p_idx]
            else:
                p_city_mask = np.ones((game_state.map.width, game_state.map.height, len(self.ct_actions)), dtype=bool)
                
            city_assignments = self._global_city_scheduler(game_state, player, clusters, p_city_mask)
            
            player_data[p_idx] = {
                "mineable": mineable,
                "city_assignments": city_assignments,
                "worker_assignments": worker_assignments
            }
        
        for entity, mask in available_actions_mask.items():
            prior = np.zeros_like(mask, dtype=np.float32)
            
            for p_idx in [0, 1]:
                player = game_state.players[p_idx]
                pdata = player_data[p_idx]
                mineable = pdata["mineable"]
                
                if entity == "worker":
                    for unit in player.units:
                        if unit.is_worker():
                            x, y = unit.pos.x, unit.pos.y
                            if 0 <= x < prior.shape[2] and 0 <= y < prior.shape[3]:
                                target_cluster = pdata["worker_assignments"].get(unit.id)
                                p_bias = self._compute_worker_prior(unit, player, game_state, mineable, target_cluster, is_night, remaining_night, turns_until_night)
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
                            assigned_action = pdata["city_assignments"].get(f"{x},{y}")
                            p_bias = self._compute_city_tile_prior(ct, player, game_state, assigned_action, is_night, remaining_night, turns_until_night)
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
