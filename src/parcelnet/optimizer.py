"""Simulated-annealing heuristic for service-point (SP) and parcel-locker (APL) location.

Consolidated port of ``legacy/SA4_00.py`` (SP-only) and ``legacy/SA5.py``
(SP + APL). The model, move set, move probabilities, capacity rule, penalty
scheme and acceptance rule are those of the originals. Changes (all listed in
docs/PROVENANCE.md):

* **Seeded RNG.** Every random draw goes through ``random.Random(seed)``; the
  originals used the unseeded module-level ``random``.
* **APL positions are solution state.** In ``SA5._relocate_apl`` the move
  wrote the new position into ``self.apl_locations`` (optimizer state), so a
  *rejected* relocation still moved the locker. Positions now live in
  ``solution["apl_nodes"]`` and are copied with the rest of the solution.
* **No VRP in the objective.** The OR-Tools VRP that ``SA5.py`` called inside
  ``evaluate_solution`` (added 2025-06-17, after all saved outputs) had index,
  unit and cost-period bugs and overwrote the delivery cost; it is moved to
  ``experimental/`` and not used. Delivery cost is the distance-based cost
  used by both SA4_00 and SA5 before that edit.
* **Missing greedy start.** ``SA5._create_existing_solution`` fell back to an
  undefined ``_create_greedy_solution``; the SA4_00 greedy start is ported.
* **Vectorised evaluation.** Same formulas as the original loops, computed
  with numpy (checked against a hand-computed case in tests/test_cost.py).
* **Cost breakdown = objective.** ``_validate_solution`` in SA5 charged APL
  customers only for their delivery parcels while ``evaluate_solution``
  charged them for all parcels; :meth:`cost_breakdown` uses the objective's
  formulas so the reported numbers add up to the optimised cost.

Model, as implemented (and its limitations, see README):
  cost = F_SP * #open SPs + apl_fixed_cost * #open APLs
       + C_cap * sum(capacity)                    (capacity in parcels/day, C_cap per year)
       + 365 * C_hd * sum_c d(c, f(c)) * q(c)     q = delivery parcels for SP customers,
                                                    all parcels for APL customers
       + 1e6 * constraint violations (assignment, closed facility, SP/city bounce, APL capacity)
  with a *fixed* pickup share (``pickup_ratio``) for every customer, independent of distance.
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional

import numpy as np

from .instance import Instance

Solution = Dict[str, np.ndarray]

DEFAULT_PARAMETERS = {
    "F_SP": 50000,   # fixed cost per service point per year (EUR)
    "C_cap": 36.50,  # yearly cost per unit of daily capacity (EUR 0.10/day * 365)
    "C_hd": 1.50,    # home-delivery cost per km per parcel (EUR)
    "BR_sp": 0.02,   # max bounce rate per service point
    "BR_cw": 0.01,   # max city-wide bounce rate
    "M": 200000,     # max capacity per service point
}

# Move tables of the originals: (name, probability)
MOVES_SP_ONLY = [("toggle_service_point", 0.3), ("reassign_customer", 0.4),
                 ("adjust_capacity", 0.2), ("relocate_service_point", 0.1)]          # SA4_00.py:865-870
MOVES_SP_APL = [("toggle_service_point", 0.3), ("reassign_customer", 0.2),
                ("adjust_capacity", 0.15), ("relocate_service_point", 0.2),
                ("toggle_apl", 0.05), ("relocate_apl", 0.1)]                        # SA5.py:935-942

PENALTY_WEIGHT = 1e6


class ServicePointOptimizer:
    def __init__(self, instance: Instance, use_apls: bool = True, seed: Optional[int] = None,
                 parameters: Optional[dict] = None, pickup_ratio: float = 0.57,
                 apl_fixed_cost: float = 500, apl_capacity: float = 60, verbose: bool = False):
        self.inst = instance
        self.use_apls = use_apls
        self.seed = seed
        self.rng = random.Random(seed)
        self.parameters = dict(DEFAULT_PARAMETERS)
        if parameters:
            self.parameters.update(parameters)
        # Hard-coded 0.57/0.43 split in both originals (SA4_00.py:25, SA5.py:29).
        self.pickup_ratio = pickup_ratio
        self.delivery_ratio = 1.0 - pickup_ratio
        self.apl_fixed_cost = apl_fixed_cost   # SA5.py:25
        self.apl_capacity = apl_capacity       # SA5.py:26
        self.verbose = verbose

        self.sp_nodes = np.array([instance.node_to_idx[int(s)] for s in instance.candidate_sp], dtype=int)
        self.apl_slots_initial = (np.array([instance.node_to_idx[int(a)] for a in instance.apl_candidates], dtype=int)
                                  if use_apls else np.zeros(0, dtype=int))
        self.n_sp = len(self.sp_nodes)
        self.n_apl = len(self.apl_slots_initial)
        self.D = instance.dist_km
        self.moves = MOVES_SP_APL if use_apls else MOVES_SP_ONLY

        self.current_solution: Optional[Solution] = None
        self.best_solution: Optional[Solution] = None
        self.best_cost = float("inf")
        self.cost_history: List[float] = []
        self.stats: dict = {}

    # ------------------------------------------------------------------ demand
    @property
    def pickup(self) -> np.ndarray:
        return self.inst.demand_total * self.pickup_ratio

    @property
    def delivery(self) -> np.ndarray:
        return self.inst.demand_total * self.delivery_ratio

    def _log(self, *a):
        if self.verbose:
            print(*a)

    # ------------------------------------------------------------ construction
    def _empty(self) -> Solution:
        n_c = self.inst.n_customers
        return {"y": np.zeros(self.n_sp, dtype=bool),
                "apl_open": np.zeros(self.n_apl, dtype=bool),
                "apl_nodes": self.apl_slots_initial.copy(),
                "x_assigned": np.zeros((n_c, self.n_sp + self.n_apl), dtype=bool),
                "capacity": np.zeros(self.n_sp + self.n_apl, dtype=np.float64)}

    def create_initial_solution(self, strategy: str = "existing") -> Solution:
        sol = self._empty()
        if strategy == "existing" and self.inst.existing_assignment:
            return self._create_existing_solution(sol)
        if strategy in ("existing", "greedy"):
            return self._create_greedy_solution(sol)
        if strategy == "random":
            return self._create_random_solution(sol)
        raise ValueError(f"unknown strategy {strategy!r}")

    def _create_existing_solution(self, sol: Solution) -> Solution:
        """SA5.py:598-672 / SA4_00.py:624-668."""
        cand_pos = {int(s): k for k, s in enumerate(self.inst.candidate_sp)}
        existing = self.inst.existing_assignment
        for sp_id in set(existing.values()):
            if sp_id in cand_pos:
                sol["y"][cand_pos[sp_id]] = True
        if self.use_apls:
            n_random_apls = 10  # SA5.py:619
            closed = [i for i in range(self.n_apl) if not sol["apl_open"][i]]
            k = min(n_random_apls - int(sol["apl_open"].sum()), len(closed))
            for a in self.rng.sample(closed, max(k, 0)):
                sol["apl_open"][a] = True
        pickup_by_sp: Dict[int, float] = {}
        for c, nid in enumerate(self.inst.node_ids):
            sp_id = existing.get(int(nid))
            if sp_id in cand_pos and sol["y"][cand_pos[sp_id]]:
                j = cand_pos[sp_id]
                sol["x_assigned"][c, j] = True
                pickup_by_sp[j] = pickup_by_sp.get(j, 0.0) + self.pickup[c]
            else:
                # SA5.py:652 passes a dict without 'apl_open', so only SPs are
                # considered here; reproduced with include_apls=False.
                self._assign_to_nearest(sol, c, fallback=True, include_apls=False)
        for j, p in pickup_by_sp.items():
            sol["capacity"][j] = min(p / (1 - self.parameters["BR_sp"] * 0.5), self.parameters["M"])
        for a in range(self.n_apl):
            if sol["apl_open"][a]:
                sol["capacity"][self.n_sp + a] = self.apl_capacity
        return sol

    def _create_greedy_solution(self, sol: Solution) -> Solution:
        """SA4_00.py:556-622 (SP-only greedy within 5 km)."""
        unassigned = set(range(self.inst.n_customers))
        total = self.inst.demand_total
        while unassigned:
            best_loc, best_score = None, -1.0
            ua = np.fromiter(unassigned, dtype=int)
            for j in range(self.n_sp):
                if sol["y"][j]:
                    continue
                d = self.D[ua, self.sp_nodes[j]]
                m = d <= 5.0
                score = float(np.sum(total[ua][m] / (1 + d[m])))
                if score > best_score:
                    best_score, best_loc = score, j
            if best_loc is None:
                break
            sol["y"][best_loc] = True
            d = self.D[ua, self.sp_nodes[best_loc]]
            served = ua[d <= 5.0]
            sol["x_assigned"][served, best_loc] = True
            unassigned.difference_update(served.tolist())
            p = self.pickup[served].sum()
            if p > 0:
                sol["capacity"][best_loc] = min(p / (1 - self.parameters["BR_sp"]) * 1.1, self.parameters["M"])
            if len(served) == 0 and sol["y"].all():
                break
        return sol

    def _create_random_solution(self, sol: Solution) -> Solution:
        """SA5.py:674-770 (30 SPs + 20 APLs) or SA4_00.py:737-785 (20-50 SPs).

        The originals assumed at least 30 (SA5) / 20 (SA4_00) candidates; here
        the counts are capped by the number of candidates.
        """
        if self.use_apls:
            n_open = min(30, self.n_sp)
        else:
            lo = min(20, self.n_sp)
            n_open = self.rng.randint(lo, min(50, self.n_sp))
        open_sp = self.rng.sample(range(self.n_sp), n_open)
        sol["y"][open_sp] = True
        if self.use_apls and self.n_apl:
            for a in self.rng.sample(range(self.n_apl), min(20, self.n_apl)):
                sol["apl_open"][a] = True
        for c in range(self.inst.n_customers):
            self._assign_to_nearest(sol, c, fallback=False)
        p = self._pickup_per_sp(sol)
        for j in open_sp:
            if p[j] > 0:
                sol["capacity"][j] = min(p[j] / (1 - self.parameters["BR_sp"]) * 1.1, self.parameters["M"])
        for a in range(self.n_apl):
            if sol["apl_open"][a]:
                sol["capacity"][self.n_sp + a] = self.apl_capacity
        return sol

    # ------------------------------------------------------------- evaluation
    def _facility_nodes(self, sol: Solution) -> np.ndarray:
        return np.concatenate([self.sp_nodes, sol["apl_nodes"]]).astype(int)

    def _pickup_per_sp(self, sol: Solution) -> np.ndarray:
        return self.pickup @ sol["x_assigned"][:, :self.n_sp]

    def cost_breakdown(self, sol: Solution) -> Dict[str, float]:
        """Objective components (SA5.evaluate_solution without the VRP block)."""
        P = self.parameters
        y, x, cap = sol["y"], sol["x_assigned"], sol["capacity"]
        apl_open = sol["apl_open"]
        fixed = float(y.sum()) * P["F_SP"]
        apl_fixed = float(apl_open.sum()) * self.apl_fixed_cost
        capacity_cost = float(cap.sum()) * P["C_cap"]

        cust, col = np.nonzero(x)
        fac = self._facility_nodes(sol)
        dist = self.D[cust, fac[col]]
        qty = np.where(col < self.n_sp, self.delivery[cust], self.inst.demand_total[cust])
        delivery = float(np.sum(qty * dist)) * P["C_hd"] * 365

        w = PENALTY_WEIGHT
        penalty = float(np.abs(x.sum(axis=1) - 1).sum()) * w
        open_mask = np.concatenate([y, apl_open])
        penalty += float(x[:, ~open_mask].sum()) * w
        pick = self._pickup_per_sp(sol)
        if np.any(pick > 0):
            bounces = np.maximum(0, pick - cap[:self.n_sp])
            rates = bounces / np.maximum(pick, 1e-8)
            exc = rates > P["BR_sp"]
            penalty += float(np.sum((rates[exc] - P["BR_sp"]) * pick[exc] * w))
            tot = pick.sum()
            if tot > 0 and bounces.sum() / tot > P["BR_cw"]:
                penalty += (bounces.sum() / tot - P["BR_cw"]) * tot * w
        if self.n_apl:
            apl_dem = self.inst.demand_total @ x[:, self.n_sp:]
            over = np.maximum(0, apl_dem - cap[self.n_sp:]) * apl_open
            penalty += float(over.sum()) * w
        total = fixed + apl_fixed + capacity_cost + delivery + penalty
        return {"fixed_sp": fixed, "fixed_apl": apl_fixed, "capacity": capacity_cost,
                "delivery": delivery, "penalty": penalty, "total": total}

    def evaluate_solution(self, sol: Solution) -> float:
        return self.cost_breakdown(sol)["total"]

    # ------------------------------------------------------------------ moves
    def generate_neighbor(self, sol: Solution) -> Solution:
        nb = {k: v.copy() for k, v in sol.items()}
        r = self.rng.random()
        cum = 0.0
        for move, wgt in self.moves:
            cum += wgt
            if r <= cum:
                break
        getattr(self, "_" + move)(nb)
        self._adjust_capacity_after_assignment(nb)
        return nb

    def _toggle_service_point(self, sol: Solution):
        y = sol["y"]
        open_sp, closed_sp = np.where(y)[0], np.where(~y)[0]
        if self.rng.random() < 0.5 and len(open_sp) > 1:
            j = int(self.rng.choice(list(open_sp)))
            y[j] = False
            sol["capacity"][j] = 0
            for c in np.where(sol["x_assigned"][:, j])[0]:
                sol["x_assigned"][c, j] = False
                self._assign_to_nearest(sol, c, fallback=True)
        elif len(closed_sp) > 0:
            j = int(self.rng.choice(list(closed_sp)))
            y[j] = True
            sol["capacity"][j] = 100  # adjusted afterwards (SA5.py:1118)

    def _reassign_customer(self, sol: Solution):
        c = self.rng.randint(0, self.inst.n_customers - 1)
        sol["x_assigned"][c, :] = False
        self._assign_to_nearest(sol, c, fallback=True)

    def _adjust_capacity(self, sol: Solution):
        # Kept for fidelity. Note: generate_neighbor always calls
        # _adjust_capacity_after_assignment afterwards, which overwrites every
        # open SP's capacity, so this move has no effect on the evaluated
        # neighbour (same behaviour as the originals).
        open_sp = np.where(sol["y"])[0]
        if len(open_sp):
            j = int(self.rng.choice(list(open_sp)))
            new = min(sol["capacity"][j] * self.rng.uniform(0.8, 1.2), self.parameters["M"])
            sol["capacity"][j] = max(new, 1.0)

    def _relocate_service_point(self, sol: Solution):
        y = sol["y"]
        open_sp, closed_sp = np.where(y)[0], np.where(~y)[0]
        if len(open_sp) > 1 and len(closed_sp) > 0:
            j = int(self.rng.choice(list(open_sp)))
            y[j] = False
            sol["capacity"][j] = 0
            near = [k for k in closed_sp if self.D[self.sp_nodes[j], self.sp_nodes[k]] <= 2.0]
            if near:
                k = int(self.rng.choice(near))
                y[k] = True
                sol["capacity"][k] = 100
            for c in np.where(sol["x_assigned"][:, j])[0]:
                sol["x_assigned"][c, j] = False
                self._assign_to_nearest(sol, c, fallback=True)

    def _toggle_apl(self, sol: Solution):
        if self.n_apl == 0:
            return
        a_open = sol["apl_open"]
        open_a, closed_a = np.where(a_open)[0], np.where(~a_open)[0]
        if self.rng.random() < 0.5 and len(open_a) > 0:
            a = int(self.rng.choice(list(open_a)))
            a_open[a] = False
            col = self.n_sp + a
            sol["capacity"][col] = 0
            for c in np.where(sol["x_assigned"][:, col])[0]:
                sol["x_assigned"][c, col] = False
                self._assign_to_nearest(sol, c, fallback=False)
        elif len(closed_a) > 0:
            a = int(self.rng.choice(list(closed_a)))
            a_open[a] = True
            sol["capacity"][self.n_sp + a] = self.apl_capacity

    def _relocate_apl(self, sol: Solution):
        """SA5.py:1004-1047 with the position stored in the solution (bug fix)."""
        if self.n_apl == 0:
            return
        open_a = np.where(sol["apl_open"])[0]
        if len(open_a) == 0:
            return
        a = int(self.rng.choice(list(open_a)))
        used = set(int(n) for n in sol["apl_nodes"])
        options = [int(n) for n in self.sp_nodes if int(n) not in used]
        if not options:
            return
        new_node = self.rng.choice(options)
        col = self.n_sp + a
        affected = np.where(sol["x_assigned"][:, col])[0]
        sol["x_assigned"][:, col] = False
        sol["apl_nodes"][a] = new_node          # was: self.apl_locations[a] = new_id
        sol["capacity"][col] = self.apl_capacity
        for c in affected:
            self._assign_to_nearest(sol, c, fallback=False)

    def _assign_to_nearest(self, sol: Solution, c: int, fallback: bool, include_apls: bool = True):
        """Nearest open SP or APL (SPs win ties). SA5.py:1049-1093 / 1178-1227."""
        best_d, best_col = np.inf, None
        for j in np.where(sol["y"])[0]:
            d = self.D[c, self.sp_nodes[j]]
            if d < best_d and np.isfinite(d):
                best_d, best_col = d, j
        if self.n_apl and include_apls:
            for a in np.where(sol["apl_open"])[0]:
                d = self.D[c, sol["apl_nodes"][a]]
                if d < best_d and np.isfinite(d):
                    best_d, best_col = d, self.n_sp + a
        if best_col is None and fallback:
            open_sp = np.where(sol["y"])[0]
            if len(open_sp):
                best_col = open_sp[0]
        if best_col is not None:
            sol["x_assigned"][c, best_col] = True

    def _adjust_capacity_after_assignment(self, sol: Solution):
        """SA5.py:1229-1250: capacity = pickup / (1 - 0.75*BR_sp) for open SPs."""
        p = self._pickup_per_sp(sol)
        target = self.parameters["BR_sp"] * 0.75
        cap = np.minimum(p / (1 - target), self.parameters["M"])
        cap = np.where(sol["y"] & (p > 0), cap, 0.0)
        sol["capacity"][:self.n_sp] = cap

    # --------------------------------------------------------------- SA loop
    def optimize(self, initial_temp: float = 10000, cooling_rate: float = 0.995,
                 max_iterations: int = 100000, time_limit: float = 3600,
                 initial_strategy: str = "existing",
                 stagnation_limit: Optional[int] = None) -> bool:
        """Geometric-cooling SA (SA5.py:1252-1325).

        ``stagnation_limit`` defaults to SA4_00's ``max_iterations // 10`` in
        SP-only mode and to SA5's ``max_iterations`` in SP+APL mode.
        """
        if stagnation_limit is None:
            stagnation_limit = max_iterations if self.use_apls else max(1, max_iterations // 10)
        start = time.time()
        self.current_solution = self.create_initial_solution(initial_strategy)
        cur = self.evaluate_solution(self.current_solution)
        self.best_solution = {k: v.copy() for k, v in self.current_solution.items()}
        self.best_cost = cur
        self.cost_history = [cur]
        T, it, accepted, last_imp = initial_temp, 0, 0, 0
        while (time.time() - start < time_limit and it < max_iterations
               and it - last_imp < stagnation_limit):
            nb = self.generate_neighbor(self.current_solution)
            nc = self.evaluate_solution(nb)
            diff = nc - cur
            accept = diff <= 0
            if not accept and math.isfinite(diff) and T > 0:
                accept = self.rng.random() < math.exp(-diff / T)
            if accept:
                self.current_solution, cur = nb, nc
                accepted += 1
                if nc < self.best_cost:
                    self.best_solution = {k: v.copy() for k, v in nb.items()}
                    self.best_cost = nc
                    last_imp = it
            T *= cooling_rate
            it += 1
            if it % 1000 == 0:
                self.cost_history.append(cur)
                self._log(f"it {it}: cost={cur:,.0f} best={self.best_cost:,.0f} T={T:.3g}")
        self.stats = {"iterations": it, "accepted": accepted,
                      "acceptance_rate": accepted / it if it else 0.0,
                      "seconds": time.time() - start, "final_temperature": T}
        return True

    # ------------------------------------------------------------- reporting
    def summarize(self, sol: Optional[Solution] = None) -> dict:
        sol = sol if sol is not None else self.best_solution
        x = sol["x_assigned"]
        counts = x.sum(axis=1)
        pick = self._pickup_per_sp(sol)
        cap = sol["capacity"]
        bounces = np.maximum(0, pick - cap[:self.n_sp])
        cust, col = np.nonzero(x)
        d = self.D[cust, self._facility_nodes(sol)[col]]
        return {
            "n_service_points": int(sol["y"].sum()),
            "n_apls": int(sol["apl_open"].sum()),
            "sp_capacity_per_day": float(cap[:self.n_sp].sum()),
            "total_capacity_per_day": float(cap.sum()),
            "unassigned_customers": int((counts == 0).sum()),
            "multi_assigned_customers": int((counts > 1).sum()),
            "customers_at_apls": int(x[:, self.n_sp:].any(axis=1).sum()) if self.n_apl else 0,
            "city_bounce_rate": float(bounces.sum() / pick.sum()) if pick.sum() > 0 else 0.0,
            "max_sp_bounce_rate": float(np.max(bounces / np.maximum(pick, 1e-8))) if self.n_sp else 0.0,
            "avg_distance_km": float(d.mean()) if len(d) else float("nan"),
            "open_sp_nodes": [int(self.inst.node_ids[n]) for n in self.sp_nodes[sol["y"]]],
            "open_apl_nodes": [int(self.inst.node_ids[n]) for n in sol["apl_nodes"][sol["apl_open"]]],
            "cost": self.cost_breakdown(sol),
        }

    def is_feasible(self, sol: Optional[Solution] = None) -> bool:
        """True if the solution carries no penalty (all constraints satisfied)."""
        sol = sol if sol is not None else self.best_solution
        return self.cost_breakdown(sol)["penalty"] == 0.0
