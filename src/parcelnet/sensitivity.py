"""One-at-a-time parameter sensitivity and robustness runs.

Port of ``legacy/sensitivity_analysis.py`` (SensitivityAnalyzer). Same test
grid, same SA settings per test type and the same text report format.
Changes (docs/PROVENANCE.md):

* every SA run gets a deterministic seed (``seed + run index``), so the whole
  analysis is reproducible; the original SA runs were unseeded, which is why
  its "1.0x demand" and "0% noise" runs differ from its own baseline
  (+3.0% and +6.3% in results/course_instance/sensitivity_report.txt);
* ``avg_distance_change`` excludes the zero diagonal (the original divided by
  zero and stored NaN);
* plotting is reduced to three panels (tornado, demand, distance); the
  original six-panel figure code remains in legacy/.

Still one SA run per scenario, as in the original: differences between
scenarios mix the parameter effect with run-to-run SA noise.
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from .instance import Instance
from .optimizer import ServicePointOptimizer

DEFAULT_VARIATIONS = {  # sensitivity_analysis.py:52-59
    "F_SP": [25000, 37500, 50000, 62500, 75000],
    "C_hd": [0.75, 1.125, 1.50, 1.875, 2.25],
    "BR_sp": [0.01, 0.015, 0.02, 0.025, 0.03],
    "BR_cw": [0.005, 0.0075, 0.01, 0.0125, 0.015],
    "apl_fixed_cost": [250, 375, 500, 625, 750],
    "pickup_ratio": [0.4, 0.485, 0.57, 0.655, 0.74],
}
DEFAULT_DEMAND = [0.7, 0.85, 1.0, 1.15, 1.3]
DEFAULT_NOISE = [0.0, 0.05, 0.1, 0.15, 0.2]
DEFAULT_STRESS = [(1.2, 0.1), (0.8, 0.15), (1.3, 0.2)]


class SensitivityAnalyzer:
    def __init__(self, instance: Instance, use_apls: bool = True, seed: int = 0,
                 optimizer_kwargs: Optional[dict] = None):
        self.inst = instance
        self.use_apls = use_apls
        self.seed = seed
        self.kw = optimizer_kwargs or {}
        self._run = 0
        self.sensitivity_results: dict = {}
        self.robustness_results: dict = {}

    def _next_seed(self) -> int:
        self._run += 1
        return self.seed + self._run

    def _solve(self, inst: Instance, overrides: Dict[str, float], initial_temp: float,
               max_iterations: int, time_limit: float) -> dict:
        kw = dict(self.kw)
        params = dict(kw.pop("parameters", {}) or {})
        for k, v in overrides.items():
            if k in ("apl_fixed_cost", "apl_capacity", "pickup_ratio"):
                kw[k] = v
            else:
                params[k] = v
        opt = ServicePointOptimizer(inst, use_apls=self.use_apls, seed=self._next_seed(),
                                    parameters=params, **kw)
        t0 = time.time()
        opt.optimize(initial_temp=initial_temp, cooling_rate=0.995,
                     max_iterations=max_iterations, time_limit=time_limit,
                     initial_strategy="random")
        s = opt.summarize()
        dem_total = float(inst.demand_total.sum())
        return {"success": True, "cost": opt.best_cost, "seed": opt.seed,
                "n_service_points": s["n_service_points"], "n_apls": s["n_apls"],
                "total_capacity": s["total_capacity_per_day"],
                "optimization_time": time.time() - t0,
                "avg_distance_to_facility": s["avg_distance_km"],
                # same definition as the original: total demand / total capacity
                "capacity_utilization": dem_total / max(s["total_capacity_per_day"], 1),
                "city_bounce_rate": s["city_bounce_rate"],
                "penalty": s["cost"]["penalty"], "parameters_used": overrides}

    # ------------------------------------------------------------ analyses
    def run_parameter_sensitivity_analysis(self, parameter_variations=None,
                                           max_iterations: int = 5000, time_limit: float = 600) -> dict:
        variations = parameter_variations or DEFAULT_VARIATIONS
        baseline = self._solve(self.inst, {}, 1000, max_iterations, time_limit)
        tests = {}
        for name, values in variations.items():
            tests[name] = {v: self._solve(self.inst, {name: v}, 1000, max_iterations, time_limit)
                           for v in values}
        self.sensitivity_results = {"baseline": baseline, "parameter_tests": tests}
        return self.sensitivity_results

    @staticmethod
    def _perturb(dist: np.ndarray, noise_level: float):
        rng = np.random.default_rng(42)  # the original also fixed this seed (np.random.seed(42))
        noise = rng.normal(1.0, noise_level, dist.shape)
        out = dist.copy()
        finite = np.isfinite(dist)
        out[finite] *= noise[finite]
        out = np.maximum(out, 0)
        np.fill_diagonal(out, 0.0)
        off = finite & (dist > 0)
        change = float(np.mean(np.abs(out[off] - dist[off]) / dist[off])) if off.any() else 0.0
        return out, change

    def run_robustness_analysis(self, demand_scenarios=None, distance_perturbations=None,
                                stress_scenarios=None, max_iterations: int = 2000,
                                time_limit: float = 600) -> dict:
        res = {"demand_scenarios": {}, "distance_perturbations": {}, "combined_stress_test": {}}
        for m in demand_scenarios or DEFAULT_DEMAND:
            inst = replace(self.inst, demand_total=self.inst.demand_total * m)
            res["demand_scenarios"][m] = self._solve(inst, {}, 5000, max_iterations, time_limit)
        for nl in distance_perturbations if distance_perturbations is not None else DEFAULT_NOISE:
            d, ch = self._perturb(self.inst.dist_km, nl)
            r = self._solve(replace(self.inst, dist_km=d), {}, 5000, max_iterations, time_limit)
            r["avg_distance_change"] = ch
            res["distance_perturbations"][nl] = r
        for m, nl in stress_scenarios or DEFAULT_STRESS:
            d, _ = self._perturb(self.inst.dist_km, nl)
            inst = replace(self.inst, dist_km=d, demand_total=self.inst.demand_total * m)
            res["combined_stress_test"][f"demand_{m:.1f}_noise_{nl:.1%}"] = \
                self._solve(inst, {}, 5000, max_iterations, time_limit)
        self.robustness_results = res
        return res

    # -------------------------------------------------------------- report
    def generate_sensitivity_report(self, save_path: Optional[str] = None) -> str:
        """Same layout as the original report (sensitivity_analysis.py:622-714)."""
        if not self.sensitivity_results:
            return "No sensitivity analysis results available."
        b = self.sensitivity_results["baseline"]
        L = ["=" * 80, "SENSITIVITY ANALYSIS REPORT", "=" * 80,
             f"Instance: {self.inst.meta.get('source', 'unknown')}  base seed: {self.seed}",
             "\nBASELINE SOLUTION:",
             f"  Total Cost: EUR {b['cost']:,.0f}",
             f"  Service Points: {b['n_service_points']}",
             f"  APLs: {b['n_apls']}",
             f"  Average Distance: {b['avg_distance_to_facility']:.2f} km",
             f"  City Bounce Rate: {b['city_bounce_rate']:.1%}",
             f"  Penalty included in cost: EUR {b['penalty']:,.0f}",
             "\nPARAMETER SENSITIVITY RESULTS:", "-" * 50]
        for name, results in self.sensitivity_results["parameter_tests"].items():
            costs = {v: r["cost"] for v, r in results.items() if r["success"]}
            L.append(f"\nParameter: {name}")
            if not costs:
                L.append("  No successful optimizations for this parameter")
                continue
            lo, hi = min(costs.values()), max(costs.values())
            L.append(f"  Cost Range: EUR {lo:,.0f} - EUR {hi:,.0f} ({(hi - lo) / b['cost'] * 100:.1f}% variation)")
            L.append(f"  Maximum Impact: {max(abs(c - b['cost']) / b['cost'] * 100 for c in costs.values()):.1f}%")
            L.append(f"  Best Value: {min(costs, key=costs.get)} (EUR {lo:,.0f})")
            L.append(f"  Worst Value: {max(costs, key=costs.get)} (EUR {hi:,.0f})")
        if self.robustness_results:
            L += ["\nROBUSTNESS ANALYSIS RESULTS:", "-" * 50, "\nDemand Scenario Robustness:"]
            dc = []
            for m, r in self.robustness_results["demand_scenarios"].items():
                L.append(f"  {m:.2f}x demand: EUR {r['cost']:,.0f} ({(r['cost'] - b['cost']) / b['cost'] * 100:+.1f}%)")
                dc.append(r["cost"])
            if dc:
                L.append(f"  Demand Volatility: {(max(dc) - min(dc)) / b['cost'] * 100:.1f}%")
            L.append("\nDistance Perturbation Robustness:")
            for nl, r in self.robustness_results["distance_perturbations"].items():
                L.append(f"  {nl:.1%} noise: EUR {r['cost']:,.0f} ({(r['cost'] - b['cost']) / b['cost'] * 100:+.1f}%)"
                         f"  penalty EUR {r['penalty']:,.0f}")
            L.append("\nCombined Stress Tests:")
            for k, r in self.robustness_results["combined_stress_test"].items():
                L.append(f"  {k}: EUR {r['cost']:,.0f}")
        text = "\n".join(L)
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        return text

    def plot(self, save_path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        b = self.sensitivity_results["baseline"]["cost"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        names, lows, highs = [], [], []
        for name, res in self.sensitivity_results["parameter_tests"].items():
            vals = sorted(res)
            names.append(name)
            lows.append((res[vals[0]]["cost"] - b) / b * 100)
            highs.append((res[vals[-1]]["cost"] - b) / b * 100)
        axes[0].barh(names, lows, color="tab:red", label="lowest value")
        axes[0].barh(names, highs, color="tab:green", alpha=0.6, label="highest value")
        axes[0].set_xlabel("cost change vs baseline (%)")
        axes[0].set_title("One-at-a-time sensitivity (single run each)")
        axes[0].legend()
        if self.robustness_results:
            ds = self.robustness_results["demand_scenarios"]
            axes[1].plot(list(ds), [r["cost"] for r in ds.values()], "o-")
            axes[1].set_xlabel("demand multiplier"); axes[1].set_ylabel("cost (EUR)")
            axes[1].set_title("Demand scenarios")
            dp = self.robustness_results["distance_perturbations"]
            axes[2].plot([k * 100 for k in dp], [r["cost"] for r in dp.values()], "o-")
            axes[2].set_xlabel("distance noise (%)"); axes[2].set_title("Distance noise")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
