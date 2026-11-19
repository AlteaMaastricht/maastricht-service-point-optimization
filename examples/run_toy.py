"""End-to-end run on the synthetic toy instance (no course data needed).

    python3 examples/run_toy.py --seed 1 --out toy_output

Writes solution.json, solution.png and (with --sensitivity) a sensitivity
report to the output directory. Numbers are for a generated grid, not Maastricht.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from parcelnet.optimizer import ServicePointOptimizer  # noqa: E402
from parcelnet.plotting import plot_solution  # noqa: E402
from parcelnet.sensitivity import SensitivityAnalyzer  # noqa: E402
from parcelnet.toy import make_toy_instance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-apls", action="store_true", help="SP-only model (SA4_00 variant)")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--initial-temp", type=float, default=15.0)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--out", default="toy_output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    inst = make_toy_instance(seed=0)
    opt = ServicePointOptimizer(inst, use_apls=not a.no_apls, seed=a.seed)
    opt.optimize(initial_temp=a.initial_temp, cooling_rate=0.99995,
                 max_iterations=a.iterations, time_limit=600, initial_strategy="existing")
    summary = opt.summarize()
    summary["stats"] = opt.stats
    summary["seed"] = a.seed
    summary["instance"] = inst.meta["source"]
    with open(os.path.join(a.out, "solution.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_solution(opt, os.path.join(a.out, "solution.png"))
    print(json.dumps({k: summary[k] for k in ("n_service_points", "n_apls", "cost", "stats")}, indent=2))

    if a.sensitivity:
        sa = SensitivityAnalyzer(inst, use_apls=not a.no_apls, seed=a.seed)
        sa.run_parameter_sensitivity_analysis(max_iterations=300)
        sa.run_robustness_analysis(max_iterations=300)
        print(sa.generate_sensitivity_report(os.path.join(a.out, "sensitivity_report.txt")))
        sa.plot(os.path.join(a.out, "sensitivity.png"))


if __name__ == "__main__":
    main()
