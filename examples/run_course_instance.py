"""Run the optimiser on a workbook in the course schema (NOT distributed).

    python3 examples/run_course_instance.py path/to/workbook.xlsx --seed 1 --out course_output

Only usable if you have legitimate access to the original case workbook (or
build one from open data, see docs/DATA.md). The all-pairs distance matrix for
~8,000 nodes is computed in memory (about 0.5 GB). With the current pure-
Python moves, each SA iteration on the full instance takes noticeably longer
than on the toy instance; runtime on the full instance was not measured for
this version.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from parcelnet.instance import load_course_workbook  # noqa: E402
from parcelnet.optimizer import ServicePointOptimizer  # noqa: E402
from parcelnet.plotting import plot_solution  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-apls", action="store_true")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--initial-temp", type=float, default=100.0)
    ap.add_argument("--out", default="course_output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = load_course_workbook(a.workbook, seed=a.seed)
    opt = ServicePointOptimizer(inst, use_apls=not a.no_apls, seed=a.seed, verbose=True)
    # SA5.py:1716-1722 settings by default (T0=100, cooling 0.995, 1000 iterations)
    opt.optimize(initial_temp=a.initial_temp, cooling_rate=0.995, max_iterations=a.iterations,
                 time_limit=3600, initial_strategy="existing")
    s = opt.summarize()
    s["stats"] = opt.stats
    with open(os.path.join(a.out, "solution.json"), "w") as f:
        json.dump(s, f, indent=2)
    plot_solution(opt, os.path.join(a.out, "solution.png"))


if __name__ == "__main__":
    main()
