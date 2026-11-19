# Service-point and parcel-locker network design with simulated annealing

This repository locates parcel service points (SPs) and parcel lockers (APLs) on a city road graph. It runs
all-pairs Dijkstra and then a simulated-annealing (SA) heuristic, and it adds a one-at-a-time sensitivity
analysis. It contains the author's **individual contributions** to a 7-person university consulting
project (2025), rewritten and tested. Shared work and teammates' work are identified in
[`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) and are not claimed.

## 1. Overview

The project redesigns the 35-service-point parcel network of a fictional carrier ("Post&L") in Maastricht.
It has two variants: SPs only, and SPs plus automated parcel lockers. The repository contains:

- `src/parcelnet/`, a packaged, seeded and tested version of the canonical pipeline:
  graph -> distances -> SA optimiser -> sensitivity analysis;
- a synthetic toy instance, so that everything runs without the course data, which cannot be distributed;
- `legacy/`, the original scripts kept verbatim for comparison;
- `experimental/`, a MILP formulation and a VRP add-on, labelled as not producing any solution;
- `results/synthetic_toy/`, reproducible outputs of the packaged pipeline on the synthetic instance;
- `results/course_instance/`, the aggregate text report saved from the 2025 run on the real (non-distributed)
  instance. Its figures are not distributed, because the solution plot uses course-workbook coordinates.

## 2. Problem

The model is a capacitated facility-location problem. It chooses which candidate SPs to open and how much
daily pickup capacity each gets. Optionally it also opens lockers. Every road-graph node is a customer and is
assigned to one facility. The goal is to minimise annual cost:

- EUR 50,000 fixed cost per SP per year;
- EUR 0.10 per parcel of daily capacity per day (EUR 36.50 per year);
- EUR 1.50 per km of home delivery per parcel;
- EUR 500 per locker per year, with a locker capacity of 60 parcels/day.

The model must meet two bounce-rate limits: at most 2% per SP and at most 1% city-wide. All parameter values
are those hard-coded in the original code.

## 3. Approach

1. **Graph and distances.** The road graph is undirected. Edge lengths are in metres. All-pairs shortest
   paths come from scipy's Dijkstra (`graph.py`).
2. **Demand.** Annual volume is split across nodes using the household count of each node's CBS 500 m square,
   increased for squares with a high share of high-income households (`instance.estimate_demand`). A fixed
   share of each node's parcels is picked up (default 0.57); the rest is delivered.
3. **SA optimiser** (`optimizer.py`):
   - **Moves:** toggle SP, reassign customer, ±20% capacity, relocate SP within 2 km, toggle APL, relocate
     APL. Move probabilities are those of the originals.
   - **Capacities:** after every move, each SP's capacity is reset to pickup/(1 − 0.75·BR_sp).
   - **Constraints:** violations add a penalty of 1e6.
   - **Search:** geometric cooling, Metropolis acceptance, stop on iterations, time or stagnation.
   - **Starts:** "existing" (nearest current SP), "greedy" (5 km coverage) or "random".
4. **Sensitivity** (`sensitivity.py`): one-at-a-time grids for F_SP, C_hd, BR_sp, BR_cw, APL fixed cost and
   pickup share, plus demand multipliers, multiplicative distance noise and three combined stress tests.
   Each scenario gets one seeded SA run.
5. **Data cleaning** (`analysis/cbs_cleaning.py`): CBS squares are cleaned with kNN-median imputation, count
   columns are converted to percentages, and features are screened by VIF.

## 4. Architecture

```
 Nodes/Edges table ──► graph.all_pairs_distances ──► Instance (dist_km, demand, candidates)
 CBS squares ──────► instance.estimate_demand ──┘            │
 (or toy.make_toy_instance — synthetic)                      ▼
                                   optimizer.ServicePointOptimizer (SA, seed)
                                         │ summarize / cost_breakdown
                          ┌──────────────┴──────────────┐
                 plotting.plot_solution        sensitivity.SensitivityAnalyzer
                                                 (report .txt + figure)
```

| Path | Content |
|---|---|
| `src/parcelnet/` | package (graph, instance, toy, optimizer, sensitivity, plotting) |
| `analysis/cbs_cleaning.py` | CBS cleaning / imputation / VIF |
| `examples/` | `run_toy.py` (runs anywhere), `run_course_instance.py` (needs the course workbook) |
| `tests/` | unittest suite (19 tests) |
| `legacy/` | original scripts, verbatim, with a table of known issues |
| `experimental/` | unsolved MILP formulation and unverified VRP extract, with explanation |
| `results/synthetic_toy/` | reproducible toy-instance outputs (`examples/run_toy.py --seed 1 --sensitivity`) |
| `results/course_instance/` | saved 2025 sensitivity report (text, aggregates only) with provenance |
| `docs/` | `PROVENANCE.md` (file mapping, every code change), `DATA.md` (schema, open-data sources) |

## 5. Technologies

Python 3, numpy, scipy (sparse Dijkstra), pandas, networkx (used only in tests, as a reference), matplotlib,
and scikit-learn (kNN imputation). `openpyxl` is needed only for reading `.xlsx` files. The experimental files
need gurobipy (with a licence) or OR-Tools; neither is needed by the package.

## 6. Installation & running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m unittest discover -s tests -t .      # from the repository root
```

These steps were verified only in part. The test command ran in an environment that already had the
dependencies installed (Python 3.14, numpy, scipy, pandas, networkx, scikit-learn, matplotlib). The
`pip install` step itself was not run during preparation.

## 7. Example usage

```bash
# SP + locker model on the synthetic 12x12 grid; writes solution.json / solution.png
python3 examples/run_toy.py --seed 1 --out toy_output

# SP-only variant, plus the full sensitivity grid (short runs)
python3 examples/run_toy.py --seed 1 --no-apls --sensitivity --out toy_output_sp

# Real instance (only if you have a workbook in the documented schema; not distributed)
python3 examples/run_course_instance.py path/to/workbook.xlsx --seed 1 --out course_output
```

```python
from parcelnet.toy import make_toy_instance
from parcelnet.optimizer import ServicePointOptimizer

inst = make_toy_instance(seed=0)
opt = ServicePointOptimizer(inst, use_apls=True, seed=1)
opt.optimize(initial_temp=15, cooling_rate=0.99995, max_iterations=2000, initial_strategy="existing")
print(opt.summarize()["cost"])
```

## 8. Results

**On the course instance (2025 run, legacy code, data not included).** `results/course_instance/`
holds the saved sensitivity report, a copy of an output of the original scripts. It cannot be
regenerated here. Details and caveats are in `results/course_instance/README.md`.

- The saved solution figure from the same work is **not distributed**, because it plots course-workbook
  coordinates. Its title reads "24 SPs, 20 APLs, TotCap: 2,158 parcels/day" (SP capacity only), with
  24 SP and 20 APL markers counted by eye during preparation.
- `sensitivity_report.txt`:
  - baseline EUR 2,611,116 with 14 SPs, 8 APLs and an average assigned distance of 3.19 km;
  - the "1.0x demand" and "0% noise" reruns of the same data cost +3.0% and +6.3%, which shows how much
    single unseeded SA runs vary;
  - three scenarios are in the tens of millions (10% noise: EUR 120.4 M; 15% noise: EUR 21.8 M;
    pickup share 0.4: EUR 41.2 M). These are penalty-dominated values, not operating costs.

No objective value from the course run can be compared with an optimum, because no exact solution exists:
the MILP in `experimental/` was never solved. The saved costs are also not comparable with the packaged
code. That code fixes the doubled distances, and it is not known whether those doubled distances affected
the run.

**On the synthetic instance.** The test suite checks behaviour, not performance. It checks five things:
- distances match networkx;
- the objective matches hand-computed values;
- SA solutions are feasible (penalty 0) for all three start strategies;
- runs are identical for the same seed;
- rejected moves leave the current solution unchanged.

`results/synthetic_toy/` contains one reproducible run:
`python3 examples/run_toy.py --seed 1 --sensitivity --out results/synthetic_toy`.
- Files: `solution.json`, `solution.png`, `sensitivity_report.txt`, `sensitivity.png`.
- Re-running with the same seed reproduced `sensitivity_report.txt` byte for byte. `solution.json` differed
  only in its wall-clock `seconds` field.
- The run opened 1 SP and 2 APLs, total cost EUR 77,381, penalty 0.

These costs describe a generated grid and say nothing about Maastricht.

### What was implemented vs what the team report describes

The team's final business report (not included) presents the results differently from what the code and
saved outputs support. The table is based on reading the report text during preparation.

| Report statement | What the code / saved outputs show |
|---|---|
| Customer pickup probability was modelled "based on factors like distance and age" | The final SA uses a **fixed** pickup share (0.57) for every node (`legacy/SA5.py:29`). Only an early, excluded SA variant had a distance-pickup curve. |
| Hybrid proposal "24 service points and 20 APLs", about EUR 2.75 M per year | 24 SPs / 20 APLs matches the saved 2025 solution figure (not distributed; see section 8). No saved file contains the EUR 2.75 M figure. |
| 16-SP proposal at about EUR 3 M; SA "pushed several sites to the south-west, south-east and far-east fringe" | No saved output shows a 16-SP solution. The candidate set is the 35 existing SP sites (`legacy/SA5.py:85-87`), so SA can only re-select existing sites and cannot create new fringe locations. |
| SA with initial temperature 15, decay 0.9995, up to 1000 iterations | `legacy/SA4_00.py:1296-1298` uses T0 = 15, cooling 0.99995 and 2000 iterations. `legacy/SA5.py:1717-1719` uses T0 = 100, cooling 0.995 and 1000 iterations. |
| Sensitivity: pickup 80% -> 65% raises cost from EUR 2.40 M to 3.40 M; 10% distance error -> +11%; +20% demand with 10% noise -> EUR 3.45 M | `sensitivity_report.txt` tests pickup shares 0.40-0.74 (baseline 0.57) and reports +4511.8% for 10% noise. Its stress test "demand 1.2 / noise 10%" is about EUR 3.03 M in the saved figure. None of the report's numbers appear in the outputs. |
| "A full sensitivity analysis was outside the scope", followed by a sensitivity section | A sensitivity script exists and ran once per scenario, unseeded. |

## 9. Limitations

These limitations are deliberately **not** fixed, because fixing them would change the model:

- **No distance-dependent pickup.** The case's central behavioural driver is missing. Closing an SP never
  loses pickups, so the main trade-off is not modelled.
- **APL customers pay distance cost on all their parcels.** This is inherited from `SA5.py:806`, and it
  makes lockers look worse than they should. On the toy instance, the SP+APL run ends with a higher cost
  than the SP-only run.
- **Candidate sites are the existing SP locations only** (35 in the course instance). APL slots are drawn
  from the same set.
- **The temperature is tiny compared with the costs.** T0 is 15-5000, while costs are about EUR 10^6 and
  the penalty weight is 1e6. In practice the search behaves like a randomised descent.
- **The `adjust_capacity` move does nothing.** It is overwritten by the capacity reset that follows every
  move. Opening an SP also does not reassign customers, so "open" moves are almost always rejected.
- **Demand is double-counted.** Every node in a CBS square receives the square's full household count, so
  squares with many road nodes are over-weighted. The 1.4 M parcels/year and the 0.57 pickup share are
  hard-coded inputs with undocumented derivation.
- **The sensitivity analysis uses one SA run per scenario.** Differences between scenarios mix parameter
  effects with SA noise. Distance noise is applied per matrix entry, which breaks symmetry and the triangle
  inequality.
- **The packaged optimiser is slow at full scale.** Moves are pure-Python loops over customers and
  facilities. Runtime on the full 8,098-node instance was not measured. The dense distance matrix needs
  about 0.5 GB.
- **No exact baseline exists.** Solution quality cannot be measured against an optimum.

**Future work:** add a distance-dependent pickup curve to the objective; add an open-source MILP baseline
(HiGHS/CBC) on a reduced instance; use incremental cost evaluation; run replicated multi-seed sensitivity
with confidence intervals; rebuild the instance from OSM and CBS open data.

## 10. Reproducibility

| Item | Status |
|---|---|
| Toy instance, tests, example | Fully reproducible offline. Instance seed `toy.make_toy_instance(seed=…)`; SA seed `ServicePointOptimizer(seed=…)`. Running `run_toy.py` twice with the same seed gives identical `solution.json` (apart from wall-clock time). |
| Course-instance results | **Not reproducible here.** They need the non-distributed course workbook. They were produced by unseeded legacy code, and the generating version of `SA5.py` was later overwritten. |
| Real-world rebuild | Possible from open data (OSM via OARBench or osmnx, CBS squares; see `docs/DATA.md`), but parcel volumes would have to be assumed. |

## 11. Context, attribution & licence

- **Context and contributions.** Collaborative project of 7 students (course eLab II, Business case 2,
  2025). Teammates are not named. This repository includes only work attributed to
  the author; the full breakdown is in [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md).
  - **Individual (author):** exploratory analysis and CBS data cleaning, all-pairs Dijkstra preprocessing,
    the SA models with and without lockers, an unsolved Gurobi formulation, and the sensitivity analysis.
  - **Shared (team):** problem framing, cost parameters and assumptions, and the business report.
  - **Teammates (not included):** a first SA attempt, an initial cost script, the pickup regression model,
    and most of the business report.
  - **Evidence level:** the split comes from the team's own documents. There is no version control, so
    file-level authorship is **inferred, not proven**.
- **AI assistance.**
  - **Original 2025 code (author's part):** whether and how AI was used is unspecified.
  - **2026 preparation:** the refactoring, bug fixes, seeding, tests, synthetic instance and documentation
    were done with Claude Code under my direction. I reviewed the changes.
- **Data sources.** The road network is from © OpenStreetMap contributors (ODbL), extracted with OARBench.
  Demographics are from CBS 500 m squares (CC BY 4.0, "Bron: CBS"). No data from either source is
  redistributed here. The course workbook, brief and report are not included.
- **License:** none chosen yet. The code includes shared team work; see [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md).
