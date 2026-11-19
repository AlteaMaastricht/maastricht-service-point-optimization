# Provenance

## Origin

This repository comes from a team consulting case for a university course: eLab II (course code EBC2182,
according to the portfolio audit), second year of a Business Analytics programme, May-June 2025. The team
had **7** members. Teammates are not named here:
their naming preferences have not been recorded (see `CONTRIBUTIONS.md`).

The team's working division, as recorded in the team's own documents and summarised in the portfolio audit:
- **The author** did a large part of the exploratory data analysis, including cleaning the CBS demographic
  data and handling multicollinearity. The author also computed the node-to-service-point assignments and
  the all-pairs Dijkstra distances, modelled both network variants (with and without parcel lockers) using
  simulated annealing, attempted an LP/MILP formulation in Gurobi, and ran the sensitivity analysis.
- **Other team members** wrote a first SA attempt, an initial cost-calculation script, and the
  pickup-probability / logistic-regression model. They also wrote most of the business report.

There is no version control and no per-file attribution in the original folder. The file-level split below
is therefore **inferred from the team documents and the audit, not proven**. Only files the audit attributes
to the author are included. Early SA variants that may be teammates' work are excluded, along with the
regression model and the mapping/EDA scripts whose authorship is unclear.

AI assistance: Whether and how AI was used in the original 2025 scripts is unspecified; it is not inferred
here from code style. One recorded fact: the original folder contains a file named `gurobi_grok.py`
(excluded from this repository). The 2026 refactoring, tests and documentation were done with Claude Code
under the author's direction.

## Source files -> staged files

The source folder is `Year 2/eLab2/CASE 2 Post&L Operational Management/`. The originals were not modified.

| Source | Staged as | Change |
|---|---|---|
| `dijstra_matrix.py` | `legacy/dijstra_matrix.py` | verbatim |
| `SA4_00.py` | `legacy/SA4_00.py` | verbatim |
| `SA5.py` | `legacy/SA5.py` | verbatim |
| `sensitivity_analysis.py` | `legacy/sensitivity_analysis.py` | verbatim |
| `Cleaning_FillingNA.py` | `legacy/Cleaning_FillingNA.py` | verbatim |
| `gurobi2.py` | `experimental/gurobi2_milp_formulation.py` | verbatim (renamed) |
| `SA5.py` lines 1625-1691 and 876-917 | `experimental/vrp_ortools_from_SA5.py` | verbatim extract with an explanatory header |
| `sensitivity_report.txt` | `results/course_instance/` | byte-identical copy |
| `sensitivity_analysis.png`, `service_points_solution.png` | — (excluded 18 Nov 2026, decision D13) | Originally copied. Removed from the publication copy because the solution plot uses course-workbook node coordinates. Kept outside the repo in `_PORTFOLIO_AUDIT/EIGHT_REPOSITORIES/excluded/postnl-network-optimization/`. |
| — (new) | `results/synthetic_toy/` | Generated 18 Nov 2026 by the staged code: `python3 examples/run_toy.py --seed 1 --sensitivity --out results/synthetic_toy` (system Python 3.14, 17 s). A rerun reproduced `sensitivity_report.txt` byte for byte; `solution.json` differed only in its `seconds` field. |
| `dijstra_matrix.py` | `src/parcelnet/graph.py` | refactor + fixes G1-G3 |
| `SA5.py`, `SA4_00.py` (`load_data`, `_estimate_demand`) | `src/parcelnet/instance.py` | loading separated from the optimizer; demand weighting ported unchanged; fix I1 |
| `SA5.py` + `SA4_00.py` (optimizer) | `src/parcelnet/optimizer.py` | consolidated; fixes S1-S7 |
| `SA5.visualize_solution_static` | `src/parcelnet/plotting.py` | fix P1 |
| `sensitivity_analysis.py` | `src/parcelnet/sensitivity.py` | fixes A1-A3 |
| `Cleaning_FillingNA.py` | `analysis/cbs_cleaning.py` | function + CLI; fix C1 |
| *(new)* | `src/parcelnet/toy.py`, `tests/`, `examples/`, docs | written for this repository |

## Code changes and why

Each change fixes a defect found while reading the code or makes the code runnable without course data. No
change was made to improve the results.

**Graph (`graph.py`)**
- **G1.** Each edge weight was added four times as COO entries: the edge list held both directions, and
  each direction was then inserted as `(i, j)` and `(j, i)`. `scipy.sparse.csr_matrix` sums duplicate
  entries, so every road length was doubled. Two pieces of evidence confirm this:
  - a 2-node reproduction gives 200 m for a 100 m edge;
  - the saved output `shortest_paths_optimized_400.csv` gives 70.53 m for the 35.26 m edge 1-5295.

  The fix builds a symmetric matrix once and keeps the minimum over parallel edges. A regression test
  covers it.
- **G2.** The walking speed was `speed_mps = 5.0`, commented as "5 km/h", but 5 m/s is 18 km/h. It is now a
  `walking_speed_kmh` parameter (default 5). Walking times are not used by the optimiser.
- **G3.** `max_distance_km=3000` was compared against metres. The parameter is renamed `max_distance_m`.
  Zero-length edges get a 1e-9 m weight so the sparse constructor cannot drop them.

**Instance (`instance.py`)**
- **I1.** The random APL slots (`random.sample`, unseeded, `SA5.py:108`) now use a seeded generator.
- The silent fallbacks are removed: a uniform 2 km matrix if loading failed, and 2 km for missing pairs.
  Distances are computed in memory, so no 1.2 GB CSV is needed.
- No script in the original folder produced the "existing assignment" file. It is now derived as
  nearest-SP by shortest path (`nearest_assignment`).

**Optimiser (`optimizer.py`)**
- **S1.** A seeded `random.Random(seed)` is used for every draw. The originals used the unseeded module
  `random`.
- **S2.** APL positions are now part of the solution (`solution["apl_nodes"]`). `SA5._relocate_apl` wrote
  into `self.apl_locations`, so rejected moves still relocated lockers and corrupted the search. Two
  regression tests cover this.
- **S3.** The OR-Tools VRP block is removed from the objective. It is moved to `experimental/`; see the
  README there for the defects. The objective is the distance-based cost used before 2025-06-17.
- **S4.** `SA5._create_existing_solution` called an undefined `_create_greedy_solution`. The greedy start
  is now ported from `SA4_00.py`.
- **S5.** `_create_random_solution` assumed at least 30 (SA5) or 20 (SA4_00) candidates. The counts are now
  capped by the number of candidates, so the toy instance works.
- **S6.** The cost function is vectorised with numpy using the same formulas. A hand-computed test checks
  it.
- **S7.** `summarize()` and `cost_breakdown()` now use the objective's formulas.
  `SA5._validate_solution` charged APL customers only for delivery parcels, while `evaluate_solution`
  charged them for all parcels. The objective itself is unchanged: APL customers still pay distance cost
  for all their parcels, as in `SA5.py:806`.

What was **not** changed, on purpose, is listed in the README "Limitations" section. It includes the fixed
pickup share, the candidate set, the temperature scale, the no-op `adjust_capacity` move, the penalty
weight and the demand double-counting.

**Plotting (`plotting.py`)**
- **P1.** The APLs to plot were taken from the boolean `apl_open` vector instead of the node ids. Plotting
  now uses the node ids.

**Sensitivity (`sensitivity.py`)**
- **A1.** Each SA run gets a deterministic seed (`base seed + run index`).
- **A2.** `avg_distance_change` divided by the zero diagonal. In the original workbook output,
  `sensitivity_analysis_results.xlsx` (not distributed), this column is NaN for every noise level. The
  diagonal is now excluded.
- **A3.** The six-panel figure is reduced to three panels. The original plotting code remains in
  `legacy/sensitivity_analysis.py`. The report now states the penalty contained in each cost.

**Cleaning (`analysis/cbs_cleaning.py`)**
- **C1.** VIF falls back to an equivalent numpy implementation when `statsmodels` is not installed (it was
  not installed in the validation environment). The thresholds and the mean/median quirks of the original
  are kept and documented in the module docstring.

## Excluded from the original folder

| Item | Reason |
|---|---|
| `data_Maastricht_2024.xlsx`, `data_Maastricht_2024_NODES_EDGES.xlsx`, `cleaned_*.xlsx`, `vif*.xlsx`, `detailed_square_analysis.csv`, `edges.json`, `nodes.json`, `*_grid_*.csv` | course-provided data or direct derivatives of it |
| `distance_matrix*.csv` (1.2 GB, 316 MB), `shortest_paths_optimized*.csv` (up to 403 MB), `sop500_updated_prb*.csv`, `distance_cache_*.pkl`, `optimization_results.pkl` | large generated intermediates; the pickles were never opened (unsafe to unpickle) |
| `case2.pdf`, `ELab Case 2.pdf`, `Project Plan Draft.docx`, `for completions, extension of the report....docx`, final report PDF, `team feedback.pdf` | course brief (instructor copyright), team notes naming members, the report (lists student IDs), confidential peer evaluation |
| `Service Point Optimizer_ Comprehensive Code Explanation.pdf` | explains the unsolved `gurobi2.py`; not needed |
| `SA_postnnl.py`, `SA_postnl*.py`, `SA_case2*.py`, `postnl_SA5.py`, `sa_case_two_4th*(1).py`, `sa_case_2_final00.py`, `PostandL.py` | earlier or parallel SA versions; several possibly by teammates; near-duplicates |
| `gurobi_grok.py`, `googleOR.py`, `lagrarian.py` | non-functional and not attributed to the author (see experimental/README.md) |
| `map*.py`, `plot*.py`, `servicepoint*.py`, `regression2.py`, `create_sp_summary.py`, `hardcoded_summary_plots.py`, `file-p.py`, `path_shorts.py`, `traveltimes_&plots00.py` | EDA/mapping with unclear authorship; `path_shorts.py` near-duplicates `dijstra_matrix.py`; `traveltimes_&plots00.py` is corrupted (SyntaxError) |
| ~120 EDA PNGs, `*.html` maps (3-20 MB), `lib/` (vendored vis-network/tom-select) | plots of course data; generated pyvis/folium output; third-party JS |
| `.DS_Store`, `~$*` lock files | system files |

## Repository name (18 Nov 2026)
- Renamed from the working name `postnl-network-optimization` to **`maastricht-service-point-optimization`** (author decision D4). The old name used a carrier brand; the case itself used the fictional "Post&L".
- The team-case-study framing was replaced by explicit individual / shared / teammates' contribution statements (author decision). See `CONTRIBUTIONS.md`.

- `LICENSE` added (19 Nov 2026, author decision): MIT, copyright 2026 Arthur J. Maarschalkerweerd; README licence line updated.
