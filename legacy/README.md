# legacy/ — original scripts (verbatim, not maintained)

These are unmodified copies of the author's scripts from the 2025 course
project, kept so that the packaged code in `src/parcelnet/` can be compared
with what was actually written at the time. They are **not** meant to be run:
they expect the course workbook `data_Maastricht_2024.xlsx` and generated
intermediates (`distance_matrix.csv`, 1.2 GB; `node_to_service_point_assignments_max5kmh.csv`)
in the working directory, none of which are distributed.

| File | Role | Last modified | Known issues (details in docs/PROVENANCE.md) |
|---|---|---|---|
| `dijstra_matrix.py` | road graph + all-pairs Dijkstra (scipy) | 2025-06-04 | edge weights summed twice (all distances x2); `speed_mps = 5.0` labelled "5 km/h"; `max_distance_km=3000` is metres; file modified after `distance_matrix.csv` was written |
| `SA4_00.py` | SA, service points only | 2025-06-06 | unseeded RNG; calls undefined `visualize_solution_static` (line 1347), so its `__main__` never reaches the pickle dump |
| `SA5.py` | SA, service points + lockers (APL) | 2025-06-17 | unseeded RNG; `_relocate_apl` mutates optimizer state; OR-Tools VRP added after the saved outputs (see experimental/); missing `_create_greedy_solution`; APL plot uses booleans as node ids |
| `sensitivity_analysis.py` | one-at-a-time sensitivity + robustness (imports `SA5`) | 2025-06-07 | one unseeded SA run per scenario; `avg_distance_change` divides by zero on the diagonal (NaN) |
| `Cleaning_FillingNA.py` | CBS square cleaning, kNN imputation, VIF | 2025-05-27 | comment says "> 4 missing" but code keeps `<= 6`; fallback uses mean, comment says median |

Common to SA4_00/SA5: fixed `pickup_ratio = 0.57` for every node (no distance
dependence), candidate set = the 35 existing SP locations, initial temperature
15-5000 against costs of ~EUR 10^6 and a penalty weight of 1e6.
