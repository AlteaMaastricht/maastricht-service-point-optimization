# experimental/ — formulations that did not produce a solution

Nothing in this folder is used by `parcelnet`, and nothing here produced a
result that appears anywhere in this repository. The files are kept so the
history of the work is visible. None of them was executed while preparing
this repository: `gurobipy` and `ortools` are not installed, and the original
input files are course material that is not distributed.

## `gurobi2_milp_formulation.py` (verbatim copy of `gurobi2.py`)

This is a capacitated facility-location MILP written for Gurobi. **It is a formulation that was not solved.** The
original project folder holds no solver log, no `.sol`/`.lp` file and no
result file from it, and the script itself only prints to the console
(`gurobi2.py`: no file is written anywhere). Its input file
(`shortest_paths_optimized_1000m.csv`, 1,015,376 node pairs, both directions)
does exist in the original folder, so the problem was not missing data. The
reason is not recorded. One likely factor is the model size: about 2.0 million
continuous assignment variables plus 8,098 binaries and a linking constraint
per assignment variable, with a Gurobi licence required.

Modelling problems found when reading the code (line numbers refer to the original `gurobi2.py`):

- Every customer is forced to exactly 65% pickup and 35% home delivery (lines 186-189). The pickup share
  therefore cannot respond to service-point distance, which was the central question of the case.
- The delivery cost is `C_hd * D[customer] * x_h` (lines 162-167). It has no distance term, so the delivery
  cost is a constant and SP locations do not affect it.
- Home-delivery assignments must point to an open SP (`x_h <= y`, lines 196-198). This forces SPs open for
  delivery-only customers.
- Demand is uniform: 1.4 M / 365 / 8,098 parcels per node (line 350). All 8,098 nodes are candidate sites
  (line 353).
- `MIPGap` is set to 0.08 but commented "Accept 5% gap" (line 254). Units are converted with
  `v/1000 if v > 100 else v` (line 37), which is fragile.

## `vrp_ortools_from_SA5.py` (verbatim extract from `SA5.py`)

This is an OR-Tools routing model that `SA5.py` called inside its cost function. It was added in the last
edit of `SA5.py` (2025-06-17), which came after every saved output in the original folder (the outputs are
dated 2025-06-06 and 2025-06-07). It was never verified. Defects found by reading the code:

- `distance_callback` indexes the global distance matrix with local routing indices 0..k. It never uses
  `service_point_idx` or `customer_indices`, so it routes over the wrong nodes.
- The model has one vehicle and `num_drivers` is hard-coded to 1. `parcel_counts` is never used.
- The route length, which is already in km, is divided by 1000 again. Total route time in hours is
  multiplied by a per-parcel drop-off cost.
- Any infeasible route returns `inf`, and the 8-hour limit with one vehicle makes larger SPs infeasible.
- The resulting **daily** cost **overwrites** the annual distance-based delivery cost and is added to annual
  fixed costs (`SA5.py:917-920`).

`parcelnet.optimizer` therefore uses the distance-based delivery cost instead, the same cost that
`SA4_00.py` and `SA5.py` used before this edit.

## Not included

These files are listed here with the reasons they were left out:

- `gurobi_grok.py` is a second Gurobi variant. It adds a distance term to the delivery cost and caps the
  model at 40 SPs, and it reads the 402 MB `shortest_paths_optimized_5000.csv`. It has no recorded output.
  The audit does not attribute it to the author.
- `googleOR.py` is an OR-Tools CBC model. It reads an `.xlsx` workbook with `csv.DictReader` (lines 17-18
  and 36-37), sets every demand to 100 and hard-codes the candidate sites `J = [101, 102]` (line 165). It
  cannot run as written, and its authorship is unknown.
- `lagrarian.py` is called a "Lagrangian relaxation", but it is not one. The assignment subgradient is
  identically zero (line 218), only the first 50 nodes are facilities (line 29), and the "dual" is
  maximised over primal costs (lines 97-98). It has no recorded output, and its authorship is unknown.
