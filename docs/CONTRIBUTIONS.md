# Contributions — maastricht-service-point-optimization

Collaborative university project: eLab II, Business case 2, 2025, **7 students**. Teammates are not named.
**Consent:** the author confirms that all required tutor and teammate consent was obtained. Its date,
wording and conditions are not recorded here.

| Category | Work | Evidence | Confidence |
|---|---|---|---|
| Individual (author) | Exploratory data analysis; CBS demographic cleaning and multicollinearity handling (`analysis/cbs_cleaning.py`) | team working-division record + audit | medium (inferred) |
| Individual (author) | Node-to-service-point assignment and all-pairs Dijkstra distances (`src/parcelnet/graph.py`, from `dijstra_matrix.py`) | same | medium (inferred) |
| Individual (author) | Simulated-annealing models with and without parcel lockers (`src/parcelnet/optimizer.py`, from `SA4_00.py` / `SA5.py`) | same | medium (inferred) |
| Individual (author) | LP/MILP formulation attempted in Gurobi, **never solved** (`experimental/`) | same + a code-explanation PDF authored by the author | medium |
| Individual (author) | One-at-a-time sensitivity analysis (`src/parcelnet/sensitivity.py`) | same | medium (inferred) |
| Shared (team) | Problem framing, cost parameters and assumptions, the final business report (not included) | team documents | high |
| Teammates (excluded) | First SA attempt, initial cost-calculation script, pickup-probability / logistic-regression model, most of the business report | team documents + audit | high |
| 2026 preparation | Refactoring, bug fixes (incl. doubled edge lengths), seeding, tests, synthetic instance, documentation | `docs/PROVENANCE.md` | high |

**Not established:**
- exact per-file authorship (there is no version control);
- whether teammates want to be named.

Nothing in this repository should be described as sole work of the author beyond the "Individual" rows.
