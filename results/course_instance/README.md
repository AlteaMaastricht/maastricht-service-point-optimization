# results/course_instance/ — saved output from the 2025 course run

`sensitivity_report.txt` is a byte-identical copy of an output found in the original
project folder. It was produced on the **course instance** (the Maastricht road graph and parcel
data supplied for the case, not distributed here) by the **legacy** scripts, not by
`src/parcelnet`. It cannot be regenerated from this repository without that data, and the exact
code version that produced it is not preserved (see below). It contains only aggregate costs and
counts, no input data.

| File | Timestamp (original) | Produced by |
|---|---|---|
| `sensitivity_report.txt` | 2025-06-07 20:19 | `sensitivity_analysis.py` (it imports `SA5`). The file was saved 2025-06-07 20:23, four minutes after the report. |

**Not distributed:** the 2025 run also saved two figures, `service_points_solution.png` (2025-06-06,
an SA5-type run) and `sensitivity_analysis.png` (same run as the report). The solution figure plots
node coordinates taken from the course workbook, so neither figure is included here. Reproducible
figures produced by this repository on the synthetic instance are in `results/synthetic_toy/`.
The facts read from the excluded solution figure during preparation are summarised in the main README
(section 8). They come from that figure, not from any file in this repository.

## What the report shows (read directly from the file)

**`sensitivity_report.txt`**
- Baseline: EUR 2,611,116 with 14 SPs, 8 APLs and an average distance of 3.19 km.
- The "1.0x demand" scenario gives EUR 2,689,038 (+3.0%) and the "0.0% noise" scenario gives
  EUR 2,775,437 (+6.3%). Both use the unchanged baseline data. They differ from the baseline only in SA
  randomness and initial temperature (5000 instead of 1000, `sensitivity_analysis.py:232` vs `:314`/`:404`).
  The gaps therefore reflect run-to-run variation, not a parameter effect.
- "10.0% noise" gives EUR 120,419,086 (+4511.8%) and "15.0% noise" gives EUR 21,792,886 (+734.6%).
  `pickup_ratio = 0.4` gives EUR 41,191,336.
- These three values are far larger than any operating-cost component could be. For comparison,
  26 SPs x EUR 50k = EUR 1.3 M. They must therefore contain the 1e6-weighted constraint penalty. This is an
  inference: the run saved only total cost, not the penalty separately.

## Caveats

- **Single runs.** Each scenario is one unseeded SA run.
- **Unverified distances.** The distance matrix behind these runs was produced by a version of
  `dijstra_matrix.py` that sums every edge twice when it is run as written. Whether the saved
  `distance_matrix.csv` used by these runs carried that error could not be checked: the 1.2 GB file was
  deliberately not opened.
- **Disagreement with the team report.** The final team report quotes different sensitivity numbers (for
  example "+11%" for 10% distance noise). Those numbers do not match this file. See the README section
  "What was implemented vs what the team report describes".
