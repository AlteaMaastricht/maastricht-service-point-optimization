# Data

## What is distributed

No real data. The repository ships only:
- a **synthetic generator** (`src/parcelnet/toy.py`): grid road graph, random
  edge-length noise, random household counts per 500 m "square";
- the saved **sensitivity report** of the 2025 course run (`results/course_instance/`),
  which contains aggregate costs/counts only, not input data. The run's figures, including a scatter plot
  of course-workbook node coordinates, are not distributed;
- reproducible **synthetic outputs** in `results/synthetic_toy/`.

## What the original project used (not distributed)

The case workbook `data_Maastricht_2024.xlsx` was supplied by the course. It
bundles open data with course-specific parcel volumes for a fictionalised
carrier ("Post&L"). It is course material and is **not** included. If you
have legitimate access to it, `parcelnet.instance.load_course_workbook` reads
it; otherwise you can rebuild the open parts yourself (below).

### Input schema expected by `load_course_workbook`

| Sheet | Required columns | Notes |
|---|---|---|
| `Nodes` | `NODE ID` (int), `X`, `Y` (planar metres), `SQUARE` (int, CBS square id) | course instance: 8,098 nodes, ids 1-8485 (not contiguous) |
| `Edges` | `V1`, `V2` (node ids), `DIST` (metres) | course instance: 10,877 edges, 33 of length 0, 234 duplicate undirected pairs; direction flag ignored (walking graph) |
| `CBS Squares` (optional) | `Square`, `Population`, `Households`, `Percentage high income households` | used for demand weights; missing -> uniform demand |
| `Service Point Locations` (optional) | `Location ID` (node id) | candidate SP sites; missing -> every node is a candidate |
| `APL Locations` (optional) | `Location ID` | initial locker slots; missing -> 20 sampled (seeded) from the SP candidates |

The counts above were obtained by reading the workbook during validation
(see the validation report); the workbook itself was not copied.

Other parameters that the original code hard-coded and that `parcelnet` keeps
as defaults: annual demand 1,400,000 parcels (`instance.DEFAULT_ANNUAL_DEMAND`),
pickup share 0.57, APL capacity 60 parcels/day, APL fixed cost EUR 500/year.
Their derivation is not documented in the original code.

### Cleaning input (`analysis/cbs_cleaning.py`)

CBS square statistics with columns such as `Population`, `Male`, `Female`,
`Age0-14` ... `Age65+`, `Households`, the four household-type counts,
`Houses`, `Home ownership %`, `Rental %`, `Avg. home value k€`,
`Median household income`, `Percentage low/high income households`,
`Distance nearest supermarket in km`, `X`, `Y`. Missing columns are skipped.

## Rebuilding from open data

Both underlying open sources can be obtained directly; nothing was downloaded
for this repository.

- **Road network: OpenStreetMap.** The case graph was extracted from
  OpenStreetMap with OARBench (https://github.com/Olibear/OARBench), per the
  case description. OSM data is © OpenStreetMap contributors and licensed
  under the **Open Database License (ODbL) 1.0**
  (https://www.openstreetmap.org/copyright). Any derived graph you publish
  must carry that attribution and remain under ODbL. Alternatives: export a
  Maastricht extract (e.g. via a Geofabrik download or osmnx) and convert
  intersections/road segments to the `Nodes`/`Edges` schema above.
- **Demographics: CBS 500 m x 500 m squares.** Statistics Netherlands (CBS)
  publishes "Kaart van 500 meter bij 500 meter met statistieken"
  (https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/kaart-van-500-meter-bij-500-meter-met-statistieken).
  CBS open data is published under **CC BY 4.0**; attribute "Bron: CBS".
  Some cells are suppressed for privacy in low-population squares, which is
  why the cleaning script imputes values.
- **Parcel volumes and existing SP locations** are course-specific and have
  no open equivalent; use the synthetic generator or your own assumptions.

Licence statements above are the publishers' general terms; check the
current terms on the linked pages before redistributing derived data.
