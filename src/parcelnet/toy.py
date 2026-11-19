"""Synthetic toy instance (grid road graph + random demand).

Added for this repository so the pipeline can run end to end without the
course data. It has the same structure as the real instance (nodes with X/Y,
edges with lengths in metres, 500 m "squares" with household counts), but all
values are generated. Numbers produced on it say nothing about Maastricht.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import graph
from .instance import Instance, estimate_demand, nearest_assignment


def make_grid_tables(rows: int = 12, cols: int = 12, spacing_m: float = 100.0,
                     drop_fraction: float = 0.1, seed: int = 0
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (nodes, edges, cbs) tables in the documented input schema.

    A rows x cols grid; a fraction of edges is removed at random (keeping the
    graph connected) and edge lengths get +/-20% noise so shortest paths are
    not trivial.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, rows * cols + 1)
    xs = np.tile(np.arange(cols), rows) * spacing_m
    ys = np.repeat(np.arange(rows), cols) * spacing_m
    square = (ys // 500).astype(int) * 100 + (xs // 500).astype(int)
    nodes = pd.DataFrame({"NODE ID": ids, "X": xs, "Y": ys, "SQUARE": square})

    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((ids[i], ids[i + 1]))
            if r + 1 < rows:
                edges.append((ids[i], ids[i + cols]))
    edges = np.array(edges)
    lengths = spacing_m * rng.uniform(0.8, 1.2, len(edges))
    keep = np.ones(len(edges), dtype=bool)
    for k in rng.permutation(len(edges))[: int(drop_fraction * len(edges))]:
        keep[k] = False
        e = pd.DataFrame({"V1": edges[keep, 0], "V2": edges[keep, 1], "DIST": lengths[keep]})
        d = graph.all_pairs_distances(ids, e)
        if not np.isfinite(d).all():
            keep[k] = True  # removal would disconnect the grid
    edge_df = pd.DataFrame({"EDGE ID": np.arange(keep.sum()) + 1,
                            "V1": edges[keep, 0], "V2": edges[keep, 1],
                            "DIST": lengths[keep]})
    squares = np.unique(square)
    pop = rng.integers(50, 1500, len(squares))
    cbs = pd.DataFrame({"Square": squares, "Population": pop,
                        "Households": (pop * rng.uniform(0.35, 0.55, len(squares))).round(),
                        "Percentage high income households": rng.uniform(0, 40, len(squares)).round(1)})
    return nodes, edge_df, cbs


def make_toy_instance(rows: int = 12, cols: int = 12, n_candidates: int = 12,
                      n_apl: int = 6, annual_demand: float = 40_000,
                      seed: int = 0) -> Instance:
    """Build a small :class:`Instance` (default 144 nodes, 12 SP candidates)."""
    nodes, edges, cbs = make_grid_tables(rows, cols, seed=seed)
    ids = nodes["NODE ID"].to_numpy()
    rng = np.random.default_rng(seed + 1)
    cand = sorted(rng.choice(ids, size=n_candidates, replace=False).tolist())
    apl = sorted(rng.choice(cand, size=min(n_apl, len(cand)), replace=False).tolist())
    dist_km = graph.all_pairs_distances(ids, edges) / 1000.0
    demand = estimate_demand(nodes, cbs, annual_demand)
    return Instance(node_ids=ids, coords=nodes[["X", "Y"]].to_numpy(float),
                    dist_km=dist_km, demand_total=demand,
                    candidate_sp=[int(c) for c in cand], apl_candidates=[int(a) for a in apl],
                    existing_assignment=nearest_assignment(dist_km, ids, cand),
                    meta={"source": f"synthetic grid {rows}x{cols}, seed={seed}"})
