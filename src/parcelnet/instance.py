"""Problem instance: nodes, distances, demand, candidate sites.

The original scripts (``legacy/SA4_00.py``, ``legacy/SA5.py``) loaded
everything inside the optimizer class from a course-provided Excel workbook
that cannot be redistributed. Here loading is separated from optimisation:

- :class:`Instance` holds plain numpy arrays.
- :func:`load_course_workbook` reads a workbook with the documented schema
  (see docs/DATA.md) if you have legitimate access to one.
- :func:`estimate_demand` is a port of ``_estimate_demand`` (same weighting).
- :mod:`parcelnet.toy` builds a synthetic instance with the same structure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import graph

# Hard-coded in the original code (SA4_00.py:348, SA5.py:392). The origin of
# this figure is not documented in the code; it is kept as a parameter.
DEFAULT_ANNUAL_DEMAND = 1_400_000


@dataclass
class Instance:
    node_ids: np.ndarray                 # (n,) node identifiers; every node is a customer
    coords: np.ndarray                   # (n, 2) planar coordinates (for plotting only)
    dist_km: np.ndarray                  # (n, n) shortest-path distances in km, np.inf if unreachable
    demand_total: np.ndarray             # (n,) expected parcels per day per node
    candidate_sp: List[int]              # node ids that may host a service point
    apl_candidates: List[int]            # node ids for the initial APL slots
    existing_assignment: Optional[Dict[int, int]] = None  # node id -> SP node id
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.node_ids = np.asarray(self.node_ids)
        self.node_to_idx = {int(n): i for i, n in enumerate(self.node_ids)}
        n = len(self.node_ids)
        if self.dist_km.shape != (n, n):
            raise ValueError(f"dist_km must be {(n, n)}, got {self.dist_km.shape}")
        if self.demand_total.shape != (n,):
            raise ValueError("demand_total must have one entry per node")
        for nid in list(self.candidate_sp) + list(self.apl_candidates):
            if int(nid) not in self.node_to_idx:
                raise ValueError(f"candidate node {nid} is not in node_ids")

    @property
    def n_customers(self) -> int:
        return len(self.node_ids)


def estimate_demand(nodes: pd.DataFrame,
                    cbs: Optional[pd.DataFrame],
                    annual_demand: float = DEFAULT_ANNUAL_DEMAND) -> np.ndarray:
    """Daily demand per node, weighted by the household count of its CBS square.

    Port of ``_estimate_demand`` (SA4_00.py:343-427). Weight per node =
    households of the node's 500 m square x (1 + 0.5 * share of high-income
    households). Missing values fall back to population 100 and households =
    0.4 x population, as in the original.

    Known limitation kept on purpose (see README): every node in a square gets
    the *full* household count of that square, so squares with many road nodes
    receive proportionally more demand.
    """
    daily = annual_demand / 365.0
    ids = nodes["NODE ID"].astype(int).tolist()
    if cbs is None:
        return np.full(len(ids), daily / len(ids))
    merged = nodes.merge(cbs, left_on="SQUARE", right_on="Square", how="left")
    weights: Dict[int, float] = {}
    for _, row in merged.iterrows():
        population = row.get("Population", 100)
        population = population if pd.notna(population) else 100
        households = row.get("Households", population * 0.4)
        households = households if pd.notna(households) else population * 0.4
        population = max(population, 1)
        households = max(households, 1)
        mult = 1.0
        hi = row.get("Percentage high income households", 0)
        if pd.notna(hi) and hi > 0:
            mult = 1.0 + (hi / 100 * 0.5)
        weights[int(row["NODE ID"])] = max(households * mult, 1.0)
    w = np.array([weights.get(i, 1.0) for i in ids], dtype=float)
    demand = w / w.sum() * daily
    return np.maximum(demand, 0.01)


def nearest_assignment(inst_dist_km: np.ndarray, node_ids: Sequence[int],
                       sp_ids: Sequence[int]) -> Dict[int, int]:
    """Assign every node to its nearest service point (by shortest path).

    Replaces the missing generator of ``node_to_service_point_assignments_max5kmh.csv``
    (no script in the original folder produces that file).
    """
    idx = {int(n): i for i, n in enumerate(node_ids)}
    cols = [idx[int(s)] for s in sp_ids]
    sub = inst_dist_km[:, cols]
    best = np.argmin(np.where(np.isfinite(sub), sub, np.inf), axis=1)
    return {int(n): int(sp_ids[b]) for n, b in zip(node_ids, best)}


def load_course_workbook(path: str,
                         annual_demand: float = DEFAULT_ANNUAL_DEMAND,
                         n_random_apls: int = 20,
                         seed: Optional[int] = None,
                         max_distance_m: Optional[float] = None) -> Instance:
    """Build an :class:`Instance` from a workbook with the course schema.

    The workbook itself is NOT distributed with this repository (course
    material). Required sheets/columns are documented in docs/DATA.md.
    Distances are computed in memory with :mod:`parcelnet.graph` instead of
    the 1.2 GB ``distance_matrix.csv`` intermediate the original code read.
    The dense n x n matrix needs ~0.5 GB for 8,098 nodes (float64).
    """
    xls = pd.ExcelFile(path)
    nodes = pd.read_excel(xls, "Nodes")
    edges = pd.read_excel(xls, "Edges")
    try:
        cbs = pd.read_excel(xls, "CBS Squares")
    except ValueError:
        cbs = None
    node_ids = nodes["NODE ID"].astype(int).tolist()
    try:
        sp = pd.read_excel(xls, "Service Point Locations")
        candidates = sp["Location ID"].dropna().astype(int).tolist()
    except ValueError:
        candidates = list(node_ids)
    rng = random.Random(seed)
    try:
        apl = pd.read_excel(xls, "APL Locations")
        apl_candidates = apl["Location ID"].dropna().astype(int).tolist()
    except ValueError:
        # Same rule as SA5.py:105-112, but with a seeded generator.
        apl_candidates = (rng.sample(candidates, n_random_apls)
                          if len(candidates) >= n_random_apls else candidates[:n_random_apls])
    dist_km = graph.all_pairs_distances(node_ids, edges, max_distance_m=max_distance_m) / 1000.0
    demand = estimate_demand(nodes, cbs, annual_demand)
    existing = nearest_assignment(dist_km, node_ids, candidates)
    return Instance(node_ids=np.array(node_ids),
                    coords=nodes[["X", "Y"]].to_numpy(float),
                    dist_km=dist_km, demand_total=demand,
                    candidate_sp=candidates, apl_candidates=apl_candidates,
                    existing_assignment=existing,
                    meta={"source": "course workbook (not distributed)"})
