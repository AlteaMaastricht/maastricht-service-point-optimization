"""Road-graph construction and all-pairs shortest paths.

Ported from ``legacy/dijstra_matrix.py``. Same approach (undirected graph,
scipy's C implementation of Dijkstra on a sparse matrix), with two fixes that
are documented in docs/PROVENANCE.md:

1. Edge weights are no longer summed twice. The original code appended every
   edge in both directions *and* added both (i, j) and (j, i) entries for each
   of those, and ``scipy.sparse.csr_matrix`` sums duplicate entries. Every road
   length was therefore doubled (verified on a 2-node example and against the
   saved output ``shortest_paths_optimized_400.csv``, see the validation
   report). Parallel edges between the same pair are now reduced to the
   shortest one instead of being summed.
2. Walking speed is an explicit parameter in km/h. The original used
   ``speed_mps = 5.0`` (5 m/s = 18 km/h) with the comment "5 km/h".
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_adjacency(node_ids: Sequence[int],
                    edges: pd.DataFrame,
                    u_col: str = "V1",
                    v_col: str = "V2",
                    w_col: str = "DIST") -> csr_matrix:
    """Build a symmetric sparse adjacency matrix (weights in metres).

    Edges whose endpoints are not in ``node_ids`` are ignored. Direction flags
    are ignored (walking graph), as in the original code. For parallel edges
    the minimum length is kept.
    """
    node_to_idx = {int(n): i for i, n in enumerate(node_ids)}
    n = len(node_ids)
    best: dict[Tuple[int, int], float] = {}
    for u, v, w in zip(edges[u_col].astype(int), edges[v_col].astype(int), edges[w_col].astype(float)):
        if u not in node_to_idx or v not in node_to_idx or u == v:
            continue
        i, j = node_to_idx[u], node_to_idx[v]
        key = (min(i, j), max(i, j))
        if w < best.get(key, np.inf):
            best[key] = w
    if not best:
        return csr_matrix((n, n))
    rows = np.fromiter((k[0] for k in best), dtype=np.int64, count=len(best))
    cols = np.fromiter((k[1] for k in best), dtype=np.int64, count=len(best))
    data = np.fromiter(best.values(), dtype=float, count=len(best))
    # Zero-length roads exist in the source data (33 in the course instance).
    # scipy treats explicit zeros in sparse input as edges, so keep them, but
    # add a tiny epsilon so the sparse constructor cannot drop them.
    data = np.where(data <= 0, 1e-9, data)
    upper = csr_matrix((data, (rows, cols)), shape=(n, n))
    return (upper + upper.T).tocsr()


def all_pairs_distances(node_ids: Sequence[int],
                        edges: pd.DataFrame,
                        max_distance_m: Optional[float] = None,
                        **cols) -> np.ndarray:
    """All-pairs shortest-path distances in metres (``np.inf`` = unreachable/cut off).

    ``max_distance_m`` replaces the original ``max_distance_km`` argument,
    which was in fact compared against metres.
    """
    adj = build_adjacency(node_ids, edges, **cols)
    dist = dijkstra(adj, directed=False)
    if max_distance_m is not None:
        dist[dist > max_distance_m] = np.inf
    return dist


def walking_times_s(dist_m: np.ndarray, walking_speed_kmh: float = 5.0) -> np.ndarray:
    """Convert distances (m) to walking times (s) at ``walking_speed_kmh``."""
    return dist_m / (walking_speed_kmh / 3.6)


def to_edge_list(node_ids: Sequence[int], dist_m: np.ndarray,
                 walking_speed_kmh: float = 5.0) -> pd.DataFrame:
    """Upper-triangular (source, target, distance_m, time_s) table of finite pairs.

    Equivalent to the ``shortest_paths_optimized.csv`` format of the original.
    """
    n = len(node_ids)
    iu, ju = np.triu_indices(n, k=1)
    d = dist_m[iu, ju]
    keep = np.isfinite(d)
    ids = np.asarray(node_ids)
    return pd.DataFrame({
        "source": ids[iu[keep]],
        "target": ids[ju[keep]],
        "distance_m": d[keep],
        "time_s": walking_times_s(d[keep], walking_speed_kmh),
    })
