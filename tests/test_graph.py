import unittest

import networkx as nx
import numpy as np
import pandas as pd

from tests import _path  # noqa: F401
from parcelnet import graph
from parcelnet.toy import make_grid_tables


class TestGraph(unittest.TestCase):
    def test_matches_networkx_on_toy_graph(self):
        nodes, edges, _ = make_grid_tables(8, 9, seed=3)
        ids = nodes["NODE ID"].tolist()
        D = graph.all_pairs_distances(ids, edges)
        G = nx.Graph()
        G.add_nodes_from(ids)
        for u, v, w in zip(edges.V1, edges.V2, edges.DIST):
            G.add_edge(int(u), int(v), weight=float(w))
        ref = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        for i, u in enumerate(ids):
            for j, v in enumerate(ids):
                self.assertAlmostEqual(D[i, j], ref[u][v], places=6)

    def test_edge_weight_not_doubled(self):
        # Regression test for the duplicate-summation bug of dijstra_matrix.py
        e = pd.DataFrame({"V1": [1], "V2": [2], "DIST": [100.0]})
        D = graph.all_pairs_distances([1, 2], e)
        self.assertEqual(D[0, 1], 100.0)

    def test_parallel_and_reverse_edges_take_minimum(self):
        e = pd.DataFrame({"V1": [1, 2, 1], "V2": [2, 1, 2], "DIST": [100.0, 80.0, 120.0]})
        D = graph.all_pairs_distances([1, 2], e)
        self.assertEqual(D[0, 1], 80.0)

    def test_zero_length_edge_connects(self):
        e = pd.DataFrame({"V1": [1, 2], "V2": [2, 3], "DIST": [0.0, 50.0]})
        D = graph.all_pairs_distances([1, 2, 3], e)
        self.assertAlmostEqual(D[0, 2], 50.0, places=6)

    def test_cutoff_in_metres_and_walking_time(self):
        e = pd.DataFrame({"V1": [1, 2], "V2": [2, 3], "DIST": [600.0, 600.0]})
        D = graph.all_pairs_distances([1, 2, 3], e, max_distance_m=1000)
        self.assertTrue(np.isinf(D[0, 2]))
        self.assertEqual(D[0, 1], 600.0)
        # 5 km/h: 600 m takes 432 s
        self.assertAlmostEqual(graph.walking_times_s(np.array([600.0]), 5.0)[0], 432.0)

    def test_edge_list_is_upper_triangular(self):
        e = pd.DataFrame({"V1": [1, 2], "V2": [2, 3], "DIST": [10.0, 20.0]})
        D = graph.all_pairs_distances([1, 2, 3], e)
        el = graph.to_edge_list([1, 2, 3], D)
        self.assertEqual(len(el), 3)
        self.assertTrue((el.source < el.target).all())


if __name__ == "__main__":
    unittest.main()
