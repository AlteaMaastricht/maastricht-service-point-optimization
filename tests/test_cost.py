"""Hand-computed checks of the objective (see comments for the arithmetic)."""
import unittest

import numpy as np

from tests import _path  # noqa: F401
from parcelnet.instance import Instance
from parcelnet.optimizer import ServicePointOptimizer


def line_instance():
    # Nodes 1 - 2 - 3 on a line, 1 km apart; 10 parcels/day each.
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    return Instance(node_ids=np.array([1, 2, 3]), coords=np.array([[0, 0], [1, 0], [2, 0]], float),
                    dist_km=D, demand_total=np.array([10.0, 10.0, 10.0]),
                    candidate_sp=[1, 3], apl_candidates=[3])


class TestCost(unittest.TestCase):
    def test_sp_only(self):
        opt = ServicePointOptimizer(line_instance(), use_apls=False, pickup_ratio=0.5)
        sol = opt._empty()
        sol["y"][0] = True
        sol["x_assigned"][:, 0] = True
        sol["capacity"][0] = 15.0             # = pickup 3 x 5, no bounce
        c = opt.cost_breakdown(sol)
        self.assertEqual(c["fixed_sp"], 50000)
        self.assertAlmostEqual(c["capacity"], 15 * 36.5)             # 547.5
        self.assertAlmostEqual(c["delivery"], 5 * (0 + 1 + 2) * 1.5 * 365)  # 8212.5
        self.assertEqual(c["penalty"], 0)
        self.assertAlmostEqual(c["total"], 58760.0)

    def test_sp_plus_apl(self):
        opt = ServicePointOptimizer(line_instance(), use_apls=True, pickup_ratio=0.5)
        sol = opt._empty()
        sol["y"][0] = True
        sol["apl_open"][0] = True
        sol["x_assigned"][0, 0] = sol["x_assigned"][1, 0] = True
        sol["x_assigned"][2, 2] = True        # node 3 -> APL at node 3 (column n_sp + 0)
        sol["capacity"][0] = 10.0
        sol["capacity"][2] = 60.0
        c = opt.cost_breakdown(sol)
        # fixed 50000 + APL 500 + capacity 70*36.5=2555 + delivery 5*(0+1)*1.5*365=2737.5
        self.assertAlmostEqual(c["total"], 55792.5)
        self.assertEqual(c["penalty"], 0)

    def test_bounce_penalties(self):
        opt = ServicePointOptimizer(line_instance(), use_apls=False, pickup_ratio=0.5)
        sol = opt._empty()
        sol["y"][0] = True
        sol["x_assigned"][:, 0] = True
        sol["capacity"][0] = 14.0             # pickup 15 -> 1 bounced, rate 1/15
        c = opt.cost_breakdown(sol)
        # per-SP: (1/15 - 0.02) * 15 * 1e6 = 0.7e6 ; city: (1/15 - 0.01) * 15 * 1e6 = 0.85e6
        self.assertAlmostEqual(c["penalty"], 1.55e6, places=3)

    def test_assignment_and_closed_and_apl_capacity_penalties(self):
        opt = ServicePointOptimizer(line_instance(), use_apls=True, pickup_ratio=0.5)
        sol = opt._empty()
        sol["y"][0] = True
        sol["x_assigned"][0, 0] = True        # node 1 OK
        sol["x_assigned"][1, 1] = True        # node 2 -> closed SP (node 3): +1e6
        sol["apl_open"][0] = True
        sol["x_assigned"][2, 2] = True        # node 3 -> APL with capacity 5 < 10: +5e6
        sol["capacity"][0] = 5.0
        sol["capacity"][2] = 5.0
        c = opt.cost_breakdown(sol)
        # closed-facility 1e6 + APL overflow (10-5)*1e6 = 5e6, plus bounce terms, because
        # the closed SP column still carries node 2's pickup (5) with capacity 0 (as in
        # the original): per-SP (1-0.02)*5*1e6 = 4.9e6, city (5/10-0.01)*10*1e6 = 4.9e6
        self.assertAlmostEqual(c["penalty"], 15.8e6, places=3)
        sol["x_assigned"][1, 1] = False       # node 2 unassigned: +1e6 instead
        self.assertAlmostEqual(opt.cost_breakdown(sol)["penalty"], 6e6, places=3)

    def test_capacity_rule(self):
        opt = ServicePointOptimizer(line_instance(), use_apls=False, pickup_ratio=0.5)
        sol = opt._empty()
        sol["y"][0] = True
        sol["x_assigned"][:, 0] = True
        opt._adjust_capacity_after_assignment(sol)
        self.assertAlmostEqual(sol["capacity"][0], 15 / (1 - 0.015))


if __name__ == "__main__":
    unittest.main()
