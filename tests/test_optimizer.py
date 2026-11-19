import unittest

import numpy as np

from tests import _path  # noqa: F401
from parcelnet.optimizer import ServicePointOptimizer
from parcelnet.toy import make_toy_instance

INST = make_toy_instance(seed=0)


def run(seed, use_apls=True, strategy="existing", iters=400):
    opt = ServicePointOptimizer(INST, use_apls=use_apls, seed=seed)
    opt.optimize(initial_temp=15, cooling_rate=0.99995, max_iterations=iters,
                 time_limit=120, initial_strategy=strategy)
    return opt


class TestOptimizer(unittest.TestCase):
    def test_feasible_solution_sp_apl(self):
        opt = run(1)
        s = opt.summarize()
        self.assertTrue(opt.is_feasible())
        self.assertEqual(s["unassigned_customers"], 0)
        self.assertEqual(s["multi_assigned_customers"], 0)
        self.assertLessEqual(s["max_sp_bounce_rate"], 0.02)
        self.assertGreaterEqual(s["n_service_points"], 1)
        self.assertLessEqual(opt.best_cost, opt.cost_history[0])

    def test_feasible_solution_sp_only_and_strategies(self):
        for strategy in ("existing", "greedy", "random"):
            opt = run(2, use_apls=False, strategy=strategy, iters=200)
            self.assertTrue(opt.is_feasible(), strategy)
            self.assertEqual(opt.summarize()["n_apls"], 0)

    def test_reproducible_with_seed(self):
        a, b = run(7), run(7)
        self.assertEqual(a.best_cost, b.best_cost)
        for k in a.best_solution:
            np.testing.assert_array_equal(a.best_solution[k], b.best_solution[k])
        self.assertEqual(a.stats["accepted"], b.stats["accepted"])

    def test_relocate_apl_does_not_mutate_parent(self):
        # Regression test for SA5._relocate_apl writing into optimizer state.
        opt = ServicePointOptimizer(INST, use_apls=True, seed=3)
        parent = opt.create_initial_solution("existing")
        before = {k: v.copy() for k, v in parent.items()}
        initial_slots = opt.apl_slots_initial.copy()
        for _ in range(20):
            child = {k: v.copy() for k, v in parent.items()}
            opt._relocate_apl(child)
        for k in parent:
            np.testing.assert_array_equal(parent[k], before[k])
        np.testing.assert_array_equal(opt.apl_slots_initial, initial_slots)

    def test_rejected_moves_leave_current_unchanged(self):
        opt = ServicePointOptimizer(INST, use_apls=True, seed=5)
        cur = opt.create_initial_solution("existing")
        snap = {k: v.copy() for k, v in cur.items()}
        for _ in range(50):
            opt.generate_neighbor(cur)        # neighbour discarded (= rejected)
        for k in cur:
            np.testing.assert_array_equal(cur[k], snap[k])


if __name__ == "__main__":
    unittest.main()
