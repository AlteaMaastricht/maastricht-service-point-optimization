import unittest

import numpy as np
import pandas as pd

from tests import _path  # noqa: F401
import cbs_cleaning
from parcelnet.sensitivity import SensitivityAnalyzer
from parcelnet.toy import make_toy_instance


class TestSensitivity(unittest.TestCase):
    def test_small_run_is_deterministic(self):
        inst = make_toy_instance(rows=6, cols=6, n_candidates=6, n_apl=3, annual_demand=10_000, seed=2)
        kw = dict(parameter_variations={"F_SP": [25000, 75000]}, max_iterations=60)
        rob = dict(demand_scenarios=[0.8, 1.2], distance_perturbations=[0.0, 0.1],
                   stress_scenarios=[(1.2, 0.1)], max_iterations=60)
        reports = []
        for _ in range(2):
            sa = SensitivityAnalyzer(inst, seed=11)
            sa.run_parameter_sensitivity_analysis(**kw)
            sa.run_robustness_analysis(**rob)
            reports.append(sa.generate_sensitivity_report())
        self.assertEqual(reports[0], reports[1])
        self.assertIn("BASELINE SOLUTION", reports[0])
        ch = sa.robustness_results["distance_perturbations"][0.1]["avg_distance_change"]
        self.assertTrue(np.isfinite(ch) and ch > 0)


class TestCleaning(unittest.TestCase):
    def test_vif_numpy_orthogonal_is_one(self):
        X = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
        np.testing.assert_allclose(cbs_cleaning.vif_numpy(X), [1.0, 1.0])

    def test_clean_imputes_everything(self):
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({"Square": range(n), "Population": rng.integers(10, 500, n).astype(float),
                           "X": rng.uniform(0, 5000, n), "Y": rng.uniform(0, 5000, n),
                           "Age0-14": rng.integers(0, 50, n).astype(float),
                           "Age15-24": rng.integers(0, 50, n).astype(float),
                           "Home ownership %": rng.uniform(0, 100, n),
                           "Median household income": rng.uniform(20, 60, n)})
        df.loc[[3, 7], "Home ownership %"] = np.nan
        df.loc[[0], "Population"] = np.nan
        imputed, final, vif_df = cbs_cleaning.clean(df)
        self.assertEqual(len(imputed), n - 1)
        self.assertEqual(int(imputed.isna().sum().sum()), 0)
        self.assertIn("Age0-14_perc", final.columns)
        self.assertTrue(len(vif_df) > 0)


if __name__ == "__main__":
    unittest.main()
