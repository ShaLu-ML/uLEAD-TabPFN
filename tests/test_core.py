import unittest

import numpy as np

from config import CONFIG
from lead import LeadTabPFN


class CoreBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.original_config = CONFIG.copy()
        CONFIG["verbose"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_paper_latent_dimension_rule(self):
        model = LeadTabPFN(seed=0)
        self.assertEqual(model._determine_latent_dim(5), 5)
        self.assertEqual(model._determine_latent_dim(101), 100)

    def test_context_set_is_normal_only_and_budgeted(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1_200, 5))
        y = np.zeros(1_200, dtype=int)
        y[-100:] = 1

        model = LeadTabPFN(seed=0)
        context_indices, purity, initial_indices = model._gen_context_set(X, y)

        self.assertEqual(len(initial_indices), 550)
        self.assertEqual(len(context_indices), CONFIG["context_cap"])
        self.assertEqual(len(np.unique(context_indices)), len(context_indices))
        self.assertTrue(np.all(y[context_indices] == 0))
        self.assertEqual(purity, 1.0)

    def test_cached_ddm_matrix_uses_nll_aggregation(self):
        CONFIG["use_ddm"] = True
        CONFIG["ddm_aggregation"] = "mean"
        nll_matrix = np.array([[1.0, 3.0], [2.0, 6.0]])

        model = LeadTabPFN.__new__(LeadTabPFN)
        scores = model.compute_scores_from_cached_artifacts(nll_matrix)

        np.testing.assert_allclose(scores, np.array([2.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
