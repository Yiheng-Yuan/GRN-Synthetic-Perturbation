import unittest

import numpy as np

from grn_experiment.metrics import (
    control_adjusted_rmse,
    edge_average_precision,
    eliminated_fraction,
    paired_network_bootstrap_ci,
    signed_edge_f1,
    sliced_wasserstein,
)


class MetricsTests(unittest.TestCase):
    def test_unpaired_distance_is_order_invariant(self):
        cells = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        self.assertAlmostEqual(sliced_wasserstein(cells, cells[::-1]), 0.0)

    def test_control_adjustment(self):
        treated = np.array([[2.0, 1.0], [4.0, 1.0]])
        control = np.array([[1.0, 1.0], [3.0, 1.0]])
        self.assertAlmostEqual(
            control_adjusted_rmse(treated, control, treated[::-1], control), 0.0
        )

    def test_average_precision_groups_ties(self):
        true = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        constant = np.ones((3, 3))
        self.assertAlmostEqual(edge_average_precision(constant, true), 1 / 6)

    def test_repressed_true_edge_ranks_as_edge_not_absence(self):
        true = np.array([[0, 1, -1], [0, 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(edge_average_precision(true, true), 1.0)

    def test_signed_f1_and_registry(self):
        true = np.array([[0, 1, -1], [0, 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(signed_edge_f1(true, true, threshold=0.5), 1.0)
        self.assertAlmostEqual(eliminated_fraction(["a", "b", "c"], ["a"]), 2 / 3)
        self.assertTrue(np.isnan(eliminated_fraction([], [])))

    def test_bootstrap_keeps_family_pairs(self):
        differences = np.array([[1.0, 3.0], [3.0, 5.0]])
        result = paired_network_bootstrap_ci(differences, seed=2)
        self.assertEqual(result.n_networks, 2)
        self.assertAlmostEqual(result.mean, 3.0)
        self.assertGreaterEqual(result.lower, 2.0)
        self.assertLessEqual(result.upper, 4.0)


if __name__ == "__main__":
    unittest.main()
