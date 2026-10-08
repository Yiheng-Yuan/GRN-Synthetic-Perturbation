"""Pure-function tests for the independent GNW pilot checker; no Java run."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_path = Path(__file__).parents[1] / "tools" / "gnw" / "check_acceptance.py"
_spec = importlib.util.spec_from_file_location("gnw_acceptance", _path)
gnw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gnw)


def fixture_truth():
    return {"genes": ["G1", "G2"], "gene_parameters": {
        "G1": {"inputs": [], "parameters": {"delta": 0.125, "max": 0.125, "a_0": 0.4}},
        "G2": {"inputs": ["G1"], "parameters": {"delta": 0.125, "max": 0.125,
                "a_0": 0.4, "a_1": 0.52, "bindsAsComplex_1": 0,
                "numActivators_1": 1, "numDeactivators_1": 0, "k_1": 0.5, "n_1": 1}},
    }}


class GNWFunctionTests(unittest.TestCase):
    def test_probability_weighting_respects_module_bit_order(self):
        # Deliberately non-additive alpha makes an accidental summed shortcut fail.
        expected = 0.1 * 0.8 * 0.3 + 0.2 * 0.2 * 0.3 + 0.4 * 0.8 * 0.7 + 0.9 * 0.2 * 0.7
        self.assertAlmostEqual(gnw.weighted_activation([0.2, 0.7], [0.1, 0.2, 0.4, 0.9]), expected)
        self.assertEqual(gnw.weighted_activation([], [0.4]), 0.4)

    def test_module_complex_and_independent_binding_differ(self):
        x, k, n = np.array([0.5, 1.0]), np.array([0.5, 0.5]), np.ones(2)
        self.assertAlmostEqual(gnw.module_activation(x, k, n, 1, False), 1 / 6)
        self.assertAlmostEqual(gnw.module_activation(x, k, n, 1, True), 1 / 4)

    def test_actual_parameters_determine_function_and_clock(self):
        model = gnw.normalize_gene_parameters(fixture_truth())
        x = np.array([0.5, 0.2])
        np.testing.assert_allclose(gnw.production(x, model, 8), [0.4, 0.46])
        np.testing.assert_allclose(gnw.vector_field(x, model, 8), [-0.1, 0.26])
        np.testing.assert_allclose(gnw.production(x, model, 8, "G1", 0.5, 0.8), [0.24, 0.46])
        # No clipping or near-zero edge substitution.
        np.testing.assert_allclose(gnw.production(x, model, 8, "G1", 1, 1), [0, 0.46])

    def test_finite_difference_matches_known_local_derivative(self):
        model = gnw.normalize_gene_parameters(fixture_truth())
        x = np.array([0.5, 0.2])
        expected = np.array([[-1.0, 0], [0.06, -1.0]])
        np.testing.assert_allclose(gnw.central_jacobian(x, lambda state: gnw.vector_field(state, model, 8)),
                                   expected, atol=1e-9)

    def test_invalid_states_and_module_tables_rejected(self):
        model = gnw.normalize_gene_parameters(fixture_truth())
        for x in ([-0.1, 0.2], [float("nan"), 0.2], [0.1]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                gnw.production(np.array(x), model, 8)
        with self.assertRaises(ValueError):
            gnw.weighted_activation([0.5], [0.4])
        with self.assertRaises(ValueError):
            gnw.weighted_activation([0.5], [0.4, 1.1])
        truth = fixture_truth()
        truth["gene_parameters"]["G2"]["inputs"] = []
        with self.assertRaises(ValueError):
            gnw.normalize_gene_parameters(truth)

    def test_exact_root_trajectory_includes_baseline_and_complete_knockout(self):
        times = np.array([0, 0.125, 4, 8])
        knockdown = gnw.analytic_root_path(times, 0.4, 0.4, 1, 0.6)
        self.assertEqual(knockdown[0], 0.4)
        np.testing.assert_allclose(knockdown, 0.24 + 0.16 * np.exp(-times))
        np.testing.assert_allclose(gnw.analytic_root_path(times, 0.4, 0.4, 1, 0), 0.4 * np.exp(-times))

    def test_trajectory_reader_rejects_duplicate_times_and_condition_changes(self):
        template = {"run_id": "x", "initial_id": "0", "target": "G1", "dose": "0.5",
                    "rho": "0.8", "remaining_fraction": "0.6", "G1": "0.4", "G2": "0.5"}
        rows = [{**template, "time": "0"}, {**template, "time": "1"}]
        groups = gnw.trajectory_groups(rows, ["G1", "G2"])
        np.testing.assert_array_equal(groups["x"]["time"], [0, 1])
        with self.assertRaises(ValueError):
            gnw.trajectory_groups([rows[0], rows[0]], ["G1", "G2"])
        with self.assertRaises(ValueError):
            gnw.trajectory_groups([rows[0], {**rows[1], "dose": "0.7"}], ["G1", "G2"])


if __name__ == "__main__":
    unittest.main()
