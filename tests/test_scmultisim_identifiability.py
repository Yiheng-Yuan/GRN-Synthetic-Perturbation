"""Regression checks of a concrete exact M-only parameter symmetry."""

import json
import unittest

import numpy as np

from grn_experiment.scmultisim import (
    CellState, KineticParameters, PositiveMapping, Simulator, development_parameters,
    observation_counts,
)
from grn_experiment.scmultisim_identifiability import (
    canonicalize_rate_order, equivalence_invariants, parameter_equivalence_report,
    swap_splicing_degradation,
)


def parameters():
    values = development_parameters(101, n_genes=4).to_dict()
    # All rates are in the development [0.7,1.3] interval. Start with beta>=gamma
    # to make the transformed hidden state nonnegative for every nonnegative U,M.
    values.update(beta=[1.05, 1.1, 1.03, 1.0], gamma=[1.0, 1.05, 1.0, 0.98],
                  b=[-0.05, -0.05, -0.05, -0.05])
    return KineticParameters.from_dict(values)


def state():
    rng = np.random.default_rng(31)
    return CellState(rng.integers(0, 2, (12, 4)), rng.uniform(0.2, 5, (12, 4)),
                     rng.uniform(0.2, 5, (12, 4)))


class RateSwapTests(unittest.TestCase):
    def test_vector_field_conjugacy_for_every_gene_and_perturbation(self):
        original, initial = parameters(), state()
        swapped = swap_splicing_degradation(original, np.ones(4, dtype=bool), initial)
        ratio = original.beta / original.gamma
        for target, dose in ((None, 0), (0, 0.3), (1, 0.7), (2, 0.9), (0, 1.0)):
            with self.subTest(target=target, dose=dose):
                dU, dM = original.derivative(initial, target, dose)
                new_dU, new_dM = swapped.parameters.derivative(swapped.initial, target, dose)
                np.testing.assert_allclose(new_dU, ratio * dU + (ratio - 1) * dM,
                                           atol=1e-14, rtol=1e-13)
                np.testing.assert_allclose(new_dM, dM, atol=1e-14, rtol=1e-13)
                np.testing.assert_allclose(
                    swapped.parameters.production(initial.M, target, dose),
                    ratio * original.production(initial.M, target, dose), rtol=1e-14,
                )

    def test_mature_rna_path_and_integer_counts_across_doses_and_feedback_grids(self):
        original, initial = parameters(), state()
        changed = swap_splicing_degradation(original, np.ones(4, dtype=bool), initial)
        for tau in (1 / 16, 1 / 32):
            for target, dose in ((None, 0), (0, 0.3), (0, 0.7), (0, 0.9)):
                left = Simulator(original, initial, seed=431, tau=tau, target=target, dose=dose)
                right = Simulator(changed.parameters, changed.initial, seed=431, tau=tau,
                                  target=target, dose=dose)
                for time in (0.125, 0.5, 2.0, 8.0):
                    with self.subTest(tau=tau, target=target, dose=dose, time=time):
                        expected, actual = left.advance(time), right.advance(time)
                        np.testing.assert_array_equal(actual.G, expected.G)
                        np.testing.assert_allclose(actual.M, expected.M, atol=2e-13, rtol=1e-13)
                        np.testing.assert_array_equal(observation_counts(actual), observation_counts(expected))

    def test_partial_gene_swap_is_involution_with_nondefault_mapping_slope(self):
        original, initial = parameters(), state()
        values = original.to_dict()
        values["mapping"] = PositiveMapping(0.4, 0.7, 1.8).to_dict()
        original = KineticParameters.from_dict(values)
        mask = np.array([True, False, True, False])
        once = swap_splicing_degradation(original, mask, initial)
        twice = swap_splicing_degradation(once.parameters, mask, once.initial)
        self.assertEqual(once.swapped_genes, (0, 2))
        for name in ("A", "b", "beta", "gamma", "kon", "koff", "rho"):
            np.testing.assert_allclose(getattr(twice.parameters, name), getattr(original, name), atol=1e-15)
        for name in ("G", "U", "M"):
            np.testing.assert_allclose(getattr(twice.initial, name), getattr(initial, name), atol=1e-14)
        np.testing.assert_array_equal(once.parameters.b[~mask], original.b[~mask])

    def test_negative_hidden_state_is_rejected_without_clipping(self):
        values = parameters().to_dict()
        values.update(beta=[0.7] * 4, gamma=[1.3] * 4)
        original = KineticParameters.from_dict(values)
        initial = CellState(np.ones((1, 4)), np.zeros((1, 4)), np.ones((1, 4)))
        with self.assertRaisesRegex(ValueError, "negative/nonfinite hidden U"):
            swap_splicing_degradation(original, np.ones(4, dtype=bool), initial)

    def test_supplied_bounds_must_admit_both_parameter_sets(self):
        with self.assertRaisesRegex(ValueError, "transformed parameters outside"):
            swap_splicing_degradation(parameters(), np.ones(4, dtype=bool),
                                       bounds={"b": (-0.1, 0.0)})
        allowed = swap_splicing_degradation(parameters(), np.ones(4, dtype=bool),
                                            bounds={"b": (-0.2, 0.2), "beta": (0.7, 1.3),
                                                    "gamma": (0.7, 1.3)})
        self.assertEqual(allowed.swapped_genes, (0, 1, 2, 3))

    def test_shape_and_mask_validation(self):
        for mask in ([1, 0, 0, 0], [True], [True, False, False, True, True]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "Boolean vector"):
                swap_splicing_degradation(parameters(), mask)
        with self.assertRaisesRegex(ValueError, "dimension"):
            swap_splicing_degradation(parameters(), [True] * 4,
                                       CellState(np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1))))

    def test_full_formal_bounds_use_effective_A_alias_and_reject_ambiguous_aliases(self):
        bounds = {"effective_A": [-2, 2], "b": [-0.2, 0.2], "kon": [0.7, 1.3],
                  "koff": [0.7, 1.3], "beta": [0.7, 1.3], "gamma": [0.7, 1.3],
                  "rho": [0.6, 1.0]}
        transformed = swap_splicing_degradation(parameters(), [True] * 4, bounds=bounds)
        report = parameter_equivalence_report(transformed.parameters, parameters(), bounds=bounds)
        self.assertTrue(report["admissibility"]["parameter_bounds_accepted"])
        with self.assertRaisesRegex(ValueError, "only one of A"):
            swap_splicing_degradation(parameters(), [False] * 4, bounds={**bounds, "A": [-2, 2]})
        with self.assertRaisesRegex(ValueError, "effective_A"):
            swap_splicing_degradation(parameters(), [False] * 4,
                                       bounds={**bounds, "effective_A": [-0.01, 0.01]})


class EquivalenceScoringTests(unittest.TestCase):
    def test_raw_errors_remain_while_orbit_and_invariant_errors_vanish(self):
        reference, initial = parameters(), state()
        estimate = swap_splicing_degradation(reference, [True, False, True, False], initial)
        report = parameter_equivalence_report(
            estimate.parameters, reference, initial.M,
            bounds={"b": (-0.2, 0.2), "beta": (0.7, 1.3), "gamma": (0.7, 1.3)},
            estimate_initial=estimate.initial, reference_initial=initial,
        )
        self.assertGreater(report["raw_rmse"]["beta"], 0)
        self.assertGreater(report["raw_rmse"]["b"], 0)
        self.assertGreater(report["jacobian_errors"]["raw_Js_rmse"], 0)
        for value in report["canonical_rmse"].values():
            self.assertLess(value, 1e-14)
        for value in report["invariant_rmse"].values():
            self.assertLess(value, 1e-14)
        self.assertLess(report["jacobian_errors"]["canonical_Js_rmse"], 1e-14)
        self.assertLess(report["jacobian_errors"]["invariant_Js_over_gamma_rmse"], 1e-14)
        self.assertEqual(report["estimate_canonical_swaps"], [0, 2])
        self.assertTrue(report["admissibility"]["parameter_bounds_accepted"])
        self.assertTrue(report["admissibility"]["initial_pair_accepted"])
        json.dumps(report, allow_nan=False)

    def test_jacobian_normalization_is_invariant_but_absolute_amplitude_is_not(self):
        reference = parameters()
        changed = swap_splicing_degradation(reference, [True] * 4).parameters
        M = np.array([[0.1, 1, 3, 10], [2, 3, 0.5, 8]])
        Js = reference.production_jacobian(M)
        changed_Js = changed.production_jacobian(M)
        np.testing.assert_allclose(changed_Js, (reference.beta / reference.gamma)[:, None] * Js,
                                   rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(changed_Js / changed.gamma[:, None], Js / reference.gamma[:, None],
                                   rtol=1e-14, atol=1e-14)

    def test_canonical_order_and_invariants_do_not_enumerate_combinations(self):
        original = development_parameters(83)
        result = canonicalize_rate_order(original)
        self.assertTrue(np.all(result.parameters.beta >= result.parameters.gamma))
        for key, value in equivalence_invariants(original).items():
            np.testing.assert_allclose(equivalence_invariants(result.parameters)[key], value, atol=1e-14)

    def test_no_initial_state_or_bounds_does_not_certify_admissibility(self):
        report = parameter_equivalence_report(parameters(), parameters())
        self.assertIsNone(report["admissibility"]["initial_pair_accepted"])
        self.assertIsNone(report["admissibility"]["parameter_bounds_accepted"])
        self.assertIsNone(report["jacobian_errors"])

    def test_comparison_keeps_domain_failures_visible(self):
        reference = parameters()
        estimate = swap_splicing_degradation(reference, [True] * 4).parameters
        report = parameter_equivalence_report(estimate, reference, bounds={"b": (-0.1, 0.0)})
        self.assertFalse(report["admissibility"]["parameter_bounds_accepted"])
        self.assertIn("outside supplied bounds", report["admissibility"]["failure"])
        self.assertLess(report["canonical_rmse"]["b"], 1e-14)

    def test_negative_initial_does_not_mislabel_valid_parameter_bounds(self):
        reference = parameters()
        estimate = swap_splicing_degradation(reference, [True] * 4).parameters
        bad_initial = CellState(np.ones((12, 4)), np.zeros((12, 4)), np.ones((12, 4)))
        report = parameter_equivalence_report(
            estimate, reference, bounds={"b": [-0.2, 0.2], "beta": [0.7, 1.3],
                                        "gamma": [0.7, 1.3]},
            estimate_initial=bad_initial, reference_initial=state(),
        )
        self.assertTrue(report["admissibility"]["parameter_bounds_accepted"])
        self.assertFalse(report["admissibility"]["initial_pair_accepted"])
        self.assertIn("negative/nonfinite hidden U", report["admissibility"]["failure"])

    def test_jacobian_uses_off_diagonal_pairs_and_one_gene_has_no_pairs(self):
        reference = parameters()
        estimate = swap_splicing_degradation(reference, [True] * 4).parameters
        M = state().M
        difference = estimate.production_jacobian(M) - reference.production_jacobian(M)
        expected = float(np.sqrt(np.mean(difference[:, ~np.eye(4, dtype=bool)] ** 2)))
        report = parameter_equivalence_report(estimate, reference, M)
        self.assertAlmostEqual(report["jacobian_errors"]["raw_Js_rmse"], expected)
        single = KineticParameters(A=np.zeros((1, 1)), b=np.zeros(1), kon=np.ones(1),
                                   koff=np.ones(1), beta=np.ones(1), gamma=np.ones(1),
                                   rho=np.full(1, 0.8), regulators=(0,), K=20)
        report = parameter_equivalence_report(single, single, np.ones((2, 1)))
        self.assertEqual(report["jacobian_errors"]["directed_pairs_per_state"], 0)
        for name in ("raw_Js_rmse", "canonical_Js_rmse", "invariant_Js_over_gamma_rmse"):
            self.assertIsNone(report["jacobian_errors"][name])

    def test_different_model_architecture_is_not_silently_quotiented(self):
        values = parameters().to_dict()
        values["K"] *= 2
        with self.assertRaisesRegex(ValueError, "identical frozen"):
            parameter_equivalence_report(KineticParameters.from_dict(values), parameters())
        with self.assertRaisesRegex(ValueError, "both initial"):
            parameter_equivalence_report(parameters(), parameters(), estimate_initial=state())
        with self.assertRaisesRegex(ValueError, "at least one"):
            parameter_equivalence_report(parameters(), parameters(), np.empty((0, 4)))

    def test_real_parameter_discrepancy_is_not_removed(self):
        reference = parameters()
        values = reference.to_dict()
        values["A"][1][0] *= 0.4
        values["rho"][0] = 0.61
        estimate = KineticParameters.from_dict(values)
        report = parameter_equivalence_report(estimate, reference)
        self.assertGreater(report["canonical_rmse"]["A"], 0)
        self.assertGreater(report["canonical_rmse"]["rho"], 0)


if __name__ == "__main__":
    unittest.main()
