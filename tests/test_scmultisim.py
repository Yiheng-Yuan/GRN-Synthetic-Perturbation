"""Scientific regression checks for the scMultiSim-derived kinetic extension.

These small synthetic fixtures test the implemented stochastic fluid mechanism,
not biological fidelity, upstream R equivalence, or formal calibration.
"""

from decimal import Decimal, localcontext
import json
import unittest

import numpy as np

from grn_experiment.scmultisim import (
    CellState,
    KineticParameters,
    PositiveMapping,
    Simulator,
    development_parameters,
    exact_rna_segment,
    observation_counts,
    stable_seed,
)


def constant_parameters(n_genes=1, **overrides):
    values = dict(
        A=np.zeros((n_genes, n_genes)),
        b=np.zeros(n_genes),
        kon=np.full(n_genes, 1.0),
        koff=np.full(n_genes, 2.0),
        beta=np.full(n_genes, 2.0),
        gamma=np.full(n_genes, 1.5),
        rho=np.full(n_genes, 0.8),
        regulators=tuple(range(n_genes)),
        K=5.0,
        mapping=PositiveMapping(log10_offset=1.0, log10_slope=0.25, scale=1.0),
    )
    values.update(overrides)
    return KineticParameters(**values)


def initial_state(cells, genes):
    rng = np.random.default_rng(734)
    return CellState(
        rng.integers(0, 2, size=(cells, genes)),
        rng.uniform(0.1, 2.0, size=(cells, genes)),
        rng.uniform(0.1, 2.0, size=(cells, genes)),
    )


def decimal_segment(u, m, production, beta, gamma, dt):
    """Independent high-precision convolution solution, including equal rates."""
    with localcontext() as ctx:
        ctx.prec = 80
        u, m, p, b, g, t = map(
            lambda value: Decimal(str(value)), (u, m, production, beta, gamma, dt)
        )
        eb, eg = (-b * t).exp(), (-g * t).exp()
        convolution = t * eb if b == g else (eb - eg) / (g - b)
        unspliced = u * eb + p * (1 - eb) / b
        spliced = m * eg + b * u * convolution + p * ((1 - eg) / g - convolution)
        return float(unspliced), float(spliced)


class AnalyticRnaTests(unittest.TestCase):
    def test_matches_high_precision_solution_in_difficult_regimes(self):
        # Near-equal rates and very short segments are cancellation-sensitive.
        for beta, gamma in ((2.0, 2.0), (2.0, 2.0 + 1e-12), (0.03, 8.0), (8.0, 0.03)):
            for dt in (0.0, 1e-12, 1e-6, 0.2, 10.0):
                for u, m, p in ((0.0, 0.0, 3.0), (2.3, 4.1, 0.0), (2.3, 4.1, 7.0)):
                    with self.subTest(beta=beta, gamma=gamma, dt=dt, u=u, m=m, p=p):
                        got_u, got_m = exact_rna_segment(
                            np.array([[u]]), np.array([[m]]), np.array([[p]]),
                            np.array([beta]), np.array([gamma]), dt,
                        )
                        expected_u, expected_m = decimal_segment(u, m, p, beta, gamma, dt)
                        np.testing.assert_allclose(got_u, expected_u, atol=0, rtol=3e-12)
                        np.testing.assert_allclose(got_m, expected_m, atol=0, rtol=3e-12)
                        self.assertGreaterEqual(float(got_u[0, 0]), 0.0)
                        self.assertGreaterEqual(float(got_m[0, 0]), 0.0)

    def test_constant_segment_semigroup_and_fractional_state(self):
        U = np.array([[0.01, 2.0], [4.0, 0.0]])
        M = np.array([[0.02, 4.0], [2.0, 0.0]])
        p = np.array([[0.0, 10.0], [10.0, 0.0]])
        beta, gamma = np.array([1.0, 2.0]), np.array([1.0, 0.7])
        one_u, one_m = exact_rna_segment(U, M, p, beta, gamma, 0.7)
        first_u, first_m = exact_rna_segment(U, M, p, beta, gamma, 0.3)
        two_u, two_m = exact_rna_segment(first_u, first_m, p, beta, gamma, 0.4)
        np.testing.assert_allclose(two_u, one_u, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(two_m, one_m, atol=1e-14, rtol=1e-13)
        self.assertGreater(float(one_m[0, 0]), 0.0)
        self.assertLess(float(one_m[0, 0]), 1.0)

    def test_count_conversion_occurs_only_at_observation(self):
        state = CellState(np.array([[1, 0]]), np.array([[0.3, 2.1]]), np.array([[0.9, 2.9]]))
        counts = observation_counts(state)
        np.testing.assert_array_equal(counts, [[0, 2]])
        self.assertTrue(np.issubdtype(counts.dtype, np.integer))
        np.testing.assert_array_equal(state.M, [[0.9, 2.9]])


class ProductionMechanismTests(unittest.TestCase):
    def test_target_production_multiplier_does_not_delete_edges_or_shrink_rna(self):
        params = development_parameters(31, n_genes=4)
        M = np.array([[0.2, 2.0, 5.0, 8.0], [0.1, 3.0, 6.0, 9.0]])
        target = params.regulators[0]
        unperturbed = params.production(M)
        saved_A = params.A.copy()
        for dose in (0.0, 0.3, 0.5, 0.7, 0.9):
            got = params.production(M, target=target, dose=dose)
            expected = unperturbed.copy()
            expected[:, target] *= 1 - params.rho[target] * dose
            np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0)
            np.testing.assert_array_equal(params.A, saved_A)
        state = initial_state(10, 4)
        simulation = Simulator(params, state, seed=25, tau=1 / 32, target=target, dose=0.7)
        zero = simulation.advance(0.0)
        for name in ("G", "U", "M"):
            np.testing.assert_array_equal(getattr(zero, name), getattr(state, name))

    def test_jacobian_orientation_and_sign_agree_with_finite_difference(self):
        A = np.array([[0.0, -0.8, 0.0], [0.0, 0.0, 0.6], [0.9, 0.0, 0.0]])
        params = constant_parameters(3, A=A)
        M = np.array([0.7, 2.0, 4.0])
        actual = params.production_jacobian(M)
        finite = np.zeros((3, 3))
        eps = 1e-5
        for source in range(3):
            shift = np.zeros(3)
            shift[source] = eps
            finite[:, source] = (params.production(M + shift) - params.production(M - shift)) / (2 * eps)
        np.testing.assert_allclose(actual, finite, atol=1e-9, rtol=1e-7)
        np.testing.assert_array_equal(np.sign(actual), np.sign(A))
        self.assertLess(actual[0, 1], 0)  # Source 1 inhibits target 0.
        self.assertEqual(actual[1, 0], 0)  # The reverse edge does not exist.

    def test_production_remains_positive_under_inhibition(self):
        params = constant_parameters(2, A=np.array([[0.0, -4.0], [-3.0, 0.0]]))
        rates = params.production(np.array([[0.0, 0.0], [1e6, 1e6]]))
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertTrue(np.all(rates > 0))
        self.assertTrue(np.all(rates[1] < rates[0]))

    def test_full_efficiency_full_dose_allows_zero_production_without_negative_rna(self):
        params = constant_parameters(rho=np.array([1.0]))
        state = initial_state(12, 1)
        rates = params.production(state.M, target=0, dose=1.0)
        np.testing.assert_array_equal(rates, np.zeros_like(rates))
        result = Simulator(params, state, seed=15, tau=1 / 32, target=0, dose=1.0).advance(2.0)
        expected_u, expected_m = exact_rna_segment(
            state.U, state.M, np.zeros_like(state.U), params.beta, params.gamma, 2.0
        )
        np.testing.assert_allclose(result.U, expected_u, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.M, expected_m, rtol=1e-12, atol=1e-14)

    def test_parameter_json_round_trip_preserves_mechanism(self):
        params = development_parameters(43, n_genes=4)
        restored = KineticParameters.from_dict(json.loads(json.dumps(params.to_dict())))
        M = initial_state(8, 4).M
        np.testing.assert_array_equal(restored.production(M), params.production(M))
        np.testing.assert_array_equal(restored.production_jacobian(M), params.production_jacobian(M))


class StochasticKernelTests(unittest.TestCase):
    def test_snapshot_requests_do_not_change_random_path_or_feedback(self):
        params = development_parameters(42, n_genes=4)
        state = initial_state(32, 4)
        direct = Simulator(params, state, seed=892, tau=1 / 32, target=0, dose=0.7)
        observed = Simulator(params, state, seed=892, tau=1 / 32, target=0, dose=0.7)
        final = direct.advance(2.0)
        for time in (0.0, 0.0001, 0.025, 0.03125, 0.1, 0.37, 0.999, 1.5, 2.0):
            result = observed.advance(time)
        for name in ("G", "U", "M"):
            np.testing.assert_array_equal(getattr(result, name), getattr(final, name))

    def test_json_checkpoint_preserves_pending_events_and_feedback_phase(self):
        params = development_parameters(11, n_genes=4)
        state = initial_state(20, 4)
        reference = Simulator(params, state, seed=309, tau=1 / 32, target=1, dose=0.5)
        split = Simulator(params, state, seed=309, tau=1 / 32, target=1, dose=0.5)
        expected = reference.advance(2.1)
        split.advance(0.137)  # Deliberately neither a feedback-grid boundary nor t=0.
        saved = json.loads(json.dumps(split.checkpoint(), allow_nan=False))
        resumed = Simulator.from_checkpoint(saved)
        actual = resumed.advance(2.1)
        for name in ("G", "U", "M"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_numpy_integer_target_normalizes_and_resumes_through_json(self):
        params = development_parameters(11, n_genes=4)
        state = initial_state(8, 4)
        reference = Simulator(params, state, seed=309, tau=1 / 32,
                              target=1, dose=0.5).advance(1.1)
        for target in (np.int32(1), np.int64(1), np.uint64(1)):
            with self.subTest(target_type=type(target).__name__):
                simulator = Simulator(params, state, seed=309, tau=1 / 32,
                                      target=target, dose=np.float64(0.5))
                simulator.advance(0.137)
                checkpoint = simulator.checkpoint()
                self.assertIs(type(checkpoint["target"]), int)
                self.assertIs(type(checkpoint["dose"]), float)
                restored = Simulator.from_checkpoint(json.loads(json.dumps(checkpoint, allow_nan=False)))
                actual = restored.advance(1.1)
                for name in ("G", "U", "M"):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))

    def test_same_seed_reproducible_and_distinct_seed_changes_events(self):
        params = development_parameters(5, n_genes=4)
        state = initial_state(64, 4)
        first = Simulator(params, state, seed=81, tau=1 / 16).advance(2.0)
        second = Simulator(params, state, seed=81, tau=1 / 16).advance(2.0)
        other = Simulator(params, state, seed=82, tau=1 / 16).advance(2.0)
        for name in ("G", "U", "M"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertFalse(np.array_equal(first.M, other.M))

    def test_passed_initial_state_and_returned_snapshots_do_not_alias_simulator(self):
        params = development_parameters(8, n_genes=4)
        state = initial_state(8, 4)
        saved = state.copy()
        simulator = Simulator(params, state, seed=63, tau=1 / 16)
        snapshot = simulator.advance(0.1)
        snapshot.M[:] = 999.0
        snapshot.U[:] = 999.0
        state.M[:] = 888.0
        reference = Simulator(params, saved, seed=63, tau=1 / 16).advance(1.0)
        actual = simulator.advance(1.0)
        np.testing.assert_array_equal(actual.M, reference.M)
        np.testing.assert_array_equal(actual.U, reference.U)

    def test_telegraph_fluid_stationary_moments_have_correct_scale(self):
        # Analytic moment equations for a two-state promoter followed by two
        # linear filters. No birth/death shot noise is part of this mechanism.
        params = constant_parameters()
        cells = 6000
        state = CellState(np.zeros((cells, 1), dtype=int), np.zeros((cells, 1)), np.zeros((cells, 1)))
        result = Simulator(params, state, seed=9823, tau=0.5).advance(30.0)
        on, off, beta, gamma, production = 1.0, 2.0, 2.0, 1.5, 10.0
        switching = on + off
        occupancy = on / switching
        var_g = occupancy * (1 - occupancy)
        mean_u, mean_m = production * occupancy / beta, production * occupancy / gamma
        cov_gu = production * var_g / (switching + beta)
        var_u = production * cov_gu / beta
        cov_gm = beta * cov_gu / (switching + gamma)
        cov_um = (production * cov_gm + beta * var_u) / (beta + gamma)
        var_m = beta * cov_um / gamma
        self.assertAlmostEqual(float(result.G.mean()), occupancy, delta=0.025)
        self.assertAlmostEqual(float(result.U.mean()), mean_u, delta=0.06 * mean_u)
        self.assertAlmostEqual(float(result.M.mean()), mean_m, delta=0.05 * mean_m)
        self.assertAlmostEqual(float(result.U.var()), var_u, delta=0.09 * var_u)
        self.assertAlmostEqual(float(result.M.var()), var_m, delta=0.09 * var_m)

    def test_no_edge_case_is_independent_of_feedback_resolution(self):
        params = constant_parameters(2)
        state = initial_state(40, 2)
        coarse = Simulator(params, state, seed=52, tau=0.25).advance(2.0)
        fine = Simulator(params, state, seed=52, tau=0.03125).advance(2.0)
        np.testing.assert_array_equal(coarse.G, fine.G)
        np.testing.assert_allclose(coarse.U, fine.U, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(coarse.M, fine.M, atol=1e-12, rtol=1e-12)

    def test_feedback_step_refinement_reduces_error_on_smooth_fixture(self):
        # Rare promoter switches isolate feedback approximation error here.
        # This fixture cannot select the calibrated tau for formal networks.
        params = constant_parameters(
            2, A=np.array([[0.0, 0.0], [1.2, 0.0]]),
            kon=np.full(2, 1e-12), koff=np.full(2, 1e-12),
        )
        state = CellState(np.ones((1, 2), dtype=int), np.zeros((1, 2)), np.zeros((1, 2)))
        coarse = Simulator(params, state, seed=82, tau=0.25).advance(2.0)
        refined = Simulator(params, state, seed=82, tau=0.125).advance(2.0)
        reference = Simulator(params, state, seed=82, tau=1 / 1024).advance(2.0)
        np.testing.assert_array_equal(reference.G, [[1, 1]])
        coarse_error = np.linalg.norm(coarse.M - reference.M)
        refined_error = np.linalg.norm(refined.M - reference.M)
        self.assertGreater(coarse_error, 1e-5)
        self.assertLess(refined_error, 0.7 * coarse_error)

    def test_baseline_and_perturbation_share_path_but_not_existing_rna_scaling(self):
        params = constant_parameters()
        state = initial_state(32, 1)
        ctrl = Simulator(params, state, seed=21, tau=1 / 32).advance(0.01)
        pert = Simulator(params, state, seed=21, tau=1 / 32, target=0, dose=0.7).advance(0.01)
        np.testing.assert_array_equal(ctrl.G, pert.G)
        # Existing M cannot instantly shrink by the full production multiplier.
        self.assertTrue(np.all(pert.M > 0.9 * ctrl.M))
        self.assertTrue(np.all(pert.M <= ctrl.M + 1e-14))
        self.assertTrue(np.any(pert.U < ctrl.U))


class KernelInputTests(unittest.TestCase):
    def test_parameters_reject_invalid_domains(self):
        for overrides in (
            {"K": 0}, {"K": float("nan")}, {"K": True},
            {"beta": np.array([0.0])}, {"gamma": np.array([-1.0])},
            {"kon": np.array([float("inf")])}, {"koff": np.array([-1.0])},
            {"rho": np.array([0.5])}, {"rho": np.array([1.1])},
            {"A": np.zeros((1, 2))}, {"b": np.array([float("nan")])},
            {"regulators": (True,)}, {"regulators": (0, 0)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, TypeError)):
                constant_parameters(**overrides)

    def test_mapping_rejects_nonpositive_slope_and_scale(self):
        for arguments in (
            {"log10_slope": 0.0}, {"log10_slope": -0.1},
            {"scale": 0.0}, {"scale": float("inf")},
            {"log10_offset": float("nan")},
        ):
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                PositiveMapping(**arguments)

    def test_state_rejects_nonfinite_negative_and_nonbinary_values(self):
        for G, U, M in (
            ([[2]], [[0]], [[0]]), ([[0.2]], [[0]], [[0]]),
            ([[0]], [[-0.1]], [[0]]), ([[0]], [[0]], [[float("nan")]]),
            ([[0]], [[0, 1]], [[0]]), ([0], [0], [0]),
        ):
            with self.subTest(G=G, U=U, M=M), self.assertRaises((ValueError, TypeError)):
                CellState(np.asarray(G), np.asarray(U), np.asarray(M))

    def test_intervention_target_dose_and_time_validation(self):
        params = constant_parameters()
        state = initial_state(2, 1)
        for arguments in (
            {"tau": 0}, {"tau": float("nan")}, {"tau": True},
            {"target": None, "dose": 0.7}, {"target": 0, "dose": 1.1},
            {"target": 0, "dose": float("nan")}, {"target": 0, "dose": True},
            {"target": True, "dose": 0.7}, {"target": 2, "dose": 0.7},
        ):
            arguments = {"tau": 1 / 32, **arguments}
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                Simulator(params, state, seed=31, **arguments)
        simulator = Simulator(params, state, seed=31, tau=1 / 32)
        for time in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(time=time), self.assertRaises((ValueError, TypeError)):
                simulator.advance(time)
        simulator.advance(0.5)
        with self.assertRaises((ValueError, TypeError)):
            simulator.advance(0.4)

    def test_incomplete_checkpoint_is_rejected(self):
        with self.assertRaises((ValueError, TypeError, KeyError)):
            Simulator.from_checkpoint({})

    def test_condition_seed_is_order_independent_and_unambiguous(self):
        first = stable_seed(41, "initial", 0, 0.7, 0.25, 1, "control")
        stable_seed(41, "active", 4, 0.5, 1.0, 0, "perturbed")
        self.assertEqual(first, stable_seed(41, "initial", 0, 0.7, 0.25, 1, "control"))
        self.assertNotEqual(first, stable_seed(41, "initial", 0, 0.7, 0.25, 1, "perturbed"))
        self.assertNotEqual(stable_seed(41, "ab", "c"), stable_seed(41, "a", "bc"))


if __name__ == "__main__":
    unittest.main()
