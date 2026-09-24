"""Small deterministic tests; these do not create the formal network cohort."""

import unittest

import numpy as np

from grn_experiment.simulation import (
    NetworkSpec,
    dynamics_diagnostics,
    make_dynamics,
    make_network,
    network_diagnostics,
    validate_network,
)


class NetworkTests(unittest.TestCase):
    def test_seeded_network_has_signed_motifs_and_degree_bound(self) -> None:
        for seed in (0, 7, 31):
            network = make_network(seed, n_genes=8, n_regulators=5, n_edges=12)
            report = validate_network(network, n_edges=12)
            self.assertTrue(report["has_chain"])
            self.assertTrue(report["has_feedforward"])
            self.assertTrue(report["has_positive_feedback"])
            self.assertTrue(report["has_negative_feedback"])
            self.assertLessEqual(int(np.max(report["indegree"])), 3)
            self.assertLessEqual(int(np.max(report["outdegree"])), 5)
            self.assertTrue(all(report["indegree"][gene] >= 1 for gene in report["nonregulators"]))
            self.assertTrue(np.all(np.diag(network.weights) == 0))
            self.assertTrue(np.array_equal(network.weights, make_network(seed, 8, 5, 12).weights))
            self.assertTrue(np.all(np.abs(network.weights[network.weights != 0]) >= 0.4))
            self.assertTrue(np.all(np.abs(network.weights[network.weights != 0]) <= 1.2))
            self.assertEqual(report["n_positive"] + report["n_negative"], 12)

    def test_capacity_matching_is_seed_robust_on_small_dense_networks(self) -> None:
        # Deliberately close to the degree limits, yet only eight genes; no
        # formal 24-gene experimental network or dataset is generated.
        for seed in range(32):
            with self.subTest(seed=seed):
                network = make_network(seed, n_genes=8, n_regulators=5, n_edges=20)
                report = validate_network(network, n_edges=20)
                self.assertLessEqual(int(np.max(report["indegree"])), 3)
                self.assertLessEqual(int(np.max(report["outdegree"])), 5)
                self.assertTrue(all(report["indegree"][gene] >= 1 for gene in report["nonregulators"]))

    def test_network_validation_rejects_self_loop_and_bad_configuration(self) -> None:
        with self.assertRaises(ValueError):
            make_network(0, n_genes=4, n_regulators=4, n_edges=7)
        with self.assertRaises(ValueError):
            make_network(0, n_genes=8, n_regulators=5, n_edges=25)
        with self.assertRaises(ValueError):
            make_network(0, n_genes=8, n_regulators=5, n_edges=9)
        weights = np.zeros((5, 5))
        weights[0, 0] = 1.0
        with self.assertRaises(ValueError):
            NetworkSpec(weights, (0, 1, 2, 3, 4), 0)

    def test_validator_rejects_excess_outdegree_and_uncovered_gene(self) -> None:
        network = make_network(2, n_genes=8, n_regulators=5, n_edges=12)
        weights = network.weights.copy()
        uncovered = network_diagnostics(network)["nonregulators"][0]
        weights[uncovered] = 0
        with self.assertRaisesRegex(ValueError, "non-regulator"):
            validate_network(NetworkSpec(weights, network.regulators, 2), require_motifs=False)

        weights = np.zeros((8, 8))
        for target in range(1, 7):
            weights[target, 0] = 1 if target % 2 else -1
        with self.assertRaisesRegex(ValueError, "outdegree"):
            validate_network(NetworkSpec(weights, (0, 1, 2, 3, 4), 3), require_motifs=False)


class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = make_network(14, n_genes=8, n_regulators=5, n_edges=12)

    def test_positive_unique_steady_state_and_time_zero_semantics(self) -> None:
        for family in ("sigmoid", "hill"):
            with self.subTest(family=family):
                dynamics = make_dynamics(self.network, family, seed=101)
                self.assertLessEqual(dynamics.max_contraction_ratio, 0.65 + 1e-12)
                baseline = dynamics.steady_state()
                target = self.network.regulators[0]
                self.assertGreater(float(np.min(baseline)), 0.0)
                self.assertLess(float(np.max(np.abs(dynamics.vector_field(baseline)))), 1e-9)
                self.assertTrue(np.array_equal(baseline, dynamics.integrate(baseline, 0, target, 0.7)))
                perturbed = dynamics.integrate(baseline, 1.0, target, 0.7)
                self.assertTrue(np.all(perturbed >= 0))
                self.assertFalse(np.allclose(perturbed, baseline))
                equilibrium = dynamics.steady_state(target, 0.7)
                alternate = dynamics.steady_state(target, 0.7, initial=np.full(8, 10.0))
                np.testing.assert_allclose(equilibrium, alternate, atol=1e-9, rtol=0)
                self.assertLess(float(np.max(np.abs(dynamics.vector_field(equilibrium, target, 0.7)))), 1e-9)
                report = dynamics_diagnostics(dynamics)
                self.assertGreaterEqual(report["min_state"], 0)
                self.assertLess(report["step_halving_error"], 1e-4)

    def test_fixed_efficiency_scales_production_at_every_dose(self) -> None:
        for family in ("sigmoid", "hill"):
            dynamics = make_dynamics(self.network, family, seed=13)
            target = self.network.regulators[1]
            x = dynamics.steady_state()
            baseline_rate = dynamics.production(x)
            for dose in (0.3, 0.5, 0.7, 0.9):
                perturbed_rate = dynamics.production(x, target, dose)
                self.assertAlmostEqual(
                    perturbed_rate[target] / baseline_rate[target],
                    1.0 - dynamics.rho[target] * dose,
                    places=13,
                )
                unaffected = np.arange(self.network.n_genes) != target
                np.testing.assert_array_equal(perturbed_rate[unaffected], baseline_rate[unaffected])

    def test_analytic_jacobian_matches_finite_differences_and_edge_signs(self) -> None:
        for family in ("sigmoid", "hill"):
            dynamics = make_dynamics(self.network, family, seed=38)
            x = dynamics.steady_state()
            target = self.network.regulators[2]
            analytic = dynamics.jacobian(x, target, 0.5)
            epsilon = 1e-6
            finite = np.zeros_like(analytic)
            for gene in range(self.network.n_genes):
                shift = np.zeros_like(x)
                shift[gene] = epsilon
                finite[:, gene] = (
                    dynamics.vector_field(x + shift, target, 0.5)
                    - dynamics.vector_field(x - shift, target, 0.5)
                ) / (2 * epsilon)
            np.testing.assert_allclose(analytic, finite, atol=1e-8, rtol=1e-7)
            regulatory = analytic + np.diag(dynamics.gamma)
            edges = self.network.weights != 0
            np.testing.assert_array_equal(
                np.sign(regulatory[edges]), np.sign(self.network.weights[edges])
            )

    def test_batch_integral_and_input_safety(self) -> None:
        dynamics = make_dynamics(self.network, "sigmoid", seed=22)
        x = dynamics.steady_state()
        cells = np.stack([x, 0.8 * x, 1.2 * x])
        target = self.network.regulators[0]
        actual = dynamics.integrate(cells, 0.3, target, 0.7)
        expected = np.stack([dynamics.integrate(cell, 0.3, target, 0.7) for cell in cells])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        with self.assertRaises(ValueError):
            dynamics.integrate(x, -0.1, target, 0.7)
        with self.assertRaises(ValueError):
            dynamics.production(x, None, 0.7)
        with self.assertRaises(ValueError):
            dynamics.vector_field(x, target, 1.1)
        with self.assertRaises(ValueError):
            dynamics.integrate(-x, 0.1)


if __name__ == "__main__":
    unittest.main()
