"""Tiny snapshot checks; the formal 4+20 network experiment is not run."""

import unittest

import numpy as np

from grn_experiment.observations import (
    ObservationConfig,
    SnapshotSampler,
    protocol_conditions,
)
from grn_experiment.protocol import COMMON_BASELINE, Condition, make_split
from grn_experiment.simulation import make_dynamics, make_network


class SnapshotSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        network = make_network(4, n_genes=8, n_regulators=5, n_edges=12)
        cls.dynamics = make_dynamics(network, "sigmoid", seed=5)
        cls.target_a, cls.target_b = network.regulators[:2]
        cls.config = ObservationConfig(
            cells_per_replicate=4,
            cell_log_sd=0.2,
            replicate_log_sd=0.08,
            observation_log_sd=0.05,
            integration_step=0.1,
        )

    def test_common_baseline_once_and_nonnegative_shape(self) -> None:
        sampler = SnapshotSampler(self.dynamics, seed=9, config=self.config)
        condition = Condition(self.target_a, 0.7, 0.25)
        snapshots = sampler.sample_many([COMMON_BASELINE, condition, COMMON_BASELINE])
        self.assertEqual(tuple(snapshots), (COMMON_BASELINE, condition))
        baseline = snapshots[COMMON_BASELINE]
        treated = snapshots[condition]
        self.assertIsNone(baseline.matched_control)
        self.assertEqual(baseline.treated.shape, (3, 4, 8))
        self.assertEqual(treated.treated.shape, (3, 4, 8))
        self.assertEqual(treated.matched_control.shape, (3, 4, 8))
        self.assertTrue(np.all(baseline.treated >= 0))
        self.assertTrue(np.all(treated.treated >= 0))
        self.assertTrue(np.all(treated.matched_control >= 0))

    def test_condition_order_does_not_change_shared_outcomes(self) -> None:
        first = Condition(self.target_a, 0.3, 0.25)
        second = Condition(self.target_b, 0.7, 0.5)
        sampler_a = SnapshotSampler(self.dynamics, seed=23, config=self.config)
        sampler_b = SnapshotSampler(self.dynamics, seed=23, config=self.config)
        forward = sampler_a.sample_many([first, second])
        reverse = sampler_b.sample_many([second, first])
        for condition in (COMMON_BASELINE, first, second):
            np.testing.assert_array_equal(
                forward[condition].treated, reverse[condition].treated
            )
            if condition.target is not None:
                np.testing.assert_array_equal(
                    forward[condition].matched_control,
                    reverse[condition].matched_control,
                )

    def test_time_and_treatment_arms_are_unpaired(self) -> None:
        sampler = SnapshotSampler(self.dynamics, seed=7, config=self.config)
        first = sampler.sample_condition(Condition(self.target_a, 0.7, 0.25))
        later = sampler.sample_condition(Condition(self.target_a, 0.7, 1.0))
        another = sampler.sample_condition(Condition(self.target_b, 0.7, 0.25))
        self.assertFalse(np.array_equal(first.treated, first.matched_control))
        self.assertFalse(np.array_equal(first.treated, later.treated))
        self.assertFalse(np.array_equal(first.matched_control, another.matched_control))
        self.assertFalse(np.array_equal(first.treated[0], first.treated[1]))

    def test_hidden_efficiency_is_not_an_observation_field(self) -> None:
        sampler = SnapshotSampler(self.dynamics, seed=5, config=self.config)
        observation = sampler.sample_condition(Condition(self.target_a, 0.5, 0.5))
        self.assertFalse(hasattr(observation, "rho"))
        self.assertFalse(hasattr(observation, "weights"))
        self.assertFalse(hasattr(observation, "initial_cells"))
        invalid_target = next(
            gene for gene in range(self.dynamics.network.n_genes)
            if gene not in self.dynamics.network.regulators
        )
        with self.assertRaises(ValueError):
            sampler.sample_condition(Condition(invalid_target, 0.5, 0.5))

    def test_config_rejects_invalid_noise_and_step(self) -> None:
        with self.assertRaises(ValueError):
            ObservationConfig(cell_log_sd=-0.1)
        with self.assertRaises(ValueError):
            ObservationConfig(observation_log_sd=float("nan"))
        with self.assertRaises(ValueError):
            ObservationConfig(integration_step=0.5)
        with self.assertRaises(ValueError):
            ObservationConfig(cells_per_replicate=2.5)


class ProtocolKeyTests(unittest.TestCase):
    def test_protocol_keys_deduplicate_baseline_and_controls(self) -> None:
        split = make_split(tuple(range(16)), seed=3)
        keys = protocol_conditions(split)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys.count(COMMON_BASELINE), 1)
        self.assertEqual(len(keys), 163)
        self.assertIn(Condition(None, 0, 2), keys)
        self.assertIn(Condition(None, 0, 8), keys)
        self.assertEqual(sum(condition.target is not None and condition.time == 0 for condition in keys), 0)


if __name__ == "__main__":
    unittest.main()
