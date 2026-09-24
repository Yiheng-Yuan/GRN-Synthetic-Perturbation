"""Small, non-experimental checks for the three comparator interfaces."""

import unittest

import numpy as np
import torch

from grn_experiment.baselines import (
    BlackBoxRNAODE,
    EndpointResponsePredictor,
    KnownFormRNAODE,
    fit_blackbox_snapshots,
    fit_known_form_snapshots,
)
from grn_experiment.learning import TrainConfig
from grn_experiment.protocol import Condition, ReplicatedSnapshot


def _fixture(check_value: float = 1.0) -> tuple[ReplicatedSnapshot, dict[Condition, ReplicatedSnapshot]]:
    baseline = ReplicatedSnapshot(np.full((3, 4, 3), 1.0))
    treated = np.full((3, 4, 3), 1.0)
    control = np.full((3, 4, 3), 1.0)
    treated[:2, :, 0] = 0.8
    treated[2] = check_value
    return baseline, {Condition(0, 0.3, 0.25): ReplicatedSnapshot(treated, control)}


class EndpointTests(unittest.TestCase):
    def test_latest_released_endpoint_and_unseen_target_fallback(self):
        _, observed = _fixture()
        treated_late = np.full((3, 4, 3), 1.0)
        treated_late[:2, :, 0] = 0.7
        observed[Condition(0, 0.3, 4.0)] = ReplicatedSnapshot(
            treated_late, np.full((3, 4, 3), 1.0)
        )
        model = EndpointResponsePredictor().fit(observed)
        baseline_cells = np.ones((5, 3))
        predicted = model.predict(baseline_cells, Condition(0, 0.3, 1.0))
        np.testing.assert_allclose(predicted[:, 0], 0.7)
        self.assertEqual(model.seen_targets, frozenset({0}))
        np.testing.assert_allclose(
            model.predict(baseline_cells, Condition(2, 0.7, 8.0)), baseline_cells
        )

    def test_check_replicate_is_not_used(self):
        baseline, first = _fixture(check_value=0.0)
        _, second = _fixture(check_value=100.0)
        cells = baseline.fit_treated.reshape(-1, 3)
        left = EndpointResponsePredictor().fit(first).predict(cells, Condition(0, 0.3, 1.0))
        right = EndpointResponsePredictor().fit(second).predict(cells, Condition(0, 0.3, 1.0))
        np.testing.assert_allclose(left, right)


class ODEComparatorTests(unittest.TestCase):
    def test_blackbox_and_known_form_are_positive_and_preintervention_at_zero(self):
        x = torch.ones(4, 3)
        for model in (BlackBoxRNAODE(3, hidden=4), KnownFormRNAODE("sigmoid", 3), KnownFormRNAODE("hill", 3)):
            torch.testing.assert_close(model.integrate(x, 0, 0, 0.7), x)
            self.assertTrue(bool(torch.all(model.synthesis(x) > 0)))
            treated = model.integrate(x, 0.25, 0, 0.7, step=0.25)
            self.assertTrue(bool(torch.all(treated >= 0)))
            self.assertGreaterEqual(float(model.efficiencies.detach().min()), model.min_efficiency)
            self.assertLessEqual(float(model.efficiencies.detach().max()), 1.0)

    def test_known_form_equations_match_declared_forms(self):
        x = torch.tensor([[0.5, 1.0, 2.0]], dtype=torch.float32)
        for family in ("sigmoid", "hill"):
            model = KnownFormRNAODE(family, 3)
            with torch.no_grad():
                model.weights.copy_(torch.tensor([[0.0, 0.3, -0.2], [0.4, 0.0, 0.0], [0.0, -0.1, 0.0]]))
                weights = model.edge_weights
                basal = torch.nn.functional.softplus(model.basal_raw) + 1e-8
                amplitude = torch.nn.functional.softplus(model.amplitude_raw) + 1e-8
                coupling = torch.sigmoid(model.coupling_raw)
                if family == "sigmoid":
                    expected = basal + amplitude * torch.sigmoid(
                        model.bias + coupling * ((x - 1.5) @ weights.T)
                    )
                else:
                    k = torch.nn.functional.softplus(model.hill_k_raw) + 1e-8
                    signal = x.square() / (k.square() + x.square())
                    drive = signal @ torch.relu(weights).T + (1 - signal) @ torch.relu(-weights).T
                    expected = basal + amplitude * (0.5 + coupling * drive)
                torch.testing.assert_close(model.synthesis(x), expected)

    def test_fitting_uses_only_fit_replicates(self):
        baseline, first = _fixture(check_value=0.0)
        _, second = _fixture(check_value=100.0)
        config = TrainConfig(epochs=1, conditions_per_epoch=1, cells_per_condition=4, projections=2, step=0.25, seed=11)
        for model_factory, fitter in (
            (lambda: BlackBoxRNAODE(3, hidden=4), fit_blackbox_snapshots),
            (lambda: KnownFormRNAODE("hill", 3), fit_known_form_snapshots),
        ):
            torch.manual_seed(123)
            left = model_factory()
            torch.manual_seed(123)
            right = model_factory()
            first_report = fitter(left, baseline, first, config=config)
            second_report = fitter(right, baseline, second, config=config)
            self.assertEqual(first_report.losses, second_report.losses)
            self.assertEqual(first_report.conditions_seen, 1)


if __name__ == "__main__":
    unittest.main()
