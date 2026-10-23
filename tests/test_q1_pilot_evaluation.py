"""Fit/check barriers and unpaired descriptive scores of the Q1 pilot."""

import json
import unittest

import numpy as np

from grn_experiment.protocol import Condition, ReplicatedSnapshot
from grn_experiment.q1_pilot_evaluation import (
    DirectOnlyTransferBaseline, LinearEffectBaseline, NoResponseBaseline, aggregate_scores,
    marginal_wasserstein1, score_prediction,
)


def snapshot(effect, *, baseline=(10.0, 10.0, 10.0), poison=0.0, cells=5):
    control = np.broadcast_to(np.array(baseline, dtype=float), (3, cells, len(baseline))).copy()
    treated = control + np.asarray(effect)
    treated[2] += poison
    control[2] += poison / 2
    return ReplicatedSnapshot(treated, control)


def observations(poison=0.0):
    return {
        Condition(0, 0.7, 0.25): snapshot((-1.0, 1.0, 0.0), poison=poison),
        Condition(0, 0.7, 4.0): snapshot((-4.0, 3.0, 0.0), poison=poison),
    }


def transfer_observations(poison=0.0):
    """Two training targets with deliberately different baseline scales."""
    values = {}
    for target, scale, slope in ((0, 2.0, 0.1), (1, 4.0, 0.3)):
        for dose in (0.3, 0.7):
            for time in (0.25, 1.0, 2.0, 4.0, 8.0, 48.0):
                effect = np.full(3, 8.0)
                effect[target] = -scale * slope * time * dose / 0.7
                values[Condition(target, dose, time)] = snapshot(
                    effect, baseline=(100, 100, 100), poison=poison)
    return values


class DirectOnlyTransferTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.array([[8, 16, 28], [12, 24, 32]], dtype=float)

    def test_unseen_target_receives_mean_standardized_self_effect_only(self):
        model = DirectOnlyTransferBaseline.fit(self.baseline, transfer_observations(), seed=4)
        treated, control = model.predict(Condition(2, 0.7, 1))
        # std=(2,4,2); mean training effect=(-.1-.3)/2=-.2 std units.
        np.testing.assert_allclose(treated[:, 2] - control[:, 2], -0.4)
        np.testing.assert_array_equal(treated[:, :2], control[:, :2])
        self.assertEqual(model.prediction_note(Condition(2, 0.7, 1)), "direct_only_standardized_transfer")

    def test_third_replicate_poisoning_has_no_effect(self):
        clean = DirectOnlyTransferBaseline.fit(self.baseline, transfer_observations(), seed=4)
        poison = DirectOnlyTransferBaseline.fit(self.baseline, transfer_observations(10000), seed=4)
        for time in (0.125, 1, 3, 48, 96):
            for left, right in zip(clean.predict(Condition(2, 0.7, time)), poison.predict(Condition(2, 0.7, time))):
                np.testing.assert_array_equal(left, right)

    def test_time_interpolation_zero_anchor_and_endpoint_hold(self):
        model = DirectOnlyTransferBaseline.fit(self.baseline, transfer_observations())
        for time in (0.125, 3):
            treated, control = model.predict(Condition(2, 0.7, time))
            np.testing.assert_allclose(treated[:, 2] - control[:, 2], -0.4 * time)
        np.testing.assert_array_equal(model.predict(Condition(2, 0.7, 96))[0],
                                      model.predict(Condition(2, 0.7, 48))[0])
        self.assertEqual(model.prediction_note(Condition(2, 0.7, 96)), "direct_only_transfer_endpoint_hold")

    def test_unseen_dose_and_untreated_condition_are_no_response(self):
        model = DirectOnlyTransferBaseline.fit(self.baseline, transfer_observations())
        for condition, note in ((Condition(2, 0.9, 4), "unseen_dose_no_response"),
                                (Condition(None, 0, 0), "untreated_baseline")):
            treated, control = model.predict(condition)
            np.testing.assert_array_equal(treated, control)
            self.assertEqual(model.prediction_note(condition), note)

    def test_requires_complete_common_grid_for_every_target_dose(self):
        missing_time = transfer_observations()
        del missing_time[Condition(1, 0.3, 2)]
        missing_dose = {key: value for key, value in transfer_observations().items()
                        if not (key.target == 1 and key.dose == 0.3)}
        for incomplete in (missing_time, missing_dose):
            with self.assertRaisesRegex(ValueError, "complete common time grid"):
                DirectOnlyTransferBaseline.fit(self.baseline, incomplete)

    def test_rejects_empty_input_bad_gene_dimension_and_missing_control(self):
        with self.assertRaisesRegex(ValueError, "training target"):
            DirectOnlyTransferBaseline.fit(self.baseline, {})
        with self.assertRaisesRegex(ValueError, "gene dimension"):
            DirectOnlyTransferBaseline.fit(np.ones((5, 2)), transfer_observations())
        with self.assertRaisesRegex(ValueError, "outside"):
            DirectOnlyTransferBaseline.fit(self.baseline, {Condition(3, 0.7, 1): snapshot((0, 0, 0))})
        with self.assertRaisesRegex(ValueError, "matched controls"):
            DirectOnlyTransferBaseline.fit(self.baseline, {
                Condition(0, 0.7, 1): ReplicatedSnapshot(np.ones((3, 5, 3)))})

    def test_validates_registered_grid_and_clamps_small_baseline_scales(self):
        for times in ((), (1, 1), (2, 1), (0, 1), (float("nan"),), (True,)):
            with self.assertRaises(ValueError):
                DirectOnlyTransferBaseline.fit(self.baseline, transfer_observations(), times=times)
        model = DirectOnlyTransferBaseline.fit(np.full((4, 3), 100.0), transfer_observations())
        treated, control = model.predict(Condition(2, 0.7, 1))
        np.testing.assert_allclose(treated[:, 2] - control[:, 2], -0.7)
        with self.assertRaisesRegex(ValueError, "outside"):
            model.predict(Condition(3, 0.7, 1))


class PilotBaselineTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.full((10, 3), 10.0)

    def test_linear_fit_ignores_poisoned_third_replicate(self):
        clean = LinearEffectBaseline.fit(self.baseline, observations(), cells=13, seed=91)
        poisoned = LinearEffectBaseline.fit(self.baseline, observations(10000), cells=13, seed=91)
        for time in (0.125, 0.25, 1, 4, 8):
            for actual, expected in zip(poisoned.predict(Condition(0, 0.7, time)), clean.predict(Condition(0, 0.7, time))):
                np.testing.assert_array_equal(actual, expected)

    def test_interpolates_at_time_one_with_zero_anchor(self):
        model = LinearEffectBaseline.fit(self.baseline, observations(), cells=7)
        treated, control = model.predict(Condition(0, 0.7, 1.0))
        np.testing.assert_allclose(treated.mean(0) - control.mean(0), (-1.6, 1.4, 0.0))
        early, ctrl = model.predict(Condition(0, 0.7, 0.125))
        np.testing.assert_allclose(early.mean(0) - ctrl.mean(0), (-0.5, 0.5, 0.0))

    def test_unseen_target_and_dose_are_explicit_no_response(self):
        model = LinearEffectBaseline.fit(self.baseline, observations())
        for condition, expected_note in (
            (Condition(1, 0.7, 1), "unseen_target_no_response"),
            (Condition(0, 0.3, 1), "unseen_dose_no_response"),
        ):
            treated, control = model.predict(condition)
            np.testing.assert_array_equal(treated, control)
            self.assertEqual(model.prediction_note(condition), expected_note)

    def test_late_time_holds_endpoint_and_names_the_limitation(self):
        model = LinearEffectBaseline.fit(self.baseline, observations())
        np.testing.assert_array_equal(model.predict(Condition(0, 0.7, 8))[0], model.predict(Condition(0, 0.7, 4))[0])
        self.assertEqual(model.prediction_note(Condition(0, 0.7, 8)), "late_time_endpoint_hold")

    def test_clipping_changes_realized_mean_effect_not_hidden_negative_counts(self):
        model = LinearEffectBaseline.fit(np.full((6, 3), 0.1), observations(), cells=6)
        treated, control = model.predict(Condition(0, 0.7, 4))
        self.assertTrue((treated >= 0).all())
        self.assertAlmostEqual(treated[:, 0].mean() - control[:, 0].mean(), -0.1)
        np.testing.assert_allclose(treated[:, 1], 3.1)

    def test_no_response_never_reads_observation_replicates(self):
        model = NoResponseBaseline.fit(self.baseline, observations(10000), cells=9)
        treated, control = model.predict(Condition(0, 0.7, 1))
        np.testing.assert_array_equal(treated, control)
        self.assertEqual(treated.shape, (9, 3))
        treated[:] = 0
        self.assertTrue((model.predict(Condition(0, 0.7, 1))[0] == 10).all())

    def test_rejects_interpolation_check_time_in_fit_data(self):
        data = observations()
        data[Condition(0, 0.7, 1)] = snapshot((-2, 2, 0))
        with self.assertRaisesRegex(ValueError, "only targeted"):
            LinearEffectBaseline.fit(self.baseline, data)

    def test_requires_both_fit_endpoints_for_every_target_dose(self):
        with self.assertRaisesRegex(ValueError, "both"):
            LinearEffectBaseline.fit(self.baseline, {Condition(0, 0.7, 0.25): snapshot((-1, 1, 0))})

    def test_validates_gene_dimension_and_target(self):
        with self.assertRaisesRegex(ValueError, "gene dimension"):
            LinearEffectBaseline.fit(np.ones((3, 2)), observations())
        model = NoResponseBaseline.fit(self.baseline)
        with self.assertRaisesRegex(ValueError, "outside"):
            model.predict(Condition(3, 0.7, 1))


class PilotScoresTests(unittest.TestCase):
    def test_exact_prediction_zero_errors_with_unequal_population_sizes(self):
        condition = Condition(0, 0.7, 1)
        observed = snapshot((-2, 1, 0), cells=5)
        result = score_prediction(np.tile((8, 11, 10), (9, 1)), np.full((7, 3), 10), condition, observed)
        for name in ("mean_rmse", "delta_rmse_all", "delta_rmse_target", "target_effect_mae", "delta_rmse_non_target", "treated_marginal_wasserstein1", "control_marginal_wasserstein1"):
            self.assertEqual(result[name], 0.0)
        self.assertEqual(result["observed_abs_target_effect"], 2.0)
        self.assertEqual(result["pred_target_effect"], -2.0)
        json.dumps(result, allow_nan=False)

    def test_all_target_non_target_effect_errors_are_distinct(self):
        result = score_prediction(np.full((4, 3), 10), np.full((4, 3), 10), Condition(0, 0.7, 1), snapshot((-2, 1, 0)))
        self.assertAlmostEqual(result["delta_rmse_all"], np.sqrt(5 / 3))
        self.assertEqual(result["delta_rmse_target"], 2)
        self.assertEqual(result["target_effect_mae"], result["delta_rmse_target"])
        self.assertAlmostEqual(result["delta_rmse_non_target"], np.sqrt(1 / 2))
        self.assertEqual(result["pred_target_effect"], 0)

    def test_independent_row_permutations_preserve_every_score(self):
        rng = np.random.default_rng(13)
        treated = rng.integers(0, 20, (3, 11, 4))
        control = rng.integers(0, 20, (3, 11, 4))
        pred_t = rng.integers(0, 20, (7, 4))
        pred_c = rng.integers(0, 20, (13, 4))
        condition = Condition(2, 0.7, 1)
        first = score_prediction(pred_t, pred_c, condition, ReplicatedSnapshot(treated, control))
        second = score_prediction(pred_t[rng.permutation(7)], pred_c[rng.permutation(13)], condition,
                                  ReplicatedSnapshot(treated[:, rng.permutation(11)], control[:, rng.permutation(11)]))
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            if isinstance(first[name], float):
                self.assertAlmostEqual(first[name], second[name])
            else:
                self.assertEqual(first[name], second[name])

    def test_defaults_to_check_replicate(self):
        observed = snapshot((0, 0, 0), poison=4)
        prediction = np.full((8, 3), 10)
        result = score_prediction(prediction, prediction, Condition(0, 0.7, 1), observed)
        self.assertEqual(result["mean_rmse"], 4)
        self.assertEqual(result["delta_rmse_all"], 2)
        self.assertEqual(score_prediction(prediction, prediction, Condition(0, 0.7, 1), observed, replicate=0)["mean_rmse"], 0)

    def test_single_gene_non_target_error_is_not_applicable(self):
        observed = snapshot((-2,), baseline=(10,), cells=3)
        result = score_prediction(np.ones((2, 1)) * 8, np.ones((2, 1)) * 10, Condition(0, 0.7, 1), observed)
        self.assertIsNone(result["delta_rmse_non_target"])
        report = aggregate_scores([result])
        self.assertNotIn("delta_rmse_non_target", report["mean_metrics"])
        json.dumps(report, allow_nan=False)

    def test_empirical_distance_with_unequal_sample_sizes_is_exact(self):
        self.assertEqual(marginal_wasserstein1(np.array([[0], [2]]), np.array([[1]])), 1)
        self.assertAlmostEqual(marginal_wasserstein1(np.array([[0], [3], [6]]), np.array([[1], [5]])), 4 / 3)

    def test_fixed_directions_detect_covariance_difference(self):
        observed_t = np.tile(np.array([[0, 0], [2, 2]])[None], (3, 1, 1))
        observed = ReplicatedSnapshot(observed_t, observed_t)
        opposite = np.array([[0, 2], [2, 0]])
        result = score_prediction(opposite, opposite, Condition(0, 0.7, 1), observed,
                                  directions=np.array([[1, 1]]))
        self.assertEqual(result["treated_marginal_wasserstein1"], 0)
        self.assertGreater(result["treated_sliced_wasserstein1"], 0)

    def test_invalid_projection_direction_is_rejected(self):
        prediction = np.full((4, 3), 10)
        with self.assertRaisesRegex(ValueError, "zero norm"):
            score_prediction(prediction, prediction, Condition(0, 0.7, 1), snapshot((0, 0, 0)), directions=np.zeros((2, 3)))

    def test_aggregate_is_condition_average_without_inferential_statistics(self):
        report = aggregate_scores([{"mean_rmse": 1.0, "delta_rmse_target": None}, {"mean_rmse": 3.0, "delta_rmse_target": 4.0}])
        self.assertEqual(report["n_conditions"], 2)
        self.assertEqual(report["mean_metrics"], {"mean_rmse": 2.0, "delta_rmse_target": 4.0})
        self.assertEqual(report["metric_condition_counts"]["delta_rmse_target"], 1)

    def test_target_effect_mae_alias_preserves_old_metric_values(self):
        scores = [score_prediction(np.full((4, 3), 10), np.full((4, 3), 10),
                                   Condition(0, 0.7, 1), snapshot((-effect, 0, 0)))
                  for effect in (1, 3)]
        report = aggregate_scores(scores)
        self.assertEqual(report["mean_metrics"]["target_effect_mae"], 2.0)
        self.assertEqual(report["mean_metrics"]["target_effect_mae"], report["mean_metrics"]["delta_rmse_target"])
        untreated = score_prediction(np.full((4, 3), 10), np.full((4, 3), 10),
                                     Condition(None, 0, 1), snapshot((0, 0, 0)))
        self.assertIsNone(untreated["target_effect_mae"])


if __name__ == "__main__":
    unittest.main()
