"""Tiny-array checks of information barriers, not a stationary simulation."""

import unittest

import numpy as np

from grn_experiment.protocol import (
    ACTIVE_BUDGET,
    COMMON_BASELINE,
    UNTREATED_STATIONARY,
    BlindStore,
    Condition,
    ReplicatedSnapshot,
    SteadyKey,
    candidate_grid,
    initial_grid,
    make_split,
    required_test_conditions,
    stationary_grid,
    steady_grid,
    test_grids as final_test_grids,
    validation_grid,
)


def snapshot(value=1.0, *, control=True, cells=4, genes=2):
    treated = np.full((3, cells, genes), value)
    return ReplicatedSnapshot(treated, treated + 1 if control else None)


def fixture(*, extras=False):
    split = make_split(tuple(range(16)), 7)
    conditions = (COMMON_BASELINE,) + initial_grid(split) + candidate_grid(split) + validation_grid(split)
    conditions += required_test_conditions(split)
    observations = {
        condition: snapshot(control=condition.target is not None)
        for condition in conditions
    }
    stationary = {key: snapshot(control=key.target is not None) for key in stationary_grid(split)}
    if extras:
        for grid in (initial_grid(split), candidate_grid(split), validation_grid(split)):
            for condition in grid:
                stationary[SteadyKey(condition.target, condition.dose)] = snapshot()
    return split, observations, stationary


def make_store(split, observations, stationary=None, **kwargs):
    return BlindStore(
        split, observations, n_genes=2, cells_per_replicate=4,
        stationary_observations=stationary, **kwargs,
    )


def complete_time_stage(store):
    for condition in candidate_grid(store.split)[:ACTIVE_BUDGET]:
        store.acquire(condition)
    store.lock_predictions({
        condition: np.zeros((4, 2)) for condition in required_test_conditions(store.split)
    })


def long_predictions(split):
    return {key: np.zeros((4, 2)) for key in stationary_grid(split)}


class StationaryKeyTests(unittest.TestCase):
    def test_stationary_key_has_no_time_and_normalizes_numeric_values(self):
        key = SteadyKey(np.int64(3), np.float64(0.7))
        self.assertEqual(key, SteadyKey(3, 0.7))
        self.assertIs(type(key.target), int)
        self.assertIs(type(key.dose), float)
        self.assertFalse(hasattr(key, "time"))
        self.assertEqual(UNTREATED_STATIONARY, SteadyKey(None, 0))
        self.assertNotEqual(UNTREATED_STATIONARY, COMMON_BASELINE)

    def test_invalid_target_dose_and_control_are_rejected(self):
        cases = [
            (True, 0.3), (-1, 0.3), (2.0, 0.3), ("2", 0.3),
            (1, 0), (1, -0.1), (1, 1.1), (None, 0.3),
            (1, float("nan")), (1, float("inf")), (None, float("inf")),
            (1, True), (1, "0.3"), (1, 0.3 + 0j),
        ]
        for target, dose in cases:
            with self.subTest(target=target, dose=dose), self.assertRaises(ValueError):
                SteadyKey(target, dose)

    def test_stationary_grid_collapses_22_targeted_tests_to_17_keys(self):
        split, _, _ = fixture()
        targeted_conditions = [condition for grid in final_test_grids(split).values() for condition in grid]
        self.assertEqual(len(targeted_conditions), 22)
        expected = {SteadyKey(condition.target, condition.dose) for condition in targeted_conditions}
        self.assertEqual(len(expected), 16)
        self.assertEqual(set(stationary_grid(split)), expected | {UNTREATED_STATIONARY})
        self.assertEqual(len(stationary_grid(split)), 17)
        self.assertEqual(stationary_grid(split), steady_grid(split))
        self.assertEqual(len(required_test_conditions(split)), 24)
        self.assertEqual(len(candidate_grid(split)), 90)
        self.assertEqual(ACTIVE_BUDGET, 12)


class StationaryStoreConstructionTests(unittest.TestCase):
    def test_old_store_without_stationary_data_still_works(self):
        split, observations, _ = fixture()
        store = make_store(split, observations)
        self.assertEqual(store.available_stationary_keys(), ())
        complete_time_stage(store)
        self.assertIn(Condition(None, 0, 8), store.available_conditions())
        with self.assertRaises(RuntimeError):
            store.lock_stationary_predictions(long_predictions(split))

    def test_alias_input_is_supported_but_both_inputs_are_rejected(self):
        split, observations, stationary = fixture()
        store = make_store(split, observations, steady_observations=stationary)
        self.assertEqual(store.available_stationary_keys(), ())
        with self.assertRaises(ValueError):
            make_store(split, observations, stationary, steady_observations=stationary)

    def test_incomplete_stationary_source_cannot_enable_partial_reveal(self):
        split, observations, stationary = fixture()
        for source in ({}, {key: value for key, value in stationary.items() if key != UNTREATED_STATIONARY}):
            with self.subTest(size=len(source)), self.assertRaises(ValueError):
                make_store(split, observations, source)

    def test_outside_protocol_or_wrong_key_type_is_rejected(self):
        split, observations, stationary = fixture()
        for key in (SteadyKey(split.active[0], 0.9), SteadyKey(999, 0.7), COMMON_BASELINE):
            source = stationary | {key: snapshot()}
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_store(split, observations, source)

    def test_stationary_snapshot_type_and_dimensions_are_checked(self):
        split, observations, stationary = fixture()
        key = stationary_grid(split)[0]
        for value in (None, snapshot(genes=3), snapshot(cells=5)):
            source = stationary | {key: value}
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                make_store(split, observations, source)

    def test_targeted_stationary_requires_matching_control(self):
        split, observations, stationary = fixture()
        key = stationary_grid(split)[0]
        with self.assertRaises(ValueError):
            make_store(split, observations, stationary | {key: snapshot(control=False)})

    def test_untreated_reference_must_not_contain_a_second_control(self):
        split, observations, stationary = fixture()
        with self.assertRaises(ValueError):
            make_store(split, observations, stationary | {UNTREATED_STATIONARY: snapshot(control=True)})

    def test_nonfinite_or_negative_mutated_source_is_revalidated(self):
        for arm, invalid in (("treated", -1), ("treated", np.nan), ("matched_control", np.inf)):
            split, observations, stationary = fixture()
            key = stationary_grid(split)[0]
            array = getattr(stationary[key], arm)
            array.setflags(write=True)
            array[0, 0, 0] = invalid
            with self.subTest(arm=arm, invalid=invalid), self.assertRaises(ValueError):
                make_store(split, observations, stationary)

    def test_duplicate_stationary_items_are_rejected(self):
        split, observations, stationary = fixture()

        class RepeatedItems(dict):
            def items(self):
                values = list(super().items())
                return values + [values[0]]

        with self.assertRaises(ValueError):
            make_store(split, observations, RepeatedItems(stationary))


class StationaryGateTests(unittest.TestCase):
    def setUp(self):
        self.split, self.observations, self.stationary = fixture(extras=True)
        self.store = make_store(self.split, self.observations, self.stationary)

    def assert_all_stationary_sealed(self):
        self.assertEqual(self.store.available_stationary_keys(), ())
        for key in self.stationary:
            with self.subTest(key=key), self.assertRaises(PermissionError):
                self.store.steady_get(key)
        self.assertFalse(any(isinstance(key, SteadyKey) for key in self.store.available_conditions()))

    def test_initial_candidate_validation_and_untreated_responses_stay_sealed(self):
        self.assert_all_stationary_sealed()
        for condition in candidate_grid(self.split)[:ACTIVE_BUDGET]:
            self.store.acquire(condition)
            self.assert_all_stationary_sealed()
        self.assertTrue(self.store.validation_open)
        self.store.get(validation_grid(self.split)[0])
        self.assert_all_stationary_sealed()

    def test_stationary_keys_cannot_be_acquired_or_merged_into_time_store(self):
        key = stationary_grid(self.split)[0]
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.store.acquire(key)
        self.assertEqual(self.store.active_cost.conditions, 0)
        time_key = candidate_grid(self.split)[0]
        self.store.acquire(time_key)
        with self.assertRaises(ValueError):
            self.store.acquire(time_key)
        with self.assertRaises(ValueError):
            make_store(self.split, self.observations | {key: snapshot()})

    def test_long_term_lock_requires_all_12_acquisitions(self):
        for condition in candidate_grid(self.split)[:ACTIVE_BUDGET - 1]:
            self.store.acquire(condition)
        with self.assertRaises(RuntimeError):
            self.store.lock_stationary_predictions(long_predictions(self.split))
        self.assertFalse(self.store.stationary_predictions_locked)
        self.assert_all_stationary_sealed()

    def test_long_term_lock_requires_time_prediction_commitment(self):
        for condition in candidate_grid(self.split)[:ACTIVE_BUDGET]:
            self.store.acquire(condition)
        with self.assertRaises(RuntimeError):
            self.store.lock_stationary_predictions(long_predictions(self.split))
        self.assert_all_stationary_sealed()

    def test_time_prediction_lock_alone_never_reveals_stationary_samples(self):
        complete_time_stage(self.store)
        self.assertTrue(self.store.predictions_locked)
        self.assertFalse(self.store.stationary_predictions_locked)
        self.assert_all_stationary_sealed()

    def test_final_time_responses_and_controls_wait_for_both_prediction_locks(self):
        complete_time_stage(self.store)
        final = required_test_conditions(self.split)
        self.assertTrue(self.store.predictions_locked)
        self.assertFalse(set(final) & set(self.store.available_conditions()))
        self.assertIn(Condition(None, 0, 2), final)
        self.assertIn(Condition(None, 0, 8), final)
        for condition in final:
            with self.subTest(condition=condition), self.assertRaises(PermissionError):
                self.store.get(condition)
        # Validation still opens at round 12; only final responses stay sealed.
        self.store.get(validation_grid(self.split)[0])
        self.store.lock_stationary_predictions(long_predictions(self.split))
        self.assertTrue(set(final).issubset(self.store.available_conditions()))
        for condition in final:
            self.store.get(condition)

    def test_missing_extra_and_wrong_prediction_keys_leave_everything_sealed(self):
        complete_time_stage(self.store)
        predictions = long_predictions(self.split)
        missing = {key: value for key, value in predictions.items() if key != UNTREATED_STATIONARY}
        extra = predictions | {SteadyKey(self.split.active[0], 0.3): np.zeros((4, 2))}
        wrong = predictions | {COMMON_BASELINE: np.zeros((4, 2))}
        for values in ({}, missing, extra, wrong):
            with self.subTest(size=len(values)), self.assertRaises(ValueError):
                self.store.lock_stationary_predictions(values)
            self.assertFalse(self.store.stationary_predictions_locked)
            self.assert_all_stationary_sealed()

    def test_invalid_long_term_arrays_do_not_partially_commit(self):
        complete_time_stage(self.store)
        key = stationary_grid(self.split)[0]
        invalid = (
            None, np.zeros(2), np.zeros((0, 2)), np.zeros((4, 3)),
            np.zeros((3, 4, 2)), np.full((4, 2), -1),
            np.full((4, 2), np.nan), np.full((4, 2), np.inf),
            np.full((4, 2), 1 + 1j),
        )
        for value in invalid:
            predictions = long_predictions(self.split) | {key: value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.lock_stationary_predictions(predictions)
            self.assertFalse(self.store.stationary_predictions_locked)
        self.assert_all_stationary_sealed()

    def test_full_lock_exposes_copied_snapshots_without_changing_time_cost(self):
        complete_time_stage(self.store)
        cost = self.store.active_cost
        self.store.lock_steady_predictions(long_predictions(self.split))
        self.assertTrue(self.store.stationary_predictions_locked)
        self.assertEqual(set(self.store.available_stationary_keys()), set(self.stationary))
        self.assertEqual(self.store.active_cost, cost)
        key = stationary_grid(self.split)[0]
        item = self.store.steady_get(key)
        self.assertEqual(item.treated.shape, (3, 4, 2))
        self.assertEqual(item.fit_treated.shape, (2, 4, 2))
        self.assertEqual(item.check_treated.shape, (4, 2))
        self.assertIsNotNone(item.matched_control)
        with self.assertRaises(RuntimeError):
            self.store.lock_stationary_predictions(long_predictions(self.split))

    def test_observations_and_predictions_do_not_alias_source_or_returned_arrays(self):
        complete_time_stage(self.store)
        predictions = long_predictions(self.split)
        with self.assertRaises(PermissionError):
            self.store.committed_stationary_prediction(UNTREATED_STATIONARY)
        self.store.lock_stationary_predictions(predictions)
        key = stationary_grid(self.split)[0]
        self.stationary[key].treated.setflags(write=True)
        self.stationary[key].matched_control.setflags(write=True)
        self.stationary[key].treated[:] = 999
        self.stationary[key].matched_control[:] = 999
        item = self.store.steady_get(key)
        np.testing.assert_array_equal(item.treated, 1)
        np.testing.assert_array_equal(item.matched_control, 2)
        item.treated.setflags(write=True)
        item.matched_control.setflags(write=True)
        item.treated[:] = 888
        item.matched_control[:] = 888
        np.testing.assert_array_equal(self.store.steady_get(key).treated, 1)
        np.testing.assert_array_equal(self.store.steady_get(key).matched_control, 2)
        predictions[key][:] = 777
        stored = self.store.committed_stationary_prediction(key)
        np.testing.assert_array_equal(stored, 0)
        stored[:] = 666
        np.testing.assert_array_equal(self.store.committed_stationary_prediction(key), 0)


if __name__ == "__main__":
    unittest.main()
