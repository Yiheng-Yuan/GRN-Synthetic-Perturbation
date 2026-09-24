import unittest

import numpy as np

from grn_experiment.protocol import (
    ACTIVE_BUDGET,
    COMMON_BASELINE,
    BlindStore,
    Condition,
    ReplicatedSnapshot,
    Split,
    candidate_grid,
    initial_grid,
    make_split,
    required_test_conditions,
    test_grids as final_test_grids,
    validation_grid,
)


def snapshot(value=1.0, *, control=True):
    treated = np.stack([np.full((4, 2), value + i) for i in range(3)])
    matched = treated + 10 if control else None
    return ReplicatedSnapshot(treated, matched)


def fixture():
    split = make_split(tuple(range(16)), 7)
    observations = {COMMON_BASELINE: snapshot(control=False)}
    observations.update({condition: snapshot() for condition in initial_grid(split)})
    observations.update({condition: snapshot() for condition in candidate_grid(split)})
    observations.update({condition: snapshot() for condition in validation_grid(split)})
    observations.update({condition: snapshot() for condition in required_test_conditions(split) if condition.target is not None})
    observations[Condition(None, 0, 2)] = snapshot(control=False)
    observations[Condition(None, 0, 8)] = snapshot(control=False)
    return split, observations


class ConditionAndGridTests(unittest.TestCase):
    def test_time_zero_is_common_baseline_only(self):
        self.assertEqual(COMMON_BASELINE, Condition(None, 0.0, 0.0))
        for target, dose, time in ((1, 0.3, 0), (None, 0.3, 0), (1, 0, 1), (1, 1.2, 1)):
            with self.subTest(target=target, dose=dose, time=time):
                with self.assertRaises(ValueError):
                    Condition(target, dose, time)
        with self.assertRaises(ValueError):
            Condition(1, 0.3, float("nan"))

    def test_split_is_reproducible_and_disjoint(self):
        split = make_split(tuple(range(16)), 17)
        self.assertEqual(split, make_split(list(range(16)), 17))
        self.assertEqual((len(split.initial), len(split.active), len(split.validation), len(split.test)), (6, 6, 2, 2))
        self.assertEqual(set(split.initial + split.active + split.validation + split.test), set(range(16)))
        with self.assertRaises(ValueError):
            make_split([0] * 16, 17)
        with self.assertRaises(ValueError):
            Split(tuple(range(6)), tuple(range(6, 12)), (12, 13), (13, 14))

    def test_fixed_grids_and_sealed_controls(self):
        split = make_split(tuple(range(16)), 17)
        initial = set(initial_grid(split))
        candidates = set(candidate_grid(split))
        validation = set(validation_grid(split))
        grids = final_test_grids(split)
        self.assertEqual((len(initial), len(candidates), len(validation)), (36, 90, 12))
        self.assertEqual({name: len(grid) for name, grid in grids.items()}, {
            "new_target": 2,
            "new_dose": 6,
            "new_time_interp": 6,
            "new_time_extrap": 6,
            "combination": 2,
        })
        self.assertFalse(initial & candidates)
        self.assertFalse(initial & validation)
        self.assertFalse(candidates & validation)
        self.assertNotIn(COMMON_BASELINE, initial | candidates | validation)
        self.assertIn(Condition(None, 0, 2), required_test_conditions(split))
        self.assertIn(Condition(None, 0, 8), required_test_conditions(split))
        self.assertEqual(len(required_test_conditions(split)), 24)


class SnapshotTests(unittest.TestCase):
    def test_replicate_partition_and_shape_validation(self):
        item = snapshot(2.0)
        np.testing.assert_allclose(item.fit_treated[:, 0, 0], [2, 3])
        self.assertEqual(item.check_treated[0, 0], 4)
        np.testing.assert_allclose(item.fit_control[:, 0, 0], [12, 13])
        self.assertEqual(item.check_control[0, 0], 14)
        exposed = item.fit_treated
        exposed[0, 0, 0] = 999
        self.assertEqual(item.treated[0, 0, 0], 2)
        with self.assertRaises(ValueError):
            ReplicatedSnapshot(np.zeros((2, 4, 2)))
        with self.assertRaises(ValueError):
            ReplicatedSnapshot(np.zeros((3, 4, 2)), np.zeros((3, 4, 3)))
        with self.assertRaises(ValueError):
            ReplicatedSnapshot(np.full((3, 4, 2), -1))


class BlindStoreTests(unittest.TestCase):
    def setUp(self):
        self.split, self.observations = fixture()
        self.store = BlindStore(self.split, self.observations, n_genes=2, cells_per_replicate=4)

    def test_no_hidden_data_in_listing_or_get(self):
        candidate = candidate_grid(self.split)[0]
        validation = validation_grid(self.split)[0]
        test = final_test_grids(self.split)["new_target"][0]
        available = set(self.store.available_conditions())
        self.assertIn(COMMON_BASELINE, available)
        self.assertTrue(set(initial_grid(self.split)).issubset(available))
        self.assertNotIn(candidate, available)
        self.assertNotIn(validation, available)
        self.assertNotIn(test, available)
        self.assertNotIn(Condition(None, 0, 2), available)
        self.assertNotIn(Condition(None, 0, 8), available)
        self.assertFalse(hasattr(self.store, "data"))
        self.assertFalse(hasattr(self.store, "observations"))
        for condition in (candidate, validation, test, Condition(None, 0, 2), Condition(None, 0, 8)):
            with self.subTest(condition=condition), self.assertRaises(PermissionError):
                self.store.get(condition)

    def test_acquisition_cost_controls_and_commitment_gate(self):
        candidates = candidate_grid(self.split)
        first = candidates[0]
        early_control = Condition(None, 0, first.time)
        with self.assertRaises(PermissionError):
            self.store.get(early_control)
        acquired = self.store.acquire(first)
        self.assertIsNotNone(acquired.matched_control)
        self.assertEqual(self.store.active_cost.conditions, 1)
        self.assertEqual(self.store.active_cost.treated_cells, 12)
        self.assertEqual(self.store.active_cost.matched_control_cells, 12)
        self.assertEqual(self.store.common_baseline_cells, 12)
        self.assertNotIn(early_control, self.store.available_conditions())
        with self.assertRaises(ValueError):
            self.store.acquire(first)
        with self.assertRaises(ValueError):
            self.store.acquire(initial_grid(self.split)[0])
        for condition in candidates[1:ACTIVE_BUDGET]:
            self.store.acquire(condition)
        self.assertTrue(self.store.validation_open)
        self.assertEqual(self.store.active_cost.total_cells, ACTIVE_BUDGET * 3 * 4 * 2)
        self.assertIn(validation_grid(self.split)[0], self.store.available_conditions())
        with self.assertRaises(RuntimeError):
            self.store.acquire(candidates[ACTIVE_BUDGET])
        with self.assertRaises(ValueError):
            self.store.lock_predictions({})
        with self.assertRaises(ValueError):
            self.store.lock_predictions(
                {condition: None for condition in required_test_conditions(self.split)}
            )
        predictions = {condition: np.zeros((4, 2)) for condition in required_test_conditions(self.split)}
        self.store.lock_predictions(predictions)
        self.assertTrue(self.store.predictions_locked)
        np.testing.assert_allclose(self.store.committed_prediction(Condition(None, 0, 8)), np.zeros((4, 2)))
        predictions[Condition(None, 0, 8)][0] = 99
        np.testing.assert_allclose(self.store.committed_prediction(Condition(None, 0, 8)), np.zeros((4, 2)))
        self.assertIn(Condition(None, 0, 8), self.store.available_conditions())
        self.assertIn(final_test_grids(self.split)["new_target"][0], self.store.available_conditions())
        self.store.get(Condition(None, 0, 8))
        with self.assertRaises(RuntimeError):
            self.store.lock_predictions(predictions)

    def test_store_does_not_share_source_or_returned_arrays(self):
        first = initial_grid(self.split)[0]
        original = self.store.get(first).treated[0, 0, 0]
        self.observations[first].treated.setflags(write=True)
        self.observations[first].treated[0, 0, 0] = 999
        self.assertEqual(self.store.get(first).treated[0, 0, 0], original)
        returned = self.store.get(first)
        returned.treated.setflags(write=True)
        returned.treated[0, 0, 0] = 111
        self.assertEqual(self.store.get(first).treated[0, 0, 0], original)


if __name__ == "__main__":
    unittest.main()
