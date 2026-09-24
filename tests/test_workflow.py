import unittest

import numpy as np

from grn_experiment.protocol import (
    ACTIVE_BUDGET,
    COMMON_BASELINE,
    Condition,
    ReplicatedSnapshot,
    candidate_grid,
    initial_grid,
    make_split,
    required_test_conditions,
    validation_grid,
)
from grn_experiment.workflow import (
    CaseWorkflow,
    FAMILIES,
    STRATEGIES,
    StudyManifest,
    make_manifest,
    paired_blind_differences,
)


def small_snapshot(*, target=True):
    treated = np.ones((3, 4, 2), dtype=float)
    control = np.ones_like(treated) if target else None
    return ReplicatedSnapshot(treated, control)


def case_fixture():
    plan = make_manifest(123).development[0]
    split = make_split(tuple(range(16)), plan.split_seed)
    observations = {COMMON_BASELINE: small_snapshot(target=False)}
    for condition in (
        initial_grid(split)
        + candidate_grid(split)
        + validation_grid(split)
        + required_test_conditions(split)
    ):
        observations[condition] = small_snapshot(target=condition.target is not None)
    return plan, split, observations


class ManifestTests(unittest.TestCase):
    def test_manifest_precommits_networks_not_cells(self):
        manifest = make_manifest(42)
        self.assertEqual(manifest, make_manifest(42))
        self.assertIsInstance(manifest, StudyManifest)
        self.assertEqual((len(manifest.development), len(manifest.blind)), (4, 20))
        self.assertEqual(manifest.n_statistical_units, 20)
        self.assertEqual(len({plan.network_seed for plan in manifest.networks}), 24)
        self.assertTrue(all(plan.families == FAMILIES for plan in manifest.networks))
        self.assertTrue(all(plan.role == "development" for plan in manifest.development))
        self.assertTrue(all(plan.role == "blind" for plan in manifest.blind))

    def test_network_paired_score_matrix(self):
        manifest = make_manifest(9)
        scores = {}
        for index, plan in enumerate(manifest.blind):
            for family in FAMILIES:
                scores[(plan.identifier, family, "active")] = index + 2
                scores[(plan.identifier, family, "uniform")] = index + 1
        differences = paired_blind_differences(manifest, scores, "active", "uniform")
        self.assertEqual(differences.shape, (20, 2))
        np.testing.assert_allclose(differences, 1)
        del scores[(manifest.blind[0].identifier, FAMILIES[0], "active")]
        with self.assertRaises(ValueError):
            paired_blind_differences(manifest, scores, "active", "uniform")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.split, self.observations = case_fixture()
        self.case = CaseWorkflow(
            self.plan,
            "sigmoid",
            self.split,
            self.observations,
            {"network-a": frozenset({(1, 0)}), "network-b": frozenset({(2, 0)})},
            n_genes=2,
            cells_per_replicate=4,
        )

    def test_shared_start_distinct_acquisitions_same_cost(self):
        for strategy in STRATEGIES:
            run = self.case.run(strategy)
            self.assertEqual(run.initial_registry_size, 2)
            self.assertEqual(set(run.available_conditions()) & set(candidate_grid(self.split)), set())
            np.testing.assert_allclose(run.get(initial_grid(self.split)[0]).fit_treated, 1)
            for _ in range(ACTIVE_BUDGET):
                run.acquire_next()  # active must record a no-rival fallback
            self.assertEqual(len(set(run.acquired_conditions)), ACTIVE_BUDGET)
            self.assertEqual(run.active_cost.total_cells, ACTIVE_BUDGET * 3 * 4 * 2)
            self.assertTrue(run.validation_open)
        active = self.case.run("active")
        random = self.case.run("random")
        uniform = self.case.run("uniform")
        self.assertEqual(active.acquired_conditions, uniform.acquired_conditions)
        self.assertNotEqual(random.acquired_conditions, uniform.acquired_conditions)
        self.assertEqual(active.fallback_count, ACTIVE_BUDGET)
        self.assertEqual(random.fallback_count, 0)
        self.assertEqual(uniform.fallback_count, 0)

    def test_validation_selection_then_prediction_commitment(self):
        run = self.case.run("uniform")
        validation = validation_grid(self.split)[0]
        test = required_test_conditions(self.split)[0]
        with self.assertRaises(PermissionError):
            run.get(validation)
        with self.assertRaises(PermissionError):
            run.get(test)
        with self.assertRaises(PermissionError):
            run.get(Condition(None, 0, 8))
        with self.assertRaises(RuntimeError):
            run.select_model({"a": 0.2})
        with self.assertRaises(RuntimeError):
            run.lock_predictions({})
        for _ in range(ACTIVE_BUDGET):
            run.acquire_next()
        run.get(validation)
        with self.assertRaises(PermissionError):
            run.get(test)
        self.assertEqual(run.select_model({"model-b": 0.5, "model-a": 0.5}), "model-a")
        with self.assertRaises(ValueError):
            run.lock_predictions({})
        predictions = {condition: np.zeros((4, 2)) for condition in required_test_conditions(self.split)}
        run.lock_predictions(predictions)
        self.assertTrue(run.scoring_open)
        run.get(test)
        run.get(Condition(None, 0, 8))
        self.assertEqual(run.selected_model, "model-a")
        self.assertEqual(run.audit[-1].action, "predictions_locked")
        self.assertEqual(run.audit[-1].detail, "model=model-a;count=24")

    def test_rival_denominator_and_audit_are_fixed(self):
        run = self.case.run("active")
        run.update_survivors({"network-a"})
        self.assertEqual(run.initial_registry_size, 2)
        self.assertEqual(run.eliminated_fraction, 0.5)
        with self.assertRaises(ValueError):
            run.update_survivors({"network-a", "network-b"})
        run.acquire_next()
        self.assertEqual(run.fallback_count, 1)
        self.assertEqual(run.audit[0].action, "start")
        self.assertEqual(run.audit[-1].action, "acquired")
        self.assertEqual(len(self.case.audit_records()), sum(len(self.case.run(s).audit) for s in STRATEGIES))


if __name__ == "__main__":
    unittest.main()
