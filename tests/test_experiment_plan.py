"""Design-only tests: no simulator, generated cells, or model training."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_experiment.experiment_plan import (  # noqa: E402
    PlanValidationError,
    _no_duplicate_keys,
    formal_readiness_errors,
    load_experiment_plan,
    main,
    validate_plan,
)
from grn_experiment import protocol as protocol_api  # noqa: E402


CONFIG = ROOT / "configs" / "scmultisim-main.json"


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_experiment_plan(CONFIG)

    def test_registered_plan_is_structurally_valid_but_not_formally_ready(self) -> None:
        self.assertEqual(self.plan["schema_version"], "scmultisim-v1")
        self.assertEqual(self.plan["status"], "approved_design_not_calibrated")
        errors = formal_readiness_errors(self.plan)
        self.assertIn("calibration.mapping is missing", errors)
        self.assertIn("calibration.K is missing", errors)
        self.assertIn("calibration.final_tau is missing", errors)
        self.assertIn("calibration.burnin is missing", errors)
        self.assertIn("calibration.parameter_bounds is missing", errors)
        self.assertIn("calibration.compatibility_thresholds is missing", errors)
        self.assertIn("calibration.model_config is missing", errors)
        self.assertIn("calibration.runtime_checks is missing", errors)
        self.assertIn("evidence.baseline_archive is missing", errors)

    def test_primary_counts_and_costs_are_registered(self) -> None:
        protocol = self.plan["main_protocol"]
        split = self.plan["target_split"]
        self.assertEqual(split["initial"] * len(protocol["initial_doses"]) * len(protocol["initial_times"]), 36)
        self.assertEqual(split["active"] * len(protocol["candidate_doses"]) * len(protocol["candidate_times"]), 90)
        self.assertEqual(split["validation"] * len(protocol["validation_doses"]) * len(protocol["validation_times"]), 12)
        self.assertEqual(protocol["replicates"] * protocol["cells_per_replicate_per_arm"], 384)
        self.assertEqual(2 * protocol["replicates"] * protocol["cells_per_replicate_per_arm"], 768)
        self.assertEqual(protocol["all_steady_observations"], "sealed_for_final_scoring_only")

    def test_registered_grids_match_protocol_keys_without_sampling(self) -> None:
        design = self.plan["main_protocol"]
        split = protocol_api.make_split(tuple(range(16)), seed=17)
        self.assertEqual(
            tuple(map(len, (split.initial, split.active, split.validation, split.test))),
            (6, 6, 2, 2),
        )
        self.assertEqual(protocol_api.N_GENES, self.plan["network"]["genes"])
        self.assertEqual(protocol_api.N_REGULATORS, self.plan["network"]["perturbable_regulators"])
        self.assertEqual(protocol_api.N_REPLICATES, design["replicates"])
        self.assertEqual(protocol_api.CELLS_PER_REPLICATE, design["cells_per_replicate_per_arm"])
        self.assertEqual(protocol_api.ACTIVE_BUDGET, design["active_conditions_per_strategy"])
        self.assertEqual(protocol_api.COMMON_BASELINE.time, design["common_preintervention_baseline_time"])

        for api_grid, group, doses, times, count in (
            (protocol_api.initial_grid, split.initial, "initial_doses", "initial_times", 36),
            (protocol_api.candidate_grid, split.active, "candidate_doses", "candidate_times", 90),
            (protocol_api.validation_grid, split.validation, "validation_doses", "validation_times", 12),
        ):
            with self.subTest(grid=api_grid.__name__):
                expected = {
                    protocol_api.Condition(target, dose, time)
                    for target in group
                    for dose in design[doses]
                    for time in design[times]
                }
                actual = api_grid(split)
                self.assertEqual(len(actual), count)
                self.assertEqual(set(actual), expected)

        expected_final: set[protocol_api.Condition] = set()
        for category in design["final_test_grids"].values():
            for target in getattr(split, category["target_group"]):
                for dose in category["doses"]:
                    for time in category["times"]:
                        expected_final.add(protocol_api.Condition(target, dose, time))
        expected_final.update(
            protocol_api.Condition(None, 0, time) for time in design["sealed_control_times"]
        )
        actual_final = protocol_api.required_test_conditions(split)
        self.assertEqual(len(actual_final), 24)
        self.assertEqual(set(actual_final), expected_final)
        self.assertEqual(set(protocol_api.SEALED_TIMES), set(design["sealed_control_times"]))

        expected_stationary = {
            protocol_api.SteadyKey(condition.target, condition.dose)
            for condition in expected_final
            if condition.target is not None
        }
        expected_stationary.add(protocol_api.UNTREATED_STATIONARY)
        actual_stationary = protocol_api.stationary_grid(split)
        self.assertEqual(len(actual_stationary), 17)
        self.assertEqual(set(actual_stationary), expected_stationary)

    def test_unknown_keys_are_rejected_at_every_level(self) -> None:
        for path in (
            ("new_root",),
            ("network", "new_network_field"),
            ("main_protocol", "free_steady"),
            ("main_protocol", "final_test_grids", "new_target", "new_field"),
            ("calibration", "new_calibration"),
            ("evidence", "new_evidence"),
        ):
            with self.subTest(path=path):
                fake = deepcopy(self.plan)
                section = fake
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = "not registered"
                with self.assertRaises(PlanValidationError):
                    validate_plan(fake)

    def test_grid_budget_and_steady_policy_cannot_change_silently(self) -> None:
        changes = (
            ("candidate_condition_count", 108),
            ("candidate_times", [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]),
            ("active_conditions_per_strategy", 13),
            ("all_steady_observations", "free_with_first_timepoint"),
            ("efficiency_range", [0.7, 1.0]),
            ("cells_per_replicate_per_arm", 64),
        )
        for key, value in changes:
            with self.subTest(key=key):
                fake = deepcopy(self.plan)
                fake["main_protocol"][key] = value
                with self.assertRaises(PlanValidationError):
                    validate_plan(fake)

    def test_generator_identity_and_no_euler_step_are_fixed(self) -> None:
        for section, key, value in (
            ("generator", "feedback_tau_candidate", 0.125),
            ("generator", "euler_step", 0.01),
            ("generator", "regulator_scale_reference", "blind_controls"),
            ("supplementary_protocol", "status", "enabled"),
        ):
            with self.subTest(section=section, key=key):
                fake = deepcopy(self.plan)
                fake[section][key] = value
                with self.assertRaises(PlanValidationError):
                    validate_plan(fake)

    def test_false_claim_of_frozen_status_does_not_pass_formal_gate(self) -> None:
        fake = deepcopy(self.plan)
        fake["status"] = "frozen_for_blind"
        self.assertTrue(formal_readiness_errors(fake))
        fake["status"] = ["frozen_for_blind"]
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)

    def test_calibration_must_have_valid_shape_not_just_a_boolean(self) -> None:
        fake = deepcopy(self.plan)
        fake["calibration"]["K"] = True
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)
        fake = deepcopy(self.plan)
        fake["calibration"]["final_tau"] = 0.25
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)
        fake = deepcopy(self.plan)
        fake["calibration"]["mapping"] = {"passed": True}
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)

    def test_K_is_one_common_positive_scalar(self) -> None:
        self.assertEqual(
            self.plan["generator"]["regulator_scale_scope"],
            "one_shared_scalar_all_regulators_and_networks",
        )
        fake = deepcopy(self.plan)
        fake["calibration"]["K"] = 1.25
        validate_plan(fake)
        for invalid in ([1.25] * 16, 0, -0.1, True, float("nan")):
            with self.subTest(invalid=invalid):
                fake["calibration"]["K"] = invalid
                with self.assertRaises(PlanValidationError):
                    validate_plan(fake)

    def test_mapping_rejects_flat_response_segment(self) -> None:
        fake = deepcopy(self.plan)
        fake["calibration"]["mapping"] = {
            "kind": "positive_monotone_lookup",
            "x_knots": [-1.0, 0.0, 1.0],
            "y_knots": [0.1, 0.2, 0.3],
        }
        validate_plan(fake)
        fake["calibration"]["mapping"]["y_knots"] = [0.1, 0.2, 0.2]
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)

    def test_parameter_bounds_allow_signed_background_but_not_narrowed_rho(self) -> None:
        fake = deepcopy(self.plan)
        fake["calibration"]["parameter_bounds"] = {
            "effective_A": [-1.2, 1.2],
            "b": [-2.0, 2.0],
            "kon": [0.1, 3.0],
            "koff": [0.1, 3.0],
            "beta": [0.1, 3.0],
            "gamma": [0.1, 3.0],
            "rho": [0.6, 1.0],
        }
        validate_plan(fake)
        fake["calibration"]["parameter_bounds"]["rho"] = [0.7, 1.0]
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)
        fake["calibration"]["parameter_bounds"]["rho"] = [0.6, 1.0]
        fake["calibration"]["parameter_bounds"]["kon"] = [0.0, 3.0]
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)

    def test_evidence_reference_needs_path_and_sha256(self) -> None:
        fake = deepcopy(self.plan)
        fake["evidence"]["mapping"] = {"path": "../outside.json", "sha256": "0" * 64}
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)
        fake["evidence"]["mapping"] = {"path": "docs/validation/mapping.json", "sha256": "short"}
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)
        fake["evidence"]["mapping"] = {"path": "docs/./mapping.json", "sha256": "0" * 64}
        with self.assertRaises(PlanValidationError):
            validate_plan(fake)

    def test_existing_evidence_file_with_correct_sha_has_no_reference_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "docs" / "validation" / "mapping.txt"
            artifact.parent.mkdir(parents=True)
            payload = b"development mapping evidence fixture"
            artifact.write_bytes(payload)
            fake = deepcopy(self.plan)
            fake["evidence"]["mapping"] = {
                "path": "docs/validation/mapping.txt",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            errors = formal_readiness_errors(fake, evidence_root=root)
            self.assertFalse(any(error.startswith("evidence.mapping") for error in errors))
            self.assertIn("calibration.mapping is missing", errors)

    def test_existing_evidence_file_with_wrong_sha_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mapping.txt"
            artifact.write_bytes(b"actual evidence")
            fake = deepcopy(self.plan)
            fake["evidence"]["mapping"] = {"path": "mapping.txt", "sha256": "0" * 64}
            errors = formal_readiness_errors(fake, evidence_root=root)
            self.assertIn("evidence.mapping SHA-256 does not match", errors)

    def test_missing_evidence_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = deepcopy(self.plan)
            fake["evidence"]["mapping"] = {"path": "missing.txt", "sha256": "0" * 64}
            errors = formal_readiness_errors(fake, evidence_root=directory)
            self.assertIn("evidence.mapping file is missing or outside the evidence root", errors)

    def test_evidence_symlink_outside_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            target = Path(outside) / "evidence.txt"
            payload = b"outside evidence"
            target.write_bytes(payload)
            (root / "linked.txt").symlink_to(target)
            fake = deepcopy(self.plan)
            fake["evidence"]["mapping"] = {
                "path": "linked.txt",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            errors = formal_readiness_errors(fake, evidence_root=root)
            self.assertIn("evidence.mapping file is missing or outside the evidence root", errors)

    def test_json_duplicate_key_is_rejected(self) -> None:
        with self.assertRaises(PlanValidationError):
            json.loads('{"status":"a","status":"b"}', object_pairs_hook=_no_duplicate_keys)

    def test_cli_is_design_only_and_formal_flag_rejects_current_plan(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--config", str(CONFIG)]), 0)
        self.assertIn("No data generation, training, or scientific validation", output.getvalue())
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--config", str(CONFIG), "--require-formal"]), 2)
        self.assertIn("calibration.mapping is missing", output.getvalue())


if __name__ == "__main__":
    unittest.main()
