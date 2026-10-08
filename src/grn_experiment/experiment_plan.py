"""Structural preflight for the proposed scMultiSim-based study.

This module validates a *design document*.  It neither runs a simulator nor
establishes that a scientific calibration or a blind experiment has passed.
The approved primary protocol is intentionally fixed; changing its grid or
budget requires a new schema version and a new, reviewed study design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA_VERSION = "scmultisim-v1"
STATUSES = {
    "approved_design_not_calibrated",
    "development_calibrated",
    "frozen_for_blind",
}
CALIBRATION_KEYS = (
    "mapping",
    "K",
    "final_tau",
    "burnin",
    "parameter_bounds",
    "compatibility_thresholds",
    "model_config",
    "runtime_checks",
)
EVIDENCE_KEYS = (
    "baseline_archive",
    *CALIBRATION_KEYS,
    "development_acceptance",
    "blind_scorer_isolation",
    "frozen_manifest",
)

# Exact constants guard the original 90-condition primary protocol.  The
# proposed 108-condition, paid-steady extension is separately disabled.
FIXED_DESIGN: dict[str, Any] = {
    "network": {
        "development_count": 4,
        "blind_count": 20,
        "genes": 24,
        "perturbable_regulators": 16,
        "signed_directed_edges": 40,
        "weight_orientation": "target_by_regulator",
    },
    "target_split": {"initial": 6, "active": 6, "validation": 2, "test": 2},
    "main_protocol": {
        "intervention_start": "t=0+",
        "intervention_duration": "sustained",
        "common_preintervention_baseline_time": 0,
        "common_baseline_cells": 384,
        "initial_doses": [0.3, 0.7],
        "initial_times": [0.25, 1.0, 4.0],
        "initial_condition_count": 36,
        "candidate_doses": [0.3, 0.5, 0.7],
        "candidate_times": [0.125, 0.25, 0.5, 1.0, 4.0],
        "candidate_condition_count": 90,
        "validation_doses": [0.3, 0.7],
        "validation_times": [0.25, 1.0, 4.0],
        "validation_condition_count": 12,
        "active_conditions_per_strategy": 12,
        "active_conditions_unique": True,
        "strategies": ["active", "random", "uniform"],
        "strategies_share_initial_data_and_candidate_pool": True,
        "replicates": 3,
        "fit_replicate_indices": [0, 1],
        "check_replicate_index": 2,
        "cells_per_replicate_per_arm": 128,
        "cells_per_targeted_condition": 768,
        "treated_and_control_cells_unpaired": True,
        "timepoints_independently_sampled": True,
        "efficiency_range": [0.6, 1.0],
        "efficiency_sharing": "fixed_per_target_across_dose_time_and_replicates",
        "validation_open_after_acquisitions": 12,
        "final_predictions_lock_before_scoring": True,
        "sealed_control_times": [2.0, 8.0],
        "all_steady_observations": "sealed_for_final_scoring_only",
        "final_test_grids": {
            "new_target": {"target_group": "test", "doses": [0.7], "times": [1.0]},
            "new_dose": {"target_group": "initial", "doses": [0.9], "times": [1.0]},
            "new_time_interpolation": {
                "target_group": "initial", "doses": [0.7], "times": [2.0]
            },
            "new_time_extrapolation": {
                "target_group": "initial", "doses": [0.7], "times": [8.0]
            },
            "combination": {"target_group": "test", "doses": [0.9], "times": [8.0]},
        },
    },
    "supplementary_protocol": {
        "status": "planned_disabled",
        "timecourse_candidates": 90,
        "paid_steady_candidates": 18,
        "total_candidates": 108,
        "steady_free_with_timecourse": False,
        "shares_primary_inference": False,
    },
    "generator": {
        "regulator_scale_reference": "common_development_no_feedback",
        "regulator_scale_scope": "one_shared_scalar_all_regulators_and_networks",
        "mapping": "positive_frozen_pure_function",
        "promoter_events": "exact",
        "rna_between_events": "exact_segment",
        "feedback_update": "synchronous",
        "feedback_tau_candidate": 0.03125,
        "feedback_tau_halving_required": True,
        "euler_step": None,
        "knockdown_location": "target_production_after_mapping_and_scaling",
        "observation_species": "spliced_rna",
        "integerization": "floor_once_at_final_observation",
        "primary_technical_noise": "none",
        "technical_noise": "stress_test_only",
    },
    "research_questions": {
        "q1": {
            "unknown_generator_form": True,
            "latent_dimension_candidates": [48, 96],
            "training_objective": "unpaired_snapshot_distribution_loss",
            "regulatory_mask": "shared_local_mask",
            "generator_family_label_visible": False,
        },
        "q2": {
            "known_architecture": "continuous_time_extension",
            "unknown_parameters": ["effective_A", "b", "kon", "koff", "beta", "gamma", "rho"],
            "network_and_efficiency_truth_visible": False,
        },
    },
}


class PlanValidationError(ValueError):
    """The JSON does not match the registered study-design schema."""


def _object(value: Any, expected_keys: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanValidationError(f"{path} must be an object")
    missing = expected_keys - value.keys()
    unknown = value.keys() - expected_keys
    if missing or unknown:
        raise PlanValidationError(
            f"{path} keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _match_fixed(value: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        obj = _object(value, set(expected), path)
        for key, subexpected in expected.items():
            _match_fixed(obj[key], subexpected, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise PlanValidationError(f"{path} must equal the registered list {expected!r}")
        for index, (item, subexpected) in enumerate(zip(value, expected)):
            _match_fixed(item, subexpected, f"{path}[{index}]")
        return
    if type(value) is not type(expected) or value != expected:
        raise PlanValidationError(f"{path} must equal registered value {expected!r}")


def _finite_number(value: Any, path: str, *, positive: bool = False) -> float:
    if type(value) not in (float, int):
        raise PlanValidationError(f"{path} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise PlanValidationError(f"{path} must be a finite number") from exc
    if not math.isfinite(number):
        raise PlanValidationError(f"{path} must be a finite number")
    if positive and number <= 0:
        raise PlanValidationError(f"{path} must be positive")
    return number


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise PlanValidationError(f"{path} must be a positive integer")
    return value


def _safe_relative_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise PlanValidationError(f"{path} must be a nonempty POSIX relative path")
    parts = value.split("/")
    if value.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise PlanValidationError(f"{path} must remain within the evidence root")
    if PurePosixPath(value).is_absolute():
        raise PlanValidationError(f"{path} must be relative")
    return value


def _hex(value: Any, path: str, *, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not (minimum <= len(value) <= maximum):
        raise PlanValidationError(f"{path} must be a hexadecimal identifier")
    if re.fullmatch(r"[0-9a-f]+", value) is None:
        raise PlanValidationError(f"{path} must use lowercase hexadecimal")
    return value


def _validate_calibration(calibration: dict[str, Any]) -> None:
    mapping = calibration["mapping"]
    if mapping is not None:
        mapping = _object(mapping, {"kind", "x_knots", "y_knots"}, "calibration.mapping")
        _match_fixed(mapping["kind"], "positive_monotone_lookup", "calibration.mapping.kind")
        xs, ys = mapping["x_knots"], mapping["y_knots"]
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys) or len(xs) < 2:
            raise PlanValidationError("calibration.mapping needs at least two paired knots")
        xs = [_finite_number(x, "calibration.mapping.x_knots") for x in xs]
        ys = [_finite_number(y, "calibration.mapping.y_knots", positive=True) for y in ys]
        if any(right <= left for left, right in zip(xs, xs[1:])) or any(
            right <= left for left, right in zip(ys, ys[1:])
        ):
            raise PlanValidationError("calibration.mapping knots must strictly increase in x and y")

    k_value = calibration["K"]
    if k_value is not None:
        _finite_number(k_value, "calibration.K", positive=True)

    tau = calibration["final_tau"]
    if tau is not None and not 0 < _finite_number(tau, "calibration.final_tau") <= 0.03125:
        raise PlanValidationError("calibration.final_tau must be in (0, 1/32]")

    burnin = calibration["burnin"]
    if burnin is not None:
        burnin = _object(
            burnin,
            {
                "minimum_time", "maximum_time", "window_time", "mean_tolerance",
                "variance_tolerance", "distribution_tolerance", "independent_initializations",
            },
            "calibration.burnin",
        )
        for key in (
            "minimum_time", "maximum_time", "window_time", "mean_tolerance",
            "variance_tolerance", "distribution_tolerance",
        ):
            _finite_number(burnin[key], f"calibration.burnin.{key}", positive=True)
        _positive_integer(burnin["independent_initializations"], "calibration.burnin.independent_initializations")
        if burnin["minimum_time"] >= burnin["maximum_time"]:
            raise PlanValidationError("calibration.burnin.maximum_time must exceed minimum_time")

    bounds = calibration["parameter_bounds"]
    if bounds is not None:
        names = {"effective_A", "b", "kon", "koff", "beta", "gamma", "rho"}
        bounds = _object(bounds, names, "calibration.parameter_bounds")
        for name, pair in bounds.items():
            if not isinstance(pair, list) or len(pair) != 2:
                raise PlanValidationError(f"calibration.parameter_bounds.{name} needs [low, high]")
            low = _finite_number(pair[0], f"calibration.parameter_bounds.{name}[0]")
            high = _finite_number(pair[1], f"calibration.parameter_bounds.{name}[1]")
            if low >= high or (name in {"kon", "koff", "beta", "gamma"} and low <= 0):
                raise PlanValidationError(f"calibration.parameter_bounds.{name} has invalid range")
            if name == "rho" and (low, high) != (0.6, 1.0):
                raise PlanValidationError("calibration.parameter_bounds.rho must remain [0.6, 1.0]")

    thresholds = calibration["compatibility_thresholds"]
    if thresholds is not None:
        thresholds = _object(
            thresholds, {"distribution", "response", "local_dynamics"},
            "calibration.compatibility_thresholds",
        )
        for name, value in thresholds.items():
            _finite_number(value, f"calibration.compatibility_thresholds.{name}", positive=True)

    model = calibration["model_config"]
    if model is not None:
        model = _object(
            model,
            {"q1_latent_dimension", "q1_training_seed", "q2_ensemble_size", "random_rule_id", "uniform_rule_id", "tie_rule_id"},
            "calibration.model_config",
        )
        if model["q1_latent_dimension"] not in (48, 96) or type(model["q1_latent_dimension"]) is not int:
            raise PlanValidationError("calibration.model_config.q1_latent_dimension must be 48 or 96")
        if type(model["q1_training_seed"]) is not int or model["q1_training_seed"] < 0:
            raise PlanValidationError("calibration.model_config.q1_training_seed must be nonnegative")
        _positive_integer(model["q2_ensemble_size"], "calibration.model_config.q2_ensemble_size")
        for key in ("random_rule_id", "uniform_rule_id", "tie_rule_id"):
            if not isinstance(model[key], str) or not model[key].strip():
                raise PlanValidationError(f"calibration.model_config.{key} must be nonempty")

    checks = calibration["runtime_checks"]
    if checks is not None:
        checks = _object(
            checks,
            {"generator_report", "adapter_report", "protocol_rehearsal_report", "scorer_isolation_report"},
            "calibration.runtime_checks",
        )
        for key, value in checks.items():
            _safe_relative_path(value, f"calibration.runtime_checks.{key}")


def validate_plan(plan: Any) -> dict[str, Any]:
    """Reject schema drift, including extra keys and altered study grids.

    Success means only that the JSON matches the registered design.  It does
    not mean that any generator, model, calibration, or blind test has run.
    """
    root_keys = {"schema_version", "status", "provenance", *FIXED_DESIGN, "calibration", "evidence"}
    plan = _object(plan, root_keys, "plan")
    _match_fixed(plan["schema_version"], SCHEMA_VERSION, "schema_version")
    if not isinstance(plan["status"], str) or plan["status"] not in STATUSES:
        raise PlanValidationError(f"status must be one of {sorted(STATUSES)}")
    provenance = _object(
        plan["provenance"],
        {"source_package", "baseline_version", "generator_identity", "scmultisim_source_commit", "implementation_commit"},
        "provenance",
    )
    for key, expected in {
        "source_package": "scMultiSim",
        "baseline_version": "1.8.0",
        "generator_identity": "scmultisim_based_continuous_time_extension",
    }.items():
        _match_fixed(provenance[key], expected, f"provenance.{key}")
    for key in ("scmultisim_source_commit", "implementation_commit"):
        if provenance[key] is not None:
            _hex(provenance[key], f"provenance.{key}", minimum=7, maximum=64)
    for section, expected in FIXED_DESIGN.items():
        _match_fixed(plan[section], expected, section)
    calibration = _object(plan["calibration"], set(CALIBRATION_KEYS), "calibration")
    _validate_calibration(calibration)
    evidence = _object(plan["evidence"], set(EVIDENCE_KEYS), "evidence")
    for key, reference in evidence.items():
        if reference is None:
            continue
        reference = _object(reference, {"path", "sha256"}, f"evidence.{key}")
        _safe_relative_path(reference["path"], f"evidence.{key}.path")
        _hex(reference["sha256"], f"evidence.{key}.sha256", minimum=64, maximum=64)
    return plan


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_nonfinite_constant(value: str) -> None:
    raise PlanValidationError(f"non-finite JSON value is prohibited: {value}")


def load_experiment_plan(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate one JSON plan, without side effects."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            plan = json.load(
                handle,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_no_nonfinite_constant,
            )
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"invalid JSON: {exc}") from exc
    return validate_plan(plan)


def formal_readiness_errors(
    plan: Any, *, evidence_root: str | Path | None = None
) -> list[str]:
    """List missing formal-run prerequisites, not a scientific proof.

    When ``evidence_root`` is supplied, every referenced file must exist and
    match its SHA-256.  Even an empty list only means that structural and
    provenance gates passed; human review of the experiments remains required.
    """
    try:
        plan = validate_plan(plan)
    except PlanValidationError as exc:
        return [f"structural design invalid: {exc}"]
    errors: list[str] = []
    if plan["status"] != "frozen_for_blind":
        errors.append("status is not frozen_for_blind")
    for key in ("scmultisim_source_commit", "implementation_commit"):
        if plan["provenance"][key] is None:
            errors.append(f"provenance.{key} is missing")
    for key in CALIBRATION_KEYS:
        if plan["calibration"][key] is None:
            errors.append(f"calibration.{key} is missing")
    root = Path(evidence_root).resolve() if evidence_root is not None else None
    checks = plan["calibration"]["runtime_checks"]
    if root is not None and checks is not None:
        for key, relative in checks.items():
            report = (root / relative).resolve()
            if not report.is_relative_to(root) or not report.is_file():
                errors.append(f"calibration.runtime_checks.{key} report is missing")
    for key in EVIDENCE_KEYS:
        reference = plan["evidence"][key]
        if reference is None:
            errors.append(f"evidence.{key} is missing")
            continue
        if root is None:
            continue
        artifact = (root / reference["path"]).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            errors.append(f"evidence.{key} file is missing or outside the evidence root")
            continue
        try:
            digest = hashlib.sha256()
            with artifact.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != reference["sha256"]:
                errors.append(f"evidence.{key} SHA-256 does not match")
        except OSError as exc:
            errors.append(f"evidence.{key} could not be read: {exc.strerror or type(exc).__name__}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the registered scMultiSim study design")
    default_path = Path(__file__).resolve().parents[2] / "configs" / "scmultisim-main.json"
    parser.add_argument("--config", type=Path, default=default_path)
    parser.add_argument("--require-formal", action="store_true", help="Require frozen calibration and cited evidence")
    arguments = parser.parse_args(argv)
    try:
        plan = load_experiment_plan(arguments.config)
    except (OSError, PlanValidationError) as exc:
        parser.exit(2, f"design validation failed: {exc}\n")
    print(f"Structural preflight passed: {plan['schema_version']} ({plan['status']})")
    print("Primary design: 4 development + 20 blind networks; 36 initial, 90 candidates, 12 acquisitions per strategy.")
    print("Steady observations are sealed for scoring; the 108-candidate paid-steady supplement is disabled.")
    print("No data generation, training, or scientific validation was performed.")
    if not arguments.require_formal:
        return 0
    errors = formal_readiness_errors(plan, evidence_root=arguments.config.resolve().parent.parent)
    if errors:
        print("Formal-run prerequisites are incomplete:")
        for issue in errors:
            print(f"- {issue}")
        return 2
    print("Formal structural/evidence gate passed; scientific review is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
