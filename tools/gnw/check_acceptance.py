#!/usr/bin/env python3
"""Independently check a native GNW pilot; never train or generate formal data.

The truth parameter dictionary must contain each gene's actual
``compileParameters`` result and ordered ``getInputGenes`` labels.  This file
reconstructs the probability-weighted GNW module function, not the project's
existing sigmoid/Hill simulator.  SciPy and matplotlib are imported only by
the command-line runner, so pure-function unit tests require only NumPy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CHECK_EXPLANATIONS = {
    "native_engine_provenance": "原生 GNW 引擎与运行来源记录",
    "no_clipping_or_normalization": "未裁剪负值，未进行全数据归一化",
    "deterministic_native_repeat": "两次原生运行的结果文件完全一致",
    "artifact_checksums_match": "结果文件的实际校验值与记录一致",
    "finite_nonnegative_raw_states": "RNA 状态有限且非负",
    "finite_nonnegative_production_and_boundary": "原生产生率非负，零状态处不会向负丰度演化",
    "finite_nonnegative_steady_states": "单独求得的稳态丰度有限且非负",
    "sampling_grid_and_pilot_size": "小网络轨迹数量及采样时间正确",
    "common_preperturbation_time_zero": "所有条件共用扰动前初态",
    "dose_metadata": "名义强度与实际抑制比例一致",
    "fixed_target_efficiency": "同一靶点在不同强度、时间及初态下效率固定",
    "zero_dose_matches_control": "零剂量与对照轨迹重合",
    "native_clock_and_tolerance_refinement": "细化原生时钟及容差后轨迹一致",
    "independent_scipy_trajectory_reconstruction": "独立重建函数的高精度积分与 GNW 一致",
    "native_rhs_and_module_probability_reconstruction": "原生变化速度及产生率与模块概率重建一致",
    "target_local_production_scaling_and_direct_derivative": "扰动当下仅改变靶点产生项及其变化速度",
    "local_jacobian_native_match": "局部导数与原生导出及差分细化一致",
    "edge_signs_and_effective_strength": "边方向、促进抑制符号及非零实际作用正确",
    "native_exported_topology": "原生导出网络与预设矩阵一致",
    "separately_integrated_steady_state_residual": "单独延长积分得到的稳态残差足够小",
    "empirical_control_convergence_from_multiple_initials": "多个对照初态经验性趋同",
    "empirical_perturbed_convergence_from_multiple_initials": "各扰动条件在多个初态下经验性趋同",
    "sustained_knockdown_second_half_analytic_root": "后半程仍持续敲低，与根基因解析轨迹一致",
    "complete_knockout_analytic_exponential": "完全敲除后根基因按解析指数衰减",
    "nonzero_downstream_propagation": "扰动确实传播到下游基因",
}


def module_activation(
    x: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    n_activators: int,
    binds_as_complex: bool,
) -> float:
    """Reconstruct GNW RegulatoryModule.computeActivation without clipping."""
    x, k, n = (np.asarray(value, dtype=float) for value in (x, k, n))
    if x.shape != k.shape or x.shape != n.shape or x.ndim != 1:
        raise ValueError("module inputs, k, and n must be matching vectors")
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("module input states must be finite and nonnegative")
    if np.any(k <= 0) or np.any(n <= 0) or not np.all(np.isfinite(k + n)):
        raise ValueError("k and n must be finite and positive")
    if not 0 <= n_activators <= len(x):
        raise ValueError("invalid number of activators")
    xi = (x / k) ** n
    numerator = float(np.prod(xi[:n_activators]))
    if binds_as_complex:
        denominator = 1.0 + numerator
        if n_activators < len(x):
            denominator += float(np.prod(xi))
    else:
        denominator = float(np.prod(1.0 + xi))
    return numerator / denominator


def weighted_activation(activities: list[float], alpha: list[float]) -> float:
    """Use GNW's bit convention: module zero is the least significant bit."""
    if len(alpha) != 2 ** len(activities):
        raise ValueError("alpha length must equal two to the number of modules")
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in activities):
        raise ValueError("module activations must be finite probabilities")
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in alpha):
        raise ValueError("alpha values must lie in [0, 1]")
    result = 0.0
    for state, value in enumerate(alpha):
        probability = 1.0
        for module, activity in enumerate(activities):
            probability *= activity if state & (1 << module) else 1.0 - activity
        result += value * probability
    return float(result)


def normalize_gene_parameters(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate actual exported GNW parameters and retain input ordering."""
    genes = truth["genes"]
    exported = truth["gene_parameters"]
    if isinstance(exported, list):
        exported = {item.get("gene", item.get("label")): item for item in exported}
    result = []
    for label in genes:
        item = exported[label]
        inputs = item.get("inputs", item.get("input_genes"))
        parameters = item.get("parameters", item.get("params"))
        if inputs is None or parameters is None:
            raise ValueError(f"{label}: need actual ordered inputs and parameters")
        if isinstance(parameters, list):
            parameters = dict(zip(item["parameter_names"], parameters, strict=True))
        parameters = {name: float(value) for name, value in parameters.items()}
        modules = []
        input_cursor = 0
        while f"bindsAsComplex_{len(modules) + 1}" in parameters:
            module_id = len(modules) + 1
            n_activators = int(parameters[f"numActivators_{module_id}"])
            n_deactivators = int(parameters[f"numDeactivators_{module_id}"])
            count = n_activators + n_deactivators
            indices = [genes.index(value) for value in inputs[input_cursor:input_cursor + count]]
            if len(indices) != count:
                raise ValueError(f"{label}: input/module count mismatch")
            modules.append({
                "indices": indices,
                "k": np.array([parameters[f"k_{input_cursor + j + 1}"] for j in range(count)]),
                "n": np.array([parameters[f"n_{input_cursor + j + 1}"] for j in range(count)]),
                "n_activators": n_activators,
                "binds_as_complex": parameters[f"bindsAsComplex_{module_id}"] == 1.0,
            })
            input_cursor += count
        if input_cursor != len(inputs):
            raise ValueError(f"{label}: input/module count mismatch")
        alpha = [parameters[f"a_{state}"] for state in range(2 ** len(modules))]
        weighted_activation([0.5] * len(modules), alpha)
        if parameters["max"] < 0 or parameters["delta"] <= 0:
            raise ValueError(f"{label}: invalid production/decay parameters")
        result.append({"label": label, "modules": modules, "alpha": alpha,
                       "max": parameters["max"], "delta": parameters["delta"]})
    return result


def production(
    x: np.ndarray,
    model: list[dict[str, Any]],
    clock_scale: float,
    target: str = "ctrl",
    dose: float = 0.0,
    rho: float = 0.0,
) -> np.ndarray:
    """Physical-time production reconstructed from native GNW exports."""
    x = np.asarray(x, dtype=float)
    if x.shape != (len(model),) or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("RNA state must be a finite nonnegative gene vector")
    if not np.isfinite(clock_scale) or clock_scale <= 0:
        raise ValueError("clock scale must be finite and positive")
    if not (0 <= dose <= 1 and 0 <= rho <= 1):
        raise ValueError("dose and efficiency must lie in [0, 1]")
    labels = [gene["label"] for gene in model]
    if target != "ctrl" and target not in labels:
        raise ValueError("unknown perturbation target")
    values = []
    for gene in model:
        activities = [module_activation(x[module["indices"]], module["k"], module["n"],
                                        module["n_activators"], module["binds_as_complex"])
                      for module in gene["modules"]]
        factor = 1.0 - rho * dose if gene["label"] == target else 1.0
        values.append(clock_scale * gene["max"] * factor *
                      weighted_activation(activities, gene["alpha"]))
    return np.asarray(values)


def vector_field(x: np.ndarray, model: list[dict[str, Any]], clock_scale: float,
                 target: str = "ctrl", dose: float = 0.0, rho: float = 0.0) -> np.ndarray:
    return production(x, model, clock_scale, target, dose, rho) - np.asarray(x) * \
        clock_scale * np.array([gene["delta"] for gene in model])


def central_jacobian(x: np.ndarray, function: Any, epsilon: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if epsilon <= 0 or np.any(x <= epsilon):
        raise ValueError("central differences need strictly interior states")
    result = np.empty((len(x), len(x)))
    for column in range(len(x)):
        shift = np.zeros_like(x)
        shift[column] = epsilon
        result[:, column] = (function(x + shift) - function(x - shift)) / (2 * epsilon)
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def trajectory_groups(rows: list[dict[str, str]], genes: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(row)
    result = {}
    for run_id, group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["time"]))
        times = np.array([float(row["time"]) for row in ordered])
        if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError(f"{run_id}: invalid or duplicate sampling times")
        metadata = ("initial_id", "target", "dose", "rho", "remaining_fraction")
        if any(any(row[key] != ordered[0][key] for key in metadata) for row in ordered):
            raise ValueError(f"{run_id}: condition changes within trajectory")
        result[run_id] = {**{key: ordered[0][key] for key in metadata}, "time": times,
                          "x": np.array([[float(row[gene]) for gene in genes] for row in ordered])}
    return result


def analytic_root_path(times: np.ndarray, initial: float, synthesis: float,
                       decay: float, remaining_fraction: float) -> np.ndarray:
    equilibrium = synthesis * remaining_fraction / decay
    return equilibrium + (initial - equilibrium) * np.exp(-decay * np.asarray(times))


def run_acceptance(output_dir: Path) -> dict[str, Any]:
    from scipy.integrate import solve_ivp

    truth = json.loads((output_dir / "truth" / "parameters.json").read_text())
    genes, scale = truth["genes"], float(truth["clock_scale"])
    model = normalize_gene_parameters(truth)
    groups = trajectory_groups(read_csv(output_dir / "trajectories.csv"), genes)
    refined = trajectory_groups(read_csv(output_dir / "refined_trajectories.csv"), genes)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **metrics: Any) -> None:
        checks.append({"name": name, "explanation": CHECK_EXPLANATIONS[name], "passed": bool(passed), **metrics})

    provenance_path = output_dir / "truth" / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    check("native_engine_provenance", bool(provenance.get("source_commit")) and
          bool(provenance.get("jar_sha256")) and provenance.get("model_translation") is False and
          bool(provenance.get("actual_native_calls")),
          evidence=provenance, caveat="Provenance is a run record, not a cryptographic attestation.")
    check("no_clipping_or_normalization", provenance.get("clip_states") is False and
          provenance.get("normalization") is False)
    repeated = provenance.get("repeat_identical", {})
    check("deterministic_native_repeat", bool(repeated) and all(value is True for value in repeated.values()),
          file_count=len(repeated))
    digests = provenance.get("artifact_sha256", {})
    mismatches = []
    for relative, expected_digest in digests.items():
        artifact = output_dir / relative
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_digest:
            mismatches.append(relative)
    check("artifact_checksums_match", bool(digests) and not mismatches,
          file_count=len(digests), mismatches=mismatches)

    all_states = np.concatenate([path["x"] for path in groups.values()])
    check("finite_nonnegative_raw_states", np.all(np.isfinite(all_states)) and np.min(all_states) >= 0,
          min_state=float(np.min(all_states)), max_state=float(np.max(all_states)))
    expected_grid = np.arange(65) / 8.0
    check("sampling_grid_and_pilot_size", len(groups) == 53 and all(
        np.array_equal(path["time"], expected_grid) for path in groups.values()), paths=len(groups),
        snapshots=int(sum(len(path["time"]) for path in groups.values())))
    controls = {path["initial_id"]: path for path in groups.values() if path["target"] == "ctrl"}
    common_error = max(float(np.max(np.abs(path["x"][0] - controls[path["initial_id"]]["x"][0])))
                       for path in groups.values())
    check("common_preperturbation_time_zero", common_error == 0.0, max_abs_error=common_error)
    fraction_error = max(abs(float(path["remaining_fraction"]) -
                             (1.0 - float(path["rho"]) * float(path["dose"])))
                         for path in groups.values() if path["target"] != "ctrl")
    check("dose_metadata", fraction_error < 1e-13, max_abs_error=fraction_error)
    efficiency_error = max((abs(float(path["rho"]) - float(truth["rho"][genes.index(path["target"])]))
                            for run_id, path in groups.items()
                            if path["target"] != "ctrl" and run_id != "complete_ko"), default=float("inf"))
    check("fixed_target_efficiency", efficiency_error < 1e-13, max_abs_error=efficiency_error,
          exception="complete_ko deliberately sets G1 efficiency=1 for an analytic sanity check")
    zero_paths = [path for path in groups.values() if path["target"] != "ctrl" and float(path["dose"]) == 0]
    zero_error = max((float(np.max(np.abs(path["x"] - controls[path["initial_id"]]["x"])))
                      for path in zero_paths), default=float("inf"))
    check("zero_dose_matches_control", zero_error < 1e-12, max_abs_error=zero_error)

    refinement_error = 0.0
    if set(groups) != set(refined):
        refinement_error = float("inf")
    else:
        for run_id, path in groups.items():
            other = refined[run_id]
            if not np.array_equal(path["time"], other["time"]):
                refinement_error = float("inf")
                break
            refinement_error = max(refinement_error, float(np.max(np.abs(path["x"] - other["x"]))))
    check("native_clock_and_tolerance_refinement", refinement_error < 1e-7,
          max_abs_error=refinement_error, threshold=1e-7)

    scipy_error, failures = 0.0, []
    for run_id, path in groups.items():
        target, dose, rho = path["target"], float(path["dose"]), float(path["rho"])
        solution = solve_ivp(lambda _t, x: vector_field(x, model, scale, target, dose, rho),
                             (0.0, float(path["time"][-1])), path["x"][0], t_eval=path["time"],
                             method="DOP853", rtol=2e-12, atol=2e-14)
        if not solution.success or solution.y.shape != path["x"].T.shape:
            failures.append(run_id)
            scipy_error = float("inf")
        else:
            scipy_error = max(scipy_error, float(np.max(np.abs(solution.y.T - path["x"]))))
    check("independent_scipy_trajectory_reconstruction", scipy_error < 1e-7 and not failures,
          max_abs_error=scipy_error, threshold=1e-7, integration_failures=failures)

    rhs_error, production_error, direct_error, ratio_error = 0.0, 0.0, 0.0, 0.0
    probes = read_csv(output_dir / "probes.csv")
    native_productions = []
    zero_boundary_derivatives = []
    for row in probes:
        x = np.array([float(row[f"x{i+1}"]) for i in range(len(genes))])
        target, dose, rho = row["target"], float(row["dose"]), float(row["rho"])
        native_rhs = np.array([float(row[f"dx{i+1}"]) for i in range(len(genes))])
        native_production = np.array([float(row[f"production{i+1}"]) for i in range(len(genes))])
        native_productions.append(native_production)
        if np.all(x == 0):
            zero_boundary_derivatives.append(native_rhs)
        expected_production = production(x, model, scale, target, dose, rho)
        rhs_error = max(rhs_error, float(np.max(np.abs(native_rhs - vector_field(x, model, scale, target, dose, rho)))))
        production_error = max(production_error, float(np.max(np.abs(native_production - expected_production))))
        control_production = production(x, model, scale)
        control_rhs = vector_field(x, model, scale)
        if target != "ctrl":
            q = genes.index(target)
            unchanged = np.arange(len(genes)) != q
            ratio_error = max(ratio_error, abs(native_production[q] / control_production[q] - (1 - rho * dose)))
            direct_error = max(direct_error, float(np.max(np.abs(native_rhs[unchanged] - control_rhs[unchanged]))),
                               float(np.max(np.abs(native_production[unchanged] - control_production[unchanged]))),
                               abs((native_rhs[q] - control_rhs[q]) + rho * dose * control_production[q]))
    check("native_rhs_and_module_probability_reconstruction", rhs_error < 1e-12 and production_error < 1e-12 and bool(probes),
          rhs_max_abs_error=rhs_error, production_max_abs_error=production_error, probe_count=len(probes))
    production_values = np.stack(native_productions) if native_productions else np.array([float("nan")])
    boundary_values = np.stack(zero_boundary_derivatives) if zero_boundary_derivatives else np.array([float("nan")])
    check("finite_nonnegative_production_and_boundary", np.all(np.isfinite(production_values)) and
          np.min(production_values) >= 0 and np.all(np.isfinite(boundary_values)) and np.min(boundary_values) >= 0,
          min_production=float(np.min(production_values)), zero_boundary_min_derivative=float(np.min(boundary_values)),
          zero_boundary_probe_count=len(zero_boundary_derivatives))
    check("target_local_production_scaling_and_direct_derivative", direct_error < 1e-12 and ratio_error < 1e-12,
          direct_effect_max_abs_error=direct_error, ratio_max_abs_error=ratio_error)

    x_probe = np.asarray(truth["jacobian_state"], dtype=float)
    jacobian = central_jacobian(x_probe, lambda x: vector_field(x, model, scale))
    fine_jacobian = central_jacobian(x_probe, lambda x: vector_field(x, model, scale), 5e-7)
    jacobian_rows = read_csv(output_dir / "truth" / "jacobian.csv")
    indexed_jacobian = {row["target"]: row for row in jacobian_rows}
    native_jacobian = np.array([[float(indexed_jacobian[target][gene]) for gene in genes] for target in genes])
    jac_error = float(np.max(np.abs(jacobian - native_jacobian)))
    jac_refinement = float(np.max(np.abs(jacobian - fine_jacobian)))
    signed = np.asarray(truth["signed_matrix"], dtype=float)
    regulatory = jacobian + np.diag([gene["delta"] * scale for gene in model])
    edges = signed != 0
    minimum_effect = float(np.min(np.abs(regulatory[edges])))
    nonedge_error = float(np.max(np.abs(regulatory[~edges])))
    check("local_jacobian_native_match", jac_error < 1e-8 and jac_refinement < 1e-8,
          max_abs_error=jac_error, step_halving_error=jac_refinement, threshold=1e-8)
    check("edge_signs_and_effective_strength", np.array_equal(np.sign(regulatory[edges]), signed[edges]) and
          minimum_effect > 1e-3 and nonedge_error < 1e-8,
          min_edge_local_effect=minimum_effect, nonedge_max_abs_error=nonedge_error)
    edge_matrix = np.zeros_like(signed)
    exported_edges = read_csv(output_dir / "edges.csv")
    for row in exported_edges:
        value = {"+": 1.0, "-": -1.0}.get(row["sign"])
        if value is None:
            value = float(row["sign"])
        edge_matrix[genes.index(row["target"]), genes.index(row["regulator"])] = value
    check("native_exported_topology", len(exported_edges) == int(np.sum(edges)) and np.array_equal(edge_matrix, signed),
          n_edges=len(exported_edges))

    steady_rows = read_csv(output_dir / "steady_states.csv")
    steady_error, stored_residual_error = 0.0, 0.0
    steady_controls = []
    steady_groups: dict[tuple[str, float], list[np.ndarray]] = {}
    steady_states = []
    for row in steady_rows:
        x = np.array([float(row[gene]) for gene in genes])
        steady_states.append(x)
        target, dose = row["target"], float(row["dose"])
        rho = 0.0 if target == "ctrl" else float(truth["rho"][genes.index(target)])
        if dose == 1.0 and target == "G1":
            rho = 1.0
        residual = float(np.max(np.abs(vector_field(x, model, scale, target, dose, rho))))
        steady_error = max(steady_error, residual)
        stored_residual_error = max(stored_residual_error, abs(residual - float(row["residual"])))
        if target == "ctrl":
            steady_controls.append(x)
        steady_groups.setdefault((target, dose), []).append(x)
    check("separately_integrated_steady_state_residual", bool(steady_rows) and steady_error < 1e-9 and stored_residual_error < 1e-10,
          max_residual=steady_error, stored_residual_max_abs_error=stored_residual_error, threshold=1e-9)
    steady_array = np.stack(steady_states) if steady_states else np.array([float("nan")])
    check("finite_nonnegative_steady_states", np.all(np.isfinite(steady_array)) and np.min(steady_array) >= 0,
          min_state=float(np.min(steady_array)))
    convergence_error = float(np.max(np.ptp(np.stack(steady_controls), axis=0))) if len(steady_controls) >= 3 else float("inf")
    check("empirical_control_convergence_from_multiple_initials", convergence_error < 1e-8,
          max_between_initials_error=convergence_error, initial_count=len(steady_controls),
          caveat="Empirical evidence for tested initial states, not a global uniqueness proof.")
    perturbed_steady_groups = {key: value for key, value in steady_groups.items() if key[0] != "ctrl"}
    perturbed_spread = max((float(np.max(np.ptp(np.stack(values), axis=0)))
                           if len(values) >= 3 else float("inf")
                           for values in perturbed_steady_groups.values()), default=float("inf"))
    check("empirical_perturbed_convergence_from_multiple_initials", len(perturbed_steady_groups) == 16 and perturbed_spread < 1e-8,
          max_between_initials_error=perturbed_spread, condition_count=len(perturbed_steady_groups),
          caveat="Empirical evidence for tested initial states, not a global uniqueness proof.")

    root = model[0]
    if root["modules"]:
        raise ValueError("analytic G1 check requires the unregulated root gene")
    root_synthesis, root_decay = scale * root["max"] * root["alpha"][0], scale * root["delta"]
    sustained_error, ko_error = 0.0, float("inf")
    for path in groups.values():
        if path["target"] != "G1":
            continue
        expected = analytic_root_path(path["time"], path["x"][0, 0], root_synthesis, root_decay,
                                      float(path["remaining_fraction"]))
        error = np.abs(path["x"][:, 0] - expected)
        sustained_error = max(sustained_error, float(np.max(error[path["time"] >= 4.0])))
        if float(path["remaining_fraction"]) == 0.0:
            ko_error = float(np.max(error))
    check("sustained_knockdown_second_half_analytic_root", sustained_error < 1e-7,
          max_abs_error=sustained_error, threshold=1e-7)
    check("complete_knockout_analytic_exponential", ko_error < 1e-7,
          max_abs_error=ko_error, threshold=1e-7)
    propagation = max((float(np.max(np.abs(path["x"][:, 1:] - controls[path["initial_id"]]["x"][:, 1:])))
                       for path in groups.values() if path["target"] == "G1" and float(path["dose"]) > 0), default=0.0)
    check("nonzero_downstream_propagation", propagation > 1e-3, max_downstream_response=propagation,
          caveat="Downstream monotonicity is not required in feedback networks.")
    terminal_residuals = [float(np.max(np.abs(vector_field(path["x"][-1], model, scale,
                          path["target"], float(path["dose"]), float(path["rho"]))))) for path in groups.values()]
    report = {"passed": all(item["passed"] for item in checks), "checks": checks,
              "model": "native GNW RNA-only; fixed independent regulatory modules; noiseless pilot",
              "terminal_t8_max_residual": max(terminal_residuals),
              "t8_is_assumed_steady": False,
              "limitations": ["Continuous model-relative RNA abundances, not sequencing counts.",
                              "One engineered four-gene parameterization; not validation of every GNW network.",
                              "No learner training, no formal cohort, no biological validation.",
                              "Native provenance is recorded, not cryptographically attested." ]}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# GNW 小网络适配验收", "", "总体结果：" + ("通过" if report["passed"] else "存在未通过项"), "",
             "本报告仅验收一个确定性、RNA 层的小网络；不代表正式实验或生物学真实性验证。", "",
             "| 检查 | 结果 | 数值证据 |", "| --- | --- | --- |"]
    for item in checks:
        metrics = {key: value for key, value in item.items() if key not in ("name", "explanation", "passed", "evidence", "caveat")}
        lines.append(f"| {item['explanation']} | {'通过' if item['passed'] else '失败'} | {json.dumps(metrics, ensure_ascii=False)} |")
    lines.extend(["", f"t=8 最大变化速度残差：{report['terminal_t8_max_residual']:.6g}；稳态另行求得，不将 t=8 当作稳态。", "",
                  "多初态趋同只是这些初态下的经验性证据，不是全局单稳态证明。", "",
                  "输出为连续 RNA 层有效丰度，不是测序整数计数；本轮不训练模型、不生成正式网络队列。", ""])
    (output_dir / "report.md").write_text("\n".join(lines))
    render_figures(output_dir, groups, genes, truth, jacobian)
    return report


def render_figures(output_dir: Path, groups: dict[str, dict[str, Any]], genes: list[str],
                   truth: dict[str, Any], jacobian: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = next(path["initial_id"] for path in groups.values() if path["target"] == "ctrl")
    selected = [path for path in groups.values() if path["initial_id"] == initial and
                (path["target"] == "ctrl" or path["target"] == "G1")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for gene_id, axis in enumerate(axes.flat):
        for path in selected:
            label = "Control" if path["target"] == "ctrl" else f"G1 dose={float(path['dose']):g}, rho={float(path['rho']):g}"
            axis.plot(path["time"], path["x"][:, gene_id], label=label, lw=1.5)
        axis.set(title=genes[gene_id], xlabel="Elapsed simulation time", ylabel="Continuous RNA abundance")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Native GNW: common initial state, sustained G1 intervention")
    fig.savefig(output_dir / "trajectories.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5), constrained_layout=True)
    matrices = [(np.asarray(truth["signed_matrix"]), "Signed topology"),
                (np.asarray(truth["coefficient_matrix"]), "Module activation coefficients"),
                (jacobian, "Local Jacobian (includes decay)")]
    for axis, (matrix, title) in zip(axes, matrices, strict=True):
        limit = max(float(np.max(np.abs(matrix))), 1e-12)
        axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set(xticks=range(len(genes)), yticks=range(len(genes)), xticklabels=genes,
                 yticklabels=genes, xlabel="Regulator / source column", ylabel="Target row", title=title)
        axis.title.set_fontsize(11)
        for i in range(len(genes)):
            for j in range(len(genes)):
                axis.text(j, i, f"{matrix[i, j]:.3g}", ha="center", va="center", fontsize=8)
    fig.savefig(output_dir / "matrices.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run_acceptance(args.output_dir.resolve())
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "failed": [item["name"] for item in report["checks"] if not item["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
