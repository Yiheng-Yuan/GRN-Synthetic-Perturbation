"""Describe signal in opened development snapshots without fitting a learner.

Only the common baseline and initial-training RNA arrays are decoded. The
generator-side graph is used to group genes by reachability, never exported as
a learner feature. Candidate, validation, final and stationary RNA stay sealed.
The control bootstrap is a descriptive noise reference, not a significance test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np

from grn_experiment.protocol import COMMON_BASELINE, Condition
from grn_experiment.scmultisim_observations import key_id


DEFAULT_PREFIX = "grn-scmultisim-20261014-dev1/outputs/scmultisim-dev-101/"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261016
SD_FLOOR_COUNTS = 1.0  # One integer count, declared before reading responses.


def reachable_genes(A: np.ndarray, target: int) -> set[int]:
    """Return directed descendants, excluding the target even in a cycle."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with A[target, regulator] orientation")
    visited = {target}
    pending = [target]
    while pending:
        parent = pending.pop()
        children = set(np.flatnonzero(A[:, parent])) - visited
        visited.update(children)
        pending.extend(sorted(children))
    return visited - {target}


def _safe_relative(name: str) -> str:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or str(value) != name:
        raise ValueError(f"Unsafe or noncanonical archive path: {name}")
    return name


def allowed_initial_entries(manifest: dict) -> dict[str, dict]:
    """Build a strict allowlist from the registered 6 x 2 x 3 initial grid."""
    initial = manifest["split"]["initial"]
    if len(initial) != 6 or len(set(initial)) != 6:
        raise ValueError("Expected six unique initial-training targets")
    expected = [COMMON_BASELINE, *(Condition(q, d, t) for q in initial
                                        for d in (0.3, 0.7) for t in (0.25, 1.0, 4.0))]
    selected = {}
    for key in expected:
        identifier = key_id(key)
        entry = manifest["conditions"].get(identifier)
        if not entry or entry.get("status") != "complete":
            raise ValueError("An initial-training condition is missing or incomplete")
        condition = entry["condition"]
        if condition != {"kind": "timecourse", "target": key.target, "dose": key.dose, "time": key.time}:
            raise ValueError("Condition metadata does not match its initial-grid key")
        relative = _safe_relative(entry["observation"])
        if relative != f"observations/initial/{identifier}.npz":
            raise ValueError("Initial-grid observation points outside the opened namespace")
        if relative not in entry.get("files", {}):
            raise ValueError("Initial observation lacks its registered checksum")
        selected[relative] = entry
    return selected


def _read_metadata(archive: Path, prefix: str) -> tuple[dict, dict]:
    wanted = {prefix + "manifest.json", prefix + "private/generator.json"}
    found = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name in wanted:
                if not member.isfile() or member.size > 8 * 1024 * 1024:
                    raise ValueError("Metadata must be a bounded ordinary file")
                if member.name in found:
                    raise ValueError("Duplicate selected archive member")
                found[member.name] = json.load(stream.extractfile(member))
    if set(found) != wanted:
        raise ValueError("Archive lacks the requested output manifest or provenance")
    return found[prefix + "manifest.json"], found[prefix + "private/generator.json"]


def load_initial_only(archive: Path, prefix: str = DEFAULT_PREFIX) -> tuple[dict, dict, dict]:
    """Read ordinary members in memory; never extract or execute archive content."""
    prefix = _safe_relative(prefix.rstrip("/")) + "/"
    manifest, generator = _read_metadata(archive, prefix)
    if generator.get("config", {}).get("phase") != "development":
        raise ValueError("This audit may only inspect a development network")
    selected = allowed_initial_entries(manifest)
    expected_checksums = {relative: entry["files"][relative] for relative, entry in selected.items()}
    expected_checksums["private/generator.json"] = manifest["private_files"]["private/generator.json"]
    arrays = {}
    seen = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix):]
            if relative not in expected_checksums:
                continue  # Do not decode any sealed RNA or private state arrays.
            if relative in seen or not member.isfile() or member.size > 64 * 1024 * 1024:
                raise ValueError("Duplicate, nonregular or oversized selected member")
            seen.add(relative)
            payload = stream.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != expected_checksums[relative]:
                raise ValueError(f"Selected input checksum mismatch: {relative}")
            if relative not in selected:
                continue
            with np.load(io.BytesIO(payload), allow_pickle=False) as values:
                names = {"treated"} if selected[relative]["condition"]["target"] is None else {"treated", "matched_control"}
                if set(values.files) != names:
                    raise ValueError("Unexpected observation arrays")
                content = {name: values[name].copy() for name in names}
            for value in content.values():
                if (value.shape != (3, manifest["cells_per_replicate"], manifest["n_genes"])
                        or not np.issubdtype(value.dtype, np.integer) or np.any(value < 0)):
                    raise ValueError("Observation must have nonnegative integer replicate/cell/gene counts")
            arrays[relative] = content
    if seen != set(expected_checksums):
        raise ValueError("A selected archive member is missing")
    return manifest, generator, {relative: (entry, arrays[relative]) for relative, entry in selected.items()}


def contrast_statistics(treated: np.ndarray, control: np.ndarray, *,
                        rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Independent group means; no row-wise treated-control pairing is assumed.

    Each null draw samples two independent size-n cohorts from each empirical
    control replicate, preserving gene covariance within a sampled cell. The
    aggregate null averages the three replicate contrasts with equal weights.
    Its 95% absolute quantile is an exploratory control-noise reference only.
    """
    treated, control = np.asarray(treated, dtype=float), np.asarray(control, dtype=float)
    if (treated.ndim != 3 or treated.shape != control.shape or treated.shape[0] != 3
            or treated.shape[1] < 2 or draws < 20
            or not np.isfinite(treated).all() or not np.isfinite(control).all()):
        raise ValueError("Expected three finite replicates, at least two cells, and >=20 draws")
    mean_t, mean_c = treated.mean(axis=1), control.mean(axis=1)
    delta = mean_t - mean_c
    pooled_sd = np.sqrt((treated.var(axis=1, ddof=1) + control.var(axis=1, ddof=1)) / 2)
    aggregate_sd = np.maximum(np.sqrt(np.mean(pooled_sd ** 2, axis=0)), SD_FLOOR_COUNTS)
    null = np.empty((draws, 3, treated.shape[2]))
    n = control.shape[1]
    probabilities = np.full(n, 1 / n)
    for replicate in range(3):
        left = rng.multinomial(n, probabilities, size=draws)
        right = rng.multinomial(n, probabilities, size=draws)
        null[:, replicate] = ((left - right) @ control[replicate]) / n
    null_average = null.mean(axis=1)
    threshold = np.quantile(np.abs(null_average), 0.95, axis=0)
    fit_threshold = np.quantile(np.abs(null[:, :2].mean(axis=1)), 0.95, axis=0)
    check_threshold = np.quantile(np.abs(null[:, 2]), 0.95, axis=0)
    average = delta.mean(axis=0)
    return {"mean_treated": mean_t.mean(axis=0), "mean_control": mean_c.mean(axis=0),
            "delta": average, "delta_by_replicate": delta, "pooled_sd": aggregate_sd,
            "standardized_effect": average / aggregate_sd,
            "control_null95": threshold, "control_null_sd": null_average.std(axis=0, ddof=1),
            "above_control_null95": np.abs(average) > threshold,
            "fit_delta": delta[:2].mean(axis=0), "check_delta": delta[2],
            "fit_above_control_null95": np.abs(delta[:2].mean(axis=0)) > fit_threshold,
            "check_above_control_null95": np.abs(delta[2]) > check_threshold,
            "sd_floor_active": np.sqrt(np.mean(pooled_sd ** 2, axis=0)) < SD_FLOOR_COUNTS}


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"gene_condition_count": 0}
    return {"gene_condition_count": len(rows),
            "median_signed_delta_counts": float(np.median([r["delta_counts"] for r in rows])),
            "median_absolute_delta_counts": float(np.median([abs(r["delta_counts"]) for r in rows])),
            "median_absolute_standardized_effect": float(np.median([abs(r["standardized_effect"]) for r in rows])),
            "median_control_null95_counts": float(np.median([r["control_null95_counts"] for r in rows])),
            "fraction_above_control_null95": float(np.mean([r["above_control_null95"] for r in rows])),
            "fraction_at_or_below_control_null95": float(np.mean([not r["above_control_null95"] for r in rows])),
            "fraction_fit_check_direction_agrees": float(np.mean([r["fit_check_direction_agrees"] for r in rows])),
            "fraction_all_replicates_negative": float(np.mean([r["all_replicates_negative"] for r in rows]))}


def audit(archive: Path, prefix: str = DEFAULT_PREFIX, *, draws: int = BOOTSTRAP_DRAWS,
          seed: int = BOOTSTRAP_SEED) -> tuple[dict, list[dict], list[dict]]:
    manifest, generator, selected = load_initial_only(archive, prefix)
    A = np.asarray(generator["parameters"]["A"], dtype=float)
    if A.shape != (manifest["n_genes"], manifest["n_genes"]) or not np.isfinite(A).all():
        raise ValueError("Invalid private diagnostic graph")
    baseline = selected[f"observations/initial/{key_id(COMMON_BASELINE)}.npz"][1]["treated"]
    baseline_rows = [{"gene": g, "mean_counts": float(baseline[:, :, g].mean()),
                      "sd_counts": float(baseline[:, :, g].std(ddof=1)),
                      "zero_fraction": float(np.mean(baseline[:, :, g] == 0))}
                     for g in range(manifest["n_genes"])]
    rng = np.random.default_rng(seed)
    rows = []
    for relative, (entry, arrays) in selected.items():
        key = entry["condition"]
        if key["target"] is None:
            continue
        target = key["target"]
        descendants = reachable_genes(A, target)
        result = contrast_statistics(arrays["treated"], arrays["matched_control"], rng=rng, draws=draws)
        for gene in range(manifest["n_genes"]):
            category = "direct_target" if gene == target else "reachable_downstream" if gene in descendants else "unreachable"
            delta_reps = result["delta_by_replicate"][:, gene]
            rows.append({"target": target, "dose": key["dose"], "time": key["time"], "gene": gene,
                         "category": category, "mean_treated_counts": float(result["mean_treated"][gene]),
                         "mean_control_counts": float(result["mean_control"][gene]),
                         "delta_counts": float(result["delta"][gene]),
                         **{f"replicate_{r + 1}_delta_counts": float(delta_reps[r]) for r in range(3)},
                         "fit_delta_counts": float(result["fit_delta"][gene]),
                         "check_delta_counts": float(result["check_delta"][gene]),
                         "pooled_sd_counts": float(result["pooled_sd"][gene]),
                         "standardized_effect": float(result["standardized_effect"][gene]),
                         "control_null95_counts": float(result["control_null95"][gene]),
                         "control_null_sd_counts": float(result["control_null_sd"][gene]),
                         "above_control_null95": bool(result["above_control_null95"][gene]),
                         "fit_above_control_null95": bool(result["fit_above_control_null95"][gene]),
                         "check_above_control_null95": bool(result["check_above_control_null95"][gene]),
                         "fit_check_direction_agrees": bool(result["fit_delta"][gene] * result["check_delta"][gene] > 0),
                         "all_replicates_negative": bool(np.all(delta_reps < 0)),
                         "sd_floor_active": bool(result["sd_floor_active"][gene])})
    with archive.open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    grouped = []
    for dose in (0.3, 0.7):
        for time in (0.25, 1.0, 4.0):
            for category in ("direct_target", "reachable_downstream", "unreachable"):
                group = [row for row in rows if (row["dose"], row["time"], row["category"]) == (dose, time, category)]
                grouped.append({"dose": dose, "time": time, "category": category, **_summary(group)})
    report = {"schema": "scmultisim-development-signal-audit-v1", "independent_network_count": 1,
              "scope": "generator_side_development_diagnostic_not_model_training_or_final_scoring",
              "archive_sha256": archive_sha256, "archive_bytes": archive.stat().st_size,
              "output_prefix": prefix, "generator_source_sha256": generator["source_sha256"],
              "generator_fingerprint": manifest["fingerprint"], "kernel_version": generator["kernel_version"],
              "mechanism_reference_commit": generator["mechanism_reference_commit"],
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy": np.__version__, "python": sys.version.split()[0],
              "registered_timecourse_keys": manifest["planned_timecourse_keys"],
              "registered_stationary_keys": manifest["planned_stationary_keys"],
              "decoded_initial_keys": len(selected), "decoded_sealed_keys": 0,
              "selected_file_checksums_verified": len(selected) + 1,
              "bootstrap_draws": draws, "bootstrap_seed": seed, "sd_floor_counts": SD_FLOOR_COUNTS,
              "noise_reference": "95th percentile absolute mean of three independent empirical-control bootstrap contrasts",
              "threshold_role": "exploratory_descriptive_only_no_pvalues_no_multiple_testing_claim_no_freeze_decision",
              "repeats": "replicates 1-2 future_fit; replicate 3 future_parameter_independent_check; all used here only for generator diagnostics",
              "sealed_arrays_decoded": [], "private_truth_arrays_decoded": [],
              "truth_usage": "private A for directed reachability grouping only; no learner inputs produced",
              "report_access": "generator_development_only; gene category labels must not be mounted into learner jobs",
              "reachability": {"initial_targets": len(manifest["split"]["initial"]),
                               "possible_nontarget_genes_per_condition": manifest["n_genes"] - 1,
                               "mean_reachable_downstream_genes_per_target": float(np.mean([len(reachable_genes(A, q)) for q in manifest["split"]["initial"]])),
                               "min_reachable_downstream_genes_per_target": min(len(reachable_genes(A, q)) for q in manifest["split"]["initial"]),
                               "max_reachable_downstream_genes_per_target": max(len(reachable_genes(A, q)) for q in manifest["split"]["initial"])},
              "baseline": {"cell_count": int(baseline.shape[0] * baseline.shape[1]),
                           "count_sd_floor_rationale": "one integer count is a predeclared measurement-resolution floor",
                           "gene_mean_min": min(row["mean_counts"] for row in baseline_rows),
                           "gene_mean_max": max(row["mean_counts"] for row in baseline_rows),
                           "maximum_zero_fraction": max(row["zero_fraction"] for row in baseline_rows)},
              "overall_by_category": {category: _summary([row for row in rows if row["category"] == category])
                                      for category in ("direct_target", "reachable_downstream", "unreachable")},
              "by_dose_time_category": grouped,
              "limitations": ["One development network; no inference about 20 independent networks.",
                              "Noise thresholds are exploratory, not calibrated tests of edge presence.",
                              "No multiplicity adjustment, FDR interpretation, compatibility cutoff or independent-network confidence interval is supplied.",
                              "Graph reachability does not guarantee detectable response; mixed paths may cancel.",
                              "RNA response need not equal the imposed production reduction rho*d.",
                              "Independent cells and unpaired snapshots; no treated-control row pairing.",
                              "Integer flooring can hide sub-count early changes; private fluid states were not decoded.",
                              "No candidate t=0.125, validation, final t=2/8 or stationary RNA was inspected.",
                              "This audit does not perform feedback-grid refinement or stationarity calibration."]}
    return report, rows, baseline_rows


def write_outputs(output: Path, report: dict, rows: list[dict], baseline_rows: list[dict], command: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, content in (("gene_condition_effects.csv", rows), ("baseline_gene_summary.csv", baseline_rows),
                          ("dose_time_summary.csv", report["by_dose_time_category"])):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(content)
    lines = ["# 开发网络初始数据的扰动信号诊断", "", "本次只分析一张开发网络的共同基线和36个初始训练条件；未生成数据、未训练模型。",
             "候选、验证、最终测试和长期分布的 RNA 数组均未解码。图结构仅在生成器侧用于区分靶基因、可达下游和不可达基因。", "",
             "本报告及逐基因类别属于生成器开发侧材料，不得作为学习器输入或挂载到训练任务。", "",
             f"归档 SHA256：`{report['archive_sha256']}`。",
             f"生成源码指纹：`{report['generator_source_sha256']}`。",
             f"核对了 {report['selected_file_checksums_verified']} 个所用文件的登记校验值；完整归档中其余文件不在本轮重复核验范围。", "",
             "各条件有3个独立重复，每重复128个处理细胞及128个对照细胞。先逐重复计算处理与对照的均值差，再等权合并；不假定两组细胞行配对。",
             "标准化效应为三重复平均差除以均方合并的组内标准差，标准差下限预先设为1个整数计数。",
             f"噪声参照用每个对照重复独立重采两组同样大小的细胞，重复{report['bootstrap_draws']}次，计算三重复平均差绝对值的95%分位数。",
             "下表“高于对照噪声参照”仅为探索性描述，没有基因／条件的多重检验校正，不是显著性或FDR（假发现率）结论，也不是未来冻结的相容性阈值。",
             "该参照刻画给定经验对照分布下的细胞抽样波动；不构成独立网络置信区间，也不涵盖全部实验或参数不确定性。", "",
             "| 强度 | 时间 | 类别 | 基因×条件数 | 中位绝对RNA差 | 中位对照噪声95%参照 | 中位绝对标准化效应 | 高于对照噪声参照 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    labels = {"direct_target": "直接靶基因", "reachable_downstream": "可达下游", "unreachable": "不可达基因"}
    for row in report["by_dose_time_category"]:
        if row["gene_condition_count"]:
            lines.append(f"| {row['dose']:g} | {row['time']:g} | {labels[row['category']]} | {row['gene_condition_count']} | {row['median_absolute_delta_counts']:.4f} | {row['median_control_null95_counts']:.4f} | {row['median_absolute_standardized_effect']:.4f} | {100*row['fraction_above_control_null95']:.1f}% |")
    target_rows = [row for row in rows if row["category"] == "direct_target"]
    downstream = [row for row in rows if row["category"] == "reachable_downstream"]
    unreachable = [row for row in rows if row["category"] == "unreachable"]
    lines += ["", f"36个直接靶基因条件中，平均 RNA 下降的有{sum(row['delta_counts'] < 0 for row in target_rows)}个；3个重复均下降的有{sum(row['all_replicates_negative'] for row in target_rows)}个。",
              "按时间合并两种强度，直接靶基因高于对照噪声参照的条件数为：" + "；".join(f"t={time:g}：{sum(row['above_control_null95'] for row in target_rows if row['time'] == time)}/12" for time in (0.25, 1.0, 4.0)) + "。",
              f"可达下游有{sum(row['above_control_null95'] for row in downstream)}/{len(downstream)}个基因×条件高于参照；不可达基因有{sum(row['above_control_null95'] for row in unreachable)}/{len(unreachable)}个。这两个探索性比例接近，现有分析尚未证明下游传播信号足够支持网络恢复。",
              "这不证明下游没有效应，也不确定弱信号来自参数幅度、反馈路径抵消、时间尺度、启动子波动或整数取整；需另设生成器侧诊断区分。",
              "初始网格最早只有 t=0.25；t=0.125 属于封存候选池，本轮没有据此选择更早时间。",
              "早时间、小剂量和下游差异需要结合对照波动解释；不可达基因的有限样本差异可作为背景参照，但不是独立网络对照。",
              "完整逐基因结果见 `gene_condition_effects.csv`，其中拟合重复1-2与检验重复3的均值差分别保存。", "",
              "这些是生成器开发诊断，不能作为学习器最终性能。当前只检查1张网络，不能估计20张网络的统计功效。",
              "生产项抑制比例不等于成熟 RNA 的同一比例下降；本轮没有将两者强制比较。",
              "整数取整可能压低早时间亚计数变化；本轮未读取私有连续状态，不能确定各条件损失了多少信号。",
              "反馈网格减半、4张开发网络和长期平稳判据校准仍需另行执行。", "", "复现命令：", "", "```bash", command, "```", ""]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", default=BOOTSTRAP_DRAWS, type=int)
    parser.add_argument("--seed", default=BOOTSTRAP_SEED, type=int)
    args = parser.parse_args()
    result = audit(args.archive, args.prefix, draws=args.bootstrap_draws, seed=args.seed)
    import shlex
    command = "PYTHONPATH=src python tools/scmultisim/audit_development_data.py " + shlex.join(sys.argv[1:])
    write_outputs(args.output, *result, command)
    print(json.dumps({"output": str(args.output), "decoded_initial_keys": result[0]["decoded_initial_keys"],
                      "decoded_sealed_keys": result[0]["decoded_sealed_keys"], "archive_sha256": result[0]["archive_sha256"]}))


if __name__ == "__main__":
    main()
