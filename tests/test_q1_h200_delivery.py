"""CPU checks for offline delivery; these are not evidence of an H200 run."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from grn_experiment.protocol import COMMON_BASELINE, Condition


PROJECT = Path(__file__).parents[1]


def _smoke_module():
    source = PROJECT / "tools/scmultisim/q1_h200_smoke.py"
    spec = importlib.util.spec_from_file_location("q1_h200_smoke", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _two_file_dataset(directory):
    manifest = {"schema": "scmultisim-q1-modular-v1", "phase": "development",
                "formal_calibration": False, "status": "complete", "n_genes": 64,
                "cells_per_replicate": 2, "target_ids": list(range(48)),
                "q1_split": {"train": list(range(32)), "held": list(range(32, 48)),
                             "registration": "topology_before_rna"},
                "sampling_times": [.25, 1., 2., 4., 8.], "dose": .7, "conditions": {}}
    for condition in (COMMON_BASELINE, Condition(0, .7, .25)):
        record = {"kind": "timecourse", "target": condition.target,
                  "dose": condition.dose, "time": condition.time}
        identifier = hashlib.sha256(json.dumps(record, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()[:24]
        relative = f"observations/q1/{identifier}.npz"
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"treated": np.full((3, 2, 64), 5, dtype=np.int64)}
        if condition.target is not None:
            arrays["matched_control"] = arrays["treated"].copy()
            arrays["treated"][..., 0] -= 1
        np.savez_compressed(path, **arrays)
        manifest["conditions"][identifier] = {
            "condition": record, "status": "complete", "observation": relative,
            "files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_cpu_delivery_smoke_reads_only_baseline_and_one_fit_snapshot(tmp_path, monkeypatch):
    module = _smoke_module()
    manifest = _two_file_dataset(tmp_path)
    allowed = {"manifest.json", *(entry["observation"] for entry in manifest["conditions"].values())}
    opened, original = [], module._directory_payload

    def guarded(directory, relative, limit):
        assert relative in allowed
        opened.append(relative)
        return original(directory, relative, limit)

    monkeypatch.setattr(module, "_directory_payload", guarded)
    result = module.smoke(tmp_path)
    assert set(opened) == allowed and len(opened) == 3
    assert result["status"] == "passed" and result["updates"] == 1
    assert result["n_genes"] == 64 and result["target"] == 0 and result["time"] == .25
    assert np.isfinite(result["loss"])
    assert result["held_or_longterm_responses_read"] is False
    assert len(result["verified_observation_sha256"]) == 2


def test_delivery_shell_syntax_and_rejects_unknown_arguments():
    entry = PROJECT / "tools/scmultisim/run_q1_h200.sh"
    slurm = PROJECT / "tools/scmultisim/q1_h200.slurm"
    for path in (entry, slurm):
        subprocess.run(["bash", "-n", str(path)], check=True)
    result = subprocess.run(["bash", str(entry), "--unexpected"], capture_output=True, text=True)
    assert result.returncode == 64 and "Usage" in result.stderr


def test_delivery_requires_an_existing_persistent_output_root(tmp_path):
    entry = PROJECT / "tools/scmultisim/run_q1_h200.sh"
    _two_file_dataset(tmp_path / "input")
    output = tmp_path / "not-mounted"
    result = subprocess.run(["bash", str(entry), "--cpu-smoke"], capture_output=True, text=True,
                            env={**os.environ, "Q1_DATASET": str(tmp_path / "input"),
                                 "Q1_OUTPUT_ROOT": str(output)})
    assert result.returncode == 73 and "persistent storage" in result.stderr
    assert not output.exists()


def test_default_wrapper_preserves_runtime_and_log_when_cuda_is_unavailable(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("This regression checks a CPU-only host, not a GPU training run")
    entry = PROJECT / "tools/scmultisim/run_q1_h200.sh"
    _two_file_dataset(tmp_path / "input")
    output = tmp_path / "persistent-output"
    output.mkdir()
    result = subprocess.run(["bash", str(entry)], capture_output=True, text=True,
                            env={**os.environ, "Q1_DATASET": str(tmp_path / "input"),
                                 "Q1_OUTPUT_ROOT": str(output),
                                 "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]})
    assert result.returncode != 0
    run = output / "q1-modular-64-h200-seed20261022"
    runtime = json.loads((run / "runtime.json").read_text())
    assert runtime["cuda_available"] is False
    assert "refusing an accidental CPU long run" in runtime["gpu_probe_error"]
    assert "Task exit status: 1" in (run / "run.log").read_text()
    assert not (run / "results").exists()


def test_docker_delivery_has_no_generator_truth_or_data_copy():
    dockerfile = (PROJECT / "Dockerfile.q1").read_text()
    assert "@sha256:a77983eb7a3042ccf41a94da547a7f54a47fb70da15c36d67a8e8a329ebaa098" in dockerfile
    assert "numpy==2.5.3" in dockerfile and "pytest==9.1.1" in dockerfile
    copies = "\n".join(line for line in dockerfile.splitlines() if line.startswith("COPY"))
    assert "COPY ." not in copies
    assert "generate_scmultisim" not in copies and "configs/" not in copies
    wrapper = (PROJECT / "tools/scmultisim/run_q1_h200.sh").read_text()
    assert "--updates 4800 --seed 20261022 --device cuda --threads 4" in wrapper
    assert "--output \"$run_root/results\"" in wrapper
    assert "generate_scmultisim" not in wrapper
