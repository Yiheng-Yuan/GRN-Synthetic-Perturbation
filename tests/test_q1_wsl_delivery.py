"""Native-WSL delivery guard checks; no GPU-capacity claim."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).parents[1]
ENTRY = PROJECT / "tools/scmultisim/run_q1_wsl.sh"


def test_wsl_entry_syntax_and_rejects_unknown_mode():
    subprocess.run(["bash", "-n", str(ENTRY)], check=True)
    result = subprocess.run(["bash", str(ENTRY), "unknown"], capture_output=True, text=True)
    assert result.returncode == 64 and "Usage" in result.stderr


def test_wsl_missing_dataset_never_creates_output(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(ENTRY), "probe"], capture_output=True, text=True,
                            env={**os.environ, "Q1_DATASET": str(tmp_path / "missing"),
                                 "Q1_OUTPUT_ROOT": str(output)})
    assert result.returncode == 66 and "Missing public dataset" in result.stderr
    assert not output.exists()


def test_wsl_entry_preserves_scientific_batch_and_has_no_generator():
    source = ENTRY.read_text()
    assert "--updates 4800 --seed 20261022 --device cuda --threads 4" in source
    assert "--condition-microbatch 1 --checkpoint-every 60" in source
    assert "generate_scmultisim" not in source and "q1_generate" not in source
    assert "--resume" in source
    assert "manifest_sha256" in source and "source_sha256" in source


@pytest.mark.parametrize("change", [None, "source", "runtime", "dataset", "batch", "probe"])
def test_wsl_acceptance_reads_actual_probe_contract(tmp_path, monkeypatch, change):
    """Run the entry's embedded gate, with CUDA identity mocked, not a GPU fit."""
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    report = {"status": "passed", "condition_microbatch": 1, "effective_conditions": 4,
              "cells_per_condition": 64,
              "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
              "source_sha256": {
                  f"src/grn_experiment/{name}": hashlib.sha256(
                      (PROJECT / "src/grn_experiment" / name).read_bytes()).hexdigest()
                  for name in ("q1_pilot.py", "q1_pilot_data.py", "protocol.py")},
              "probe_source_sha256": hashlib.sha256(
                  (PROJECT / "tools/scmultisim/q1_gpu_probe.py").read_bytes()).hexdigest(),
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
              "device_name": "Fixture RTX 4070", "cuda_capability": [8, 9]}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fixture RTX 4070")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    if change == "source":
        report["source_sha256"]["src/grn_experiment/q1_pilot.py"] = "a" * 64
    elif change == "runtime":
        report["python_version"] = "0.0.0"
    elif change == "dataset":
        report["manifest_sha256"] = "a" * 64
    elif change == "batch":
        report["cells_per_condition"] = 2
    elif change == "probe":
        report["probe_source_sha256"] = "a" * 64
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["-", str(path), str(dataset)])
    program = ENTRY.read_text().split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    if change:
        with pytest.raises(SystemExit):
            exec(compile(program, str(ENTRY), "exec"), {})
    else:
        exec(compile(program, str(ENTRY), "exec"), {})
