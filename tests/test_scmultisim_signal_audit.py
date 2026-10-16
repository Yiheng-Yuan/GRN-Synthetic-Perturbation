"""Checks for independent-group development diagnostics and sealed-array scope."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from grn_experiment.protocol import COMMON_BASELINE, Condition
from grn_experiment.scmultisim_observations import key_id, key_record


script = Path(__file__).parents[1] / "tools/scmultisim/audit_development_data.py"
spec = importlib.util.spec_from_file_location("audit_development_data", script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_graph_reachability_handles_cycles_and_orientation():
    A = np.zeros((5, 5))
    A[1, 0], A[2, 1], A[0, 2], A[3, 0] = 1, -1, 1, 1
    assert audit.reachable_genes(A, 0) == {1, 2, 3}
    assert audit.reachable_genes(A, 4) == set()


def test_independent_means_do_not_assume_cell_row_pairing():
    rng = np.random.default_rng(4)
    control = rng.poisson(20, size=(3, 64, 2))
    treated = control + np.array([8, 0])
    first = audit.contrast_statistics(treated, control, rng=np.random.default_rng(8), draws=200)
    permuted = treated[:, rng.permutation(64)]
    second = audit.contrast_statistics(permuted, control, rng=np.random.default_rng(8), draws=200)
    for key in ("delta", "standardized_effect", "control_null95", "control_null_sd"):
        np.testing.assert_allclose(first[key], second[key])
    assert first["above_control_null95"].tolist() == [True, False]
    assert first["delta"].tolist() == [8.0, 0.0]


def test_constant_zero_counts_are_finite_and_not_exploratory_signals():
    data = np.zeros((3, 8, 2), dtype=int)
    result = audit.contrast_statistics(data, data, rng=np.random.default_rng(0), draws=50)
    np.testing.assert_array_equal(result["standardized_effect"], [0, 0])
    np.testing.assert_array_equal(result["pooled_sd"], [1, 1])
    np.testing.assert_array_equal(result["above_control_null95"], [False, False])
    assert all(np.isfinite(value).all() for value in result.values())


def test_control_bootstrap_matches_independent_mean_variance_scale():
    # Both independent control means contribute variance/n. There is no
    # paired-row variance subtraction and no false multiplication of networks.
    control = np.tile(np.arange(64, dtype=float)[None, :, None], (3, 1, 1))
    result = audit.contrast_statistics(control, control, rng=np.random.default_rng(1), draws=5000)
    expected = np.sqrt(2 * control[0, :, 0].var() / 64 / 3)
    assert result["control_null_sd"][0] == pytest.approx(expected, rel=0.04)


def make_archive(tmp_path, *, mutate_observation=None, corrupt_checksum=False, symlink=False):
    prefix = "example/outputs/dev/"
    generator = json.dumps({"config": {"phase": "development"}, "parameters": {"A": np.zeros((6, 6)).tolist()}}).encode()
    manifest = {"split": {"initial": [0, 1, 2, 3, 4, 5]}, "cells_per_replicate": 2,
                "n_genes": 6, "conditions": {},
                "private_files": {"private/generator.json": hashlib.sha256(generator).hexdigest()}}
    members = [(prefix + "private/generator.json", generator)]
    for key in [COMMON_BASELINE, *(Condition(q, d, t) for q in range(6)
                                  for d in (0.3, 0.7) for t in (0.25, 1., 4.))]:
        relative = f"observations/initial/{key_id(key)}.npz"
        payload = io.BytesIO()
        arrays = {"treated": np.zeros((3, 2, 6), dtype=int)}
        if key.target is not None:
            arrays["matched_control"] = np.zeros((3, 2, 6), dtype=int)
        np.savez_compressed(payload, **arrays)
        content = payload.getvalue()
        manifest["conditions"][key_id(key)] = {"condition": key_record(key), "status": "complete",
                                               "observation": relative,
                                               "files": {relative: "bad" if corrupt_checksum else hashlib.sha256(content).hexdigest()}}
        members.append((prefix + relative, content))
    if mutate_observation:
        mutate_observation(manifest)
    members.extend([(prefix + "manifest.json", json.dumps(manifest).encode()),
                    # Deliberately malformed sealed content: opening it as an
                    # npz would fail. The strict allowlist must never decode it.
                    (prefix + "sealed/final/secret.npz", b"not a numpy archive"),
                    (prefix + "private/truth/secret.npz", b"not a numpy archive")])
    archive = tmp_path / "example.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for index, (name, payload) in enumerate(members):
            info = tarfile.TarInfo(name)
            if symlink and index == 1:
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/not-read"
                stream.addfile(info)
            else:
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
    return archive, prefix


def test_loader_decodes_exact_initial_allowlist_and_ignores_sealed_npz(tmp_path, monkeypatch):
    archive, prefix = make_archive(tmp_path)
    original = audit.np.load
    decoded = []
    def track(source, **kwargs):
        decoded.append(source)
        return original(source, **kwargs)
    monkeypatch.setattr(audit.np, "load", track)
    manifest, generator, selected = audit.load_initial_only(archive, prefix)
    assert len(decoded) == len(selected) == 37
    assert all(name.startswith("observations/initial/") for name in selected)


def test_manifest_cannot_redirect_initial_snapshot_to_sealed_data(tmp_path):
    def redirect(manifest):
        entry = next(iter(manifest["conditions"].values()))
        entry["observation"] = "sealed/final/secret.npz"
    archive, prefix = make_archive(tmp_path, mutate_observation=redirect)
    with pytest.raises(ValueError, match="outside the opened"):
        audit.load_initial_only(archive, prefix)


def test_selected_checksum_mismatch_is_rejected(tmp_path):
    archive, prefix = make_archive(tmp_path, corrupt_checksum=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.load_initial_only(archive, prefix)


def test_selected_symlink_is_rejected_without_following_it(tmp_path):
    archive, prefix = make_archive(tmp_path, symlink=True)
    with pytest.raises(ValueError, match="nonregular"):
        audit.load_initial_only(archive, prefix)
