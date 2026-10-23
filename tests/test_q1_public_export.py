"""Small allowlist fixtures; these do not validate a real generated dataset."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    path = Path(__file__).parents[1] / "tools/scmultisim/export_q1_public.py"
    spec = importlib.util.spec_from_file_location("export_q1_public", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def public_data(tmp_path, monkeypatch, module):
    directory = tmp_path / "generator-vault"
    observation = directory / "observations/q1/example.npz"
    observation.parent.mkdir(parents=True)
    observation.write_bytes(b"small stub payload already validated by the loader")
    private = directory / "private/truth/example.npz"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"secret simulator states")
    manifest = json.dumps({"private_files": {"private/truth/example.npz": "hash-only"}}).encode()
    (directory / "manifest.json").write_bytes(manifest)
    provenance = {"manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                  "verified_observation_sha256": {
                      "observations/q1/example.npz": hashlib.sha256(observation.read_bytes()).hexdigest()}}
    monkeypatch.setattr(module, "load_modular_q1", lambda path: SimpleNamespace(provenance=provenance))
    return directory, provenance, manifest


def test_export_preserves_raw_manifest_and_never_opens_or_includes_private(
    tmp_path, monkeypatch, module, public_data,
):
    directory, _, raw_manifest = public_data
    output = tmp_path / "public.tar.gz"
    original = module._directory_payload
    opened = []

    def guarded(dataset, relative, limit):
        assert relative == "manifest.json" or relative.startswith("observations/q1/")
        opened.append(relative)
        return original(dataset, relative, limit)

    monkeypatch.setattr(module, "_directory_payload", guarded)
    result = module.export(directory, output)
    assert set(opened) == {"manifest.json", "observations/q1/example.npz"}
    with tarfile.open(output) as archive:
        assert archive.getnames() == ["q1-modular-64/manifest.json",
                                      "q1-modular-64/observations/q1/example.npz"]
        assert archive.extractfile(archive.getmembers()[0]).read() == raw_manifest
        assert all(member.isfile() and member.mode == 0o644 and member.uid == member.gid == 0
                   and not member.linkname for member in archive)
    assert result["members"] == 2 and result["private_files_included"] == 0
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["bytes"] == output.stat().st_size


@pytest.mark.parametrize("relative", ["observations/q1/example.npz", "manifest.json"])
def test_export_rechecks_payload_after_loader_and_cleans_partial_archive(
    tmp_path, module, public_data, relative,
):
    directory, _, _ = public_data
    (directory / relative).write_bytes(b"tampered after verification")
    output = tmp_path / "public.tar.gz"
    with pytest.raises(ValueError, match="changed after validation"):
        module.export(directory, output)
    assert not output.exists() and not list(tmp_path.glob(".q1-public-*"))


def test_export_does_not_allow_private_whitelist_injection(tmp_path, module, public_data):
    directory, provenance, _ = public_data
    provenance["verified_observation_sha256"] = {"private/truth/example.npz": "0" * 64}
    with pytest.raises(ValueError, match="Only allowlisted"):
        module.export(directory, tmp_path / "public.tar.gz")
    assert not (tmp_path / "public.tar.gz").exists()


def test_export_refuses_existing_or_inside_dataset_output(tmp_path, module, public_data, monkeypatch):
    directory, _, _ = public_data
    existing = tmp_path / "existing.tar.gz"
    existing.write_bytes(b"keep me")
    monkeypatch.setattr(module, "load_modular_q1", lambda path: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError, match="already exists"):
        module.export(directory, existing)
    assert existing.read_bytes() == b"keep me"
    with pytest.raises(ValueError, match="outside"):
        module.export(directory, directory / "do-not-create.tar.gz")


def test_export_rejects_symlinked_public_observation(tmp_path, module, public_data):
    directory, _, _ = public_data
    observation = directory / "observations/q1/example.npz"
    observation.unlink()
    observation.symlink_to(directory / "private/truth/example.npz")
    with pytest.raises(ValueError, match="symbolic links"):
        module.export(directory, tmp_path / "public.tar.gz")
