"""Learner allowlist and integrity checks; fixtures contain no real truth."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import patch

import numpy as np
import pytest

from grn_experiment.protocol import COMMON_BASELINE, Condition, ReplicatedSnapshot
from grn_experiment.q1_pilot_data import (
    OpenedInitialData, load_fullcourse_initial, load_modular_q1, load_opened_initial,
    split_fit_interpolation, split_registered_targets, split_targets,
)


PREFIX = "project/outputs/dev/"


def _key(condition, *, stationary=False):
    record = {"kind": "timecourse", "target": condition.target,
              "dose": condition.dose, "time": condition.time}
    if stationary:
        record["kind"] = "stationary"
        record.pop("time")
    identifier = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return record, identifier


def _diagnostic(target=None, dose=0.):
    return {"passed": True, "burnin": 48., "elapsed_time": 48., "report": {
        "schema_version": "scmultisim-stationarity-v1", "status": "diagnostic_passed",
        "target": target, "dose": dose, "tau": 1 / 32, "parameter_sha256": "a" * 64,
    }}


def _fixture(tmp_path, *, mutate_manifest=None, mutate_arrays=None, member_change=None,
             duplicate=None, missing=False, checksum_mismatch=False, fullcourse=False):
    manifest = {"schema": "scmultisim-development-v1", "formal_calibration": False,
                "status": "complete", "split": {"initial": list(range(6))},
                "cells_per_replicate": 2, "n_genes": 6, "conditions": {},
                "private_files": {"private/generator.json": "unused secret checksum"}}
    members = []
    keys = [COMMON_BASELINE, *(Condition(q, d, t) for q in range(6)
                              for d in (0.3, 0.7) for t in (0.25, 1, 4))]
    if fullcourse:
        manifest["baseline_stationarity"] = _diagnostic()
        keys.extend(Condition(q, .7, t) for q in range(6) for t in (2, 8, 48))
    for index, condition in enumerate(keys):
        stationary = condition.time == 48
        record, identifier = _key(condition, stationary=stationary)
        directory = ("sealed/stationary" if stationary else
                     "sealed/final" if condition.time in (2, 8) else "observations/initial")
        relative = f"{directory}/{identifier}.npz"
        arrays = {"treated": np.full((3, 2, 6), index, dtype=np.int64)}
        if condition.target is not None:
            arrays["matched_control"] = np.ones((3, 2, 6), dtype=np.int64)
        if mutate_arrays and index == 1:
            mutate_arrays(arrays)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        checksum = hashlib.sha256(payload).hexdigest()
        manifest["conditions"][identifier] = {
            "condition": record, "status": "complete", "observation": relative,
            "files": {relative: checksum},
        }
        if stationary:
            manifest["conditions"][identifier]["diagnostic"] = _diagnostic(condition.target, condition.dose)
        if checksum_mismatch and index == 1:
            payload += b"changed"
        if not (missing and index == 1):
            members.append((PREFIX + relative, payload, index))
    if mutate_manifest:
        mutate_manifest(manifest)
    manifest_payload = json.dumps(manifest).encode()
    members.append((PREFIX + "manifest.json", manifest_payload, -1))
    # Malformed private metadata, private arrays and sealed RNA must be ignored.
    members.extend((PREFIX + name, b"MUST NOT BE DECODED", -2) for name in (
        "private/generator.json", "private/truth/anything.npz", "sealed/final/secret.npz"))
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, payload, index in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            if member_change and index == 1:
                member_change(info)
            stream.addfile(info, io.BytesIO(payload) if info.isfile() else None)
            if ((duplicate == "observation" and index == 1)
                    or (duplicate == "baseline" and index == 0)
                    or (duplicate == "manifest" and index == -1)):
                stream.addfile(info, io.BytesIO(payload))
    return archive


def test_load_only_opened_public_arrays_and_split_time(tmp_path):
    data = load_opened_initial(_fixture(tmp_path), PREFIX)
    assert data.baseline.shape == (3, 2, 6)
    assert not data.baseline.flags.writeable
    assert len(data.observations) == 36
    assert all(value.matched_control.shape == (3, 2, 6) for value in data.observations.values())
    assert data.provenance["phase"] == "development"
    assert data.provenance["verified_observation_count"] == 37
    assert data.provenance["decoded_private_or_sealed_files"] == 0
    assert "private" not in json.dumps(data.provenance["verified_observation_sha256"])
    fit, held = split_fit_interpolation(data)
    assert len(fit) == 24 and len(held) == 12
    assert {key.time for key in fit} == {0.25, 4}
    assert {key.time for key in held} == {1}


def test_private_and_sealed_members_are_never_read(tmp_path):
    archive = _fixture(tmp_path)
    original = tarfile.TarFile.extractfile
    opened = []

    def guarded(stream, member):
        assert "/private/" not in member.name and "/sealed/" not in member.name
        opened.append(member.name)
        return original(stream, member)

    with patch.object(tarfile.TarFile, "extractfile", guarded):
        load_opened_initial(archive, PREFIX)
    assert len(opened) == 38


@pytest.mark.parametrize("change", [
    lambda m: m.update(schema="formal-v1"),
    lambda m: m.update(formal_calibration=True),
    lambda m: m.update(phase="formal"),
    lambda m: m.update(status="running"),
    lambda m: m.update(n_genes=5),
    lambda m: m.update(cells_per_replicate=True),
    lambda m: m["split"].update(initial=[0, 0, 2, 3, 4, 5]),
    lambda m: m["split"].update(initial=[0, 1, 2, 3, 4, True]),
])
def test_wrong_manifest_declarations_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_opened_initial(_fixture(tmp_path, mutate_manifest=change), PREFIX)


@pytest.mark.parametrize("destination", ["sealed/final/secret.npz", "../secret.npz", "/absolute.npz", "observations//initial/wrong.npz"])
def test_manifest_redirects_rejected(tmp_path, destination):
    def change(manifest):
        next(iter(manifest["conditions"].values()))["observation"] = destination
    with pytest.raises(ValueError):
        load_opened_initial(_fixture(tmp_path, mutate_manifest=change), PREFIX)


@pytest.mark.parametrize("duplicate", ["observation", "manifest"])
def test_duplicate_selected_members_rejected(tmp_path, duplicate):
    with pytest.raises(ValueError, match="Duplicate"):
        load_opened_initial(_fixture(tmp_path, duplicate=duplicate), PREFIX)


@pytest.mark.parametrize("change", [
    lambda info: setattr(info, "type", tarfile.SYMTYPE),
    lambda info: setattr(info, "type", tarfile.LNKTYPE),
    lambda info: setattr(info, "type", tarfile.DIRTYPE),
])
def test_selected_nonregular_members_rejected(tmp_path, change):
    with pytest.raises(ValueError, match="ordinary"):
        load_opened_initial(_fixture(tmp_path, member_change=change), PREFIX)


@pytest.mark.parametrize("mutation", [
    lambda arrays: arrays.update(treated=np.zeros((3, 1, 6), dtype=np.int64)),
    lambda arrays: arrays.update(treated=np.full((3, 2, 6), -1, dtype=np.int64)),
    lambda arrays: arrays.update(treated=np.ones((3, 2, 6), dtype=float)),
    lambda arrays: arrays.update(treated=np.full((3, 2, 6), np.nan)),
    lambda arrays: arrays.update(treated=np.array([{"secret": 1}], dtype=object)),
    lambda arrays: arrays.update(extra=np.ones(1)),
    lambda arrays: arrays.pop("matched_control"),
])
def test_invalid_npz_arrays_rejected(tmp_path, mutation):
    with pytest.raises(ValueError):
        load_opened_initial(_fixture(tmp_path, mutate_arrays=mutation), PREFIX)


def test_missing_member_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_opened_initial(_fixture(tmp_path, missing=True), PREFIX)


def test_checksum_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        load_opened_initial(_fixture(tmp_path, checksum_mismatch=True), PREFIX)


def test_observation_bound_enforced_before_decode(tmp_path):
    archive = _fixture(tmp_path)
    with patch("grn_experiment.q1_pilot_data._MAX_OBSERVATION_BYTES", 1):
        with pytest.raises(ValueError, match="bounded"):
            load_opened_initial(archive, PREFIX)


@pytest.mark.parametrize("prefix", ["/project/outputs/dev/", "../dev/", "project//outputs/dev/", "project/outputs/./dev/", "project\\outputs\\dev/"])
def test_noncanonical_prefix_rejected(tmp_path, prefix):
    with pytest.raises(ValueError, match="canonical"):
        load_opened_initial(tmp_path / "does-not-exist", prefix)


def _stationary_entry(manifest):
    return next(entry for entry in manifest["conditions"].values()
                if entry["condition"]["kind"] == "stationary")


def test_fullcourse_opens_exact_times_and_stationary_without_private_truth(tmp_path):
    data = load_fullcourse_initial(_fixture(tmp_path, fullcourse=True), PREFIX)
    assert data.baseline.shape == (3, 2, 6)
    assert len(data.observations) == 36
    assert {key.dose for key in data.observations} == {.7}
    for target in range(6):
        assert {key.time for key in data.observations if key.target == target} == {.25, 1., 2., 4., 8., 48.}
    p = data.provenance
    assert p["verified_observation_count"] == 37
    assert p["decoded_private_files"] == 0
    assert p["decoded_private_or_sealed_files"] == p["decoded_sealed_observation_files"] == 18
    assert p["newly_opened_initial_late_count"] == 12
    assert p["newly_opened_initial_stationary_count"] == 6
    assert p["longterm_duration_by_target"] == {str(q): 48. for q in range(6)}
    assert p["old_formal_held_time_purpose_changed"] is True


def test_fullcourse_never_reads_nonselected_doses_targets_or_private_members(tmp_path):
    archive = _fixture(tmp_path, fullcourse=True)
    original = tarfile.TarFile.extractfile
    expected = {PREFIX + "manifest.json"}
    for condition in [COMMON_BASELINE, *(Condition(q, .7, t) for q in range(6)
                                         for t in (.25, 1, 2, 4, 8, 48))]:
        _, identifier = _key(condition, stationary=condition.time == 48)
        directory = ("sealed/stationary" if condition.time == 48 else
                     "sealed/final" if condition.time in (2, 8) else "observations/initial")
        expected.add(PREFIX + f"{directory}/{identifier}.npz")
    opened = []

    def guarded(stream, member):
        assert member.name in expected
        opened.append(member.name)
        return original(stream, member)

    with patch.object(tarfile.TarFile, "extractfile", guarded):
        load_fullcourse_initial(archive, PREFIX)
    assert set(opened) == expected and len(opened) == 38


@pytest.mark.parametrize("change", [
    lambda d: d.update(passed=False),
    lambda d: d.update(passed=1),
    lambda d: d.update(burnin=None),
    lambda d: d.update(burnin=True),
    lambda d: d.update(burnin=8),
    lambda d: d.update(burnin=float("inf")),
    lambda d: d["report"].update(target=5),
    lambda d: d["report"].update(target=False),
    lambda d: d["report"].update(dose=.3),
    lambda d: d["report"].update(tau=1 / 16),
    lambda d: d["report"].update(tau=False),
    lambda d: d["report"].update(parameter_sha256="b" * 64),
    lambda d: d["report"].update(status="diagnostic_failed"),
])
def test_fullcourse_rejects_bad_stationary_diagnostic(tmp_path, change):
    def mutation(manifest):
        change(_stationary_entry(manifest)["diagnostic"])
    with pytest.raises(ValueError):
        load_fullcourse_initial(_fixture(tmp_path, fullcourse=True, mutate_manifest=mutation), PREFIX)


@pytest.mark.parametrize("destination", ["private/truth/stolen.npz", "../secret.npz", "sealed/final/other.npz"])
def test_fullcourse_rejects_stationary_path_redirects(tmp_path, destination):
    def mutation(manifest):
        _stationary_entry(manifest)["observation"] = destination
    with pytest.raises(ValueError):
        load_fullcourse_initial(_fixture(tmp_path, fullcourse=True, mutate_manifest=mutation), PREFIX)


def test_fullcourse_requires_every_late_time_and_accepted_baseline(tmp_path):
    def remove_time(manifest):
        identifier = _key(Condition(0, .7, 2))[1]
        manifest["conditions"].pop(identifier)
    with pytest.raises(ValueError, match="Missing"):
        load_fullcourse_initial(_fixture(tmp_path, fullcourse=True, mutate_manifest=remove_time), PREFIX)
    with pytest.raises(ValueError, match="passed"):
        load_fullcourse_initial(_fixture(tmp_path, fullcourse=True, mutate_manifest=lambda m:
                                        m["baseline_stationarity"].update(passed=False)), PREFIX)


@pytest.mark.parametrize("duplicate", ["baseline", "manifest"])
def test_fullcourse_reuses_duplicate_archive_protection(tmp_path, duplicate):
    with pytest.raises(ValueError, match="Duplicate"):
        load_fullcourse_initial(_fixture(tmp_path, fullcourse=True, duplicate=duplicate), PREFIX)


def test_fullcourse_reuses_predecode_byte_bound(tmp_path):
    archive = _fixture(tmp_path, fullcourse=True)
    with patch("grn_experiment.q1_pilot_data._MAX_OBSERVATION_BYTES", 1):
        with pytest.raises(ValueError, match="bounded"):
            load_fullcourse_initial(archive, PREFIX)


def test_target_split_is_deterministic_response_independent_and_has_no_cross_group_target(tmp_path):
    data = load_fullcourse_initial(_fixture(tmp_path, fullcourse=True), PREFIX)
    train, test = split_targets(data)
    expected = np.random.default_rng(20261022).permutation(sorted(data.provenance["initial_targets"]))
    train_targets, test_targets = {key.target for key in train}, {key.target for key in test}
    assert train_targets == set(expected[:4]) and test_targets == set(expected[4:])
    assert train_targets.isdisjoint(test_targets)
    assert len(train) == 24 and len(test) == 12
    assert set(train) | set(test) == set(data.observations)
    reordered = OpenedInitialData(data.baseline, dict(reversed(list(data.observations.items()))),
                                  {**data.provenance, "initial_targets": list(reversed(range(6)))})
    assert set(split_targets(reordered)[0]) == set(train)
    assert set(split_targets(reordered)[1]) == set(test)
    changed_responses = {
        key: ReplicatedSnapshot(np.full_like(value.treated, 1000), np.zeros_like(value.matched_control))
        for key, value in data.observations.items()
    }
    changed = OpenedInitialData(data.baseline, changed_responses, data.provenance)
    assert set(split_targets(changed)[0]) == set(train)
    assert set(split_targets(changed)[1]) == set(test)


@pytest.mark.parametrize("settings", [{"seed": True}, {"seed": -1}, {"train_count": 0}, {"train_count": 6}])
def test_invalid_group_split_settings_rejected(tmp_path, settings):
    data = load_fullcourse_initial(_fixture(tmp_path, fullcourse=True), PREFIX)
    with pytest.raises(ValueError):
        split_targets(data, **settings)


def test_group_split_rejects_observations_from_undeclared_targets(tmp_path):
    data = load_fullcourse_initial(_fixture(tmp_path, fullcourse=True), PREFIX)
    observations = dict(data.observations)
    observations[Condition(7, .7, .25)] = next(iter(observations.values()))
    with pytest.raises(ValueError, match="exactly"):
        split_targets(OpenedInitialData(data.baseline, observations, data.provenance))


def _modular_fixture(tmp_path, *, mutate_manifest=None, long_time=48.):
    """Public-only 64-gene fixtures; topology/private sentinels are unreadable."""
    directory = tmp_path / "modular"
    directory.mkdir()
    manifest = {
        "schema": "scmultisim-q1-modular-v1", "phase": "development",
        "formal_calibration": False, "status": "complete", "n_genes": 64,
        "cells_per_replicate": 2, "target_ids": list(range(48)),
        "q1_split": {"train": list(range(32)), "held": list(range(32, 48)),
                     "registration": "topology_before_rna"},
        "sampling_times": [.25, 1., 2., 4., 8.], "dose": .7,
        "common_longterm_time": long_time, "baseline_stationarity": _diagnostic(),
        "conditions": {}, "private_files": {"private/generator.json": "not-for-learner"},
    }
    for index, condition in enumerate([COMMON_BASELINE, *(Condition(q, .7, t)
                                         for q in range(48) for t in (.25, 1., 2., 4., 8., long_time))]):
        record, identifier = _key(condition)
        relative = f"observations/q1/{identifier}.npz"
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"treated": np.full((3, 2, 64), index % 7, dtype=np.int64)}
        if condition.target is not None:
            arrays["matched_control"] = np.ones((3, 2, 64), dtype=np.int64)
        np.savez_compressed(path, **arrays)
        entry = {"condition": record, "status": "complete", "observation": relative,
                 "files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()}}
        if condition.time == long_time:
            entry.update(diagnostic=_diagnostic(condition.target, .7), actual_sample_time=long_time)
        manifest["conditions"][identifier] = entry
    private = directory / "private/generator.json"
    private.parent.mkdir()
    private.write_text("NEVER DECODE: A, G, U, rho, module labels")
    if mutate_manifest:
        mutate_manifest(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_modular_loader_reads_only_allowlisted_public_files_and_keeps_real_time(tmp_path):
    directory = _modular_fixture(tmp_path, long_time=80.)
    allowed = {directory / "manifest.json", *(directory / entry["observation"] for entry in
                json.loads((directory / "manifest.json").read_text())["conditions"].values())}
    opened, original = [], Path.open

    def guarded(path, *args, **kwargs):
        assert path in allowed
        opened.append(path)
        return original(path, *args, **kwargs)

    with patch.object(Path, "open", guarded):
        data = load_modular_q1(directory)
    assert set(opened) == allowed and len(opened) == 290
    assert data.baseline.shape == (3, 2, 64)
    assert len(data.observations) == 288
    assert {key.time for key in data.observations} == {.25, 1., 2., 4., 8., 80.}
    assert data.provenance["verified_observation_count"] == 289
    assert data.provenance["common_longterm_time"] == 80.
    assert data.provenance["decoded_private_files"] == 0
    assert "module labels" not in json.dumps(data.provenance)
    assert all(not snapshot.treated.flags.writeable for snapshot in data.observations.values())
    train, held = split_registered_targets(data)
    assert len(train) == 192 and len(held) == 96
    assert {key.target for key in train} == set(range(32))
    assert {key.target for key in held} == set(range(32, 48))


@pytest.mark.parametrize("change", [
    lambda m: m.update(phase="formal"),
    lambda m: m.update(formal_calibration=True),
    lambda m: m.update(status="running"),
    lambda m: m.update(n_genes=24),
    lambda m: m.update(dose=True),
    lambda m: m.update(sampling_times=[.25, True, 2., 4., 8.]),
    lambda m: m["target_ids"].__setitem__(0, True),
    lambda m: m["q1_split"].update(registration="response_selected"),
    lambda m: m["q1_split"]["held"].__setitem__(0, 0),
    lambda m: m.update(common_longterm_time=8.),
    lambda m: m.update(common_longterm_time=True),
    lambda m: m["baseline_stationarity"].update(passed=False),
])
def test_modular_loader_rejects_incompatible_public_contract(tmp_path, change):
    with pytest.raises(ValueError):
        load_modular_q1(_modular_fixture(tmp_path, mutate_manifest=change))


def _modular_long_entry(manifest):
    return next(entry for entry in manifest["conditions"].values() if "diagnostic" in entry)


@pytest.mark.parametrize("change", [
    lambda e: e.update(actual_sample_time=47.),
    lambda e: e["diagnostic"].update(passed=False),
    lambda e: e["diagnostic"].update(burnin=64.),
    lambda e: e["diagnostic"]["report"].update(target=47),
    lambda e: e["diagnostic"]["report"].update(parameter_sha256="b" * 64),
    lambda e: e["diagnostic"]["report"].update(tau=.5),
])
def test_modular_long_endpoint_cannot_fake_time_or_stationarity(tmp_path, change):
    with pytest.raises(ValueError):
        load_modular_q1(_modular_fixture(tmp_path, mutate_manifest=lambda m: change(_modular_long_entry(m))))


def test_modular_loader_rejects_checksum_mismatch_and_symlink(tmp_path):
    directory = _modular_fixture(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / next(iter(manifest["conditions"].values()))["observation"]
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_modular_q1(directory)
    path.unlink()
    target = tmp_path / "other.npz"
    target.write_bytes(original)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        load_modular_q1(directory)


@pytest.mark.parametrize("destination", ["private/truth/secret.npz", "../secret.npz", "/absolute.npz"])
def test_modular_loader_rejects_observation_path_redirects(tmp_path, destination):
    def change(manifest):
        next(iter(manifest["conditions"].values()))["observation"] = destination
    with pytest.raises(ValueError):
        load_modular_q1(_modular_fixture(tmp_path, mutate_manifest=change))


def test_modular_registered_split_does_not_depend_on_response_or_mapping_order(tmp_path):
    data = load_modular_q1(_modular_fixture(tmp_path))
    train, held = split_registered_targets(data)
    poisoned = {key: ReplicatedSnapshot(np.full_like(value.treated, 9999),
                                        np.zeros_like(value.matched_control))
                for key, value in reversed(list(data.observations.items()))}
    changed = OpenedInitialData(data.baseline, poisoned, data.provenance)
    other_train, other_held = split_registered_targets(changed)
    assert set(train) == set(other_train) and set(held) == set(other_held)


def test_seeded_group_split_supports_more_than_six_targets():
    baseline = np.ones((3, 2, 8))
    observations = {Condition(q, .7, 1): ReplicatedSnapshot(baseline, baseline) for q in range(8)}
    data = OpenedInitialData(baseline, observations, {"initial_targets": list(range(8))})
    train, held = split_targets(data, train_count=6)
    assert len(train) == 6 and len(held) == 2
