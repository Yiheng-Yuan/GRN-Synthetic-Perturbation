"""Allowlisted development RNA loading without simulator truth.

The original pilots read bounded archives, never extracting them. The modular
pilot reads only its local public manifest and exact opened RNA paths. Neither
loading route decodes private truth, latent states or perturbation efficiencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

import numpy as np

from .protocol import COMMON_BASELINE, Condition, ReplicatedSnapshot


DEFAULT_PREFIX = "grn-scmultisim-20261014-dev1/outputs/scmultisim-dev-101/"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_OBSERVATION_BYTES = 32 * 1024 * 1024
_MAX_SELECTED_BYTES = 256 * 1024 * 1024
_MAX_ARRAY_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class OpenedInitialData:
    """Only public RNA and conditions; no simulator truth is returned."""

    baseline: np.ndarray
    observations: dict[Condition, ReplicatedSnapshot]
    provenance: dict


def _relative(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == "."):
        raise ValueError("Archive paths must be canonical relative POSIX paths")
    return name


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json(payload: bytes) -> dict:
    def invalid_constant(value: str):
        raise ValueError(f"Invalid JSON constant: {value}")

    value = json.loads(payload, object_pairs_hook=_unique_object,
                       parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Manifest must be a JSON object")
    return value


def _record(condition: Condition) -> dict:
    return {"kind": "timecourse", "target": condition.target,
            "dose": condition.dose, "time": condition.time}


def _record_id(record: dict) -> str:
    # Registered public condition-key encoding; intentionally no generator import.
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _read_member(stream: tarfile.TarFile, member: tarfile.TarInfo, limit: int) -> bytes:
    _relative(member.name)
    if not member.isfile() or member.size < 0 or member.size > limit:
        raise ValueError("Selected archive members must be bounded ordinary files")
    handle = stream.extractfile(member)
    if handle is None:
        raise ValueError("Selected archive member is unreadable")
    payload = handle.read(limit + 1)
    if len(payload) != member.size:
        raise ValueError("Selected archive member has an invalid size")
    return payload


def _manifest(archive: Path, prefix: str) -> tuple[dict, str]:
    payload = None
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            if member.name != prefix + "manifest.json":
                continue
            if payload is not None:
                raise ValueError("Duplicate selected archive member: manifest.json")
            payload = _read_member(stream, member, _MAX_MANIFEST_BYTES)
    if payload is None:
        raise ValueError("The requested manifest is missing")
    manifest = _json(payload)
    # Existing development-v1 manifests predate an explicit public phase field.
    if (manifest.get("schema") != "scmultisim-development-v1"
            or manifest.get("formal_calibration") is not False
            or manifest.get("phase", "development") != "development"):
        raise ValueError("Only an explicitly development-schema dataset is allowed")
    if manifest.get("status") not in ("complete", "complete_with_stationarity_failures"):
        raise ValueError("The development dataset is not complete")
    return manifest, hashlib.sha256(payload).hexdigest()


def _layout(manifest: dict) -> tuple[tuple[int, ...], tuple[int, int, int], dict]:
    genes, cells = manifest.get("n_genes"), manifest.get("cells_per_replicate")
    if (type(genes) is not int or type(cells) is not int or min(genes, cells) < 1
            or 3 * genes * cells > _MAX_ARRAY_ELEMENTS):
        raise ValueError("Invalid or oversized replicate/cell/gene shape")
    initial = manifest.get("split", {}).get("initial")
    if (not isinstance(initial, list) or len(initial) != 6
            or any(type(q) is not int or not 0 <= q < genes for q in initial)
            or len(set(initial)) != 6):
        raise ValueError("Expected six unique in-bounds initial targets")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("Missing public condition records")
    return tuple(initial), (3, cells, genes), conditions


def _select_entry(conditions: dict, record: dict, directory: str) -> tuple[str, str, dict]:
    identifier = _record_id(record)
    entry = conditions.get(identifier)
    if (not isinstance(entry, dict) or entry.get("status") != "complete"
            or entry.get("condition") != record):
        raise ValueError("Missing, incomplete or inconsistent selected condition")
    relative = _relative(entry.get("observation"))
    if relative != f"{directory}/{identifier}.npz":
        raise ValueError("Selected observation redirects outside its exact opened path")
    checksum = entry.get("files", {}).get(relative)
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Selected observation lacks a valid SHA-256 checksum")
    return relative, checksum, entry


def _selected(manifest: dict) -> tuple[dict[str, tuple[Condition, str]], tuple[int, ...], tuple[int, int, int]]:
    initial, shape, conditions = _layout(manifest)
    expected = [COMMON_BASELINE, *(Condition(q, d, t) for q in initial
                                  for d in (0.3, 0.7) for t in (0.25, 1.0, 4.0))]
    selected = {}
    for condition in expected:
        relative, checksum, _ = _select_entry(conditions, _record(condition), "observations/initial")
        selected[relative] = (condition, checksum)
    return selected, initial, shape


def _diagnostic(value: object, target: int | None, dose: float) -> tuple[float, dict]:
    """Check public diagnostic identity, not the hidden generator parameters."""
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise ValueError("A passed stationary diagnostic is required")
    duration, report = value.get("burnin"), value.get("report")
    if (type(duration) not in (int, float) or not np.isfinite(duration) or duration <= 0
            or not isinstance(report, dict)):
        raise ValueError("Stationary diagnostic needs a finite positive duration and report")
    actual_target, actual_dose, tau = report.get("target"), report.get("dose"), report.get("tau")
    valid_target = actual_target is None if target is None else type(actual_target) is int and actual_target == target
    if ("target" not in report or not valid_target or type(actual_dose) not in (int, float) or actual_dose != dose
            or type(tau) not in (int, float) or not np.isfinite(tau) or tau <= 0
            or report.get("schema_version") != "scmultisim-stationarity-v1"
            or report.get("status") != "diagnostic_passed"
            or not isinstance(report.get("parameter_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", report["parameter_sha256"])):
        raise ValueError("Stationary diagnostic has inconsistent intervention, status, tau or fingerprint")
    return float(duration), report


def _selected_fullcourse(manifest: dict) -> tuple[dict, tuple[int, ...], tuple[int, int, int], dict]:
    initial, shape, conditions = _layout(manifest)
    _, baseline_report = _diagnostic(manifest.get("baseline_stationarity"), None, 0.)
    selected, durations = {}, {}
    for condition in [COMMON_BASELINE, *(Condition(q, .7, t) for q in initial
                                         for t in (.25, 1., 2., 4., 8.))]:
        directory = "sealed/final" if condition.time in (2., 8.) else "observations/initial"
        relative, checksum, _ = _select_entry(conditions, _record(condition), directory)
        selected[relative] = (condition, checksum)
    for target in initial:
        record = {"kind": "stationary", "target": target, "dose": .7}
        relative, checksum, entry = _select_entry(conditions, record, "sealed/stationary")
        duration, report = _diagnostic(entry.get("diagnostic"), target, .7)
        if (duration <= 8 or report["tau"] != baseline_report["tau"]
                or report["parameter_sha256"] != baseline_report["parameter_sha256"]):
            raise ValueError("Stationary endpoint must follow time 8 and match baseline tau and parameters")
        durations[str(target)] = duration
        selected[relative] = (Condition(target, .7, duration), checksum)
    return selected, initial, shape, durations


def _decode(payload: bytes, condition: Condition, shape: tuple[int, int, int]) -> ReplicatedSnapshot:
    expected = {"treated"} if condition == COMMON_BASELINE else {"treated", "matched_control"}
    # Check ZIP names, duplicates and expansion before np.load allocates arrays.
    with zipfile.ZipFile(io.BytesIO(payload)) as compressed:
        entries = compressed.infolist()
        if ({entry.filename for entry in entries} != {name + ".npy" for name in expected}
                or len(entries) != len(expected)
                or any(entry.file_size > 8 * _MAX_ARRAY_ELEMENTS + 4096 for entry in entries)):
            raise ValueError("Unexpected or oversized observation arrays")
        for entry in entries:
            with compressed.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise ValueError("Unsupported observation NPY format")
                declared_shape, _, dtype = readers[version](stream, max_header_size=4096)
                if (declared_shape != shape or not np.issubdtype(dtype, np.integer)
                        or dtype.itemsize > 8
                        or entry.file_size != stream.tell() + int(np.prod(shape)) * dtype.itemsize):
                    raise ValueError("Observation header disagrees with bounded integer RNA shape")
    with np.load(io.BytesIO(payload), allow_pickle=False) as values:
        arrays = {name: values[name] for name in expected}
    for array in arrays.values():
        if (array.shape != shape or not np.issubdtype(array.dtype, np.integer)
                or not np.isfinite(array).all() or np.any(array < 0)):
            raise ValueError("RNA must be nonnegative finite integer counts with the registered shape")
    return ReplicatedSnapshot(arrays["treated"], arrays.get("matched_control"))


def _read_snapshots(archive: Path, prefix: str, selected: dict,
                    shape: tuple[int, int, int]) -> dict[Condition, ReplicatedSnapshot]:
    """One bounded/checksummed decoding path shared by both explicit allowlists."""
    seen, snapshots = set(), {}
    total = 0
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            if not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix):]
            if relative not in selected:
                continue
            if relative in seen:
                raise ValueError("Duplicate selected archive member")
            seen.add(relative)
            payload = _read_member(stream, member, _MAX_OBSERVATION_BYTES)
            total += len(payload)
            if total > _MAX_SELECTED_BYTES:
                raise ValueError("Selected observations exceed the memory limit")
            condition, checksum = selected[relative]
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f"Observation checksum mismatch: {relative}")
            snapshots[condition] = _decode(payload, condition, shape)
    if seen != set(selected):
        raise ValueError("A selected initial observation is missing")
    return snapshots


def _load_initial(archive: Path, prefix: str, *, fullcourse: bool) -> OpenedInitialData:
    prefix = _relative(prefix.removesuffix("/")) + "/"
    archive = Path(archive)
    manifest, manifest_hash = _manifest(archive, prefix)
    if fullcourse:
        selected, initial, shape, durations = _selected_fullcourse(manifest)
    else:
        selected, initial, shape = _selected(manifest)
    snapshots = _read_snapshots(archive, prefix, selected, shape)
    baseline = snapshots.pop(COMMON_BASELINE).treated
    sealed_count = sum(path.startswith("sealed/") for path in selected)
    provenance = {
        "phase": "development", "schema": manifest["schema"],
        "manifest_sha256": manifest_hash,
        "verified_observation_sha256": {path: checksum for path, (_, checksum) in sorted(selected.items())},
        "verified_observation_count": len(selected), "initial_targets": list(initial),
        "n_genes": shape[2], "cells_per_replicate": shape[1],
        "decoded_private_or_sealed_files": sealed_count,
    }
    if fullcourse:
        provenance.update({
            "observation_scope": "initial_targets_fullcourse_dose_0.7",
            "decoded_private_files": 0, "decoded_sealed_observation_files": sealed_count,
            "newly_opened_initial_late_count": 12,
            "newly_opened_initial_stationary_count": 6,
            "longterm_duration_by_target": durations,
            "longterm_interpretation": "finite_time_samples_after_passed_distribution_diagnostic_not_fixed_points",
            "old_formal_held_time_purpose_changed": True,
            "test_status": "development_target_holdout_not_a_fresh_formal_blind_test",
        })
    return OpenedInitialData(baseline, snapshots, provenance)


def load_opened_initial(archive: Path, prefix: str = DEFAULT_PREFIX) -> OpenedInitialData:
    """Decode only the baseline and 36 originally opened initial conditions.

    Other split identities, private metadata and sealed arrays are not returned
    or decoded. Checksums verify integrity relative to the supplied manifest,
    not publisher authenticity. The tar file is never extracted.
    """
    return _load_initial(archive, prefix, fullcourse=False)


def load_fullcourse_initial(archive: Path, prefix: str = DEFAULT_PREFIX) -> OpenedInitialData:
    """Explicitly open six initial targets at dose .7, including late endpoints.

    Each target has .25/1/2/4/8 and its diagnostic-accepted long duration; the
    preperturbation baseline occurs once. Nothing under private or another
    target/dose is decoded. This deliberately changes the old held-time task.
    """
    return _load_initial(archive, prefix, fullcourse=True)


def _directory_payload(directory: Path, relative: str, limit: int) -> bytes:
    """Read one bounded ordinary public file without following path redirects."""
    relative = _relative(relative)
    path = directory / relative
    current = directory
    for component in PurePosixPath(relative).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("Selected public files must not follow symbolic links")
    if not path.resolve().is_relative_to(directory) or not path.is_file():
        raise ValueError(f"A selected public observation is missing or outside its directory: {relative}")
    if path.stat().st_size > limit:
        raise ValueError("Selected public files must have bounded size")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Selected public files must have bounded size")
    return payload


def _target_list(value: object, *, count: int, genes: int, name: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or len(value) != count
            or any(type(q) is not int or not 0 <= q < genes for q in value)
            or len(set(value)) != count):
        raise ValueError(f"{name} must contain {count} unique in-bounds targets")
    return tuple(value)


def _modular_layout(manifest: dict) -> tuple[tuple[int, ...], tuple[int, int, int], dict, dict]:
    """Check a public target registration, never an adjacency matrix or modules."""
    if (manifest.get("schema") != "scmultisim-q1-modular-v1"
            or manifest.get("phase") != "development"
            or manifest.get("formal_calibration") is not False
            or manifest.get("status") != "complete"):
        raise ValueError("A complete development modular-Q1 dataset is required")
    genes, cells = manifest.get("n_genes"), manifest.get("cells_per_replicate")
    if (type(genes) is not int or genes != 64 or type(cells) is not int or cells < 1
            or 3 * cells * genes > _MAX_ARRAY_ELEMENTS):
        raise ValueError("Invalid or oversized modular replicate/cell/gene shape")
    targets = _target_list(manifest.get("target_ids"), count=48, genes=genes, name="target_ids")
    split = manifest.get("q1_split")
    if not isinstance(split, dict) or split.get("registration") != "topology_before_rna":
        raise ValueError("The target split must be registered from topology before RNA generation")
    train = _target_list(split.get("train"), count=32, genes=genes, name="train")
    held = _target_list(split.get("held"), count=16, genes=genes, name="held")
    if set(train) & set(held) or set(train) | set(held) != set(targets):
        raise ValueError("Registered train/held targets must disjointly cover target_ids")
    times, dose = manifest.get("sampling_times"), manifest.get("dose")
    if (not isinstance(times, list) or any(type(t) not in (int, float) for t in times)
            or tuple(times) != (.25, 1., 2., 4., 8.)
            or type(dose) not in (int, float) or dose != .7):
        raise ValueError("Expected dose 0.7 and the registered .25/1/2/4/8 time grid")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("Missing public condition records")
    # Only these identities, not true module labels or topology, reach training.
    registration = {"train": list(train), "held": list(held),
                    "registration": "topology_before_rna"}
    return targets, (3, cells, genes), conditions, registration


def load_modular_q1(directory: Path) -> OpenedInitialData:
    """Load the 64-gene public Q1 RNA grid and its pre-RNA target split.

    The common long endpoint is an actual finite sampling time, not a forced
    fixed point. Each target must have passed a compatible distributional
    diagnostic by that time. Checksums ensure consistency with the supplied
    manifest; they do not authenticate the publisher or prove stationarity.
    The learner receives no matrix, module identity, latent G/U, or efficiency.
    """
    directory = Path(directory).resolve()
    payload = _directory_payload(directory, "manifest.json", _MAX_MANIFEST_BYTES)
    manifest = _json(payload)
    targets, shape, conditions, registration = _modular_layout(manifest)
    _, baseline_report = _diagnostic(manifest.get("baseline_stationarity"), None, 0.)
    long_time = manifest.get("common_longterm_time")
    if type(long_time) not in (int, float) or not np.isfinite(long_time) or long_time <= 8:
        raise ValueError("The actual common long-term sampling time must be finite and greater than 8")
    selected = {}
    grid = [COMMON_BASELINE, *(Condition(q, .7, t) for q in targets
                               for t in (*manifest["sampling_times"], float(long_time)))]
    for condition in grid:
        relative, checksum, entry = _select_entry(conditions, _record(condition), "observations/q1")
        if condition.time == long_time:
            duration, report = _diagnostic(entry.get("diagnostic"), condition.target, .7)
            actual_time = entry.get("actual_sample_time")
            if (type(actual_time) not in (int, float) or actual_time != long_time
                    or duration > long_time or report["tau"] != baseline_report["tau"]
                    or report["parameter_sha256"] != baseline_report["parameter_sha256"]):
                raise ValueError("Long-term RNA must use its actual time and compatible passed diagnostic")
        selected[relative] = (condition, checksum)
    snapshots, total = {}, 0
    for relative, (condition, checksum) in selected.items():
        observation = _directory_payload(directory, relative, _MAX_OBSERVATION_BYTES)
        total += len(observation)
        if total > _MAX_SELECTED_BYTES:
            raise ValueError("Selected observations exceed the memory limit")
        if hashlib.sha256(observation).hexdigest() != checksum:
            raise ValueError(f"Observation checksum mismatch: {relative}")
        snapshots[condition] = _decode(observation, condition, shape)
    baseline = snapshots.pop(COMMON_BASELINE).treated
    provenance = {
        "phase": "development", "schema": manifest["schema"],
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "verified_observation_sha256": {path: checksum for path, (_, checksum) in sorted(selected.items())},
        "verified_observation_count": len(selected), "initial_targets": list(targets),
        "n_genes": shape[2], "cells_per_replicate": shape[1],
        "registered_split": registration, "decoded_private_files": 0,
        "decoded_private_or_sealed_files": 0,
        "observation_scope": "modular_q1_all_registered_targets_dose_0.7",
        "common_longterm_time": float(long_time),
        "longterm_interpretation": "finite_time_samples_after_passed_distribution_diagnostic_not_fixed_points",
        "test_status": "development_pre_registered_target_holdout_not_formal_network_blind_test",
    }
    return OpenedInitialData(baseline, snapshots, provenance)


def split_registered_targets(data: OpenedInitialData
                             ) -> tuple[dict[Condition, ReplicatedSnapshot], dict[Condition, ReplicatedSnapshot]]:
    """Apply the pre-RNA public split without inspecting any response values."""
    split = data.provenance.get("registered_split")
    if not isinstance(split, dict) or split.get("registration") != "topology_before_rna":
        raise ValueError("A topology-before-RNA target registration is required")
    genes = data.baseline.shape[-1]
    train = set(_target_list(split.get("train"), count=32, genes=genes, name="train"))
    held = set(_target_list(split.get("held"), count=16, genes=genes, name="held"))
    if train & held or {key.target for key in data.observations} != train | held:
        raise ValueError("Observations must contain exactly the disjoint registered targets")
    return ({key: value for key, value in data.observations.items() if key.target in train},
            {key: value for key, value in data.observations.items() if key.target in held})


def split_targets(data: OpenedInitialData, seed: int = 20261022, train_count: int = 4
                  ) -> tuple[dict[Condition, ReplicatedSnapshot], dict[Condition, ReplicatedSnapshot]]:
    """Response-independent target holdout; every time/replicate stays together."""
    targets = data.provenance.get("initial_targets")
    genes = data.baseline.shape[-1]
    if (not isinstance(targets, list) or len(targets) < 2
            or any(type(q) is not int or not 0 <= q < genes for q in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError("Expected at least two unique declared initial targets")
    if type(seed) is not int or seed < 0 or type(train_count) is not int or not 1 <= train_count < len(targets):
        raise ValueError("A nonnegative integer seed and a nonempty train/test split are required")
    if {key.target for key in data.observations} != set(targets):
        raise ValueError("Observations must contain exactly the declared initial targets")
    ordered = np.random.default_rng(seed).permutation(sorted(targets))
    train_targets = set(ordered[:train_count].tolist())
    train = {key: value for key, value in data.observations.items() if key.target in train_targets}
    test = {key: value for key, value in data.observations.items() if key.target not in train_targets}
    return train, test


def split_fit_interpolation(data: OpenedInitialData) -> tuple[dict[Condition, ReplicatedSnapshot], dict[Condition, ReplicatedSnapshot]]:
    """Pilot-only time split: .25/4 fit, time 1 held for interpolation.

    This reuses opened development observations, not sealed protocol tests.
    Replicate splitting remains the caller's responsibility: only replicates
    zero and one may enter parameter fitting; replicate two is a check.
    """
    fit = {key: value for key, value in data.observations.items() if key.time in (0.25, 4.0)}
    held = {key: value for key, value in data.observations.items() if key.time == 1.0}
    if len(fit) != 24 or len(held) != 12 or len(fit) + len(held) != len(data.observations):
        raise ValueError("Expected the complete 36-condition initial grid")
    return fit, held
