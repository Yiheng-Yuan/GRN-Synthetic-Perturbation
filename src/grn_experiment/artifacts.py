"""Small, role-scoped run ledger for portable experiment artifacts.

Use distinct filesystem roots for learner and scorer.  A learner registry never
lists or opens scorer artifacts through this API; Python objects and filesystem
permissions are not a security boundary if both roots are handed to one process.
No observations, simulator truth, or model tensors are embedded in ledger JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Literal, Mapping

SCHEMA_VERSION = 1
Role = Literal["learner", "scorer"]
EventKind = Literal["selected", "acquired", "checkpoint", "validation", "final_lock"]
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash a JSON-serializable, versioned protocol/config payload canonically."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_create(path: Path, payload: bytes) -> None:
    """Publish a complete file without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-clobber publish on local filesystems.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"Invalid {label}")
    return value


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Artifact path must be a relative POSIX path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or parts[0] == ".ledger":
        raise ValueError("Artifact path escapes or uses the reserved ledger directory")
    return value


def _under_root(root: Path, relative: str) -> Path:
    path = root.joinpath(*relative.split("/"))
    # Reject a pre-existing symlink in any path component, including metadata.
    node = root
    for part in relative.split("/"):
        node = node / part
        if node.is_symlink():
            raise ValueError("Symlinked artifact paths are not allowed")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Artifact path escapes the supplied root")
    return path


@dataclass(frozen=True)
class Environment:
    hostname: str
    platform: str
    python_version: str
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available: bool | None = None

    @classmethod
    def capture(cls, *, include_torch: bool = False) -> Environment:
        version = cuda = available = None
        if include_torch:
            import torch  # Optional and deliberately never imported at module load.

            version = str(torch.__version__)
            cuda = str(torch.version.cuda) if torch.version.cuda is not None else None
            available = bool(torch.cuda.is_available())
        return cls(platform.node(), platform.platform(), platform.python_version(), version, cuda, available)


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    created_utc: str
    git_sha: str
    protocol_hash: str
    config_hash: str
    dependency_lock_hash: str
    seed: int
    environment: Environment

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported manifest schema")
        _validate_id(self.run_id, "run_id")
        if _GIT_SHA.fullmatch(self.git_sha) is None:
            raise ValueError("git_sha must be a full hexadecimal commit ID")
        if any(_HEX_64.fullmatch(value) is None for value in
               (self.protocol_hash, self.config_hash, self.dependency_lock_hash)):
            raise ValueError("Protocol, config, and dependency lock hashes must be SHA-256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        stamp = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            raise ValueError("created_utc must be in UTC")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        git_sha: str,
        protocol_hash: str,
        config_hash: str,
        dependency_lock_hash: str,
        seed: int,
        include_torch: bool = False,
    ) -> RunManifest:
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        return cls(SCHEMA_VERSION, run_id, now, git_sha, protocol_hash, config_hash,
                   dependency_lock_hash, seed,
                   Environment.capture(include_torch=include_torch))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_json_bytes(asdict(self))).hexdigest()

    def write(self, root: str | Path) -> Path:
        path = Path(root) / "manifest.json"
        _atomic_create(path, _json_bytes(asdict(self)))
        return path

    @classmethod
    def load(cls, root: str | Path, *, expected_fingerprint: str | None = None) -> RunManifest:
        raw = (Path(root) / "manifest.json").read_bytes()
        if expected_fingerprint is not None and hashlib.sha256(raw).hexdigest() != expected_fingerprint:
            raise ValueError("Run manifest fingerprint mismatch")
        data = json.loads(raw)
        data["environment"] = Environment(**data["environment"])
        manifest = cls(**data)
        if raw != _json_bytes(asdict(manifest)):
            raise ValueError("Run manifest is not canonical")
        return manifest


@dataclass(frozen=True)
class ArtifactRecord:
    schema_version: int
    role: Role
    path: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.role not in ("learner", "scorer"):
            raise ValueError("Invalid artifact schema or role")
        _safe_relative(self.path)
        if _HEX_64.fullmatch(self.sha256) is None or self.byte_size < 0:
            raise ValueError("Invalid artifact digest or size")


class ArtifactRegistry:
    """Immutable artifacts under one caller-owned, role-specific root.

    Metadata contains only path, digest, byte count, and role.  Passing the
    scorer root to learner code would defeat role separation; keep those roots
    in separate processes and apply OS permissions for an actual boundary.
    """

    def __init__(self, root: str | Path, *, role: Role) -> None:
        if role not in ("learner", "scorer"):
            raise ValueError("Invalid artifact role")
        self.root = Path(root).resolve()
        self.role = role
        marker = self.root / ".ledger" / "role.json"
        if not marker.exists():
            try:
                _atomic_create(marker, _json_bytes({"role": role}))
            except FileExistsError:
                pass
        if marker.read_bytes() != _json_bytes({"role": role}):
            raise ValueError("Artifact root was initialized for a different role")

    def _metadata_path(self, name: str) -> Path:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return self.root / ".ledger" / "artifacts" / f"{digest}.json"

    def write(self, name: str, payload: bytes) -> ArtifactRecord:
        name = _safe_relative(name)
        if not isinstance(payload, bytes):
            raise TypeError("Artifact payload must be bytes")
        path = _under_root(self.root, name)
        record = ArtifactRecord(SCHEMA_VERSION, self.role, name, hashlib.sha256(payload).hexdigest(), len(payload))
        if self._metadata_path(name).exists():
            raise FileExistsError(name)
        if not path.exists():
            try:
                _atomic_create(path, payload)
            except FileExistsError:
                pass  # Another writer or interrupted publication; compare bytes below.
        # A crash between artifact and metadata publication leaves an orphan.
        # The same bytes may finalize it; different bytes must never replace it.
        if path.stat().st_size != len(payload) or sha256_file(path) != record.sha256:
            raise FileExistsError(f"Unregistered artifact exists with different bytes: {name}")
        _atomic_create(self._metadata_path(name), _json_bytes(asdict(record)))
        return record

    def get(self, name: str) -> ArtifactRecord:
        name = _safe_relative(name)
        raw = self._metadata_path(name).read_bytes()
        record = ArtifactRecord(**json.loads(raw))
        if record.role != self.role or record.path != name or raw != _json_bytes(asdict(record)):
            raise ValueError("Artifact registry metadata mismatch")
        path = _under_root(self.root, record.path)
        if path.stat().st_size != record.byte_size or sha256_file(path) != record.sha256:
            raise ValueError(f"Artifact content mismatch: {name}")
        return record

    def verify(self, record: ArtifactRecord) -> None:
        if record.role != self.role or self.get(record.path) != record:
            raise ValueError("Artifact record mismatch")

    def records(self) -> tuple[ArtifactRecord, ...]:
        directory = self.root / ".ledger" / "artifacts"
        if not directory.exists():
            return ()
        result = []
        for path in sorted(directory.glob("*.json")):
            record = ArtifactRecord(**json.loads(path.read_bytes()))
            if path != self._metadata_path(record.path):
                raise ValueError("Artifact metadata filename mismatch")
            result.append(self.get(record.path))
        return tuple(result)


def _condition(value: Any) -> dict[str, int | float]:
    try:
        target, dose, time = value.target, float(value.dose), float(value.time)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Expected a targeted Condition") from exc
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise ValueError("Selection target must be a nonnegative integer")
    if not (math.isfinite(dose) and 0 < dose <= 1 and math.isfinite(time) and time > 0):
        raise ValueError("Selection dose/time invalid")
    return {"target": target, "dose": dose, "time": time}


class EventLog:
    """One strategy's atomic, hash-chained event files and deterministic replay.

    Hashes catch accidental corruption; they are not signatures and cannot
    prevent a writer with root access from rewriting an entire chain.
    """

    def __init__(self, root: str | Path, *, strategy: str, artifacts: ArtifactRegistry,
                 expected_budget: int = 12) -> None:
        self.root = Path(root).resolve()
        self.strategy = _validate_id(strategy, "strategy")
        self.artifacts = artifacts
        if isinstance(expected_budget, bool) or not isinstance(expected_budget, int) or expected_budget < 1:
            raise ValueError("expected_budget must be positive")
        self.expected_budget = expected_budget
        if artifacts.root != self.root:
            raise ValueError("Event log and artifact registry must share one role-specific root")
        self.directory = self.root / ".ledger" / "events" / strategy

    def events(self) -> tuple[dict[str, Any], ...]:
        if not self.directory.exists():
            return ()
        files = sorted(self.directory.glob("*.json"))
        state: dict[str, Any] = {"pending": None, "selected": set(), "acquired": set(),
                                 "checkpoints": {}, "validated_models": {}, "locked": False}
        previous = "0" * 64
        result = []
        for sequence, file in enumerate(files, 1):
            if file.name != f"{sequence:06d}.json":
                raise ValueError("Event sequence gap or unexpected filename")
            raw = file.read_bytes()
            event = json.loads(raw)
            if raw != _json_bytes(event):
                raise ValueError("Event file is not canonical")
            if (event.get("schema_version") != SCHEMA_VERSION or event.get("strategy") != self.strategy
                    or event.get("expected_budget") != self.expected_budget):
                raise ValueError("Event schema, strategy, or budget mismatch")
            if event.get("sequence") != sequence or event.get("previous_hash") != previous:
                raise ValueError("Event sequence or hash chain mismatch")
            claimed_hash = event.get("event_hash")
            body = {key: value for key, value in event.items() if key != "event_hash"}
            if hashlib.sha256(_json_bytes(body)).hexdigest() != claimed_hash:
                raise ValueError("Event content hash mismatch")
            self._apply(state, event["kind"], event["details"], verify_artifacts=True)
            previous = claimed_hash
            result.append(event)
        return tuple(result)

    def _apply(self, state: dict[str, Any], kind: str, details: Mapping[str, Any], *, verify_artifacts: bool) -> None:
        if state["locked"]:
            raise ValueError("Final lock forbids further events")
        if kind == "selected":
            condition = details["condition"]
            key = _json_bytes(condition)
            if (state["pending"] is not None or key in state["selected"]
                    or len(state["selected"]) >= self.expected_budget):
                raise ValueError("Condition already selected or an acquisition is pending")
            if _condition(_object_condition(condition)) != condition:
                raise ValueError("Invalid selected condition")
            state["selected"].add(key)
            state["pending"] = key
        elif kind == "acquired":
            key = _json_bytes(details["condition"])
            if state["pending"] != key or key in state["acquired"]:
                raise ValueError("Acquisition must match one fresh selection")
            self._check_artifact(details, "observation", verify_artifacts)
            state["acquired"].add(key)
            state["pending"] = None
        elif kind == "checkpoint":
            model_id = _validate_id(details["model_id"], "model_id")
            key = self._check_artifact(details, "checkpoint", verify_artifacts)
            state["checkpoints"].setdefault(model_id, set()).add(key)
        elif kind == "validation":
            if state["pending"] is not None or len(state["acquired"]) != self.expected_budget:
                raise ValueError("Validation needs the full completed acquisition budget")
            model_id = _validate_id(details["model_id"], "model_id")
            if (isinstance(details["score"], bool) or not isinstance(details["score"], (int, float))
                    or not math.isfinite(details["score"])):
                raise ValueError("Invalid validation score")
            key = self._check_artifact(details, "checkpoint", verify_artifacts)
            if key not in state["checkpoints"].get(model_id, set()):
                raise ValueError("Validation checkpoint was not registered for this model")
            if model_id in state["validated_models"]:
                raise ValueError("Model validation was already recorded")
            state["validated_models"][model_id] = key
        elif kind == "final_lock":
            model_id = _validate_id(details["model_id"], "model_id")
            if state["pending"] is not None or model_id not in state["validated_models"]:
                raise ValueError("Final lock needs a validated model and no pending selection")
            key = self._check_artifact(details, "checkpoint", verify_artifacts)
            if key != state["validated_models"][model_id]:
                raise ValueError("Final lock must use this model's validated checkpoint")
            self._check_artifact(details, "predictions", verify_artifacts)
            state["locked"] = True
        else:
            raise ValueError("Unknown event kind")

    def _check_artifact(self, details: Mapping[str, Any], prefix: str, verify: bool) -> tuple[str, str]:
        name = _safe_relative(details[f"{prefix}_path"])
        digest = details[f"{prefix}_sha256"]
        if _HEX_64.fullmatch(digest) is None:
            raise ValueError("Invalid event artifact digest")
        if verify and self.artifacts.get(name).sha256 != digest:
            raise ValueError("Event artifact hash mismatch")
        return name, digest

    def _append(self, kind: EventKind, details: dict[str, Any]) -> dict[str, Any]:
        previous_events = self.events()  # Replay before each append, including artifact verification.
        state: dict[str, Any] = {"pending": None, "selected": set(), "acquired": set(),
                                 "checkpoints": {}, "validated_models": {}, "locked": False}
        for event in previous_events:
            self._apply(state, event["kind"], event["details"], verify_artifacts=False)
        self._apply(state, kind, details, verify_artifacts=True)
        sequence = len(previous_events) + 1
        body = {"schema_version": SCHEMA_VERSION, "strategy": self.strategy,
                "expected_budget": self.expected_budget,
                "sequence": sequence, "previous_hash": previous_events[-1]["event_hash"] if previous_events else "0" * 64,
                "kind": kind, "details": details}
        event = {**body, "event_hash": hashlib.sha256(_json_bytes(body)).hexdigest()}
        _atomic_create(self.directory / f"{sequence:06d}.json", _json_bytes(event))
        return event

    def select(self, condition: Any) -> dict[str, Any]:
        return self._append("selected", {"condition": _condition(condition)})

    def acquire(self, condition: Any, observation: ArtifactRecord) -> dict[str, Any]:
        self.artifacts.verify(observation)
        return self._append("acquired", {"condition": _condition(condition),
                                         "observation_path": observation.path,
                                         "observation_sha256": observation.sha256})

    def checkpoint(self, model_id: str, artifact: ArtifactRecord) -> dict[str, Any]:
        self.artifacts.verify(artifact)
        return self._append("checkpoint", {"model_id": _validate_id(model_id, "model_id"),
                                           "checkpoint_path": artifact.path,
                                           "checkpoint_sha256": artifact.sha256})

    def validate(self, model_id: str, score: float, checkpoint: ArtifactRecord) -> dict[str, Any]:
        self.artifacts.verify(checkpoint)
        return self._append("validation", {"model_id": _validate_id(model_id, "model_id"),
                                           "score": float(score),
                                           "checkpoint_path": checkpoint.path,
                                           "checkpoint_sha256": checkpoint.sha256})

    def lock(self, model_id: str, checkpoint: ArtifactRecord,
             predictions: ArtifactRecord) -> dict[str, Any]:
        self.artifacts.verify(checkpoint)
        self.artifacts.verify(predictions)
        return self._append("final_lock", {"model_id": _validate_id(model_id, "model_id"),
                                           "checkpoint_path": checkpoint.path,
                                           "checkpoint_sha256": checkpoint.sha256,
                                           "predictions_path": predictions.path,
                                           "predictions_sha256": predictions.sha256})

    @classmethod
    def resume(
        cls, root: str | Path, *, strategy: str, artifacts: ArtifactRegistry,
        expected_manifest_fingerprint: str, expected_budget: int = 12,
    ) -> EventLog:
        RunManifest.load(root, expected_fingerprint=expected_manifest_fingerprint)
        ledger = cls(root, strategy=strategy, artifacts=artifacts,
                     expected_budget=expected_budget)
        ledger.events()  # Verify chain, state, and all referenced artifact hashes.
        artifacts.records()  # Verify even registered artifacts not mentioned in events.
        return ledger


def _object_condition(value: Mapping[str, Any]) -> Any:
    """Adapt decoded JSON to the same condition validation used by selection."""
    return SimpleNamespace(**value)
