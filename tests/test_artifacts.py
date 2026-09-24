import json
from pathlib import Path
import tempfile
import unittest

from grn_experiment.artifacts import ArtifactRegistry, EventLog, RunManifest, sha256_json
from grn_experiment.protocol import Condition


class ArtifactLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "learner"
        self.root.mkdir()
        self.manifest = RunManifest.create(
            run_id="dev-00-sigmoid-active",
            git_sha="a" * 40,
            protocol_hash=sha256_json({"protocol": 1, "budget": 12}),
            config_hash=sha256_json({"lr": 0.001}),
            dependency_lock_hash=sha256_json({"locked": []}),
            seed=42,
        )
        self.manifest.write(self.root)
        self.artifacts = ArtifactRegistry(self.root, role="learner")
        self.condition = Condition(3, 0.3, 0.25)

    def test_manifest_immutable_and_resume_verifies_manifest(self):
        self.assertEqual(RunManifest.load(self.root, expected_fingerprint=self.manifest.fingerprint), self.manifest)
        with self.assertRaises(FileExistsError):
            self.manifest.write(self.root)
        self.assertIsNone(self.manifest.environment.torch_version)
        altered = self.root / "manifest.json"
        data = json.loads(altered.read_text())
        data["seed"] = 999
        altered.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            EventLog.resume(
                self.root, strategy="active", artifacts=self.artifacts,
                expected_manifest_fingerprint=self.manifest.fingerprint,
            )

    def test_relative_paths_role_scope_and_no_overwrite(self):
        record = self.artifacts.write("checkpoints/step-01.bin", b"weights")
        self.assertEqual((record.role, record.byte_size, record.path), ("learner", 7, "checkpoints/step-01.bin"))
        self.assertEqual(self.artifacts.records(), (record,))
        with self.assertRaises(FileExistsError):
            self.artifacts.write(record.path, b"different")
        orphan = self.root / "checkpoints" / "interrupted.bin"
        orphan.write_bytes(b"recoverable")
        recovered = self.artifacts.write("checkpoints/interrupted.bin", b"recoverable")
        self.assertEqual(self.artifacts.get(recovered.path), recovered)
        different = self.root / "checkpoints" / "orphan-other.bin"
        different.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.artifacts.write("checkpoints/orphan-other.bin", b"new")
        for name in ("../truth.bin", "/tmp/truth.bin", "a/../../truth.bin", ".ledger/x"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.artifacts.write(name, b"x")
        scorer_root = Path(self.directory.name) / "scorer"
        scorer = ArtifactRegistry(scorer_root, role="scorer")
        scorer.write("truth.bin", b"secret")
        self.assertEqual(self.artifacts.records(), (record, recovered))
        with self.assertRaises(ValueError):
            ArtifactRegistry(self.root, role="scorer")
        with self.assertRaises(FileNotFoundError):
            self.artifacts.get("truth.bin")

    def test_atomic_events_duplicate_prevention_and_deterministic_resume(self):
        observation = self.artifacts.write("observations/c03.bin", b"sampled-cells")
        checkpoint = self.artifacts.write("checkpoints/model.bin", b"model")
        predictions = self.artifacts.write("predictions/test.json", b"{}")
        log = EventLog(self.root, strategy="active", artifacts=self.artifacts, expected_budget=1)
        log.select(self.condition)
        with self.assertRaises(ValueError):
            log.select(self.condition)
        resumed = EventLog.resume(
            self.root, strategy="active", artifacts=self.artifacts,
            expected_manifest_fingerprint=self.manifest.fingerprint, expected_budget=1,
        )
        resumed.acquire(self.condition, observation)
        with self.assertRaises(ValueError):
            resumed.acquire(self.condition, observation)
        with self.assertRaises(ValueError):
            resumed.select(self.condition)
        resumed.checkpoint("model-a", checkpoint)
        resumed.validate("model-a", 0.1, checkpoint)
        resumed.lock("model-a", checkpoint, predictions)
        with self.assertRaises(ValueError):
            resumed.select(Condition(4, 0.5, 1.0))
        verified = EventLog.resume(
            self.root, strategy="active", artifacts=self.artifacts,
            expected_manifest_fingerprint=self.manifest.fingerprint, expected_budget=1,
        )
        self.assertEqual([event["kind"] for event in verified.events()],
                         ["selected", "acquired", "checkpoint", "validation", "final_lock"])
        self.assertEqual([event["sequence"] for event in verified.events()], list(range(1, 6)))
        (self.root / predictions.path).write_bytes(b"altered")
        with self.assertRaises(ValueError):
            EventLog.resume(
                self.root, strategy="active", artifacts=self.artifacts,
                expected_manifest_fingerprint=self.manifest.fingerprint, expected_budget=1,
            )

    def test_artifact_and_event_tampering_detected_on_resume(self):
        observation = self.artifacts.write("observations/c03.bin", b"sampled-cells")
        log = EventLog(self.root, strategy="active", artifacts=self.artifacts, expected_budget=1)
        log.select(self.condition)
        log.acquire(self.condition, observation)
        artifact_path = self.root / observation.path
        artifact_path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            EventLog.resume(self.root, strategy="active", artifacts=self.artifacts,
                            expected_manifest_fingerprint=self.manifest.fingerprint, expected_budget=1)
        artifact_path.write_bytes(b"sampled-cells")
        event_path = self.root / ".ledger/events/active/000001.json"
        event = json.loads(event_path.read_text())
        event["details"]["condition"]["target"] = 5
        event_path.write_text(json.dumps(event))
        with self.assertRaises(ValueError):
            EventLog.resume(self.root, strategy="active", artifacts=self.artifacts,
                            expected_manifest_fingerprint=self.manifest.fingerprint, expected_budget=1)

    def test_budget_and_prediction_commitment_required(self):
        observation = self.artifacts.write("observations/c03.bin", b"sampled-cells")
        checkpoint = self.artifacts.write("checkpoints/model.bin", b"model")
        predictions = self.artifacts.write("predictions/test.json", b"{}")
        log = EventLog(self.root, strategy="active", artifacts=self.artifacts)
        log.select(self.condition)
        log.acquire(self.condition, observation)
        with self.assertRaises(ValueError):
            log.validate("model-a", 0.1, checkpoint)
        with self.assertRaises(ValueError):
            log.lock("model-a", checkpoint, predictions)
        with self.assertRaises(ValueError):
            EventLog.resume(self.root, strategy="active", artifacts=self.artifacts,
                            expected_manifest_fingerprint=self.manifest.fingerprint, expected_budget=1)

    def test_validation_and_final_lock_bind_model_to_exact_checkpoint(self):
        observation = self.artifacts.write("observations/c03.bin", b"sampled-cells")
        checkpoint_a = self.artifacts.write("checkpoints/model-a-01.bin", b"model-a-first")
        checkpoint_a_other = self.artifacts.write("checkpoints/model-a-02.bin", b"model-a-other")
        checkpoint_b = self.artifacts.write("checkpoints/model-b.bin", b"model-b")
        predictions = self.artifacts.write("predictions/test.json", b"{}")
        log = EventLog(self.root, strategy="active", artifacts=self.artifacts, expected_budget=1)
        log.select(self.condition)
        log.acquire(self.condition, observation)
        log.checkpoint("model-a", checkpoint_a)
        log.checkpoint("model-a", checkpoint_a_other)
        log.checkpoint("model-b", checkpoint_b)
        with self.assertRaises(ValueError):
            log.validate("model-b", 0.2, checkpoint_a)
        with self.assertRaises(ValueError):
            log.validate("model-a", 0.2, checkpoint_b)
        log.validate("model-a", 0.1, checkpoint_a)
        with self.assertRaises(ValueError):
            log.lock("model-b", checkpoint_b, predictions)
        with self.assertRaises(ValueError):
            log.lock("model-a", checkpoint_a_other, predictions)
        log.lock("model-a", checkpoint_a, predictions)
        resumed = EventLog.resume(self.root, strategy="active", artifacts=self.artifacts,
                                  expected_manifest_fingerprint=self.manifest.fingerprint,
                                  expected_budget=1)
        self.assertEqual(resumed.events()[-1]["details"]["checkpoint_sha256"], checkpoint_a.sha256)

    def test_symlink_escape_rejected(self):
        outside = Path(self.directory.name) / "outside"
        outside.mkdir()
        (self.root / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.artifacts.write("linked/file.bin", b"x")


if __name__ == "__main__":
    unittest.main()
