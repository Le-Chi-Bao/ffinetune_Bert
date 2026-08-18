"""Stage-13 tests: manifest validation, checksum verification, one-time test guard.

Critically, no test in this file opens the official AG News test split. The
official-test loader is never called; the guarded runner is exercised only in
--dry-run mode and through deliberate rejection cases.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

from scripts.validate_stage13_manifest import (  # noqa: E402
    ManifestError,
    validate_manifest,
    verify_checkpoints,
    verify_manifest_hash,
)
from stage12_statistics import file_sha256, payload_sha256  # noqa: E402
from stage13_official_test import (  # noqa: E402
    OfficialTestAccessError,
    assert_first_official_test_access,
    paired_prediction_contingency,
    record_official_test_access,
)
from tests.fixtures import make_tiny_checkpoint, write_json  # noqa: E402

SEEDS = [0, 1, 2]


def build_manifest(tmp: Path, **overrides) -> Path:
    """Create a locked Stage-13 manifest with real checkpoint digests."""
    models = {}
    for name in ("bert_baseline", "ldtf_selected"):
        entries = []
        for seed in SEEDS:
            checkpoint = make_tiny_checkpoint(
                tmp / name / f"seed_{seed}" / "checkpoints" / "best.pt", seed=seed
            )
            entries.append(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                }
            )
        models[name] = entries
    core = {
        "stage": 13,
        "protocol_hash": "c" * 64,
        "data_signature_hash": "d" * 64,
        "official_test_evaluated": False,
        "seeds": list(SEEDS),
        "models": models,
        "selection_note": "All locked seeds are included; no best-seed cherry-picking is permitted.",
    }
    core.update(overrides)
    return write_json({**core, "manifest_sha256": payload_sha256(core)}, tmp / "manifest.json")


class TestManifestValidation(unittest.TestCase):
    def test_valid_manifest_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = validate_manifest(build_manifest(Path(tmp)))
            self.assertEqual(resolved["seeds"], SEEDS)
            self.assertEqual(len(resolved["checkpoints"]), 6)
            self.assertEqual(resolved["models"], ["bert_baseline", "ldtf_selected"])

    def test_tampered_manifest_body_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_manifest(Path(tmp))
            manifest = json.loads(path.read_text())
            manifest["protocol_hash"] = "e" * 64  # hash no longer matches the body
            write_json(manifest, path)
            with self.assertRaises(ManifestError) as caught:
                validate_manifest(path)
            self.assertIn("checksum mismatch", str(caught.exception).lower())

    def test_missing_manifest_hash_is_rejected(self):
        with self.assertRaises(ManifestError):
            verify_manifest_hash({"stage": 13})

    def test_manifest_already_marked_evaluated_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_manifest(Path(tmp), official_test_evaluated=True)
            with self.assertRaises(ManifestError) as caught:
                validate_manifest(path)
            self.assertIn("already", str(caught.exception).lower())

    def test_non_stage13_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_manifest(Path(tmp), stage=12)
            with self.assertRaises(ManifestError):
                validate_manifest(path)

    def test_missing_manifest_file_is_rejected(self):
        with self.assertRaises(ManifestError):
            validate_manifest("/nonexistent/manifest.json")


class TestCheckpointChecksums(unittest.TestCase):
    def test_modified_checkpoint_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_manifest(Path(tmp))
            manifest = json.loads(path.read_text())
            victim = Path(manifest["models"]["ldtf_selected"][1]["checkpoint"])
            victim.write_bytes(b"tampered-checkpoint")
            with self.assertRaises(ManifestError) as caught:
                verify_checkpoints(manifest)
            self.assertIn("checkpoint checksum mismatch", str(caught.exception).lower())

    def test_missing_checkpoint_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_manifest(Path(tmp))
            manifest = json.loads(path.read_text())
            Path(manifest["models"]["bert_baseline"][0]["checkpoint"]).unlink()
            with self.assertRaises(ManifestError) as caught:
                verify_checkpoints(manifest)
            self.assertIn("missing", str(caught.exception).lower())

    def test_dropped_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_manifest(Path(tmp))
            manifest = json.loads(path.read_text())
            manifest["models"]["ldtf_selected"] = manifest["models"]["ldtf_selected"][:2]
            with self.assertRaises(ManifestError) as caught:
                verify_checkpoints(manifest)
            self.assertIn("never be dropped", str(caught.exception).lower())

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_manifest(Path(tmp))
            manifest = json.loads(path.read_text())
            entries = manifest["models"]["ldtf_selected"]
            entries[2] = dict(entries[1])
            with self.assertRaises(ManifestError):
                verify_checkpoints(manifest)


class TestOneTimeAccessGuard(unittest.TestCase):
    def test_first_access_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "access.json"
            self.assertEqual(assert_first_official_test_access(log, "hash-a"), [])

    def test_second_access_to_same_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "access.json"
            assert_first_official_test_access(log, "hash-a")
            record_official_test_access(log, "hash-a", {"models": ["m"]})
            with self.assertRaises(OfficialTestAccessError) as caught:
                assert_first_official_test_access(log, "hash-a")
            self.assertIn("already been evaluated", str(caught.exception).lower())

    def test_a_new_manifest_version_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "access.json"
            record_official_test_access(log, "hash-a", {})
            self.assertEqual(len(assert_first_official_test_access(log, "hash-b")), 1)

    def test_access_log_is_appended_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "access.json"
            record_official_test_access(log, "hash-a", {})
            record_official_test_access(log, "hash-b", {})
            self.assertEqual(len(json.loads(log.read_text())), 2)


class TestPairedPredictionComparison(unittest.TestCase):
    def test_contingency_counts_are_correct(self):
        labels = [0, 1, 2, 3]
        baseline = [0, 1, 0, 0]  # correct on 0,1
        candidate = [0, 0, 2, 0]  # correct on 0,2
        table = paired_prediction_contingency(labels, baseline, candidate)
        self.assertEqual(table["both_correct"], 1)
        self.assertEqual(table["only_baseline_correct"], 1)
        self.assertEqual(table["only_candidate_correct"], 1)
        self.assertEqual(table["both_wrong"], 1)
        self.assertEqual(table["discordant_pairs"], 2)
        self.assertAlmostEqual(table["accuracy_delta"], 0.0)

    def test_delta_sign_favors_candidate(self):
        table = paired_prediction_contingency([0, 0], [1, 1], [0, 0])
        self.assertEqual(table["only_candidate_correct"], 2)
        self.assertAlmostEqual(table["accuracy_delta"], 1.0)

    def test_no_significance_is_claimed(self):
        table = paired_prediction_contingency([0], [0], [0])
        self.assertIn("no significance test", table["statistical_claim"].lower())

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_prediction_contingency([0, 1], [0], [0, 1])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_prediction_contingency([], [], [])


class TestStage13RunnerGuards(unittest.TestCase):
    """The runner must never reach the official test split in these cases."""

    def _run(self, manifest: Path, output_root: Path, *extra: str):
        return subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "run_stage13_official_test.py"),
                "--manifest", str(manifest), "--output-root", str(output_root), *extra,
            ],
            capture_output=True, text=True, cwd=ROOT,
        )

    def test_dry_run_validates_without_loading_official_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest(Path(tmp))
            root = Path(tmp) / "stage13"
            result = self._run(manifest, root, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("official test was not loaded", result.stdout.lower())
            report = json.loads((root / "stage13_dry_run.json").read_text())
            self.assertIs(report["official_test_loaded"], False)
            self.assertEqual(report["checkpoints_verified"], 6)
            self.assertFalse((root / "official_test_access_log.json").exists())

    def test_real_run_requires_explicit_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(build_manifest(Path(tmp)), Path(tmp) / "stage13")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("acknowledgement", result.stderr.lower())

    def test_truncated_official_test_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(
                build_manifest(Path(tmp)), Path(tmp) / "stage13",
                "--i-understand-this-consumes-the-one-time-official-test",
                "--max-batches", "5",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("truncates", result.stderr.lower())

    def test_tampered_checkpoint_blocks_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest(Path(tmp))
            payload = json.loads(manifest.read_text())
            Path(payload["models"]["ldtf_selected"][0]["checkpoint"]).write_bytes(b"tampered")
            result = self._run(manifest, Path(tmp) / "stage13", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum mismatch", result.stderr.lower())

    def test_already_consumed_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest(Path(tmp))
            root = Path(tmp) / "stage13"
            root.mkdir(parents=True, exist_ok=True)
            manifest_hash = json.loads(manifest.read_text())["manifest_sha256"]
            record_official_test_access(root / "official_test_access_log.json", manifest_hash, {})
            result = self._run(manifest, root, "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already been evaluated", result.stderr.lower())


class TestNoOfficialTestLeakage(unittest.TestCase):
    def test_only_stage13_module_can_load_the_official_test(self):
        """No Stage 10/11/12 module may build an official-test loader."""
        forbidden_callers = [
            ROOT / "train.py",
            ROOT / "evaluate.py",
            ROOT / "scripts" / "run_stage11_ablation.py",
            ROOT / "scripts" / "run_stage12_multiseed.py",
            ROOT / "scripts" / "aggregate_stage12_multiseed.py",
            ROOT / "scripts" / "select_stage11_config.py",
        ]
        for path in forbidden_callers:
            text = path.read_text()
            self.assertNotIn("prepare_official_test_data", text, f"{path.name} references the official test loader")
            self.assertNotIn("stage13_official_test", text, f"{path.name} imports the Stage-13 test module")

    def test_stage12_runner_blocks_official_test_arguments(self):
        text = (ROOT / "scripts" / "run_stage12_multiseed.py").read_text()
        for flag in ("--test-path", "--official-test-path", "--run-test"):
            self.assertIn(flag, text, "runner must explicitly forbid official-test arguments")

    def test_train_cli_exposes_no_test_path_argument(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "train.py"), "--help"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0)
        for flag in ("--test-path", "--official-test-path", "--run-test"):
            self.assertNotIn(flag, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
