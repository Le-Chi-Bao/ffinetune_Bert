"""Stage-12 tests: protocol validation, locking, statistics, and a mocked run matrix.

The six-run matrix is exercised with a stub training entrypoint, so no real
training occurs and no multi-gigabyte checkpoints are produced.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

from scripts.validate_stage12_protocol import ProtocolError, validate_protocol  # noqa: E402
from stage12_statistics import (  # noqa: E402
    file_sha256,
    paired_differences,
    payload_sha256,
    summarize,
    summarize_paired,
)
from tests.fixtures import (  # noqa: E402
    make_data_quality_report,
    make_stage10_run,
    write_json,
)

SEEDS = [0, 1, 2]


def make_locked_stage11_config(path: Path, **overrides) -> Path:
    payload = {
        "stage": 11,
        "locked_for_stage": 12,
        "selected_run_name": "A0_full",
        "model_type": "ldtf",
        "ablation_variant": "A0_full",
        "training_regime": "finetune",
        "token_router_dim": 256,
        "depth_router_dim": 256,
        "scorer_type": "shared",
        "exclude_special_tokens": False,
        "model_name": "bert-base-uncased",
        "max_length": 128,
        "num_classes": 4,
        "official_test_evaluated": False,
        # Training hyperparameters inherited from the Stage-10 base run so that
        # Stage 12 can reproduce the selected architecture without re-reading it.
        "epochs": 5,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "backbone_learning_rate": 2e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "mixed_precision": "fp16",
        "early_stopping_patience": 2,
        "num_workers": 0,
        "dropout": 0.1,
    }
    payload.update(overrides)
    return write_json(payload, path)


def make_stage12_protocol(tmp: Path, **overrides) -> Path:
    """Build a complete, valid Stage-12 protocol rooted at tmp."""
    (tmp / "data" / "processed").mkdir(parents=True, exist_ok=True)
    train = tmp / "data" / "processed" / "research_train.parquet"
    validation = tmp / "data" / "processed" / "research_validation.parquet"
    train.write_bytes(b"synthetic-train")
    validation.write_bytes(b"synthetic-validation")
    report = make_data_quality_report(tmp / "data" / "reports" / "final_data_report.json")
    locked = make_locked_stage11_config(tmp / "outputs" / "stage11" / "locked_stage12_config.json")
    baseline_run = make_stage10_run(tmp / "outputs" / "stage10" / "bert_baseline_finetune_seed42")
    payload = {
        "stage": 12,
        "protocol_name": "agnews_ldtf_multiseed_v1",
        "seeds": list(SEEDS),
        "train_path": str(train),
        "validation_path": str(validation),
        "data_quality_report": str(report),
        "stage11_locked_config": str(locked),
        "output_root": str(tmp / "outputs" / "stage12"),
        "models": [
            {
                "experiment_name": "bert_baseline",
                "model_type": "bert_baseline",
                "config_source": str(baseline_run / "config.json"),
            },
            {
                "experiment_name": "ldtf_selected",
                "model_type": "from_stage11",
                "config_source": str(locked),
            },
        ],
        "selection_metric": "validation_macro_f1",
        "official_test_allowed": False,
    }
    payload.update(overrides)
    return write_json(payload, tmp / "stage12.json")


class TestProtocolAcceptance(unittest.TestCase):
    def test_valid_protocol_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = validate_protocol(make_stage12_protocol(Path(tmp)))
            self.assertEqual(resolved["seeds"], SEEDS)
            self.assertIs(resolved["official_test_allowed"], False)
            self.assertEqual(
                [model["experiment_name"] for model in resolved["models"]],
                ["bert_baseline", "ldtf_selected"],
            )

    def test_from_stage11_resolves_to_locked_model_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = validate_protocol(make_stage12_protocol(Path(tmp)))
            candidate = next(m for m in resolved["models"] if m["experiment_name"] == "ldtf_selected")
            self.assertEqual(candidate["model_type"], "ldtf")

    def test_shipped_config_locks_three_seeds_and_forbids_test(self):
        shipped = json.loads((ROOT / "configs" / "stage12_multiseed.json").read_text())
        self.assertEqual(shipped["seeds"], SEEDS)
        self.assertIs(shipped["official_test_allowed"], False)
        self.assertEqual(shipped["selection_metric"], "validation_macro_f1")


class TestProtocolRejections(unittest.TestCase):
    """Each case must be refused before any training could start."""

    def _expect_rejection(self, fragment: str, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_stage12_protocol(Path(tmp), **overrides)
            with self.assertRaises(ProtocolError) as caught:
                validate_protocol(path)
            self.assertIn(fragment.lower(), str(caught.exception).lower())

    def test_official_test_allowed_true_is_rejected(self):
        self._expect_rejection("official_test_allowed", official_test_allowed=True)

    def test_test_path_key_is_rejected(self):
        self._expect_rejection("test", test_path="data/test.parquet")

    def test_official_test_path_key_is_rejected(self):
        self._expect_rejection("test", official_test_path="data/test.parquet")

    def test_duplicate_seeds_are_rejected(self):
        self._expect_rejection("duplicate", seeds=[0, 1, 1])

    def test_empty_seeds_are_rejected(self):
        self._expect_rejection("non-empty", seeds=[])

    def test_non_integer_seeds_are_rejected(self):
        self._expect_rejection("integer", seeds=[0, "one", 2])

    def test_wrong_selection_metric_is_rejected(self):
        self._expect_rejection("selection_metric", selection_metric="validation_accuracy")

    def test_single_model_is_rejected(self):
        self._expect_rejection(
            "baseline",
            models=[{"experiment_name": "only", "model_type": "bert_baseline", "config_source": "x"}],
        )

    def test_placeholder_path_is_rejected(self):
        self._expect_rejection("placeholder", stage11_locked_config="REPLACE_WITH_PATH")

    def test_failed_data_quality_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_stage12_protocol(Path(tmp))
            config = json.loads(path.read_text())
            make_data_quality_report(Path(config["data_quality_report"]), status="FAIL", ready=False)
            with self.assertRaises(ProtocolError) as caught:
                validate_protocol(path)
            self.assertIn("data-quality gate", str(caught.exception).lower())

    def test_bad_locked_model_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_stage12_protocol(Path(tmp))
            config = json.loads(path.read_text())
            make_locked_stage11_config(Path(config["stage11_locked_config"]), model_type="resnet")
            with self.assertRaises(ProtocolError):
                validate_protocol(path)

    def test_tokenizer_mismatch_between_models_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_stage12_protocol(Path(tmp))
            config = json.loads(path.read_text())
            source = Path(config["models"][0]["config_source"])
            payload = json.loads(source.read_text())
            payload["model_name"] = "roberta-base"
            write_json(payload, source)
            with self.assertRaises(ProtocolError) as caught:
                validate_protocol(path)
            self.assertIn("mismatch", str(caught.exception).lower())


class TestProtocolLockImmutability(unittest.TestCase):
    def test_lock_hash_changes_when_any_input_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_stage12_protocol(Path(tmp))
            config = json.loads(path.read_text())
            before = file_sha256(config["train_path"])
            Path(config["train_path"]).write_bytes(b"tampered")
            self.assertNotEqual(before, file_sha256(config["train_path"]))

    def test_payload_hash_is_order_independent(self):
        self.assertEqual(
            payload_sha256({"a": 1, "b": [1, 2]}),
            payload_sha256({"b": [1, 2], "a": 1}),
        )

    def test_payload_hash_detects_value_change(self):
        self.assertNotEqual(payload_sha256({"seeds": [0, 1, 2]}), payload_sha256({"seeds": [0, 1, 3]}))


class TestMultiSeedStatistics(unittest.TestCase):
    def test_std_uses_ddof_one(self):
        summary = summarize([1.0, 2.0, 3.0])
        self.assertEqual(summary["std_ddof"], 1)
        self.assertAlmostEqual(summary["std"], 1.0)
        self.assertAlmostEqual(summary["mean"], 2.0)

    def test_single_observation_has_undefined_std(self):
        self.assertIsNone(summarize([0.9])["std"])

    def test_non_finite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize([0.9, float("nan")])

    def test_paired_differences_require_identical_seeds(self):
        with self.assertRaises(ValueError) as caught:
            paired_differences({0: 0.9, 1: 0.9}, {0: 0.91, 2: 0.92})
        self.assertIn("identical seed sets", str(caught.exception))

    def test_paired_delta_sign_favors_candidate(self):
        paired = paired_differences({0: 0.90, 1: 0.91}, {0: 0.93, 1: 0.90})
        summary = summarize_paired(paired)
        self.assertAlmostEqual(paired[0]["delta"], 0.03)
        self.assertEqual(summary["number_positive"], 1)
        self.assertEqual(summary["number_negative"], 1)
        self.assertIn("significance is not claimed", summary["statistical_claim"])


class TestMockedSixRunMatrix(unittest.TestCase):
    """Drive the real runner with a stub trainer: 2 models x 3 seeds = 6 runs."""

    STUB = textwrap.dedent(
        '''
        """Stub trainer: writes Stage-12 artifacts without training anything."""
        import argparse, json, sys
        from pathlib import Path
        import torch

        parser = argparse.ArgumentParser()
        for flag in (
            "--model-type", "--training-regime", "--model-name", "--num-classes",
            "--train-path", "--validation-path", "--output-dir", "--run-name", "--seed",
            "--epochs", "--train-batch-size", "--eval-batch-size",
            "--gradient-accumulation-steps", "--backbone-learning-rate",
            "--head-learning-rate", "--weight-decay", "--warmup-ratio", "--max-grad-norm",
            "--mixed-precision", "--early-stopping-patience", "--num-workers",
            "--max-length", "--dropout", "--token-router-dim", "--depth-router-dim",
            "--ablation-variant", "--protocol-hash",
        ):
            parser.add_argument(flag)
        parser.add_argument("--exclude-special-tokens", action="store_true")
        args, _unknown = parser.parse_known_args()

        forbidden = {"--test-path", "--official-test-path", "--run-test"}
        if forbidden & set(sys.argv):
            raise SystemExit("stub trainer received a forbidden official-test argument")

        run_dir = Path(args.output_dir) / args.run_name
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        seed = int(args.seed)
        macro_f1 = 0.93 + seed * 0.001 + (0.004 if args.model_type != "bert_baseline" else 0.0)

        def dump(payload, path):
            Path(path).write_text(json.dumps(payload, indent=2) + "\\n")

        for name in ("best.pt", "last.pt"):
            torch.save(
                {
                    "model_state_dict": {"w": torch.zeros(2, 2)},
                    "model_config": {"model_type": args.model_type, "num_classes": 4},
                    "training_regime": args.training_regime,
                    "protocol_hash": args.protocol_hash,
                    "checkpoint_kind": "slim_best" if name == "best.pt" else "resumable_last",
                },
                run_dir / "checkpoints" / name,
            )
        per_class = {
            c: {"precision": macro_f1, "recall": macro_f1, "f1": macro_f1, "support": 3000}
            for c in ("World", "Sports", "Business", "Sci/Tech")
        }
        dump(
            {
                "loss": 0.2, "accuracy": macro_f1 + 0.002, "precision_macro": macro_f1,
                "recall_macro": macro_f1, "f1_macro": macro_f1, "f1_weighted": macro_f1,
                "per_class": per_class, "label_order": list(per_class),
                "confusion_matrix": [[1, 0, 0, 0]] * 4,
            },
            run_dir / "metrics" / "best_validation_metrics.json",
        )
        dump(
            {
                "model_type": args.model_type, "training_regime": args.training_regime,
                "seed": seed, "best_epoch": 3, "best_validation_loss": 0.2,
                "best_validation_accuracy": macro_f1 + 0.002,
                "best_validation_macro_f1": macro_f1,
                "total_parameters": 109681921, "trainable_parameters": 109681921,
                "frozen_parameters": 0, "peak_vram_mb": 8000.0,
                "training_time_seconds": 4000.0, "average_epoch_time_seconds": 1300.0,
                "optimizer_updates": 3000, "samples_per_second": 77.0,
                "official_test_evaluated": False, "official_test_loaded": False,
                "stopped_early": False,
            },
            run_dir / "summary.json",
        )
        dump(
            {
                "dataset_protocol": "ag_news_stratified_90_10_research_split",
                "train_sample_count": 108000, "validation_sample_count": 11991,
                "tokenizer": "bert-base-uncased", "max_length": 128, "split_seed": 42,
                "train_manifest_checksum": "a" * 64, "validation_manifest_checksum": "b" * 64,
                "quality_gate_status": "PASS",
            },
            run_dir / "data_signature.json",
        )
        dump(
            {
                "device": "cuda", "gpu_name": "NVIDIA A100-SXM4-40GB",
                "pytorch_version": "2.11.0", "transformers_version": "5.8.0",
                "pytorch_cuda_version": "12.1", "python_version": "3.13.0",
                "git_commit": "0" * 40,
            },
            run_dir / "environment.json",
        )
        '''
    )

    def _run_matrix(self, tmp: Path):
        stub = tmp / "stub_train.py"
        stub.write_text(self.STUB)
        config_path = make_stage12_protocol(tmp)
        output_root = tmp / "outputs" / "stage12"
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "run_stage12_multiseed.py"),
                "--config", str(config_path), "--output-root", str(output_root),
                "--train-entrypoint", str(stub),
            ],
            capture_output=True, text=True, cwd=ROOT,
        )
        return result, output_root, config_path

    def test_matrix_produces_six_runs_and_a_protocol_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, root, _ = self._run_matrix(Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            registry = json.loads((root / "run_registry.json").read_text())
            self.assertEqual(len(registry["runs"]), 6)
            self.assertTrue(all(entry["status"] == "PASS" for entry in registry["runs"]))
            lock = json.loads((root / "protocol_lock.json").read_text())
            self.assertEqual(lock["seeds"], SEEDS)
            self.assertIs(lock["official_test_allowed"], False)
            self.assertEqual(len(lock["protocol_hash"]), 64)

    def test_rerun_is_skipped_and_lock_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, root, config_path = self._run_matrix(Path(tmp))
            self.assertEqual(first.returncode, 0, first.stderr)
            first_hash = json.loads((root / "protocol_lock.json").read_text())["protocol_hash"]
            second = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "run_stage12_multiseed.py"),
                    "--config", str(config_path), "--output-root", str(root),
                    "--train-entrypoint", str(Path(tmp) / "stub_train.py"), "--skip-completed",
                ],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                json.loads((root / "protocol_lock.json").read_text())["protocol_hash"], first_hash
            )

    def test_editing_a_locked_protocol_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _first, root, config_path = self._run_matrix(Path(tmp))
            config = json.loads(config_path.read_text())
            config["protocol_name"] = "tampered_v2"
            write_json(config, config_path)
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "run_stage12_multiseed.py"),
                    "--config", str(config_path), "--output-root", str(root),
                    "--train-entrypoint", str(Path(tmp) / "stub_train.py"),
                ],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("locked", result.stderr.lower())

    def test_aggregation_emits_stage13_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            _result, root, _ = self._run_matrix(Path(tmp))
            aggregate = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "aggregate_stage12_multiseed.py"),
                    "--root", str(root), "--baseline", "bert_baseline",
                    "--candidate", "ldtf_selected",
                    "--output-csv", str(root / "summary.csv"),
                    "--output-json", str(root / "summary.json"),
                ],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(aggregate.returncode, 0, aggregate.stderr)
            manifest = json.loads((root / "locked_stage13_manifest.json").read_text())
            self.assertEqual(manifest["stage"], 13)
            self.assertEqual(sorted(manifest["seeds"]), SEEDS)
            self.assertIs(manifest["official_test_evaluated"], False)
            self.assertEqual(len(manifest["manifest_sha256"]), 64)
            for entries in manifest["models"].values():
                self.assertEqual(len(entries), 3)
                for entry in entries:
                    self.assertEqual(len(entry["checkpoint_sha256"]), 64)

    def test_aggregation_refuses_a_missing_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _result, root, _ = self._run_matrix(Path(tmp))
            registry = json.loads((root / "run_registry.json").read_text())
            registry["runs"] = [r for r in registry["runs"] if int(r["seed"]) != 2]
            write_json(registry, root / "run_registry.json")
            aggregate = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "aggregate_stage12_multiseed.py"),
                    "--root", str(root), "--baseline", "bert_baseline",
                    "--candidate", "ldtf_selected",
                    "--output-csv", str(root / "s.csv"), "--output-json", str(root / "s.json"),
                ],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertNotEqual(aggregate.returncode, 0)
            self.assertIn("never dropped", aggregate.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
