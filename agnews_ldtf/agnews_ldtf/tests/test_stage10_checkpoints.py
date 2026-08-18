"""Stage-10 tests: checkpoint policy, resume contract, optimizer coverage, gates.

Uses a tiny in-memory model. Does not load the real dataset or train anything.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_factory import (  # noqa: E402
    build_data_signature,
    count_model_parameters,
    enforce_data_quality_gate,
)
from tests.fixtures import make_data_quality_report  # noqa: E402
from train import (  # noqa: E402
    CHECKPOINT_RULE,
    _is_better,
    build_resumable_checkpoint,
    build_slim_checkpoint,
)
from training_utils import atomic_torch_save, load_torch_checkpoint  # noqa: E402

RESUME_ONLY_KEYS = (
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
    "patience_counter",
    "history",
)
SHARED_METADATA_KEYS = (
    "model_state_dict",
    "model_config",
    "training_config",
    "data_signature",
    "training_regime",
    "protocol_hash",
    "checkpoint_rule",
    "epoch",
    "global_step",
    "best_val_macro_f1",
    "best_val_loss",
    "best_epoch",
)


class _Scaler:
    def state_dict(self):
        return {"scale": 65536.0}


def _training_objects():
    model = nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return model, optimizer, scheduler, _Scaler()


def _common_kwargs():
    return {
        "epoch": 3,
        "global_step": 300,
        "best_val_macro_f1": 0.93,
        "best_val_loss": 0.21,
        "best_epoch": 3,
        "model_config": {"model_type": "ldtf", "num_classes": 4},
        "training_config": {"seed": 42, "epochs": 5},
        "data_signature": {"tokenizer": "bert-base-uncased", "max_length": 128},
        "training_regime": "finetune",
        "protocol_hash": "deadbeef",
    }


class TestCheckpointPolicy(unittest.TestCase):
    def test_slim_best_excludes_all_resume_state(self):
        model, *_ = _training_objects()
        slim = build_slim_checkpoint(model=model, **_common_kwargs())
        for key in RESUME_ONLY_KEYS:
            self.assertNotIn(key, slim, f"slim best.pt must not contain {key!r}")

    def test_slim_best_contains_required_metadata(self):
        model, *_ = _training_objects()
        slim = build_slim_checkpoint(model=model, **_common_kwargs())
        for key in SHARED_METADATA_KEYS:
            self.assertIn(key, slim, f"slim best.pt is missing required metadata {key!r}")
        self.assertEqual(slim["checkpoint_kind"], "slim_best")
        self.assertFalse(slim["resumable"])
        self.assertEqual(slim["checkpoint_rule"], CHECKPOINT_RULE)
        self.assertIs(slim["official_test_evaluated"], False)

    def test_resumable_last_contains_full_state(self):
        model, optimizer, scheduler, scaler = _training_objects()
        last = build_resumable_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            patience_counter=1, dataloader_generator=None, history=[{"epoch": 1}],
            **_common_kwargs(),
        )
        for key in SHARED_METADATA_KEYS + RESUME_ONLY_KEYS:
            self.assertIn(key, last, f"resumable last.pt is missing {key!r}")
        self.assertEqual(last["checkpoint_kind"], "resumable_last")
        self.assertTrue(last["resumable"])

    def test_slim_is_materially_smaller_on_disk(self):
        model, optimizer, scheduler, scaler = _training_objects()
        slim = build_slim_checkpoint(model=model, **_common_kwargs())
        last = build_resumable_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            patience_counter=0, dataloader_generator=None, history=[], **_common_kwargs(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            slim_path, last_path = Path(tmp) / "best.pt", Path(tmp) / "last.pt"
            atomic_torch_save(slim, slim_path)
            atomic_torch_save(last, last_path)
            self.assertLess(slim_path.stat().st_size, last_path.stat().st_size)

    def test_slim_weights_roundtrip_and_are_detached(self):
        model, *_ = _training_objects()
        slim = build_slim_checkpoint(model=model, **_common_kwargs())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            atomic_torch_save(slim, path)
            loaded = load_torch_checkpoint(path, torch.device("cpu"))
        rebuilt = nn.Linear(8, 4)
        rebuilt.load_state_dict(loaded["model_state_dict"])
        for original, restored in zip(model.state_dict().values(), rebuilt.state_dict().values()):
            self.assertTrue(torch.equal(original, restored))
        for tensor in slim["model_state_dict"].values():
            self.assertFalse(tensor.requires_grad)


class TestCheckpointSelectionRule(unittest.TestCase):
    def test_higher_macro_f1_wins(self):
        self.assertTrue(_is_better({"f1_macro": 0.95, "loss": 0.9}, 0.94, 0.1))

    def test_tie_broken_by_lower_loss(self):
        self.assertTrue(_is_better({"f1_macro": 0.94, "loss": 0.05}, 0.94, 0.10))
        self.assertFalse(_is_better({"f1_macro": 0.94, "loss": 0.20}, 0.94, 0.10))

    def test_equal_metrics_keeps_earlier_epoch(self):
        self.assertFalse(_is_better({"f1_macro": 0.94, "loss": 0.10}, 0.94, 0.10))

    def test_lower_macro_f1_never_wins(self):
        self.assertFalse(_is_better({"f1_macro": 0.90, "loss": 0.001}, 0.94, 0.10))


class TestDataQualityGate(unittest.TestCase):
    def test_gate_passes_on_ready_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_data_quality_report(Path(tmp) / "report.json")
            report = enforce_data_quality_gate(report_path=path)
            self.assertEqual(report["overall_status"], "PASS")

    def test_gate_rejects_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_data_quality_report(Path(tmp) / "r.json", status="FAIL", ready=False)
            with self.assertRaises(RuntimeError):
                enforce_data_quality_gate(report_path=path)

    def test_gate_rejects_missing_report(self):
        with self.assertRaises(RuntimeError):
            enforce_data_quality_gate(report_path="/nonexistent/report.json")

    def test_smoke_override_is_marked_unverified(self):
        gate = enforce_data_quality_gate(allow_unverified_data_for_smoke_test=True)
        self.assertEqual(gate["overall_status"], "BYPASSED")
        self.assertFalse(gate["READY_FOR_OFFICIAL_TRAINING"])


class TestDataSignature(unittest.TestCase):
    def test_signature_never_mentions_test_split(self):
        signature = build_data_signature(
            train_samples=108000, validation_samples=11991,
            tokenizer_name="bert-base-uncased", max_length=128,
            label_mapping={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}, seed=42,
        )
        for key in signature:
            self.assertNotIn("test", key.lower())


class TestParameterCounting(unittest.TestCase):
    def test_frozen_parameters_are_counted_separately(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        stats = count_model_parameters(model)
        self.assertEqual(stats["total"], stats["trainable"] + stats["frozen"])
        self.assertGreater(stats["frozen"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
