"""Synthetic fixture builders for Stage 10-13 tests.

Every artifact here is fabricated in a temporary directory. Nothing in this module
loads the real dataset, trains a model, or touches the official AG News test split.
Checkpoints are tiny tensors, not real BERT weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def make_data_quality_report(path: Path, *, status: str = "PASS", ready: bool = True) -> Path:
    return write_json(
        {
            "overall_status": status,
            "READY_FOR_OFFICIAL_TRAINING": ready,
            "critical_failures": [],
            "data_cleaning_version": "v1-whitespace-normalization",
            "dataset_protocol": "ag_news_stratified_90_10_research_split",
            "split_seed": 42,
            "train_sample_count": 108000,
            "validation_sample_count": 11991,
            "label_mapping": {str(i): n for i, n in enumerate(CLASS_NAMES)},
            "train_manifest_checksum": "a" * 64,
            "validation_manifest_checksum": "b" * 64,
            "official_test_exported": False,
        },
        path,
    )


def make_data_signature() -> dict[str, Any]:
    return {
        "dataset_protocol": "ag_news_stratified_90_10_research_split",
        "train_sample_count": 108000,
        "validation_sample_count": 11991,
        "label_mapping": {str(i): n for i, n in enumerate(CLASS_NAMES)},
        "tokenizer": "bert-base-uncased",
        "max_length": 128,
        "split_seed": 42,
        "data_cleaning_version": "v1-whitespace-normalization",
        "train_manifest_checksum": "a" * 64,
        "validation_manifest_checksum": "b" * 64,
        "quality_gate_status": "PASS",
    }


def make_run_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "model_name": "bert-base-uncased",
        "num_classes": 4,
        "max_length": 128,
        "epochs": 5,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
        "backbone_learning_rate": 2e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "mixed_precision": "fp16",
        "early_stopping_patience": 2,
        "num_workers": 0,
        "dropout": 0.1,
        "seed": 42,
        "training_regime": "finetune",
        "train_path": "data/processed/research_train.parquet",
        "validation_path": "data/processed/research_validation.parquet",
    }
    config.update(overrides)
    return config


def make_metrics(*, macro_f1: float, loss: float, accuracy: float | None = None) -> dict[str, Any]:
    accuracy = macro_f1 + 0.002 if accuracy is None else accuracy
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision_macro": macro_f1 - 0.001,
        "recall_macro": macro_f1 + 0.001,
        "f1_macro": macro_f1,
        "f1_weighted": macro_f1 + 0.0005,
        "per_class": {
            name: {
                "precision": macro_f1,
                "recall": macro_f1,
                "f1": macro_f1 + 0.001 * index,
                "support": 3000,
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "label_order": list(CLASS_NAMES),
        "confusion_matrix": [[100 if i == j else 1 for j in range(4)] for i in range(4)],
    }


def make_environment(**overrides: Any) -> dict[str, Any]:
    environment = {
        "device": "cuda",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "pytorch_version": "2.11.0",
        "transformers_version": "5.8.0",
        "pytorch_cuda_version": "12.1",
        "python_version": "3.13.0",
        "git_commit": "0" * 40,
    }
    environment.update(overrides)
    return environment


def make_tiny_checkpoint(
    path: Path,
    *,
    model_config: Mapping[str, Any] | None = None,
    training_regime: str = "finetune",
    protocol_hash: str | None = None,
    kind: str = "slim_best",
    seed: int = 0,
) -> Path:
    """Write a tiny checkpoint with the real metadata contract but 4 float weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": {"classifier.weight": torch.zeros(2, 2)},
        "model_config": dict(model_config or {"model_type": "ldtf", "num_classes": 4}),
        "training_config": make_run_config(seed=seed),
        "data_signature": make_data_signature(),
        "training_regime": training_regime,
        "protocol_hash": protocol_hash,
        "checkpoint_kind": kind,
        "resumable": kind == "resumable_last",
        "epoch": 3,
        "global_step": 300,
        "best_val_macro_f1": 0.94,
        "best_val_loss": 0.18,
        "best_epoch": 3,
        "official_test_evaluated": False,
        "checkpoint_rule": "validation_macro_f1_then_lower_loss_then_earlier_epoch",
    }
    if kind == "resumable_last":
        payload.update(
            {
                "optimizer_state_dict": {"state": {}, "param_groups": []},
                "scheduler_state_dict": {"last_epoch": 3},
                "scaler_state_dict": {"scale": 1.0},
                "patience_counter": 0,
                "rng_state": {"python": None, "numpy": None, "torch_cpu": None},
                "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            }
        )
    torch.save(payload, path)
    return path


def make_stage10_run(
    run_dir: Path,
    *,
    training_regime: str = "finetune",
    macro_f1: float = 0.93,
    loss: float = 0.21,
    model_type: str = "ldtf",
    seed: int = 42,
) -> Path:
    """Create a complete, protocol-valid Stage-10 run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(make_run_config(seed=seed, training_regime=training_regime), run_dir / "config.json")
    write_json(make_data_signature(), run_dir / "data_signature.json")
    write_json(make_environment(), run_dir / "environment.json")
    write_json(make_metrics(macro_f1=macro_f1, loss=loss), run_dir / "metrics" / "best_validation_metrics.json")
    write_json(
        {
            "model_type": model_type,
            "training_regime": training_regime,
            "seed": seed,
            "best_epoch": 3,
            "best_validation_loss": loss,
            "best_validation_accuracy": macro_f1 + 0.002,
            "best_validation_macro_f1": macro_f1,
            "total_parameters": 109_681_921,
            "trainable_parameters": 109_681_921 if training_regime == "finetune" else 1_234_567,
            "frozen_parameters": 0 if training_regime == "finetune" else 108_447_354,
            "peak_vram_mb": 8123.5,
            "training_time_seconds": 4200.0,
            "average_epoch_time_seconds": 1400.0,
            "optimizer_updates": 3000,
            "samples_per_second": 77.1,
            "official_test_evaluated": False,
            "official_test_loaded": False,
            "stopped_early": False,
            "checkpoint_rule": "validation_macro_f1_then_lower_loss_then_earlier_epoch",
        },
        run_dir / "summary.json",
    )
    return run_dir


def make_stage11_ablation_run(
    run_dir: Path,
    *,
    name: str,
    variant: str,
    macro_f1: float,
    loss: float,
    training_regime: str = "finetune",
    token_router_dim: int = 256,
    depth_router_dim: int = 256,
    exclude_special_tokens: bool = False,
    trainable_parameters: int = 109_681_921,
) -> Path:
    """Create a complete Stage-11 ablation variant run directory."""
    make_stage10_run(run_dir, training_regime=training_regime, macro_f1=macro_f1, loss=loss)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["trainable_parameters"] = trainable_parameters
    summary["total_parameters"] = trainable_parameters
    write_json(summary, run_dir / "summary.json")
    ablation: dict[str, Any] = {
        "name": name,
        "variant": variant,
        "training_regime": training_regime,
        "token_router_dim": token_router_dim,
        "depth_router_dim": depth_router_dim,
    }
    if exclude_special_tokens:
        ablation["exclude_special_tokens"] = True
    write_json(ablation, run_dir / "ablation_config.json")
    return run_dir
