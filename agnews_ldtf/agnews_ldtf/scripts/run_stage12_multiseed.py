"""Stage-12 locked multi-seed runner reusing the shared Stage-10 training engine."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stage12_protocol import ProtocolError, load_json, validate_protocol  # noqa: E402
from stage12_statistics import file_sha256, payload_sha256  # noqa: E402
from training_utils import get_git_commit  # noqa: E402

FORBIDDEN_ARGUMENTS = ("--test-path", "--official-test-path", "--run-test", "--decontaminated-test-path")


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_protocol_lock(config_path: Path, config: Mapping[str, Any], resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every immutable protocol input so later drift is detectable."""
    file_hashes = {
        "stage12_config": file_sha256(config_path),
        "stage11_locked_config": file_sha256(config["stage11_locked_config"]),
        "data_quality_report": file_sha256(config["data_quality_report"]),
        "research_train": file_sha256(config["train_path"]),
        "research_validation": file_sha256(config["validation_path"]),
    }
    for model in resolved["models"]:
        file_hashes[f"config_source::{model['experiment_name']}"] = file_sha256(model["config_source"])
    core = {
        "protocol_name": config.get("protocol_name"),
        "seeds": resolved["seeds"],
        "models": [
            {"experiment_name": m["experiment_name"], "model_type": m["model_type"]}
            for m in resolved["models"]
        ],
        "file_hashes": file_hashes,
        "official_test_allowed": False,
        "selection_metric": config.get("selection_metric"),
    }
    return {
        **core,
        "protocol_hash": payload_sha256(core),
        "git_commit": get_git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def ensure_protocol_lock(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Create the lock once; afterwards refuse to run under a different protocol."""
    lock_path = root / "protocol_lock.json"
    if not lock_path.is_file():
        save_json(lock, lock_path)
        return dict(lock)
    existing = load_json(lock_path, "Existing protocol lock")
    if existing.get("protocol_hash") != lock.get("protocol_hash"):
        raise ProtocolError(
            "Protocol lock mismatch. The Stage-12 protocol changed after runs began.\n"
            f"locked={existing.get('protocol_hash')}\ncurrent={lock.get('protocol_hash')}\n"
            "Create a new protocol version instead of editing a locked one."
        )
    return existing


def resolve_run_config(
    model: Mapping[str, Any],
    locked_stage11: Mapping[str, Any],
    source_config: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
    protocol_hash: str,
) -> dict[str, Any]:
    """Materialize the full per-run configuration so runs never depend on mutable files."""
    resolved: dict[str, Any] = {
        "experiment_name": model["experiment_name"],
        "model_type": model["model_type"],
        "model_name": source_config.get("model_name", "bert-base-uncased"),
        "num_classes": source_config.get("num_classes", 4),
        "seed": seed,
        "max_length": source_config["max_length"],
        "epochs": source_config["epochs"],
        "train_batch_size": source_config["train_batch_size"],
        "eval_batch_size": source_config["eval_batch_size"],
        "gradient_accumulation_steps": source_config["gradient_accumulation_steps"],
        "effective_batch_size": source_config["train_batch_size"] * source_config["gradient_accumulation_steps"],
        "backbone_learning_rate": source_config["backbone_learning_rate"],
        "head_learning_rate": source_config["head_learning_rate"],
        "weight_decay": source_config["weight_decay"],
        "warmup_ratio": source_config["warmup_ratio"],
        "max_grad_norm": source_config["max_grad_norm"],
        "mixed_precision": source_config["mixed_precision"],
        "early_stopping_patience": source_config["early_stopping_patience"],
        "num_workers": source_config.get("num_workers", 0),
        "dropout": source_config.get("dropout", 0.1),
        "training_regime": source_config.get("training_regime", "finetune"),
        "checkpoint_rule": "validation_macro_f1_then_lower_loss_then_earlier_epoch",
        "train_path": config["train_path"],
        "validation_path": config["validation_path"],
        "protocol_hash": protocol_hash,
        "official_test_allowed": False,
    }
    if model["model_type"] != "bert_baseline":
        resolved.update(
            {
                "training_regime": locked_stage11["training_regime"],
                "ablation_variant": locked_stage11.get("ablation_variant", "A0_full"),
                "token_router_dim": locked_stage11["token_router_dim"],
                "depth_router_dim": locked_stage11["depth_router_dim"],
                "scorer_type": locked_stage11.get("scorer_type", "shared"),
                "exclude_special_tokens": bool(locked_stage11.get("exclude_special_tokens", False)),
            }
        )
    return resolved


def build_command(entrypoint: str, resolved: Mapping[str, Any], run_root: Path, run_name: str) -> list[str]:
    """Build a safe argument list; never a joined shell string."""
    command = [
        sys.executable, entrypoint,
        "--model-type", resolved["model_type"],
        "--training-regime", resolved["training_regime"],
        "--model-name", str(resolved["model_name"]),
        "--num-classes", str(resolved["num_classes"]),
        "--train-path", resolved["train_path"],
        "--validation-path", resolved["validation_path"],
        "--output-dir", str(run_root),
        "--run-name", run_name,
        "--seed", str(resolved["seed"]),
        "--epochs", str(resolved["epochs"]),
        "--train-batch-size", str(resolved["train_batch_size"]),
        "--eval-batch-size", str(resolved["eval_batch_size"]),
        "--gradient-accumulation-steps", str(resolved["gradient_accumulation_steps"]),
        "--backbone-learning-rate", str(resolved["backbone_learning_rate"]),
        "--head-learning-rate", str(resolved["head_learning_rate"]),
        "--weight-decay", str(resolved["weight_decay"]),
        "--warmup-ratio", str(resolved["warmup_ratio"]),
        "--max-grad-norm", str(resolved["max_grad_norm"]),
        "--mixed-precision", str(resolved["mixed_precision"]),
        "--early-stopping-patience", str(resolved["early_stopping_patience"]),
        "--num-workers", str(resolved["num_workers"]),
        "--max-length", str(resolved["max_length"]),
        "--dropout", str(resolved["dropout"]),
    ]
    if resolved["model_type"] != "bert_baseline":
        command += [
            "--token-router-dim", str(resolved["token_router_dim"]),
            "--depth-router-dim", str(resolved["depth_router_dim"]),
        ]
        if resolved["model_type"] == "ldtf_ablation":
            command += ["--ablation-variant", str(resolved["ablation_variant"])]
        if resolved.get("exclude_special_tokens"):
            command.append("--exclude-special-tokens")
    for forbidden in FORBIDDEN_ARGUMENTS:
        if forbidden in command:
            raise ProtocolError(f"Stage-12 command must never contain {forbidden}.")
    return command


def is_complete(run_dir: Path, seed: int, protocol_hash: str) -> bool:
    """A run counts as complete only when every required artifact is valid."""
    required = [
        run_dir / "summary.json",
        run_dir / "checkpoints" / "best.pt",
        run_dir / "checkpoints" / "last.pt",
        run_dir / "metrics" / "best_validation_metrics.json",
        run_dir / "resolved_config.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        summary = load_json(run_dir / "summary.json", "summary")
        resolved = load_json(run_dir / "resolved_config.json", "resolved config")
    except ProtocolError:
        return False
    return (
        summary.get("official_test_evaluated") is False
        and int(summary.get("seed", -1)) == seed
        and int(resolved.get("seed", -1)) == seed
        and resolved.get("protocol_hash") == protocol_hash
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked Stage-12 multi-seed experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--train-entrypoint", default="train.py")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path, "Stage-12 config")
    resolved_protocol = validate_protocol(config_path)
    root = Path(args.output_root or config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = ensure_protocol_lock(root, build_protocol_lock(config_path, config, resolved_protocol))
    protocol_hash = lock["protocol_hash"]
    locked_stage11 = resolved_protocol["locked_stage11"]

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model in resolved_protocol["models"]:
        source_config = load_json(model["config_source"], f"Config source for {model['experiment_name']}")
        for seed in resolved_protocol["seeds"]:
            experiment_root = root / model["experiment_name"]
            run_name = f"seed_{seed}"
            run_dir = experiment_root / run_name
            entry: dict[str, Any] = {
                "experiment_name": model["experiment_name"],
                "seed": seed,
                "run_directory": str(run_dir),
                "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
                "official_test_evaluated": False,
                "protocol_hash": protocol_hash,
                "status": "PENDING",
            }
            if is_complete(run_dir, seed, protocol_hash):
                if args.skip_completed:
                    entry["status"] = "SKIPPED_COMPLETED"
                    runs.append(entry)
                    continue
                entry["status"] = "SKIPPED_COMPLETED"
                runs.append(entry)
                continue

            resolved = resolve_run_config(model, locked_stage11, source_config, seed, config, protocol_hash)
            command = build_command(args.train_entrypoint, resolved, experiment_root, run_name)
            last_checkpoint = run_dir / "checkpoints" / "last.pt"
            if last_checkpoint.is_file():
                if not args.resume_incomplete:
                    entry["status"] = "RESUME_REQUIRED"
                    runs.append(entry)
                    continue
                command += ["--resume-from", str(last_checkpoint)]
                entry["resumed"] = True

            entry["status"] = "RUNNING"
            run_dir.mkdir(parents=True, exist_ok=True)
            save_json(resolved, run_dir / "resolved_config.json")
            save_json(lock, run_dir / "protocol_lock.json")
            try:
                subprocess.run(command, check=True)
                if not is_complete(run_dir, seed, protocol_hash):
                    raise RuntimeError("Training exited successfully but required artifacts are missing.")
                entry["status"] = "PASS"
            except Exception as error:  # noqa: BLE001 - recorded, not silenced
                entry["status"] = "FAIL"
                entry["error"] = str(error)
                failures.append(dict(entry))
                runs.append(entry)
                if not args.continue_on_error:
                    break
                continue
            runs.append(entry)
        else:
            continue
        break

    save_json({"protocol_hash": protocol_hash, "runs": runs}, root / "run_registry.json")
    save_json(failures, root / "failures.json")
    print(f"Stage-12 runs recorded: {len(runs)}; failures: {len(failures)}")
    print("Official test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
