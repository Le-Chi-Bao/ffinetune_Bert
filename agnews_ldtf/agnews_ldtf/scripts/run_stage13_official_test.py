"""Stage-13 locked official-test evaluation.

This is the only script in the project permitted to read the official AG News
test split, and only after:

  1. the locked Stage-13 manifest validates (structure + manifest checksum),
  2. every locked checkpoint matches its recorded SHA-256,
  3. the one-time access guard confirms this manifest has not been evaluated.

Every locked seed of every locked model is evaluated. No selection, tuning, or
filtering may be performed using these results -- the architecture and seeds were
fixed by Stages 11 and 12 before this script could run.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ID2LABEL  # noqa: E402
from evaluate import build_model_from_checkpoint  # noqa: E402
from metrics import compute_classification_metrics  # noqa: E402
from scripts.validate_stage13_manifest import ManifestError, validate_manifest  # noqa: E402
from stage13_official_test import (  # noqa: E402
    ACCESS_LOG_NAME,
    assert_first_official_test_access,
    predict_logits,
    prepare_official_test_data,
    record_official_test_access,
)
from training_utils import (  # noqa: E402
    ensure_directory,
    get_device,
    get_environment_info,
    load_torch_checkpoint,
    print_environment_info,
    save_confusion_matrix_csv,
    save_json,
    save_per_class_metrics_csv,
)


def save_predictions(payload: dict[str, list[int]], path: Path) -> None:
    """Persist per-example predictions so paired comparisons stay reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("index,label,prediction\n")
        for index, (label, prediction) in enumerate(
            zip(payload["labels"], payload["predictions"])
        ):
            handle.write(f"{index},{label},{prediction}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one-time locked Stage-13 official test evaluation."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Debug only; a truncated official test run is NOT a valid Stage-13 result.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and guard, then stop WITHOUT opening the official test.",
    )
    parser.add_argument(
        "--i-understand-this-consumes-the-one-time-official-test",
        dest="confirmed",
        action="store_true",
        help="Required acknowledgement for a real (non-dry-run) official test evaluation.",
    )
    args = parser.parse_args()

    root = ensure_directory(args.output_root)
    access_log = Path(root) / ACCESS_LOG_NAME

    resolved = validate_manifest(args.manifest)
    manifest_hash = resolved["manifest_sha256"]
    print(f"Manifest SHA-256: {manifest_hash}")
    print(f"Checkpoints verified: {len(resolved['checkpoints'])}")

    assert_first_official_test_access(access_log, manifest_hash)

    if args.dry_run:
        save_json(
            {
                "dry_run": True,
                "manifest_sha256": manifest_hash,
                "protocol_hash": resolved["protocol_hash"],
                "models": resolved["models"],
                "seeds": resolved["seeds"],
                "checkpoints_verified": len(resolved["checkpoints"]),
                "official_test_loaded": False,
                "official_test_evaluated": False,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            Path(root) / "stage13_dry_run.json",
        )
        print("DRY RUN: manifest and access guard validated.")
        print("Official test was NOT loaded or evaluated.")
        return

    if not args.confirmed:
        raise ManifestError(
            "Refusing to open the official test split without explicit acknowledgement. "
            "Pass --i-understand-this-consumes-the-one-time-official-test, or use --dry-run."
        )
    if args.max_batches is not None:
        raise ManifestError(
            "--max-batches truncates the official test set and cannot produce a valid "
            "Stage-13 result. Remove it, or use --dry-run."
        )

    device = get_device()
    environment = get_environment_info(device, False)
    print_environment_info(environment)

    data = prepare_official_test_data(
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    print(f"Official test samples: {data['test_samples']}")

    results: dict[str, dict[int, dict[str, Any]]] = {}
    for entry in resolved["checkpoints"]:
        name, seed = entry["experiment_name"], entry["seed"]
        run_dir = Path(root) / name / f"seed_{seed}"
        checkpoint = load_torch_checkpoint(entry["checkpoint"], device)
        model, model_config = build_model_from_checkpoint(checkpoint, device)
        predictions = predict_logits(model, data["test_loader"], device, args.max_batches)
        metrics = compute_classification_metrics(
            torch.tensor(predictions["labels"]),
            torch.tensor(predictions["predictions"]),
            ID2LABEL,
        )
        save_predictions(predictions, run_dir / "test_predictions.csv")
        save_json(
            {
                key: value
                for key, value in metrics.items()
                if key not in {"classification_report_text", "classification_report"}
            },
            run_dir / "test_metrics.json",
        )
        save_json(metrics["classification_report"], run_dir / "test_classification_report.json")
        save_per_class_metrics_csv(metrics["per_class"], run_dir / "test_per_class_metrics.csv")
        save_confusion_matrix_csv(
            metrics["confusion_matrix"], metrics["label_order"], run_dir / "test_confusion_matrix.csv"
        )
        save_json(
            {
                "experiment_name": name,
                "seed": seed,
                "checkpoint": entry["checkpoint"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "manifest_sha256": manifest_hash,
                "protocol_hash": resolved["protocol_hash"],
                "model_type": model_config.get("model_type"),
                "official_test_evaluated": True,
                "test_samples": len(predictions["labels"]),
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["f1_macro"],
            },
            run_dir / "test_summary.json",
        )
        results.setdefault(name, {})[seed] = {
            "run_directory": str(run_dir),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }
        print(f"{name}/seed_{seed}: accuracy={metrics['accuracy']:.6f} macro_f1={metrics['f1_macro']:.6f}")
        del model

    record_official_test_access(
        access_log,
        manifest_hash,
        {
            "models": resolved["models"],
            "seeds": resolved["seeds"],
            "output_root": str(root),
            "test_samples": data["test_samples"],
        },
    )
    save_json(
        {
            "manifest_sha256": manifest_hash,
            "protocol_hash": resolved["protocol_hash"],
            "seeds": resolved["seeds"],
            "models": {name: {str(k): v for k, v in runs.items()} for name, runs in results.items()},
            "official_test_evaluated": True,
            "test_samples": data["test_samples"],
            "selection_note": (
                "Architecture and seeds were locked before this evaluation. No model, seed, "
                "or hyperparameter was selected using these test results."
            ),
            "statistical_claim": "Descriptive results; statistical significance is not claimed.",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        Path(root) / "stage13_run_registry.json",
    )
    print(f"Recorded one-time official test access in {access_log}")
    print("Stage-13 evaluation complete. No selection was performed on test results.")


if __name__ == "__main__":
    main()
