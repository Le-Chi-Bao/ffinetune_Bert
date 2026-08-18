"""Reusable validation evaluator for Stage-10 baseline and LDTF models.

The training entry point uses validation only. The standalone CLI intentionally
exposes validation only as well, so Stage 10 cannot load an official test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import ID2LABEL
from data import prepare_research_data
from metrics import compute_classification_metrics
from model_factory import build_model, extract_logits
from training_utils import (
    autocast_context,
    ensure_directory,
    get_device,
    get_environment_info,
    load_torch_checkpoint,
    move_batch_to_device,
    print_environment_info,
    resolve_mixed_precision,
    resolve_num_batches,
    save_confusion_matrix_csv,
    save_json,
    save_per_class_metrics_csv,
)


def evaluate_model(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: Any | None = None,
    max_batches: int | None = None,
    return_predictions: bool = False,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Evaluate complete validation predictions under inference mode."""
    num_batches = resolve_num_batches(dataloader, max_batches, "evaluation")
    expected_classes = num_classes if num_classes is not None else int(model.num_classes)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions_cpu: list[torch.Tensor] = []
    labels_cpu: list[torch.Tensor] = []

    progress = tqdm(total=num_batches, desc="Validating", unit="batch", leave=False)
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= num_batches:
                break
            input_ids, attention_mask, labels, token_type_ids = move_batch_to_device(batch, device)
            if labels.dtype != torch.long:
                raise ValueError(f"Labels must be torch.long, received {labels.dtype}.")
            if labels.min().item() < 0 or labels.max().item() >= expected_classes:
                raise ValueError("Validation labels are outside the configured class range.")
            with autocast_context(amp_enabled, amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = extract_logits(outputs, expected_classes)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite validation loss at batch {batch_index}: {loss.item()!r}."
                )
            if logits.shape != (labels.shape[0], expected_classes):
                raise RuntimeError(
                    f"Unexpected validation logits shape {tuple(logits.shape)}; "
                    f"expected ({labels.shape[0]}, {expected_classes})."
                )
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions_cpu.append(logits.argmax(dim=-1).cpu())
            labels_cpu.append(labels.cpu())
            progress.set_postfix(loss=f"{loss.item():.4f}")
            progress.update(1)
    progress.close()

    if total_samples == 0:
        raise RuntimeError("Evaluation processed zero samples.")
    metrics = compute_classification_metrics(torch.cat(labels_cpu), torch.cat(predictions_cpu), ID2LABEL)
    metrics["loss"] = total_loss / total_samples
    metrics["evaluated_batches"] = num_batches
    if return_predictions:
        metrics["predictions"] = torch.cat(predictions_cpu).tolist()
        metrics["labels"] = torch.cat(labels_cpu).tolist()
    return metrics


def save_evaluation_outputs(
    evaluation: Mapping[str, Any],
    output_dir: str | Path,
    split_name: str = "validation",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist required Stage-10 validation metrics, class report, and matrix."""
    if split_name != "validation":
        raise ValueError("Stage 10 only saves validation outputs; official test is prohibited.")
    directory = ensure_directory(output_dir)
    payload = dict(metadata or {})
    payload.update(
        {
            key: value
            for key, value in evaluation.items()
            if key not in {"classification_report_text", "classification_report", "predictions", "labels"}
        }
    )
    save_json(payload, directory / "best_validation_metrics.json")
    save_json(evaluation["classification_report"], directory / "classification_report.json")
    with (directory / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(evaluation["classification_report_text"]))
    save_per_class_metrics_csv(evaluation["per_class"], directory / "per_class_metrics.csv")
    save_confusion_matrix_csv(
        evaluation["confusion_matrix"], evaluation["label_order"], directory / "confusion_matrix.csv"
    )


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[nn.Module, Mapping[str, Any]]:
    """Rebuild the exact model described by a Stage-10/12 checkpoint and load its weights.

    Works for both the slim ``best.pt`` and the resumable ``last.pt`` because both
    carry ``model_config``/``training_regime`` metadata.
    """
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint lacks Stage-10 model_config metadata.")
    training_regime = checkpoint.get("training_regime")
    model_type = model_config.get("model_type")
    if model_type not in {"bert_baseline", "ldtf", "ldtf_ablation"} or training_regime not in {"frozen", "finetune"}:
        raise ValueError("Checkpoint has unsupported Stage-10 model metadata.")
    build_kwargs = dict(model_config)
    build_kwargs.pop("model_type")
    if model_type == "bert_baseline":
        for key in (
            "classifier_dropout", "token_router_dim", "depth_router_dim",
            "projection_bias", "scorer_bias", "class_names", "variant",
            "exclude_special_tokens",
        ):
            build_kwargs.pop(key, None)
    if model_type == "ldtf":
        build_kwargs.pop("variant", None)
        build_kwargs.pop("exclude_special_tokens", None)
    model = build_model(model_type, training_regime=training_regime, **build_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage-10 validation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-path", default="data/processed/research_validation.parquet")
    parser.add_argument("--train-path", default="data/processed/research_train.parquet")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = get_device()
    amp_enabled, amp_dtype = resolve_mixed_precision(args.mixed_precision, device)
    print_environment_info(get_environment_info(device, amp_enabled))
    checkpoint = load_torch_checkpoint(args.checkpoint, device)
    model, model_config = build_model_from_checkpoint(checkpoint, device)
    data = prepare_research_data(
        train_path=args.train_path,
        validation_path=args.validation_path,
        max_length=args.max_length,
        train_batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=int(checkpoint.get("training_config", {}).get("seed", 42)),
    )
    evaluation = evaluate_model(
        model, data["val_loader"], nn.CrossEntropyLoss(), device, amp_enabled, amp_dtype,
        args.max_batches, num_classes=int(model_config["num_classes"]),
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parents[1] / "metrics"
    save_evaluation_outputs(evaluation, output_dir, metadata={"checkpoint": str(Path(args.checkpoint).resolve())})
    print(f"Validation loss: {evaluation['loss']:.6f}")
    print(f"Validation Macro F1: {evaluation['f1_macro']:.6f}")
    print("Official test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
