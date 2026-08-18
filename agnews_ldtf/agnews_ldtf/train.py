"""Stage-10 training entry point: LDTF frozen backbone versus full fine-tuning.

This module loads only ``research_train.parquet`` and
``research_validation.parquet``. It has no test-path argument and never creates
a test DataLoader.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from config import BASELINE_DROPOUT, ID2LABEL
from data import prepare_research_data
from evaluate import evaluate_model, save_evaluation_outputs
from metrics import compute_classification_metrics
from model_factory import (
    apply_training_regime,
    build_data_signature,
    build_model,
    build_optimizer,
    count_model_parameters,
    describe_optimizer_groups,
    enforce_data_quality_gate,
    extract_logits,
    get_backbone_module,
    verify_training_regime,
)
from training_utils import (
    atomic_torch_save,
    autocast_context,
    capture_rng_state,
    create_grad_scaler,
    get_device,
    get_environment_info,
    get_optimizer_learning_rates,
    load_torch_checkpoint,
    move_batch_to_device,
    peak_memory_gb,
    prepare_run_directory,
    print_environment_info,
    resolve_mixed_precision,
    resolve_num_batches,
    restore_rng_state,
    save_history,
    save_json,
    set_seed,
    set_training_mode,
    write_train_log,
)

TIE_TOLERANCE = 1e-12
CHECKPOINT_RULE = "validation_macro_f1_then_lower_loss_then_earlier_epoch"


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_dataloader: Any,
    epochs: int,
    grad_accumulation_steps: int,
    warmup_ratio: float,
    max_train_batches: int | None,
) -> tuple[Any, int, int]:
    """Build linear warmup/decay in optimizer-update units."""
    batches = resolve_num_batches(train_dataloader, max_train_batches, "training")
    updates_per_epoch = math.ceil(batches / grad_accumulation_steps)
    total_updates = updates_per_epoch * epochs
    if total_updates <= 0:
        raise ValueError("total optimizer updates must be positive.")
    warmup_updates = math.floor(total_updates * warmup_ratio)
    return (
        get_linear_schedule_with_warmup(optimizer, warmup_updates, total_updates),
        total_updates,
        warmup_updates,
    )


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "num_classes": args.num_classes,
        "dropout": args.dropout,
    }
    if args.model_type in {"ldtf", "ldtf_ablation"}:
        config.update(
            {
                "token_router_dim": args.token_router_dim,
                "depth_router_dim": args.depth_router_dim,
                "classifier_dropout": args.dropout,
                "projection_bias": args.projection_bias,
                "scorer_bias": not args.no_scorer_bias,
                "class_names": tuple(ID2LABEL[index] for index in range(args.num_classes)),
            }
        )
        if args.model_type == "ldtf_ablation":
            config.update(
                {
                    "variant": args.ablation_variant,
                    "exclude_special_tokens": args.exclude_special_tokens,
                }
            )
    return config


def _build_model_from_config(
    model_config: Mapping[str, Any], training_regime: str
) -> nn.Module:
    """Build a model while removing cross-model compatibility-only config keys."""
    kwargs = dict(model_config)
    model_type = kwargs.pop("model_type")
    if model_type == "bert_baseline":
        for key in (
            "classifier_dropout", "token_router_dim", "depth_router_dim", "projection_bias",
            "scorer_bias", "class_names", "variant", "exclude_special_tokens",
        ):
            kwargs.pop(key, None)
    return build_model(model_type, training_regime=training_regime, **kwargs)


def _forward_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None,
    num_classes: int,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    return extract_logits(outputs, num_classes)


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
    amp_dtype: Any,
    training_regime: str,
    num_classes: int,
    grad_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    max_batches: int | None,
) -> dict[str, Any]:
    """One full epoch with loss scaling, correct accumulation, and finite guards."""
    num_batches = resolve_num_batches(dataloader, max_batches, "training")
    set_training_mode(model, training_regime)
    backbone = get_backbone_module(model)
    if training_regime == "frozen" and backbone.training:
        raise RuntimeError("Frozen backbone must be in eval mode during training.")
    if training_regime == "finetune" and not backbone.training:
        raise RuntimeError("Fine-tuned backbone must be in train mode during training.")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    optimizer_steps = 0
    skipped_updates = 0
    last_gradient_norm = 0.0
    backbone_received_gradient = False
    predictions: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    progress = tqdm(total=num_batches, desc=f"Epoch {epoch} train", unit="batch")
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= num_batches:
            break
        input_ids, attention_mask, labels, token_type_ids = move_batch_to_device(batch, device)
        if labels.dtype != torch.long or labels.ndim != 1:
            raise ValueError("Labels must be a torch.long tensor with shape [B].")
        if labels.min().item() < 0 or labels.max().item() >= num_classes:
            raise ValueError("Training labels are outside [0, num_classes).")

        with autocast_context(amp_enabled, amp_dtype):
            logits = _forward_logits(model, input_ids, attention_mask, token_type_ids, num_classes)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}.")
        if logits.shape != (labels.shape[0], num_classes):
            raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}.")

        scaled_loss = loss / grad_accumulation_steps
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if training_regime == "finetune":
            backbone_received_gradient |= any(
                parameter.grad is not None for parameter in backbone.parameters() if parameter.requires_grad
            )

        update_boundary = (batch_index + 1) % grad_accumulation_steps == 0 or batch_index + 1 == num_batches
        if update_boundary:
            if amp_enabled:
                scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in trainable_parameters
            )
            if not gradients_finite:
                skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.update()
            else:
                norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
                last_gradient_norm = float(norm.detach().cpu())
                if amp_enabled:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updated = scaler.get_scale() >= scale_before
                else:
                    optimizer.step()
                    updated = True
                if updated:
                    scheduler.step()
                    optimizer_steps += 1
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach()) * labels.shape[0]
        total_samples += labels.shape[0]
        predictions.append(logits.argmax(dim=-1).detach().cpu())
        labels_list.append(labels.detach().cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}", updates=optimizer_steps)
        progress.update(1)
    progress.close()

    if total_samples == 0:
        raise RuntimeError("Training processed zero samples.")
    metrics = compute_classification_metrics(torch.cat(labels_list), torch.cat(predictions), ID2LABEL)
    metrics.update(
        {
            "loss": total_loss / total_samples,
            "optimizer_steps": optimizer_steps,
            "skipped_updates": skipped_updates,
            "gradient_norm": last_gradient_norm,
            "backbone_received_gradient": backbone_received_gradient,
            "processed_batches": num_batches,
        }
    )
    return metrics


def _is_better(candidate: Mapping[str, Any], best_f1: float, best_loss: float) -> bool:
    """Macro F1 primary, lower validation loss secondary, earlier epoch implicit."""
    candidate_f1 = float(candidate["f1_macro"])
    candidate_loss = float(candidate["loss"])
    return candidate_f1 > best_f1 + TIE_TOLERANCE or (
        abs(candidate_f1 - best_f1) <= TIE_TOLERANCE and candidate_loss < best_loss - TIE_TOLERANCE
    )


def _common_checkpoint_metadata(
    *, epoch: int, global_step: int, best_val_macro_f1: float, best_val_loss: float,
    best_epoch: int, model_config: Mapping[str, Any], training_config: Mapping[str, Any],
    data_signature: Mapping[str, Any], training_regime: str,
    protocol_hash: str | None = None,
) -> dict[str, Any]:
    """Metadata carried by BOTH checkpoint kinds so either can be identified/validated."""
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "data_signature": dict(data_signature),
        "training_regime": training_regime,
        "protocol_hash": protocol_hash,
        "checkpoint_rule": CHECKPOINT_RULE,
        "official_test_evaluated": False,
    }


def build_slim_checkpoint(
    *, model: nn.Module, epoch: int, global_step: int, best_val_macro_f1: float,
    best_val_loss: float, best_epoch: int, model_config: Mapping[str, Any],
    training_config: Mapping[str, Any], data_signature: Mapping[str, Any],
    training_regime: str, protocol_hash: str | None = None,
) -> dict[str, Any]:
    """Slim ``best.pt``: model weights plus model/config/data/protocol metadata.

    Optimizer, scheduler, scaler and RNG state are intentionally excluded. This
    checkpoint is for evaluation and Stage-13 test inference, not for resuming.
    """
    payload = _common_checkpoint_metadata(
        epoch=epoch, global_step=global_step, best_val_macro_f1=best_val_macro_f1,
        best_val_loss=best_val_loss, best_epoch=best_epoch, model_config=model_config,
        training_config=training_config, data_signature=data_signature,
        training_regime=training_regime, protocol_hash=protocol_hash,
    )
    payload["model_state_dict"] = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    payload["checkpoint_kind"] = "slim_best"
    payload["resumable"] = False
    return payload


def build_resumable_checkpoint(
    *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
    epoch: int, global_step: int, best_val_macro_f1: float, best_val_loss: float,
    best_epoch: int, patience_counter: int, model_config: Mapping[str, Any],
    training_config: Mapping[str, Any], data_signature: Mapping[str, Any],
    training_regime: str, dataloader_generator: Any, history: list[Mapping[str, Any]],
    protocol_hash: str | None = None,
) -> dict[str, Any]:
    """Fully resumable ``last.pt``: model, optimizer, scheduler, scaler, RNG, progress."""
    payload = _common_checkpoint_metadata(
        epoch=epoch, global_step=global_step, best_val_macro_f1=best_val_macro_f1,
        best_val_loss=best_val_loss, best_epoch=best_epoch, model_config=model_config,
        training_config=training_config, data_signature=data_signature,
        training_regime=training_regime, protocol_hash=protocol_hash,
    )
    payload.update(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "patience_counter": patience_counter,
            "rng_state": capture_rng_state(dataloader_generator),
            "history": list(history),
            "checkpoint_kind": "resumable_last",
            "resumable": True,
        }
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-10 LDTF-BERT train/validation training only.")
    parser.add_argument("--model-type", choices=("bert_baseline", "ldtf", "ldtf_ablation"), default="ldtf")
    parser.add_argument("--training-regime", choices=("frozen", "finetune"), default="finetune")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--train-path", default="data/processed/research_train.parquet")
    parser.add_argument("--validation-path", default="data/processed/research_validation.parquet")
    parser.add_argument("--output-dir", default="outputs/stage10")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--allow-unverified-data-for-smoke-test", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--token-router-dim", type=int, default=256)
    parser.add_argument("--depth-router-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=BASELINE_DROPOUT)
    parser.add_argument("--projection-bias", action="store_true")
    parser.add_argument("--no-scorer-bias", action="store_true")
    parser.add_argument("--ablation-variant", default="A0_full")
    parser.add_argument("--exclude-special-tokens", action="store_true")
    parser.add_argument("--base-stage10-config", default=None)
    parser.add_argument(
        "--protocol-hash",
        default=None,
        help="Stage-12 protocol hash stamped into checkpoints and summary for lock verification.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for key in ("num_classes", "epochs", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps", "max_length"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive.")
    if args.num_workers < 0 or args.early_stopping_patience < 0:
        raise ValueError("--num-workers and --early-stopping-patience must be non-negative.")
    if any(getattr(args, key) <= 0 for key in ("backbone_learning_rate", "head_learning_rate", "max_grad_norm")):
        raise ValueError("Learning rates and --max-grad-norm must be positive.")
    if not 0 <= args.weight_decay or not 0 <= args.warmup_ratio <= 1 or not 0 <= args.dropout < 1:
        raise ValueError("Invalid decay, warmup ratio, or dropout.")


def _resume(
    path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
    device: torch.device, model_config: Mapping[str, Any], training_regime: str,
    data_signature: Mapping[str, Any], dataloader_generator: Any,
    protocol_hash: str | None = None,
) -> tuple[int, int, float, float, int, int, list[Mapping[str, Any]]]:
    checkpoint = load_torch_checkpoint(path, device)
    kind = checkpoint.get("checkpoint_kind")
    if kind is not None and kind != "resumable_last":
        raise ValueError(
            f"Cannot resume from checkpoint_kind={kind!r}. Only the fully resumable "
            "'last.pt' carries optimizer/scheduler/scaler/RNG state; 'best.pt' is slim by policy."
        )
    for required in ("optimizer_state_dict", "scheduler_state_dict", "rng_state"):
        if checkpoint.get(required) is None:
            raise ValueError(
                f"Resume checkpoint is missing {required!r}; it is not a resumable checkpoint."
            )
    for key, expected in (("model_config", dict(model_config)), ("training_regime", training_regime), ("data_signature", dict(data_signature))):
        if checkpoint.get(key) != expected:
            raise ValueError(f"Resume checkpoint {key} conflicts with the requested run configuration.")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise ValueError(
            f"Resume checkpoint protocol_hash={checkpoint.get('protocol_hash')!r} conflicts with "
            f"the requested protocol_hash={protocol_hash!r}."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"), dataloader_generator)
    return (
        int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"]),
        float(checkpoint["best_val_macro_f1"]), float(checkpoint["best_val_loss"]),
        int(checkpoint["best_epoch"]), int(checkpoint["patience_counter"]), list(checkpoint.get("history", [])),
    )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if args.resume_from and args.overwrite_output_dir:
        raise ValueError("--resume-from cannot be combined with --overwrite-output-dir.")
    if args.run_name is None:
        args.run_name = f"{args.model_type}_{args.training_regime}_seed{args.seed}"

    protocol_hash = args.protocol_hash
    quality_gate = enforce_data_quality_gate(
        allow_unverified_data_for_smoke_test=args.allow_unverified_data_for_smoke_test
    )
    run_dir = prepare_run_directory(args.output_dir, args.run_name, args.overwrite_output_dir) if not args.resume_from else Path(args.resume_from).resolve().parents[1]
    device = get_device()
    amp_enabled, amp_dtype = resolve_mixed_precision(args.mixed_precision, device)
    environment = get_environment_info(device, amp_enabled)
    environment["mixed_precision"] = args.mixed_precision
    environment["start_time_utc"] = datetime.now(timezone.utc).isoformat()
    print_environment_info(environment)

    set_seed(args.seed)
    data = prepare_research_data(
        train_path=args.train_path, validation_path=args.validation_path, max_length=args.max_length,
        train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    set_seed(args.seed)
    data_signature = build_data_signature(
        train_samples=data["train_samples"], validation_samples=data["validation_samples"],
        tokenizer_name=data["tokenizer"].name_or_path, max_length=args.max_length,
        label_mapping=data["id2label"], seed=args.seed, quality_gate=quality_gate,
        train_manifest_checksum=data["train_manifest_checksum"],
        validation_manifest_checksum=data["validation_manifest_checksum"],
    )
    model_config = _model_config_from_args(args)
    model = _build_model_from_config(model_config, args.training_regime).to(device)
    regime_info = verify_training_regime(model, args.training_regime)
    optimizer = build_optimizer(
        model, training_regime=args.training_regime,
        backbone_learning_rate=args.backbone_learning_rate, head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler, total_updates, warmup_updates = build_scheduler(
        optimizer, data["train_loader"], args.epochs, args.gradient_accumulation_steps,
        args.warmup_ratio, args.max_train_batches,
    )
    scaler = create_grad_scaler(amp_enabled and amp_dtype == torch.float16)
    criterion = nn.CrossEntropyLoss()
    parameter_stats = count_model_parameters(model)
    training_config = vars(args).copy()
    training_config.update({"effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps, "total_update_steps": total_updates, "warmup_steps": warmup_updates})

    save_json(training_config, run_dir / "config.json")
    save_json(model_config, run_dir / "model_config.json")
    save_json(data_signature, run_dir / "data_signature.json")
    save_json(environment, run_dir / "environment.json")
    write_train_log(run_dir, "=" * 60)
    write_train_log(run_dir, "LDTF-BERT TRAINING")
    write_train_log(run_dir, "=" * 60)
    for line in (
        f"Model type: {args.model_type}", f"Training regime: {args.training_regime}",
        f"Backbone: {args.model_name}", f"Number of classes: {args.num_classes}",
        "Official test loaded: False", f"Train samples: {data['train_samples']}",
        f"Validation samples: {data['validation_samples']}",
        f"Effective batch size: {training_config['effective_batch_size']}",
        f"Backbone trainable: {regime_info['backbone_trainable']}",
        f"Backbone mode during training: {'eval' if args.training_regime == 'frozen' else 'train'}",
        f"Parameter stats: {parameter_stats}",
    ):
        write_train_log(run_dir, line)
    for group in describe_optimizer_groups(optimizer):
        write_train_log(run_dir, "Optimizer group: " + str(group))

    start_epoch, global_step, best_f1, best_loss, best_epoch, patience, history = 1, 0, float("-inf"), float("inf"), 0, 0, []
    if args.resume_from:
        start_epoch, global_step, best_f1, best_loss, best_epoch, patience, history = _resume(
            args.resume_from, model, optimizer, scheduler, scaler, device, model_config,
            args.training_regime, data_signature, data["dataloader_generator"],
            protocol_hash=protocol_hash,
        )
        write_train_log(run_dir, f"Resumed at epoch {start_epoch} from {args.resume_from}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    best_metrics: Mapping[str, Any] | None = None
    stopped_early = False

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model, dataloader=data["train_loader"], optimizer=optimizer, scheduler=scheduler,
            criterion=criterion, device=device, scaler=scaler, amp_enabled=amp_enabled,
            amp_dtype=amp_dtype, training_regime=args.training_regime, num_classes=args.num_classes,
            grad_accumulation_steps=args.gradient_accumulation_steps, max_grad_norm=args.max_grad_norm,
            epoch=epoch, max_batches=args.max_train_batches,
        )
        validation_metrics = evaluate_model(
            model, data["val_loader"], criterion, device, amp_enabled, amp_dtype,
            args.max_validation_batches, num_classes=args.num_classes,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        global_step += int(train_metrics["optimizer_steps"])
        improved = _is_better(validation_metrics, best_f1, best_loss)
        if improved:
            best_f1, best_loss, best_epoch, patience = float(validation_metrics["f1_macro"]), float(validation_metrics["loss"]), epoch, 0
            best_metrics = dict(validation_metrics)
        else:
            patience += 1
        rates = get_optimizer_learning_rates(optimizer)
        record = {
            "epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"], "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"], "val_f1_macro": validation_metrics["f1_macro"],
            "backbone_lr": rates["backbone_lr"], "head_lr": rates["head_lr"],
            "gradient_norm": train_metrics["gradient_norm"], "optimizer_steps": train_metrics["optimizer_steps"],
            "global_optimizer_steps": global_step, "epoch_time_seconds": epoch_seconds,
            "peak_vram_mb": peak_memory_gb(device)["peak_vram_mb"], "best_checkpoint_updated": improved,
            "patience_counter": patience,
        }
        history.append(record)
        save_history(history, run_dir)
        last_checkpoint = build_resumable_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch,
            global_step=global_step, best_val_macro_f1=best_f1, best_val_loss=best_loss,
            best_epoch=best_epoch, patience_counter=patience, model_config=model_config,
            training_config=training_config, data_signature=data_signature,
            training_regime=args.training_regime, dataloader_generator=data["dataloader_generator"],
            history=history, protocol_hash=protocol_hash,
        )
        atomic_torch_save(last_checkpoint, run_dir / "checkpoints" / "last.pt")
        if improved:
            slim_checkpoint = build_slim_checkpoint(
                model=model, epoch=epoch, global_step=global_step, best_val_macro_f1=best_f1,
                best_val_loss=best_loss, best_epoch=best_epoch, model_config=model_config,
                training_config=training_config, data_signature=data_signature,
                training_regime=args.training_regime, protocol_hash=protocol_hash,
            )
            atomic_torch_save(slim_checkpoint, run_dir / "checkpoints" / "best.pt")
            save_evaluation_outputs(best_metrics, run_dir / "metrics", metadata={"best_epoch": best_epoch})
        write_train_log(run_dir, f"Epoch {epoch}/{args.epochs} | train_loss={train_metrics['loss']:.6f} | val_loss={validation_metrics['loss']:.6f} | val_accuracy={validation_metrics['accuracy']:.6f} | val_macro_f1={validation_metrics['f1_macro']:.6f} | grad_norm={train_metrics['gradient_norm']:.4f} | epoch_time={epoch_seconds:.2f}s | best_updated={improved} | patience={patience}/{args.early_stopping_patience}")
        if patience >= args.early_stopping_patience:
            stopped_early = True
            write_train_log(run_dir, "Early stopping triggered by validation Macro F1.")
            break

    if best_metrics is None:
        # Resumed run that never improved: recover the recorded best metrics from disk
        # rather than the slim checkpoint, which stores only the selection scalars.
        metrics_path = run_dir / "metrics" / "best_validation_metrics.json"
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8") as handle:
                best_metrics = json.load(handle)
        else:
            best_checkpoint = load_torch_checkpoint(run_dir / "checkpoints" / "best.pt", device)
            best_metrics = {
                "loss": best_checkpoint["best_val_loss"],
                "f1_macro": best_checkpoint["best_val_macro_f1"],
            }
    total_seconds = time.perf_counter() - start_time
    environment["end_time_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(environment, run_dir / "environment.json")
    summary = {
        "model_type": args.model_type, "training_regime": args.training_regime, "seed": args.seed,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "best_validation_accuracy": best_metrics.get("accuracy"), "best_validation_macro_f1": best_f1,
        "total_parameters": parameter_stats["total"], "trainable_parameters": parameter_stats["trainable"],
        "frozen_parameters": parameter_stats["frozen"], "peak_vram_mb": peak_memory_gb(device)["peak_vram_mb"],
        "training_time_seconds": total_seconds, "average_epoch_time_seconds": total_seconds / len(history),
        "optimizer_updates": global_step, "samples_per_second": data["train_samples"] * len(history) / total_seconds,
        "official_test_evaluated": False, "official_test_loaded": False,
        "protocol_hash": protocol_hash, "checkpoint_rule": CHECKPOINT_RULE,
        "best_checkpoint_kind": "slim_best", "last_checkpoint_kind": "resumable_last",
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "last_checkpoint": str(run_dir / "checkpoints" / "last.pt"), "stopped_early": stopped_early,
    }
    save_json(summary, run_dir / "summary.json")
    write_train_log(run_dir, "Official test was not loaded or evaluated.")
    write_train_log(run_dir, f"Training completed. Provisional best validation Macro F1: {best_f1:.6f}; best epoch: {best_epoch}")


if __name__ == "__main__":
    main()
