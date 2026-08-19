"""Training loop with a fine-tune / frozen-encoder switch."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import config
from .utils import get_device, save_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters collected from config.py plus training switches."""

    output_dir: Path
    epochs: int = config.EPOCHS
    learning_rate: float = config.LEARNING_RATE
    weight_decay: float = config.WEIGHT_DECAY
    warmup_ratio: float = config.WARMUP_RATIO
    max_grad_norm: float = config.MAX_GRAD_NORM
    grad_accum_steps: int = config.GRAD_ACCUM_STEPS
    label_smoothing: float = config.LABEL_SMOOTHING
    freeze_encoder: bool = False
    seed: int = config.SEED


@dataclass(frozen=True)
class TrainResult:
    """Final metrics plus the per-epoch history produced by a training run."""

    best_val_accuracy: float
    best_epoch: int
    history: list[dict[str, float]]


def build_optimizer(model: nn.Module, train_config: TrainConfig) -> AdamW:
    """Return AdamW with weight decay only on multi-D parameters (BERT convention)."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    grouped = [
        {"params": decay, "weight_decay": train_config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=train_config.learning_rate)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup followed by linear decay to 0 over the remaining steps."""
    warmup_steps = max(1, int(total_steps * train_config.warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move every tensor in *batch* to *device* and return a new dict."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run *model* on *dataloader*; return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )["logits"]
            loss = criterion(logits, batch["labels"])
            total_loss += float(loss.item()) * batch["labels"].size(0)
            predictions = torch.argmax(logits, dim=-1)
            total_correct += int((predictions == batch["labels"]).sum().item())
            total_examples += batch["labels"].size(0)
    avg_loss = total_loss / max(1, total_examples)
    accuracy = total_correct / max(1, total_examples)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_config: TrainConfig,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainResult:
    """Train *model* and validate on every epoch; return the best result."""
    set_seed(train_config.seed)
    train_config.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    optimizer = build_optimizer(model, train_config)
    total_steps = (
        max(1, len(train_loader) // max(1, train_config.grad_accum_steps)) * train_config.epochs
    )
    scheduler = build_scheduler(optimizer, train_config, total_steps=total_steps)

    best_val_accuracy = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        if train_config.freeze_encoder:
            model.eval()  # BN/dropout off; gradients still flow through routers
            for name, sub in model.backbone.named_parameters():
                sub.requires_grad = False
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            forward_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                forward_kwargs["token_type_ids"] = batch["token_type_ids"]
            logits = model(**forward_kwargs)["logits"]
            loss = criterion(logits, batch["labels"]) / train_config.grad_accum_steps
            loss.backward()
            epoch_loss += float(loss.item()) * train_config.grad_accum_steps
            epoch_steps += 1

            if step % train_config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=train_config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = epoch_loss / max(1, epoch_steps)
        val_loss, val_accuracy = evaluate_loss(model, val_loader, criterion, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "freeze_encoder": train_config.freeze_encoder,
                "history": history,
            }
            torch.save(checkpoint, train_config.output_dir / "best_model.pt")
        save_json(epoch_record, train_config.output_dir / "train_log.json", indent=2)
        if progress_callback is not None:
            progress_callback(epoch, epoch_record)

    save_json(
        {
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "history": history,
            "freeze_encoder": train_config.freeze_encoder,
        },
        train_config.output_dir / "val_metrics.json",
        indent=2,
    )
    return TrainResult(best_val_accuracy=best_val_accuracy, best_epoch=best_epoch, history=history)


def aggregate_history(history: list[dict[str, float]]) -> dict[str, list[float]]:
    """Group per-epoch metric names into parallel lists for plotting."""
    aggregated: dict[str, list[float]] = defaultdict(list)
    for record in history:
        for key, value in record.items():
            aggregated[key].append(value)
    return aggregated