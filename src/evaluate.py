"""Evaluate a checkpointed LDTF-BERT model on a held-out DataLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from . import config
from .models import LdtfBert
from .train import move_batch_to_device
from .utils import get_device, save_json


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> dict:
    """Load *model* weights from *checkpoint_path* and return the raw checkpoint dict."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}.")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise KeyError(f"Checkpoint at {path} does not contain a 'model_state_dict' entry.")
    model.load_state_dict(state["model_state_dict"])
    return state


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) as NumPy arrays for the entire *dataloader*."""
    device = device or get_device()
    model.to(device)
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        forward_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
            forward_kwargs["token_type_ids"] = batch["token_type_ids"]
        logits = model(**forward_kwargs)["logits"]
        logits_list.append(logits.cpu().numpy())
        labels_list.append(batch["labels"].cpu().numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: tuple[str, ...] = config.LABEL_NAMES,
) -> dict[str, object]:
    """Compute accuracy, macro F1, weighted F1, and confusion matrix from logits/labels."""
    predictions = np.argmax(logits, axis=-1)
    confusion = confusion_matrix(labels, predictions, labels=list(range(len(label_names))))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted")),
        "confusion_matrix": confusion.tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(label_names))),
            target_names=list(label_names),
            zero_division=0,
        ),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }


def evaluate_checkpoint(
    model: torch.nn.Module,
    dataloader: DataLoader,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    label_names: tuple[str, ...] = config.LABEL_NAMES,
) -> dict[str, object]:
    """Load checkpoint, run prediction on *dataloader*, return (and optionally save) metrics."""
    state = load_checkpoint(model=model, checkpoint_path=checkpoint_path)
    logits, labels = predict_logits(model=model, dataloader=dataloader)
    metrics = compute_metrics(logits=logits, labels=labels, label_names=label_names)
    metrics["checkpoint_epoch"] = int(state.get("epoch", -1))
    metrics["checkpoint_val_accuracy"] = float(state.get("val_accuracy", float("nan")))
    if output_path is not None:
        save_json(metrics, output_path)
    return metrics


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    freeze_encoder: bool | None = None,
    cache_dir: str | None = None,
) -> LdtfBert:
    """Build an LdtfBert with the architecture implied by the checkpoint weights."""
    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if freeze_encoder is None:
        freeze_encoder = bool(state.get("freeze_encoder", False))
    model = LdtfBert(
        model_name=config.MODEL_NAME,
        freeze_encoder=freeze_encoder,
        cache_dir=cache_dir,
    )
    model.load_state_dict(state["model_state_dict"])
    return model