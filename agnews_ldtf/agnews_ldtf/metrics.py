"""Classification metrics with a fixed AG News label order."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def _to_one_dimensional_numpy(values: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert CPU/GPU tensors and common sequences to a non-empty 1-D array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D label array, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError("Cannot compute classification metrics for an empty array.")
    return array.astype(np.int64, copy=False)


def _as_python_native(value: Any) -> Any:
    """Recursively convert NumPy and tensor values so outputs are JSON-safe."""
    if isinstance(value, torch.Tensor):
        return _as_python_native(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_as_python_native(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _as_python_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python_native(item) for item in value]
    return value


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray | torch.Tensor,
    y_pred: Sequence[int] | np.ndarray | torch.Tensor,
    id2label: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Compute JSON-safe AG News metrics in a stable class order.

    Rows of the returned confusion matrix are ground-truth labels and columns
    are predicted labels.
    """
    true_array = _to_one_dimensional_numpy(y_true)
    pred_array = _to_one_dimensional_numpy(y_pred)
    if true_array.shape != pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, got "
            f"{true_array.shape} and {pred_array.shape}."
        )

    if id2label is None:
        id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    label_ids = list(id2label.keys())
    target_names = [id2label[label_id] for label_id in label_ids]
    allowed_labels = set(label_ids)
    observed_labels = set(true_array.tolist()).union(pred_array.tolist())
    unexpected_labels = observed_labels.difference(allowed_labels)
    if unexpected_labels:
        raise ValueError(
            f"Found labels outside the configured label order: {sorted(unexpected_labels)}."
        )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_array,
        pred_array,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_array,
        pred_array,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            true_array,
            pred_array,
            labels=label_ids,
            average=None,
            zero_division=0,
        )
    )
    report = classification_report(
        true_array,
        pred_array,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        true_array,
        pred_array,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    matrix = confusion_matrix(true_array, pred_array, labels=label_ids)

    per_class = {
        id2label[label_id]: {
            "precision": float(per_class_precision[index]),
            "recall": float(per_class_recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(per_class_support[index]),
        }
        for index, label_id in enumerate(label_ids)
    }
    return _as_python_native(
        {
            "accuracy": float(accuracy_score(true_array, pred_array)),
            "precision_macro": float(macro_precision),
            "recall_macro": float(macro_recall),
            "f1_macro": float(macro_f1),
            "precision_weighted": float(weighted_precision),
            "recall_weighted": float(weighted_recall),
            "f1_weighted": float(weighted_f1),
            "per_class": per_class,
            "confusion_matrix": matrix,
            "classification_report": report,
            "classification_report_text": report_text,
            "label_order": target_names,
            "num_samples": int(true_array.size),
        }
    )
