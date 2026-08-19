"""Aggregate fine-tune vs frozen metrics and produce comparison artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config                                # noqa: E402
from src.utils import load_json, save_json            # noqa: E402


EXPERIMENT_NAMES = ("finetune", "frozen")


def _load_split_metrics(name: str) -> tuple[dict, dict, dict]:
    """Return (val_metrics, test_metrics, log_records) for an experiment directory."""
    base = config.OUTPUTS_DIR / name
    val = load_json(base / "val_metrics.json")
    test = load_json(base / "test_metrics.json")
    return val, test, val.get("history", [])


def _summary_row(name: str, val: dict, test: dict) -> dict:
    """Return a one-row summary dict for an experiment."""
    return {
        "experiment": name,
        "best_val_accuracy": float(val.get("best_val_accuracy", float("nan"))),
        "best_epoch": int(val.get("best_epoch", -1)),
        "test_accuracy": float(test.get("accuracy", float("nan"))),
        "test_f1_macro": float(test.get("f1_macro", float("nan"))),
        "test_f1_weighted": float(test.get("f1_weighted", float("nan"))),
    }


def build_summary() -> pd.DataFrame:
    """Return a 2-row summary DataFrame of fine-tune vs frozen metrics."""
    rows = []
    for name in EXPERIMENT_NAMES:
        val, test, _ = _load_split_metrics(name)
        rows.append(_summary_row(name, val, test))
    return pd.DataFrame(rows)


def save_summary_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the comparison DataFrame to UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def save_summary_markdown(df: pd.DataFrame, path: Path) -> None:
    """Write a readable Markdown version of the comparison table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Fine-tune vs Frozen Encoder Comparison", ""]
    lines.append(df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("Notes:")
    lines.append("- **best_val_accuracy**: highest validation accuracy across all epochs.")
    lines.append("- **test_accuracy / test_f1_***: best-checkpoint metrics on the test split.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_training_curves(history_per_experiment: dict[str, list[dict]], path: Path) -> None:
    """Plot training loss and validation accuracy per epoch for both experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in history_per_experiment.items():
        epochs = [record["epoch"] for record in history]
        train_loss = [record["train_loss"] for record in history]
        val_acc = [record["val_accuracy"] for record in history]
        axes[0].plot(epochs, train_loss, marker="o", label=name)
        axes[1].plot(epochs, val_acc, marker="o", label=name)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(test_per_experiment: dict[str, dict], path: Path) -> None:
    """Plot confusion matrices side-by-side for fine-tune vs frozen."""
    fig, axes = plt.subplots(1, len(test_per_experiment), figsize=(5 * len(test_per_experiment), 4))
    if len(test_per_experiment) == 1:
        axes = [axes]
    for axis, (name, test_metrics) in zip(axes, test_per_experiment.items()):
        matrix = np.asarray(test_metrics["confusion_matrix"], dtype=np.float32)
        im = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"Confusion: {name}")
        axis.set_xticks(range(len(config.LABEL_NAMES)))
        axis.set_xticklabels(config.LABEL_NAMES, rotation=30, ha="right")
        axis.set_yticks(range(len(config.LABEL_NAMES)))
        axis.set_yticklabels(config.LABEL_NAMES)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Entry point that builds the comparison CSV, Markdown, and figures."""
    history = {}
    tests = {}
    for name in EXPERIMENT_NAMES:
        val, test, _ = _load_split_metrics(name)
        history[name] = val.get("history", [])
        tests[name] = test

    summary = build_summary()
    save_summary_csv(summary, config.REPORTS_DIR / "comparison.csv")
    save_summary_markdown(summary, config.REPORTS_DIR / "comparison.md")
    save_json({"summary": summary.to_dict(orient="records")}, config.REPORTS_DIR / "comparison.json")
    plot_training_curves(history, config.FIGURES_DIR / "loss_curve.png")
    plot_confusion_matrices(tests, config.FIGURES_DIR / "confusion.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()