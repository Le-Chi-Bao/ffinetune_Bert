"""Shared utilities for reproducible single-process baseline training."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import transformers


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA when available, then Apple MPS, otherwise CPU.

    AMP/GradScaler remain CUDA-only; MPS runs in float32 via resolve_mixed_precision.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_environment_info(device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    """Collect concise runtime metadata without invoking provider-specific commands."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_version": torch.version.cuda,
        "amp_enabled": amp_enabled,
        "git_commit": get_git_commit(),
    }
    if device.type == "cuda":
        current_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current_index)
        info.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "current_cuda_device": current_index,
                "gpu_name": torch.cuda.get_device_name(current_index),
                "gpu_total_memory_gb": round(properties.total_memory / (1024**3), 2),
            }
        )
    return info


def print_environment_info(info: Mapping[str, Any]) -> None:
    """Print the runtime facts most useful while diagnosing a cloud run."""
    print("\nEnvironment")
    print("=" * 50)
    ordered_keys = (
        "python_version",
        "pytorch_version",
        "transformers_version",
        "device",
        "cuda_available",
        "gpu_name",
        "cuda_device_count",
        "current_cuda_device",
        "gpu_total_memory_gb",
        "pytorch_cuda_version",
        "amp_enabled",
    )
    for key in ordered_keys:
        if key in info:
            print(f"{key}: {info[key]}")


def to_jsonable(value: Any) -> Any:
    """Convert paths, tensors, NumPy values, and nested structures for JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def ensure_directory(path: str | Path) -> Path:
    """Create a persistent output directory if it does not already exist."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def save_json(payload: Mapping[str, Any] | list[Any], path: str | Path) -> None:
    """Write JSON with only native serializable values."""
    destination = Path(path)
    ensure_directory(destination.parent)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2, ensure_ascii=False)
        file.write("\n")


def save_history(history: list[Mapping[str, Any]], output_dir: str | Path) -> None:
    """Persist epoch history after every completed epoch as JSON and CSV."""
    directory = ensure_directory(output_dir)
    save_json(history, directory / "history.json")
    if not history:
        return

    fieldnames: list[str] = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow({key: to_jsonable(record.get(key)) for key in fieldnames})


def capture_rng_state(dataloader_generator: Any | None = None) -> dict[str, Any]:
    """Capture RNG state so a checkpoint can resume as faithfully as possible."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "dataloader_generator": capture_dataloader_rng(dataloader_generator),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: Mapping[str, Any] | None,
    dataloader_generator: Any | None = None,
) -> None:
    """Restore a checkpoint RNG state, warning when legacy metadata is absent."""
    if not state:
        warnings.warn(
            "Checkpoint has no RNG state; resume remains valid but may not reproduce "
            "the exact uninterrupted run.",
            stacklevel=2,
        )
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        restore_dataloader_rng(dataloader_generator, state.get("dataloader_generator"))
    except KeyError as error:
        warnings.warn(
            f"Checkpoint RNG state is incomplete ({error}); exact resume reproducibility "
            "is unavailable.",
            stacklevel=2,
        )


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Save a checkpoint atomically within the requested persistent directory."""
    destination = Path(path)
    ensure_directory(destination.parent)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a full training checkpoint across recent and older PyTorch releases."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary, got {type(checkpoint).__name__}.")
    return checkpoint


def create_grad_scaler(amp_enabled: bool) -> Any:
    """Create a CUDA GradScaler while retaining compatibility with older PyTorch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def resolve_mixed_precision(mixed_precision: str, device: torch.device) -> tuple[bool, Any]:
    """Return (amp_enabled, dtype); AMP stays disabled on CPU per Stage-10 rules."""
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"mixed_precision must be 'no', 'fp16', or 'bf16', got {mixed_precision!r}."
        )
    if mixed_precision == "no" or device.type != "cuda":
        if mixed_precision != "no" and device.type != "cuda":
            warnings.warn(
                f"mixed_precision={mixed_precision!r} requires CUDA; running in float32.",
                stacklevel=2,
            )
        return False, torch.float32
    if mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        warnings.warn("bf16 is unsupported on this GPU; falling back to fp16.", stacklevel=2)
        return True, torch.float16
    return True, torch.bfloat16 if mixed_precision == "bf16" else torch.float16


def autocast_context(amp_enabled: bool, dtype: Any | None = None) -> Any:
    """Return CUDA autocast only when AMP is explicitly enabled."""
    if not amp_enabled:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=dtype if dtype is not None else torch.float16,
        enabled=True,
    )


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Move the supported Stage-1 batch fields without assuming token_type_ids."""
    required = {"input_ids", "attention_mask", "labels"}
    missing = required.difference(batch.keys())
    if missing:
        raise KeyError(f"Batch is missing required key(s): {sorted(missing)}.")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    token_type_ids = batch.get("token_type_ids")

    if input_ids.ndim != 2 or attention_mask.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            "Batch dimensions must be input_ids=[B,T], attention_mask=[B,T], labels=[B]."
        )
    if input_ids.shape != attention_mask.shape or input_ids.shape[0] != labels.shape[0]:
        raise ValueError("Batch input_ids, attention_mask, and labels have incompatible shapes.")
    if token_type_ids is not None and token_type_ids.shape != input_ids.shape:
        raise ValueError("token_type_ids must have the same shape as input_ids.")

    non_blocking = device.type == "cuda"
    return (
        input_ids.to(device, non_blocking=non_blocking),
        attention_mask.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
        (
            token_type_ids.to(device, non_blocking=non_blocking)
            if token_type_ids is not None
            else None
        ),
    )


def resolve_num_batches(dataloader: Any, max_batches: int | None, split_name: str) -> int:
    """Validate an optional batch cap and return actual batches to process."""
    loader_length = len(dataloader)
    if loader_length == 0:
        raise ValueError(f"{split_name} DataLoader is empty.")
    if max_batches is None:
        return loader_length
    if max_batches <= 0:
        raise ValueError(f"max_batches for {split_name} must be positive, got {max_batches}.")
    return min(loader_length, max_batches)


def set_training_mode(model: Any, encoder_mode: str) -> None:
    """Enable head dropout while keeping a frozen BERT encoder deterministic.

    Supports Stage-2 baseline (``model.bert``) and Stage-9 LDTF (``model.backbone``).
    After ``model.train()`` in the frozen regime the backbone/encoder is forced
    back to eval so BERT dropout does not inject noise into frozen features.
    """
    if encoder_mode not in {"frozen", "finetune"}:
        raise ValueError(f"Unsupported encoder mode: {encoder_mode!r}.")
    model.train()
    if encoder_mode != "frozen":
        return
    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), torch.nn.Module):
        model.backbone.eval()
        return
    if hasattr(model, "bert") and isinstance(getattr(model, "bert"), torch.nn.Module):
        model.bert.eval()
        return
    raise AttributeError(
        "Frozen training mode requires model.backbone or model.bert for eval pinning."
    )


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None when the project is not a repo."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def prepare_run_directory(
    output_dir: str | Path,
    run_name: str,
    overwrite: bool,
) -> Path:
    """Create ``output_dir/run_name`` and refuse to overwrite an existing run."""
    run_dir = Path(output_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Pass --overwrite-output-dir to replace it."
        )
    if overwrite and run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    for subdirectory in ("checkpoints", "metrics", "logs"):
        (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def write_train_log(run_dir: str | Path, message: str) -> None:
    """Append one line to the run log and print it."""
    print(message)
    log_path = Path(run_dir) / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def save_per_class_metrics_csv(
    per_class: Mapping[str, Mapping[str, Any]],
    path: str | Path,
) -> None:
    """Write per-class precision/recall/F1/support to CSV."""
    destination = Path(path)
    ensure_directory(destination.parent)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for class_name, values in per_class.items():
            writer.writerow(
                [
                    class_name,
                    values["precision"],
                    values["recall"],
                    values["f1"],
                    values["support"],
                ]
            )


def save_confusion_matrix_csv(
    matrix: Any,
    label_order: Sequence[str],
    path: str | Path,
) -> None:
    """Write a labeled confusion matrix with ground-truth rows."""
    destination = Path(path)
    ensure_directory(destination.parent)
    array = np.asarray(matrix, dtype=np.int64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] != len(label_order):
        raise ValueError(
            f"Confusion matrix shape {array.shape} does not match {len(label_order)} labels."
        )
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ground_truth\\prediction", *label_order])
        for label_name, row in zip(label_order, array.tolist()):
            writer.writerow([label_name, *row])


def capture_dataloader_rng(generator: Any | None) -> Any:
    """Serialize a DataLoader generator state when one is used."""
    if generator is None:
        return None
    return generator.get_state()


def restore_dataloader_rng(generator: Any | None, state: Any) -> None:
    """Restore a DataLoader generator state after resume."""
    if generator is None or state is None:
        return
    generator.set_state(state)


def get_optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Read separate encoder/head LRs from explicitly named optimizer groups."""
    rates: dict[str, list[float]] = {"encoder": [], "head": []}
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        if name.startswith("encoder_") or name.startswith("backbone_"):
            rates["encoder"].append(float(group["lr"]))
        elif name.startswith("head_"):
            rates["head"].append(float(group["lr"]))
    return {
        "encoder_lr": rates["encoder"][0] if rates["encoder"] else 0.0,
        "backbone_lr": rates["encoder"][0] if rates["encoder"] else 0.0,
        "head_lr": rates["head"][0] if rates["head"] else 0.0,
    }


def peak_memory_gb(device: torch.device) -> dict[str, float]:
    """Return peak CUDA memory in GB/MB, or zeros when running on CPU."""
    if device.type != "cuda":
        return {
            "peak_memory_allocated_gb": 0.0,
            "peak_memory_reserved_gb": 0.0,
            "peak_vram_mb": 0.0,
        }
    return {
        "peak_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "peak_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
    }


def iter_limited(dataloader: Any, max_batches: int | None) -> Iterator[tuple[int, Any]]:
    """Iterate a loader up to a validated limit without materializing batches."""
    limit = resolve_num_batches(dataloader, max_batches, "requested")
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= limit:
            break
        yield batch_index, batch
