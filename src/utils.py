"""Light-weight utilities for seeding, logging, and JSON I/O."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data: dict[str, Any], path: str | Path, *, indent: int = 2) -> None:
    """Write *data* as UTF-8 JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read UTF-8 JSON from *path* into a Python dict."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Context manager that prints the elapsed wall-clock time of *label*."""
    start = time.perf_counter()
    print(f"[timer] {label} ...", flush=True)
    yield
    elapsed = time.perf_counter() - start
    print(f"[timer] {label} done in {elapsed:.2f}s", flush=True)


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
