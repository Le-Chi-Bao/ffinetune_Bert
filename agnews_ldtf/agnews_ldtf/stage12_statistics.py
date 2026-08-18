"""Descriptive multi-seed statistics for Stage 12.

Sample standard deviation uses ddof=1. With fewer than two observations the
standard deviation is undefined and is reported as ``None`` rather than 0.0.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def _as_finite_float(value: Any, label: str) -> float:
    """Coerce to float and reject NaN/Inf so silent corruption cannot enter aggregates."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number, received {value!r}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite, received {numeric!r}.")
    return numeric


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """Return mean/std/min/max/median; std is None for a single observation."""
    numeric = [_as_finite_float(value, f"metric value[{index}]") for index, value in enumerate(values)]
    if not numeric:
        raise ValueError("Cannot summarize an empty sequence of metric values.")
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else None,
        "std_ddof": 1,
        "min": min(numeric),
        "max": max(numeric),
        "median": statistics.median(numeric),
        "values": numeric,
    }


def format_mean_std(summary: Mapping[str, Any], digits: int = 4) -> str:
    """Render ``mean ± std`` for tables while keeping numerics available."""
    if summary.get("std") is None:
        return f"{summary['mean']:.{digits}f} ± n/a"
    return f"{summary['mean']:.{digits}f} ± {summary['std']:.{digits}f}"


def paired_differences(
    baseline_by_seed: Mapping[int, float],
    candidate_by_seed: Mapping[int, float],
) -> list[dict[str, float]]:
    """Pair strictly by identical seed; refuse to silently drop unmatched seeds."""
    baseline_seeds = set(baseline_by_seed)
    candidate_seeds = set(candidate_by_seed)
    if baseline_seeds != candidate_seeds:
        missing_candidate = sorted(baseline_seeds - candidate_seeds)
        missing_baseline = sorted(candidate_seeds - baseline_seeds)
        raise ValueError(
            "Paired comparison requires identical seed sets. "
            f"missing_in_candidate={missing_candidate}, missing_in_baseline={missing_baseline}."
        )
    return [
        {
            "seed": seed,
            "baseline": _as_finite_float(baseline_by_seed[seed], f"baseline[seed={seed}]"),
            "candidate": _as_finite_float(candidate_by_seed[seed], f"candidate[seed={seed}]"),
            "delta": (
                _as_finite_float(candidate_by_seed[seed], f"candidate[seed={seed}]")
                - _as_finite_float(baseline_by_seed[seed], f"baseline[seed={seed}]")
            ),
        }
        for seed in sorted(baseline_seeds)
    ]


def summarize_paired(differences: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Aggregate paired deltas; candidate minus baseline, sign never inverted."""
    deltas = [float(item["delta"]) for item in differences]
    summary = summarize(deltas)
    return {
        "mean_paired_delta": summary["mean"],
        "std_paired_delta": summary["std"],
        "min_paired_delta": summary["min"],
        "max_paired_delta": summary["max"],
        "median_paired_delta": summary["median"],
        "number_positive": sum(1 for value in deltas if value > 0),
        "number_negative": sum(1 for value in deltas if value < 0),
        "number_zero": sum(1 for value in deltas if value == 0),
        "n_seeds": len(deltas),
        "interpretation": "delta = candidate - baseline; positive favors the candidate.",
        "statistical_claim": (
            "Descriptive multi-seed comparison only; statistical significance is not claimed."
        ),
    }


def file_sha256(path: str | Path) -> str:
    """Stream a SHA-256 digest for protocol and checkpoint locking."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    """Hash a JSON-serializable payload with stable key ordering."""
    import json

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
