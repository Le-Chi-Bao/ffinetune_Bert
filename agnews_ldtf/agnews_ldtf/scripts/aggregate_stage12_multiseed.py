"""Aggregate locked Stage-12 multi-seed validation results and lock the Stage-13 manifest."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stage12_protocol import ProtocolError, load_json  # noqa: E402
from stage12_statistics import (  # noqa: E402
    file_sha256,
    format_mean_std,
    paired_differences,
    payload_sha256,
    summarize,
    summarize_paired,
)

SCALAR_METRICS = (
    "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
)
ENVIRONMENT_KEYS = (
    "gpu_name", "pytorch_version", "transformers_version", "pytorch_cuda_version",
    "python_version", "git_commit",
)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ProtocolError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(root: Path, protocol_hash: str) -> dict[str, dict[int, dict[str, Any]]]:
    """Load every completed run, rejecting incomplete or protocol-violating runs."""
    registry = load_json(root / "run_registry.json", "Stage-12 run registry")
    if registry.get("protocol_hash") != protocol_hash:
        raise ProtocolError("Run registry protocol hash does not match the protocol lock.")
    collected: dict[str, dict[int, dict[str, Any]]] = {}
    for entry in registry.get("runs", []):
        status = entry.get("status")
        if status not in {"PASS", "SKIPPED_COMPLETED"}:
            raise ProtocolError(
                f"Run {entry.get('experiment_name')}/seed_{entry.get('seed')} has status {status!r}; "
                "every locked seed must complete before aggregation. Seeds must not be dropped."
            )
        run_dir = Path(entry["run_directory"])
        summary = load_json(run_dir / "summary.json", "summary")
        resolved = load_json(run_dir / "resolved_config.json", "resolved config")
        signature = load_json(run_dir / "data_signature.json", "data signature")
        metrics = load_json(run_dir / "metrics" / "best_validation_metrics.json", "validation metrics")
        environment = load_json(run_dir / "environment.json", "environment")
        if summary.get("official_test_evaluated") is not False:
            raise ProtocolError(f"{run_dir}: official test was evaluated; run is invalid for Stage 12.")
        if summary.get("official_test_loaded") not in (False, None):
            raise ProtocolError(f"{run_dir}: official test was loaded; run is invalid for Stage 12.")
        if resolved.get("protocol_hash") != protocol_hash:
            raise ProtocolError(f"{run_dir}: resolved config protocol hash mismatch.")
        seed = int(entry["seed"])
        if int(summary.get("seed", -1)) != seed or int(resolved.get("seed", -1)) != seed:
            raise ProtocolError(f"{run_dir}: seed mismatch between registry, summary, and resolved config.")
        collected.setdefault(entry["experiment_name"], {})[seed] = {
            "run_directory": run_dir,
            "summary": summary,
            "resolved": resolved,
            "signature": signature,
            "metrics": metrics,
            "environment": environment,
        }
    return collected


def assert_family_consistency(name: str, runs: Mapping[int, Mapping[str, Any]]) -> None:
    """Within one model family only the seed may differ."""
    reference_seed = min(runs)
    reference = dict(runs[reference_seed]["resolved"])
    reference.pop("seed", None)
    reference_params = runs[reference_seed]["summary"].get("total_parameters")
    reference_trainable = runs[reference_seed]["summary"].get("trainable_parameters")
    for seed, run in runs.items():
        candidate = dict(run["resolved"])
        candidate.pop("seed", None)
        if candidate != reference:
            differing = sorted(
                key for key in set(candidate) | set(reference)
                if candidate.get(key) != reference.get(key)
            )
            raise ProtocolError(f"{name}: config differs beyond the seed at seed {seed}: {differing}")
        if run["summary"].get("total_parameters") != reference_params:
            raise ProtocolError(
                f"{name}: total parameter count differs at seed {seed} "
                f"({run['summary'].get('total_parameters')} vs {reference_params}); "
                "parameter counts must never be averaged over a config mismatch."
            )
        if run["summary"].get("trainable_parameters") != reference_trainable:
            raise ProtocolError(f"{name}: trainable parameter count differs at seed {seed}.")


def check_environments(collected: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> bool:
    """Warn when hardware/software differ; efficiency must not be compared across them."""
    fingerprints = {
        (name, seed): tuple(run["environment"].get(key) for key in ENVIRONMENT_KEYS)
        for name, runs in collected.items()
        for seed, run in runs.items()
    }
    unique = set(fingerprints.values())
    if len(unique) > 1:
        warnings.warn(
            "Stage-12 runs span multiple environments; accuracy may still be aggregated but "
            "peak VRAM and training time must not be compared directly across hardware.",
            stacklevel=2,
        )
        return False
    return True


def metric_values(runs: Mapping[int, Mapping[str, Any]], key: str) -> dict[int, float]:
    return {seed: float(run["metrics"][key]) for seed, run in runs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate locked Stage-12 multi-seed results.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    lock = load_json(root / "protocol_lock.json", "Stage-12 protocol lock")
    protocol_hash = lock["protocol_hash"]
    collected = collect_runs(root, protocol_hash)
    for required in (args.baseline, args.candidate):
        if required not in collected:
            raise ProtocolError(f"Missing experiment {required!r} in the Stage-12 registry.")

    locked_seeds = set(lock["seeds"])
    for name, runs in collected.items():
        assert_family_consistency(name, runs)
        if set(runs) != locked_seeds:
            raise ProtocolError(
                f"{name}: seed set {sorted(runs)} does not match the locked seeds {sorted(locked_seeds)}. "
                "Missing seeds must be re-run, never dropped."
            )
    signatures = {
        payload_sha256(run["signature"])
        for runs in collected.values() for run in runs.values()
    }
    if len(signatures) != 1:
        raise ProtocolError("Stage-12 runs do not share an identical data signature.")
    same_environment = check_environments(collected)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    numeric: dict[str, Any] = {}
    for name, runs in collected.items():
        seeds = sorted(runs)
        stats = {metric: summarize(list(metric_values(runs, metric).values())) for metric in SCALAR_METRICS}
        numeric[name] = {"seeds": seeds, "metrics": stats}
        summary_rows.append(
            {
                "Model": name, "Seeds": len(seeds),
                **{
                    label: format_mean_std(stats[key])
                    for label, key in (
                        ("Val loss", "loss"), ("Accuracy", "accuracy"),
                        ("Macro Precision", "precision_macro"), ("Macro Recall", "recall_macro"),
                        ("Macro F1", "f1_macro"), ("Weighted F1", "f1_weighted"),
                    )
                },
                "Macro F1 min": stats["f1_macro"]["min"], "Macro F1 max": stats["f1_macro"]["max"],
            }
        )
        class_names = list(runs[seeds[0]]["metrics"]["per_class"])
        class_stats = {
            class_name: summarize([float(runs[seed]["metrics"]["per_class"][class_name]["f1"]) for seed in seeds])
            for class_name in class_names
        }
        numeric[name]["per_class_f1"] = class_stats
        per_class_rows.append({"Model": name, **{f"{c} F1": format_mean_std(class_stats[c]) for c in class_names}})
        vram = summarize([float(runs[s]["summary"]["peak_vram_mb"]) for s in seeds])
        duration = summarize([float(runs[s]["summary"]["training_time_seconds"]) for s in seeds])
        best_epoch = summarize([float(runs[s]["summary"]["best_epoch"]) for s in seeds])
        numeric[name]["efficiency"] = {"peak_vram_mb": vram, "training_time_seconds": duration, "best_epoch": best_epoch}
        efficiency_rows.append(
            {
                "Model": name,
                "Total params": runs[seeds[0]]["summary"]["total_parameters"],
                "Trainable params": runs[seeds[0]]["summary"]["trainable_parameters"],
                "Peak VRAM MB": format_mean_std(vram, 2), "Train time s": format_mean_std(duration, 2),
                "Best epoch": format_mean_std(best_epoch, 2),
                "Comparable hardware": same_environment,
            }
        )

    baseline_runs, candidate_runs = collected[args.baseline], collected[args.candidate]
    paired_rows = []
    for seed_pair in paired_differences(metric_values(baseline_runs, "f1_macro"), metric_values(candidate_runs, "f1_macro")):
        seed = int(seed_pair["seed"])
        paired_rows.append(
            {
                "seed": seed,
                "baseline_macro_f1": seed_pair["baseline"], "ldtf_macro_f1": seed_pair["candidate"],
                "delta_macro_f1": seed_pair["delta"],
                "baseline_accuracy": float(baseline_runs[seed]["metrics"]["accuracy"]),
                "ldtf_accuracy": float(candidate_runs[seed]["metrics"]["accuracy"]),
                "delta_accuracy": float(candidate_runs[seed]["metrics"]["accuracy"]) - float(baseline_runs[seed]["metrics"]["accuracy"]),
                "baseline_val_loss": float(baseline_runs[seed]["metrics"]["loss"]),
                "ldtf_val_loss": float(candidate_runs[seed]["metrics"]["loss"]),
                "delta_val_loss": float(candidate_runs[seed]["metrics"]["loss"]) - float(baseline_runs[seed]["metrics"]["loss"]),
            }
        )
    paired_summary = summarize_paired([{"delta": row["delta_macro_f1"]} for row in paired_rows])

    write_csv(summary_rows, Path(args.output_csv))
    write_csv(per_class_rows, root / "per_class_multiseed.csv")
    write_csv(efficiency_rows, root / "efficiency_multiseed.csv")
    write_csv(paired_rows, root / "paired_seed_differences.csv")
    save_json(
        {
            "protocol_hash": protocol_hash, "seeds": sorted(locked_seeds),
            "data_signature_hash": next(iter(signatures)), "official_test_evaluated": False,
            "same_environment": same_environment, "models": numeric,
            "paired_comparison": {"baseline": args.baseline, "candidate": args.candidate, "summary": paired_summary, "per_seed": paired_rows},
            "statistical_claim": "Descriptive multi-seed results; statistical significance is not claimed.",
        },
        Path(args.output_json),
    )

    manifest_models = {
        name: [
            {
                "seed": seed,
                "checkpoint": str(runs[seed]["run_directory"] / "checkpoints" / "best.pt"),
                "checkpoint_sha256": file_sha256(runs[seed]["run_directory"] / "checkpoints" / "best.pt"),
            }
            for seed in sorted(runs)
        ]
        for name, runs in collected.items()
    }
    manifest_core = {
        "stage": 13, "protocol_hash": protocol_hash, "data_signature_hash": next(iter(signatures)),
        "official_test_evaluated": False, "seeds": sorted(locked_seeds), "models": manifest_models,
        "selection_note": "All locked seeds are included; no best-seed cherry-picking is permitted.",
    }
    save_json({**manifest_core, "manifest_sha256": payload_sha256(manifest_core)}, root / "locked_stage13_manifest.json")
    save_json(
        {
            "stage": 12, "protocol_hash": protocol_hash, "models": list(collected),
            "seeds": sorted(locked_seeds), "official_test_evaluated": False,
            "paired_summary": paired_summary,
        },
        root / "stage12_summary.json",
    )
    print(f"Aggregated {sum(len(r) for r in collected.values())} runs across {len(collected)} models.")
    print(f"Mean paired delta (candidate - baseline) Macro F1: {paired_summary['mean_paired_delta']:.6f}")
    print(f"Seeds favoring candidate: {paired_summary['number_positive']}/{paired_summary['n_seeds']}")
    print("Statistical significance is not claimed. Official test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
