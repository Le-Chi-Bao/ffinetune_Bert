"""Aggregate the locked Stage-13 official-test results across all seeds.

Reports descriptive multi-seed statistics (ddof=1) and a paired, prediction-level
comparison between the baseline and the LDTF candidate. No significance is
claimed and no selection is performed here -- the manifest fixed everything.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stage13_manifest import ManifestError, load_manifest  # noqa: E402
from stage12_statistics import (  # noqa: E402
    format_mean_std,
    paired_differences,
    summarize,
    summarize_paired,
)
from stage13_official_test import paired_prediction_contingency  # noqa: E402

SCALAR_METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ManifestError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(path: Path) -> dict[str, list[int]]:
    if not path.is_file():
        raise ManifestError(f"Missing Stage-13 predictions: {path}")
    labels: list[int] = []
    predictions: list[int] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(int(row["label"]))
            predictions.append(int(row["prediction"]))
    if not labels:
        raise ManifestError(f"Empty Stage-13 predictions: {path}")
    return {"labels": labels, "predictions": predictions}


def collect(root: Path, manifest: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    """Load every locked seed's test artifacts, refusing any missing seed."""
    registry_path = root / "stage13_run_registry.json"
    if not registry_path.is_file():
        raise ManifestError(f"Stage-13 registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ManifestError("Stage-13 registry manifest hash does not match the locked manifest.")
    if registry.get("official_test_evaluated") is not True:
        raise ManifestError("Stage-13 registry does not record a completed official test evaluation.")

    locked_seeds = sorted(int(seed) for seed in manifest["seeds"])
    collected: dict[str, dict[int, dict[str, Any]]] = {}
    for name in manifest["models"]:
        for seed in locked_seeds:
            run_dir = root / name / f"seed_{seed}"
            summary_path = run_dir / "test_summary.json"
            metrics_path = run_dir / "test_metrics.json"
            if not summary_path.is_file() or not metrics_path.is_file():
                raise ManifestError(
                    f"Missing Stage-13 results for {name}/seed_{seed}. Every locked seed must be "
                    "evaluated; seeds must never be dropped."
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("manifest_sha256") != manifest.get("manifest_sha256"):
                raise ManifestError(f"{run_dir}: manifest hash mismatch.")
            collected.setdefault(name, {})[seed] = {
                "summary": summary,
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "predictions": read_predictions(run_dir / "test_predictions.csv"),
            }
        if sorted(collected.get(name, {})) != locked_seeds:
            raise ManifestError(f"{name}: seed coverage does not match the locked seeds.")
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate locked Stage-13 official test results.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    manifest = load_manifest(args.manifest)
    collected = collect(root, manifest)
    for required in (args.baseline, args.candidate):
        if required not in collected:
            raise ManifestError(f"Missing experiment {required!r} in the Stage-13 results.")

    summary_rows: list[dict[str, Any]] = []
    numeric: dict[str, Any] = {}
    for name, runs in collected.items():
        seeds = sorted(runs)
        stats = {
            metric: summarize([float(runs[seed]["metrics"][metric]) for seed in seeds])
            for metric in SCALAR_METRICS
        }
        class_names = list(runs[seeds[0]]["metrics"]["per_class"])
        class_stats = {
            class_name: summarize(
                [float(runs[seed]["metrics"]["per_class"][class_name]["f1"]) for seed in seeds]
            )
            for class_name in class_names
        }
        numeric[name] = {"seeds": seeds, "metrics": stats, "per_class_f1": class_stats}
        summary_rows.append(
            {
                "Model": name,
                "Seeds": len(seeds),
                "Test accuracy": format_mean_std(stats["accuracy"]),
                "Test Macro F1": format_mean_std(stats["f1_macro"]),
                "Test Weighted F1": format_mean_std(stats["f1_weighted"]),
                **{f"{c} F1": format_mean_std(class_stats[c]) for c in class_names},
            }
        )

    baseline_runs, candidate_runs = collected[args.baseline], collected[args.candidate]
    paired = paired_differences(
        {seed: float(run["metrics"]["f1_macro"]) for seed, run in baseline_runs.items()},
        {seed: float(run["metrics"]["f1_macro"]) for seed, run in candidate_runs.items()},
    )
    paired_summary = summarize_paired(paired)

    contingency_by_seed = {}
    for seed in sorted(baseline_runs):
        baseline_predictions = baseline_runs[seed]["predictions"]
        candidate_predictions = candidate_runs[seed]["predictions"]
        if baseline_predictions["labels"] != candidate_predictions["labels"]:
            raise ManifestError(
                f"seed {seed}: baseline and candidate were scored against different test "
                "label sequences; a paired comparison would be invalid."
            )
        contingency_by_seed[seed] = paired_prediction_contingency(
            baseline_predictions["labels"],
            baseline_predictions["predictions"],
            candidate_predictions["predictions"],
        )

    write_csv(summary_rows, Path(args.output_csv))
    write_csv(
        [
            {
                "seed": item["seed"],
                "baseline_macro_f1": item["baseline"],
                "candidate_macro_f1": item["candidate"],
                "delta_macro_f1": item["delta"],
                **{
                    key: contingency_by_seed[int(item["seed"])][key]
                    for key in (
                        "only_baseline_correct", "only_candidate_correct",
                        "discordant_pairs", "accuracy_delta",
                    )
                },
            }
            for item in paired
        ],
        root / "stage13_paired_seed_differences.csv",
    )
    save_json(
        {
            "manifest_sha256": manifest.get("manifest_sha256"),
            "protocol_hash": manifest.get("protocol_hash"),
            "seeds": sorted(int(seed) for seed in manifest["seeds"]),
            "official_test_evaluated": True,
            "models": numeric,
            "paired_comparison": {
                "baseline": args.baseline,
                "candidate": args.candidate,
                "summary": paired_summary,
                "per_seed": paired,
                "prediction_level_contingency": contingency_by_seed,
            },
            "selection_note": (
                "No model, seed, or hyperparameter was selected using these official test "
                "results; the manifest locked every choice beforehand."
            ),
            "statistical_claim": (
                "Descriptive multi-seed official-test results; statistical significance is "
                "not claimed."
            ),
        },
        Path(args.output_json),
    )
    print(f"Aggregated {sum(len(r) for r in collected.values())} official-test runs.")
    print(f"Mean paired delta (candidate - baseline) Macro F1: {paired_summary['mean_paired_delta']:.6f}")
    print(f"Seeds favoring candidate: {paired_summary['number_positive']}/{paired_summary['n_seeds']}")
    print("Statistical significance is not claimed; no selection used test results.")


if __name__ == "__main__":
    main()
