"""Compare exactly two or more compatible Stage-10 frozen/finetune runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _per_class_f1(run_dir: Path) -> dict[str, float]:
    metrics = _load_json(run_dir / "metrics" / "best_validation_metrics.json")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError(f"Missing per_class validation metrics in {run_dir}.")
    return {str(name): float(values["f1"]) for name, values in per_class.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage-10 validation-only runs.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dirs = [Path(value).resolve() for value in args.runs]
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two Stage-10 runs to compare.")
    records: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        summary = _load_json(run_dir / "summary.json")
        signature = _load_json(run_dir / "data_signature.json")
        if summary.get("official_test_evaluated") is not False:
            raise ValueError(f"Run {run_dir} evaluated official test and is invalid for Stage-10 comparison.")
        f1 = _per_class_f1(run_dir)
        records.append(
            {
                "Regime": summary["training_regime"],
                "Best epoch": summary["best_epoch"],
                "Val loss": summary["best_validation_loss"],
                "Accuracy": summary["best_validation_accuracy"],
                "Macro F1": summary["best_validation_macro_f1"],
                "Trainable params": summary["trainable_parameters"],
                "Peak VRAM MB": summary["peak_vram_mb"],
                "Total time seconds": summary["training_time_seconds"],
                "World F1": f1["World"],
                "Sports F1": f1["Sports"],
                "Business F1": f1["Business"],
                "Sci/Tech F1": f1["Sci/Tech"],
            }
        )
        signatures.append(signature)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Refusing comparison: Stage-10 runs have different data signatures.")
    regimes = {record["Regime"] for record in records}
    if not regimes.issubset({"frozen", "finetune"}):
        raise ValueError("Only frozen and finetune Stage-10 regimes may be compared.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    provisional = sorted(records, key=lambda item: (-float(item["Macro F1"]), float(item["Val loss"]))) [0]
    print("| Regime | Best epoch | Val loss | Accuracy | Macro F1 | Trainable params | Peak VRAM MB | Total time |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(f"| {row['Regime']} | {row['Best epoch']} | {row['Val loss']:.6f} | {row['Accuracy']:.6f} | {row['Macro F1']:.6f} | {row['Trainable params']:,} | {row['Peak VRAM MB']:.2f} | {row['Total time seconds']:.2f}s |")
    print("| Regime | World F1 | Sports F1 | Business F1 | Sci/Tech F1 |")
    print("|---|---:|---:|---:|---:|")
    for row in records:
        print(f"| {row['Regime']} | {row['World F1']:.6f} | {row['Sports F1']:.6f} | {row['Business F1']:.6f} | {row['Sci/Tech F1']:.6f} |")
    print(f"Provisional best regime: {provisional['Regime']} (validation-only, one seed; not statistically superior).")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
