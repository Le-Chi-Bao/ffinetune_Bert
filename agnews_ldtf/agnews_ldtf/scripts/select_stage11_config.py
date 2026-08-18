"""Generate ``locked_stage12_config.json`` from aggregated Stage-11 ablation results.

Selection is validation-only and fully deterministic:

1. Highest validation Macro F1.
2. Ties broken by lower validation loss.
3. Remaining ties broken by fewer trainable parameters, then by variant name.

The selection never reads the official test split, and the resulting locked config
is an *input* to Stage 12 -- it is written once and then treated as immutable.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage12_statistics import payload_sha256  # noqa: E402

TIE_TOLERANCE = 1e-12

# Maps the aggregated table back to concrete architecture switches.
VARIANT_TO_ARCHITECTURE = {
    "A0_full": {"model_type": "ldtf", "ablation_variant": "A0_full"},
    "A1_no_token_router": {"model_type": "ldtf_ablation", "ablation_variant": "A1_no_token_router"},
    "A2_no_depth_router": {"model_type": "ldtf_ablation", "ablation_variant": "A2_no_depth_router"},
    "A3_final_layer": {"model_type": "ldtf_ablation", "ablation_variant": "A3_final_layer"},
    "A4_shared_token_query": {"model_type": "ldtf_ablation", "ablation_variant": "A4_shared_token_query"},
    "A5_shared_depth_query": {"model_type": "ldtf_ablation", "ablation_variant": "A5_shared_depth_query"},
    "A6_class_specific_scorer": {"model_type": "ldtf_ablation", "ablation_variant": "A6_class_specific_scorer"},
}


HYPERPARAMETER_KEYS = (
    "epochs",
    "train_batch_size",
    "eval_batch_size",
    "gradient_accumulation_steps",
    "backbone_learning_rate",
    "head_learning_rate",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "mixed_precision",
    "early_stopping_patience",
    "num_workers",
    "dropout",
)


class SelectionError(RuntimeError):
    """Raised when Stage-11 results cannot produce a trustworthy locked config."""


def load_json(path: str | Path, label: str) -> Any:
    file_path = Path(path)
    if not file_path.is_file():
        raise SelectionError(f"{label} not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_name(row: Mapping[str, Any]) -> str:
    return str(row["Variant"])


def rank_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Deterministic descending ranking: Macro F1, then loss, then params, then name."""
    if not rows:
        raise SelectionError("Stage-11 aggregate contains no results to select from.")
    return sorted(
        rows,
        key=lambda row: (
            -float(row["Macro F1"]),
            float(row["Val loss"]),
            int(row["Trainable params"]),
            _run_name(row),
        ),
    )


def select_best(rows: list[Mapping[str, Any]]) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Return the selected row plus an auditable ranking trace."""
    ranked = rank_rows(rows)
    best = ranked[0]
    ties = [
        row
        for row in ranked
        if abs(float(row["Macro F1"]) - float(best["Macro F1"])) <= TIE_TOLERANCE
    ]
    trace = [
        {
            "rank": index + 1,
            "variant": _run_name(row),
            "validation_macro_f1": float(row["Macro F1"]),
            "validation_loss": float(row["Val loss"]),
            "trainable_parameters": int(row["Trainable params"]),
        }
        for index, row in enumerate(ranked)
    ]
    for entry in trace:
        entry["tied_on_macro_f1_with_best"] = any(
            entry["variant"] == _run_name(row) for row in ties
        )
    return best, trace


def build_locked_config(
    *,
    best: Mapping[str, Any],
    trace: list[dict[str, Any]],
    aggregate: Mapping[str, Any],
    ablation_config: Mapping[str, Any],
    training_regime: str,
    stage10_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the immutable Stage-12 candidate definition."""
    variant_name = _run_name(best)
    variant_entry = next(
        (item for item in ablation_config["variants"] if item["name"] == variant_name),
        None,
    )
    if variant_entry is None:
        raise SelectionError(
            f"Selected variant {variant_name!r} is absent from the Stage-11 ablation config."
        )
    base_variant = str(variant_entry["variant"])
    architecture = VARIANT_TO_ARCHITECTURE.get(base_variant)
    if architecture is None:
        raise SelectionError(f"Unknown ablation variant {base_variant!r}; cannot lock an architecture.")

    missing = [key for key in HYPERPARAMETER_KEYS if key not in stage10_config]
    if missing:
        raise SelectionError(
            f"Stage-10 base config is missing training hyperparameters {missing}; "
            "Stage 12 cannot reproduce the selected architecture without them."
        )
    hyperparameters = {key: stage10_config[key] for key in HYPERPARAMETER_KEYS}

    core = {
        "stage": 11,
        "locked_for_stage": 12,
        "selected_run_name": variant_name,
        "model_type": architecture["model_type"],
        "ablation_variant": architecture["ablation_variant"],
        "training_regime": training_regime,
        "token_router_dim": int(variant_entry.get("token_router_dim", 256)),
        "depth_router_dim": int(variant_entry.get("depth_router_dim", 256)),
        "scorer_type": "class-specific" if base_variant == "A6_class_specific_scorer" else "shared",
        "exclude_special_tokens": bool(variant_entry.get("exclude_special_tokens", False)),
        "model_name": ablation_config["base_model"],
        "max_length": int(ablation_config["max_length"]),
        "num_classes": 4,
        **hyperparameters,
        "selection_metric": "validation_macro_f1",
        "selection_rule": (
            "max validation_macro_f1; ties -> lower validation loss -> fewer trainable "
            "parameters -> variant name"
        ),
        "selection_basis": "single-seed validation-only Stage-11 ablation",
        "selected_validation_macro_f1": float(best["Macro F1"]),
        "selected_validation_loss": float(best["Val loss"]),
        "official_test_evaluated": False,
        "comparison_protocol": aggregate.get("comparison_protocol"),
        "data_signature": aggregate.get("data_signature"),
        "ranking": trace,
        "statistical_claim": (
            "Provisional single-seed selection; no statistical significance is claimed. "
            "Stage 12 re-runs the selected architecture across all locked seeds."
        ),
    }
    return {
        **core,
        "locked_config_sha256": payload_sha256(core),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the Stage-11 winner and emit locked_stage12_config.json."
    )
    parser.add_argument("--aggregate-json", required=True, help="Output of aggregate_stage11_ablation.py")
    parser.add_argument("--ablation-config", required=True, help="configs/stage11_ablation.json")
    parser.add_argument("--stage10-summary", required=True, help="Base Stage-10 run summary.json")
    parser.add_argument(
        "--stage10-config",
        default=None,
        help="Base Stage-10 run config.json (defaults to config.json beside the summary).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SelectionError(
            f"Refusing to overwrite an existing locked config: {output}. "
            "A locked config is immutable; create a new protocol version instead."
        )

    aggregate = load_json(args.aggregate_json, "Stage-11 aggregate")
    ablation_config = load_json(args.ablation_config, "Stage-11 ablation config")
    stage10_summary = load_json(args.stage10_summary, "Stage-10 summary")
    stage10_config_path = (
        Path(args.stage10_config)
        if args.stage10_config
        else Path(args.stage10_summary).resolve().parent / "config.json"
    )
    stage10_config = load_json(stage10_config_path, "Stage-10 config")

    if stage10_summary.get("official_test_evaluated") is not False:
        raise SelectionError("Base Stage-10 run evaluated the official test; selection is invalid.")
    training_regime = stage10_summary.get("training_regime")
    if training_regime not in {"frozen", "finetune"}:
        raise SelectionError(f"Stage-10 summary has an invalid training_regime: {training_regime!r}")

    rows = aggregate.get("results")
    if not isinstance(rows, list):
        raise SelectionError("Stage-11 aggregate is missing a 'results' list.")
    expected = {item["name"] for item in ablation_config["variants"]}
    present = {_run_name(row) for row in rows}
    missing = sorted(expected - present)
    if missing:
        raise SelectionError(
            f"Stage-11 aggregate is missing variants {missing}. Every configured variant must "
            "complete before a Stage-12 config can be locked; variants must not be dropped."
        )

    best, trace = select_best(rows)
    locked = build_locked_config(
        best=best, trace=trace, aggregate=aggregate,
        ablation_config=ablation_config, training_regime=training_regime,
        stage10_config=stage10_config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(locked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Selected Stage-11 variant: {locked['selected_run_name']}")
    print(f"  model_type={locked['model_type']} ablation_variant={locked['ablation_variant']}")
    print(f"  training_regime={locked['training_regime']}")
    print(f"  validation Macro F1={locked['selected_validation_macro_f1']:.6f}")
    print(f"  locked_config_sha256={locked['locked_config_sha256']}")
    print(f"Wrote {output}")
    print("Selection used validation only; official test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
