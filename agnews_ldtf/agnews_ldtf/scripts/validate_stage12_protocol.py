"""Stage-12 protocol validator: refuse to start locked multi-seed runs on a bad protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FORBIDDEN_TEST_KEYS = (
    "test_path", "official_test_path", "decontaminated_test_path", "test_split", "test_parquet",
)
PLACEHOLDER_MARKERS = ("REPLACE_WITH", "<", ">", "TODO", "FIXME", "CHANGEME")


class ProtocolError(RuntimeError):
    """Raised when the Stage-12 protocol is not safe to execute."""


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ProtocolError(f"{label} not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} must contain a JSON object: {file_path}")
    return payload


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in PLACEHOLDER_MARKERS)
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return False


def _assert_no_test_keys(payload: Mapping[str, Any], label: str) -> None:
    for key in FORBIDDEN_TEST_KEYS:
        if key in payload:
            raise ProtocolError(f"{label} must not reference an official test split: {key!r}")
    for key, value in payload.items():
        if isinstance(value, Mapping):
            _assert_no_test_keys(value, f"{label}.{key}")
        elif isinstance(value, str) and "test" in key.lower() and "path" in key.lower():
            raise ProtocolError(f"{label} contains a test path key: {key!r}")


def validate_seeds(seeds: Any) -> list[int]:
    if not isinstance(seeds, list) or not seeds:
        raise ProtocolError("seeds must be a non-empty list.")
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ProtocolError(f"seeds must all be integers, got {seeds!r}.")
    if len(set(seeds)) != len(seeds):
        raise ProtocolError(f"seeds must not contain duplicates, got {seeds!r}.")
    return list(seeds)


def _comparable_data_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tokenizer": config.get("model_name"),
        "max_length": config.get("max_length"),
        "num_classes": config.get("num_classes"),
        "train_path": config.get("train_path"),
        "validation_path": config.get("validation_path"),
    }


def validate_protocol(config_path: str | Path) -> dict[str, Any]:
    """Run every Stage-12 protocol precondition and return a resolved summary."""
    config = load_json(config_path, "Stage-12 config")
    _assert_no_test_keys(config, "Stage-12 config")
    if config.get("official_test_allowed") is not False:
        raise ProtocolError("official_test_allowed must be exactly false.")
    seeds = validate_seeds(config.get("seeds"))

    for key in ("train_path", "validation_path", "data_quality_report", "stage11_locked_config"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ProtocolError(f"Stage-12 config is missing {key!r}.")
        if _contains_placeholder(value):
            raise ProtocolError(f"Stage-12 config {key!r} still contains a placeholder: {value!r}")
        if not Path(value).is_file():
            raise ProtocolError(f"Stage-12 {key!r} does not exist: {value}")

    report = load_json(config["data_quality_report"], "Data-quality report")
    if report.get("overall_status") != "PASS" or report.get("READY_FOR_OFFICIAL_TRAINING") is not True:
        raise ProtocolError(
            "Data-quality gate failed: "
            f"overall_status={report.get('overall_status')!r}, "
            f"READY_FOR_OFFICIAL_TRAINING={report.get('READY_FOR_OFFICIAL_TRAINING')!r}."
        )

    locked = load_json(config["stage11_locked_config"], "Stage-11 locked config")
    if _contains_placeholder(locked):
        raise ProtocolError("Stage-11 locked config still contains an unresolved placeholder.")
    _assert_no_test_keys(locked, "Stage-11 locked config")
    for key in ("model_type", "training_regime"):
        if not locked.get(key):
            raise ProtocolError(f"Stage-11 locked config is missing {key!r}.")
    if locked["model_type"] not in {"ldtf", "ldtf_ablation"}:
        raise ProtocolError(f"Unsupported locked model_type: {locked['model_type']!r}")
    if locked["training_regime"] not in {"frozen", "finetune"}:
        raise ProtocolError(f"Unsupported locked training_regime: {locked['training_regime']!r}")

    models = config.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ProtocolError("Stage-12 config must define at least a baseline and a candidate model.")
    resolved_models: list[dict[str, Any]] = []
    comparable: list[tuple[str, dict[str, Any]]] = []
    for model in models:
        name = model.get("experiment_name")
        source = model.get("config_source")
        if not name or not source:
            raise ProtocolError("Each Stage-12 model needs experiment_name and config_source.")
        if _contains_placeholder(source):
            raise ProtocolError(f"Model {name!r} config_source contains a placeholder: {source!r}")
        source_config = load_json(source, f"Config source for {name}")
        _assert_no_test_keys(source_config, f"Config source for {name}")
        model_type = model.get("model_type")
        if model_type == "from_stage11":
            model_type = locked["model_type"]
        if model_type not in {"bert_baseline", "ldtf", "ldtf_ablation"}:
            raise ProtocolError(f"Model {name!r} has unsupported model_type {model_type!r}.")
        resolved_models.append({"experiment_name": name, "model_type": model_type, "config_source": source})
        comparable.append((name, _comparable_data_fields(source_config)))

    reference_name, reference_fields = comparable[0]
    for name, fields in comparable[1:]:
        for key in ("tokenizer", "max_length", "num_classes"):
            if fields.get(key) != reference_fields.get(key):
                raise ProtocolError(
                    f"Data protocol mismatch between {reference_name!r} and {name!r} for {key!r}: "
                    f"{reference_fields.get(key)!r} vs {fields.get(key)!r}."
                )

    if config.get("selection_metric") != "validation_macro_f1":
        raise ProtocolError("selection_metric must be 'validation_macro_f1'.")

    return {
        "protocol_name": config.get("protocol_name"),
        "seeds": seeds,
        "models": resolved_models,
        "locked_stage11": locked,
        "data_quality_status": report.get("overall_status"),
        "official_test_allowed": False,
        "label_mapping": report.get("label_mapping"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the locked Stage-12 protocol.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    resolved = validate_protocol(args.config)
    print(f"Protocol: {resolved['protocol_name']}")
    print(f"Seeds: {resolved['seeds']}")
    print(f"Models: {[model['experiment_name'] for model in resolved['models']]}")
    print("Official test allowed: False")
    print("STAGE 12 PROTOCOL VALIDATION: PASS")


if __name__ == "__main__":
    main()
