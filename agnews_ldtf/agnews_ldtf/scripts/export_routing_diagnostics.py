"""Export descriptive Stage-11 routing diagnostics for a trained LDTF checkpoint.

This is an *analysis* utility. It runs validation batches only, never the official
test split, and it produces descriptive attention statistics that are explicitly
not causal evidence about model behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation_diagnostics import compute_routing_diagnostics  # noqa: E402
from data import prepare_research_data  # noqa: E402
from evaluate import build_model_from_checkpoint  # noqa: E402
from training_utils import (  # noqa: E402
    get_device,
    load_torch_checkpoint,
    move_batch_to_device,
    save_json,
)


def collect_routing_diagnostics(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    max_batches: int = 8,
) -> dict[str, Any]:
    """Average routing diagnostics over a bounded number of validation batches."""
    if max_batches <= 0:
        raise ValueError(f"max_batches must be positive, got {max_batches}.")
    model.eval()
    collected: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            input_ids, attention_mask, _labels, token_type_ids = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_routing=True,
            )
            if not isinstance(outputs, dict):
                raise TypeError("Routing diagnostics require a model returning a dict.")
            missing = {"token_attention", "depth_attention"} - set(outputs)
            if missing:
                raise KeyError(
                    f"Model did not return routing tensors {sorted(missing)}; this variant "
                    "may not expose the requested router."
                )
            label_queries = _resolve_label_queries(model, outputs)
            collected.append(
                compute_routing_diagnostics(
                    token_attention=outputs["token_attention"],
                    depth_attention=outputs["depth_attention"],
                    label_queries=label_queries,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            )
    if not collected:
        raise RuntimeError("Routing diagnostics processed zero batches.")
    return _average_diagnostics(collected)


def _resolve_label_queries(model: torch.nn.Module, outputs: dict[str, Any]) -> torch.Tensor:
    """Find the label query bank on the model or in the forward outputs."""
    if "label_queries" in outputs and torch.is_tensor(outputs["label_queries"]):
        return outputs["label_queries"]
    for attribute in ("label_query_bank", "token_label_queries", "label_queries"):
        module = getattr(model, attribute, None)
        if torch.is_tensor(module):
            return module
        if module is not None:
            for inner in ("queries", "weight", "embedding"):
                tensor = getattr(module, inner, None)
                if torch.is_tensor(tensor):
                    return tensor
                if tensor is not None and torch.is_tensor(getattr(tensor, "weight", None)):
                    return tensor.weight
    raise AttributeError("Could not locate a label query bank for routing diagnostics.")


def _average_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average scalar and vector diagnostics across batches."""
    averaged: dict[str, Any] = {}
    reference = records[0]
    for key, value in reference.items():
        if isinstance(value, str):
            averaged[key] = value
        elif isinstance(value, (int, float)):
            averaged[key] = sum(float(record[key]) for record in records) / len(records)
        elif isinstance(value, list):
            stacked = torch.tensor([record[key] for record in records], dtype=torch.float64)
            averaged[key] = stacked.mean(dim=0).tolist()
    averaged["batches_analyzed"] = len(records)
    averaged["official_test_evaluated"] = False
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage-11 routing diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-path", default="data/processed/research_train.parquet")
    parser.add_argument("--validation-path", default="data/processed/research_validation.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    checkpoint = load_torch_checkpoint(args.checkpoint, device)
    model, model_config = build_model_from_checkpoint(checkpoint, device)
    if model_config.get("model_type") == "bert_baseline":
        raise ValueError("Routing diagnostics are only defined for LDTF models.")
    data = prepare_research_data(
        train_path=args.train_path,
        validation_path=args.validation_path,
        max_length=args.max_length,
        train_batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=int(checkpoint.get("training_config", {}).get("seed", 42)),
    )
    diagnostics = collect_routing_diagnostics(model, data["val_loader"], device, args.max_batches)
    diagnostics["checkpoint"] = str(Path(args.checkpoint).resolve())
    diagnostics["model_type"] = model_config.get("model_type")
    save_json(diagnostics, args.output)
    print(json.dumps({k: v for k, v in diagnostics.items() if not isinstance(v, list)}, indent=2))
    print(f"Wrote {args.output}")
    print("Descriptive routing statistics only; attention is not causal evidence.")
    print("Official test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
