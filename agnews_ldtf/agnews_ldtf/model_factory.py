"""Model construction, logit adapters, and optimizer groups for Stage 3/10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from models.bert_baseline import BertBaselineClassifier
from models.ldtf_ablation import LDTFAblationClassifier
from models.ldtf_bert import LDTFBertClassifier


DEFAULT_LDTF_CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
FINAL_DATA_REPORT_PATH = Path("data/reports/final_data_report.json")


def extract_logits(model_outputs: Any, num_classes: int | None = None) -> torch.Tensor:
    """Normalize tensor, dict, or attribute-style model outputs to logits ``[B,C]``."""
    if isinstance(model_outputs, torch.Tensor):
        logits = model_outputs
    elif isinstance(model_outputs, Mapping):
        if "logits" not in model_outputs:
            raise TypeError("Model output dictionary must contain key 'logits'.")
        logits = model_outputs["logits"]
    elif hasattr(model_outputs, "logits"):
        logits = model_outputs.logits
    else:
        raise TypeError(
            "Unsupported model output type: "
            f"{type(model_outputs).__name__}. Expected Tensor, dict, or object with .logits."
        )
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}.")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B,C], got {tuple(logits.shape)}.")
    if num_classes is not None and logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected logits with {num_classes} classes, but received {logits.shape[1]}."
        )
    return logits


def build_model(
    model_type: str,
    *,
    model_name: str = "bert-base-uncased",
    num_classes: int = 4,
    dropout: float = 0.1,
    training_regime: str = "finetune",
    token_router_dim: int = 256,
    depth_router_dim: int = 256,
    classifier_dropout: float | None = None,
    projection_bias: bool = False,
    scorer_bias: bool = True,
    class_names: tuple[str, ...] | None = None,
    variant: str = "A0_full",
    exclude_special_tokens: bool = False,
) -> nn.Module:
    """Build a baseline or LDTF classifier and apply the requested training regime."""
    if model_type not in {"bert_baseline", "ldtf", "ldtf_ablation"}:
        raise ValueError(
            "model_type must be 'bert_baseline', 'ldtf', or 'ldtf_ablation', "
            f"but received {model_type!r}."
        )
    if training_regime not in {"frozen", "finetune"}:
        raise ValueError(
            f"training_regime must be 'frozen' or 'finetune', but received {training_regime!r}."
        )

    if model_type == "bert_baseline":
        model: nn.Module = BertBaselineClassifier(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_type == "ldtf":
        resolved_dropout = dropout if classifier_dropout is None else classifier_dropout
        model = LDTFBertClassifier(
            model_name=model_name,
            num_classes=num_classes,
            token_router_dim=token_router_dim,
            depth_router_dim=depth_router_dim,
            classifier_dropout=resolved_dropout,
            projection_bias=projection_bias,
            scorer_bias=scorer_bias,
            class_names=class_names or DEFAULT_LDTF_CLASS_NAMES,
        )
    else:
        resolved_dropout = dropout if classifier_dropout is None else classifier_dropout
        model = LDTFAblationClassifier(
            model_name=model_name,
            num_classes=num_classes,
            variant=variant,
            token_router_dim=token_router_dim,
            depth_router_dim=depth_router_dim,
            classifier_dropout=resolved_dropout,
            projection_bias=projection_bias,
            scorer_bias=scorer_bias,
            exclude_special_tokens=exclude_special_tokens,
            class_names=class_names or DEFAULT_LDTF_CLASS_NAMES,
        )
    apply_training_regime(model, training_regime)
    return model


def apply_training_regime(model: nn.Module, training_regime: str) -> None:
    """Freeze or unfreeze the backbone/encoder according to the training regime."""
    if training_regime not in {"frozen", "finetune"}:
        raise ValueError(
            f"training_regime must be 'frozen' or 'finetune', but received {training_regime!r}."
        )
    if isinstance(model, (LDTFBertClassifier, LDTFAblationClassifier)):
        if training_regime == "frozen":
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        return
    if isinstance(model, BertBaselineClassifier):
        if training_regime == "frozen":
            model.freeze_encoder()
        else:
            model.unfreeze_encoder()
        return
    raise TypeError(f"Unsupported model type for training regime: {type(model).__name__}.")


def set_model_training_mode(model: nn.Module, training_regime: str) -> None:
    """Put heads in train mode; keep a frozen backbone/encoder in eval mode."""
    if training_regime not in {"frozen", "finetune"}:
        raise ValueError(
            f"training_regime must be 'frozen' or 'finetune', but received {training_regime!r}."
        )
    model.train()
    if training_regime != "frozen":
        return
    if isinstance(model, (LDTFBertClassifier, LDTFAblationClassifier)):
        model.backbone.eval()
        return
    if isinstance(model, BertBaselineClassifier):
        model.bert.eval()
        return
    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        model.backbone.eval()
        return
    if hasattr(model, "bert") and isinstance(model.bert, nn.Module):
        model.bert.eval()


def get_backbone_module(model: nn.Module) -> nn.Module:
    """Return the pretrained encoder submodule for either supported model type."""
    if isinstance(model, (LDTFBertClassifier, LDTFAblationClassifier)):
        return model.backbone
    if isinstance(model, BertBaselineClassifier):
        return model.bert
    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        return model.backbone
    if hasattr(model, "bert") and isinstance(model.bert, nn.Module):
        return model.bert
    raise AttributeError(
        f"Cannot locate a backbone/encoder submodule on {type(model).__name__}."
    )


def get_ldtf_head_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Return registered non-backbone modules for regime assertions."""
    if isinstance(model, LDTFAblationClassifier):
        return {name: module for name, module in model.named_children() if name != "backbone"}
    if isinstance(model, LDTFBertClassifier):
        return {
            "label_query_bank": model.label_query_bank,
            "token_router": model.token_router,
            "depth_router": model.depth_router,
            "class_scorer": model.class_scorer,
        }
    if isinstance(model, BertBaselineClassifier):
        return {"classifier": model.classifier}
    raise TypeError(f"Unsupported model type for head inspection: {type(model).__name__}.")


def verify_training_regime(model: nn.Module, training_regime: str) -> dict[str, Any]:
    """Assert the frozen/finetune contract on parameters and return a log-ready summary.

    Raises RuntimeError when the regime is not actually in force, so a
    misconfigured run fails before it consumes GPU hours.
    """
    if training_regime not in {"frozen", "finetune"}:
        raise ValueError(
            f"training_regime must be 'frozen' or 'finetune', but received {training_regime!r}."
        )
    backbone = get_backbone_module(model)
    backbone_parameters = list(backbone.parameters())
    if not backbone_parameters:
        raise RuntimeError("Backbone exposes no parameters; the model is misconfigured.")

    backbone_trainable = any(parameter.requires_grad for parameter in backbone_parameters)
    if training_regime == "frozen" and backbone_trainable:
        raise RuntimeError(
            "Frozen regime requires every backbone parameter to have requires_grad=False."
        )
    if training_regime == "finetune" and not all(
        parameter.requires_grad for parameter in backbone_parameters
    ):
        raise RuntimeError(
            "Fine-tuning regime requires every backbone parameter to have requires_grad=True."
        )

    head_status: dict[str, bool] = {}
    for module_name, module in get_ldtf_head_modules(model).items():
        parameters = list(module.parameters())
        if not parameters:
            raise RuntimeError(f"Head module {module_name} exposes no parameters.")
        if not any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError(f"Head module {module_name} must remain trainable.")
        head_status[module_name] = True
    return {
        "backbone_trainable": backbone_trainable,
        "head_trainable": head_status,
        "training_regime": training_regime,
    }


def _is_no_decay_parameter(parameter_name: str) -> bool:
    """Identify bias and LayerNorm scale parameters excluded from weight decay."""
    normalized_name = parameter_name.lower()
    return (
        normalized_name.endswith("bias")
        or "layernorm.weight" in normalized_name
        or "layer_norm.weight" in normalized_name
    )


def _is_backbone_parameter(model: nn.Module, parameter_name: str) -> bool:
    """Return True when a named parameter belongs to the pretrained encoder."""
    if isinstance(model, (LDTFBertClassifier, LDTFAblationClassifier)):
        return parameter_name.startswith("backbone.")
    if isinstance(model, BertBaselineClassifier):
        return parameter_name.startswith("bert.")
    return parameter_name.startswith("backbone.") or parameter_name.startswith("bert.")


def build_optimizer(
    model: nn.Module,
    *,
    training_regime: str,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build non-overlapping AdamW groups with differential backbone/head LRs."""
    if training_regime not in {"frozen", "finetune"}:
        raise ValueError(
            f"training_regime must be 'frozen' or 'finetune', but received {training_regime!r}."
        )
    if backbone_learning_rate <= 0 or head_learning_rate <= 0:
        raise ValueError("Backbone and head learning rates must both be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    buckets: dict[str, list[nn.Parameter]] = {
        "encoder_decay": [],
        "encoder_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    seen_parameter_ids: set[int] = set()
    trainable_parameter_ids: set[int] = set()

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            raise RuntimeError(
                f"Parameter appears in more than one optimizer group: {parameter_name}"
            )
        seen_parameter_ids.add(parameter_id)
        trainable_parameter_ids.add(parameter_id)

        is_backbone = _is_backbone_parameter(model, parameter_name)
        if training_regime == "frozen" and is_backbone:
            raise RuntimeError(
                "Frozen regime included a backbone parameter with requires_grad=True: "
                f"{parameter_name}"
            )
        component = "encoder" if is_backbone else "head"
        decay_group = "no_decay" if _is_no_decay_parameter(parameter_name) else "decay"
        buckets[f"{component}_{decay_group}"].append(parameter)

    group_settings = (
        ("encoder_decay", backbone_learning_rate, weight_decay),
        ("encoder_no_decay", backbone_learning_rate, 0.0),
        ("head_decay", head_learning_rate, weight_decay),
        ("head_no_decay", head_learning_rate, 0.0),
    )
    parameter_groups = [
        {
            "params": buckets[group_name],
            "lr": learning_rate,
            "weight_decay": group_weight_decay,
            "group_name": group_name,
        }
        for group_name, learning_rate, group_weight_decay in group_settings
        if buckets[group_name]
    ]
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were available for AdamW.")

    optimizer = torch.optim.AdamW(parameter_groups)
    validate_optimizer_groups(model, optimizer, training_regime=training_regime)
    return optimizer


def validate_optimizer_groups(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    training_regime: str,
) -> None:
    """Ensure optimizer coverage matches trainable parameters exactly once."""
    optimizer_ids: list[int] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer_ids.append(id(parameter))
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("Duplicate parameters detected in optimizer groups.")

    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer_id_set = set(optimizer_ids)
    if optimizer_id_set != trainable_ids:
        missing = trainable_ids.difference(optimizer_id_set)
        extra = optimizer_id_set.difference(trainable_ids)
        raise RuntimeError(
            "Optimizer parameter set does not match trainable model parameters. "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    if training_regime == "frozen":
        for group in optimizer.param_groups:
            name = str(group.get("group_name", ""))
            if name.startswith("encoder_") or name.startswith("backbone_"):
                raise RuntimeError("Frozen regime must not contain backbone optimizer groups.")


def describe_optimizer_groups(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Return a printable summary of named AdamW parameter groups."""
    summaries: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        parameters = list(group["params"])
        summaries.append(
            {
                "group_name": group.get("group_name", "unnamed"),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "num_tensors": len(parameters),
                "num_parameters": int(sum(parameter.numel() for parameter in parameters)),
            }
        )
    return summaries


def count_model_parameters(model: nn.Module) -> dict[str, Any]:
    """Return total/trainable/frozen counts, preferring model-specific helpers."""
    if hasattr(model, "count_parameters") and callable(model.count_parameters):
        stats = model.count_parameters()
        if isinstance(stats, Mapping) and "total" in stats and "trainable" in stats:
            frozen = int(stats.get("frozen", int(stats["total"]) - int(stats["trainable"])))
            return {
                "total": int(stats["total"]),
                "trainable": int(stats["trainable"]),
                "frozen": frozen,
                "by_module": stats.get("by_module"),
            }
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def enforce_data_quality_gate(
    *,
    report_path: str | Path = FINAL_DATA_REPORT_PATH,
    allow_unverified_data_for_smoke_test: bool = False,
) -> dict[str, Any]:
    """Require Stage-1 data readiness unless an explicit smoke-test override is set."""
    path = Path(report_path)
    if allow_unverified_data_for_smoke_test:
        print("DATA QUALITY GATE: BYPASSED (smoke-test override)")
        return {
            "overall_status": "BYPASSED",
            "READY_FOR_OFFICIAL_TRAINING": False,
            "allow_unverified_data_for_smoke_test": True,
            "report_path": str(path),
        }
    if not path.is_file():
        print("DATA QUALITY GATE: FAIL")
        raise RuntimeError(
            f"Data quality report not found at {path}. "
            "Run Stage-1 data preparation or pass "
            "--allow-unverified-data-for-smoke-test for non-official smoke tests only."
        )
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    overall_status = report.get("overall_status")
    ready = report.get("READY_FOR_OFFICIAL_TRAINING")
    if overall_status != "PASS" or ready is not True:
        print("DATA QUALITY GATE: FAIL")
        raise RuntimeError(
            "Data quality gate failed. "
            f"overall_status={overall_status!r}, READY_FOR_OFFICIAL_TRAINING={ready!r}."
        )
    print("DATA QUALITY GATE: PASS")
    return report


def build_data_signature(
    *,
    train_samples: int,
    validation_samples: int,
    tokenizer_name: str,
    max_length: int,
    label_mapping: Mapping[int, str],
    seed: int,
    quality_gate: Mapping[str, Any] | None = None,
    train_manifest_checksum: str | None = None,
    validation_manifest_checksum: str | None = None,
) -> dict[str, Any]:
    """Create a comparable immutable Stage-10 train/validation data signature."""
    train_checksum = train_manifest_checksum
    validation_checksum = validation_manifest_checksum
    if quality_gate is not None:
        train_checksum = train_checksum or quality_gate.get("train_manifest_checksum")
        validation_checksum = validation_checksum or quality_gate.get("validation_manifest_checksum")
    return {
        "dataset_protocol": (
            "ag_news_research_train_validation"
            if quality_gate is None
            else quality_gate.get("dataset_protocol", "ag_news_research_train_validation")
        ),
        "train_sample_count": int(train_samples),
        "validation_sample_count": int(validation_samples),
        "label_mapping": {str(key): value for key, value in label_mapping.items()},
        "tokenizer": tokenizer_name,
        "max_length": int(max_length),
        "split_seed": int(seed),
        "data_cleaning_version": (
            None if quality_gate is None else quality_gate.get("data_cleaning_version")
        ),
        "train_manifest_checksum": train_checksum,
        "validation_manifest_checksum": validation_checksum,
        "quality_gate_status": (
            None if quality_gate is None else quality_gate.get("overall_status")
        ),
    }
