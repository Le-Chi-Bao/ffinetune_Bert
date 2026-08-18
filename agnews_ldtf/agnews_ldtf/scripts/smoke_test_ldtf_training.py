"""Stage-10 LDTF smoke tests for frozen and fine-tuning regimes.

Uses synthetic tokenized batches, never loads AG News or any official test data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

from evaluate import evaluate_model  # noqa: E402
from model_factory import (  # noqa: E402
    build_model,
    build_optimizer,
    extract_logits,
    get_backbone_module,
    get_ldtf_head_modules,
    set_model_training_mode,
    validate_optimizer_groups,
    verify_training_regime,
)
from train import train_one_epoch  # noqa: E402
from training_utils import (  # noqa: E402
    atomic_torch_save,
    create_grad_scaler,
    load_torch_checkpoint,
    set_seed,
)

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches() -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(123)
    return [
        {
            "input_ids": torch.randint(100, 20_000, (2, 12), generator=generator),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.randint(100, 20_000, (2, 12), generator=generator),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
            "labels": torch.tensor([2, 3], dtype=torch.long),
        },
    ]


def _snapshot(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def _changed(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(not torch.equal(old, new.detach().cpu()) for old, new in zip(before, module.parameters()))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_regime(regime: str) -> None:
    set_seed(42)
    model = build_model(
        "ldtf", model_name="bert-base-uncased", num_classes=NUM_CLASSES,
        token_router_dim=16, depth_router_dim=16, dropout=0.1, training_regime=regime,
    ).to(DEVICE)
    verify_training_regime(model, regime)
    backbone = get_backbone_module(model)
    head_modules = get_ldtf_head_modules(model)
    _assert(getattr(backbone.bert, "pooler", None) is None, f"{regime}: backbone exposes an unused pooler")
    _assert(not [n for n, _ in model.named_parameters() if ".pooler." in n], f"{regime}: pooler parameters registered")

    set_model_training_mode(model, regime)
    _assert(model.training, f"{regime}: full model must be in train mode")
    _assert(backbone.training is (regime == "finetune"), f"{regime}: incorrect backbone mode")
    _assert(model.class_scorer.training, f"{regime}: scorer must remain in train mode")
    if regime == "frozen":
        input_ids = torch.full((2, 12), 101, dtype=torch.long, device=DEVICE)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            first = backbone(input_ids, attention_mask)["hidden_states"]
            second = backbone(input_ids, attention_mask)["hidden_states"]
        _assert(torch.allclose(first, second, atol=0, rtol=0), "frozen backbone dropout is active")

    optimizer = build_optimizer(
        model, training_regime=regime, backbone_learning_rate=2e-5,
        head_learning_rate=1e-3, weight_decay=0.01,
    )
    validate_optimizer_groups(model, optimizer, training_regime=regime)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if regime == "frozen":
        _assert(not any(id(parameter) in optimizer_ids for parameter in backbone.parameters()), "frozen backbone in optimizer")
    else:
        _assert(all(id(parameter) in optimizer_ids for parameter in backbone.parameters()), "finetune backbone missing optimizer")

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 1)
    scaler = create_grad_scaler(False)
    backbone_before = _snapshot(backbone)
    heads_before = {name: _snapshot(module) for name, module in head_modules.items()}
    metrics = train_one_epoch(
        model=model, dataloader=_batches(), optimizer=optimizer, scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(), device=DEVICE, scaler=scaler, amp_enabled=False,
        amp_dtype=torch.float32, training_regime=regime, num_classes=NUM_CLASSES,
        grad_accumulation_steps=2, max_grad_norm=1.0, epoch=1, max_batches=None,
    )
    _assert(torch.isfinite(torch.tensor(metrics["loss"])), f"{regime}: non-finite loss")
    _assert(metrics["optimizer_steps"] == 1, f"{regime}: gradient accumulation step incorrect")
    if regime == "frozen":
        _assert(not _changed(backbone_before, backbone), "frozen backbone weights changed")
        _assert(all(parameter.grad is None for parameter in backbone.parameters()), "frozen backbone has gradients")
    else:
        _assert(metrics["backbone_received_gradient"], "finetuned backbone received no gradient")
        _assert(_changed(backbone_before, backbone), "finetuned backbone weights did not change")
    for name, module in head_modules.items():
        _assert(_changed(heads_before[name], module), f"{regime}: {name} weights did not change")

    first_batch = _batches()[0]
    model.eval()
    with torch.inference_mode():
        before_logits = extract_logits(model(**{key: value.to(DEVICE) for key, value in first_batch.items() if key != "labels"}), NUM_CLASSES)
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        # Saving a full fine-tuning AdamW state requires several GiB because BERT
        # has two moment tensors per parameter. The smoke test validates the
        # serialized model round trip here; train.py persists full optimizer and
        # scheduler state in its production checkpoints.
        atomic_torch_save({"model_state_dict": model.state_dict()}, checkpoint_path)
        restored = build_model("ldtf", model_name="bert-base-uncased", num_classes=NUM_CLASSES, token_router_dim=16, depth_router_dim=16, dropout=0.1, training_regime=regime).to(DEVICE)
        checkpoint = load_torch_checkpoint(checkpoint_path, DEVICE)
        restored.load_state_dict(checkpoint["model_state_dict"])
        restored.eval()
        with torch.inference_mode():
            after_logits = extract_logits(restored(**{key: value.to(DEVICE) for key, value in first_batch.items() if key != "labels"}), NUM_CLASSES)
        _assert(torch.allclose(before_logits, after_logits, atol=1e-6, rtol=1e-5), "checkpoint logits differ")

    validation = evaluate_model(model, _batches(), nn.CrossEntropyLoss(), DEVICE, num_classes=NUM_CLASSES)
    _assert(not model.training and not backbone.training and not model.class_scorer.training, "validation mode incorrect")
    _assert(torch.isfinite(torch.tensor(validation["loss"])), "validation loss non-finite")
    print(f"PASS {regime}: regime, optimizer groups, external loss, update, checkpoint, validation")


def main() -> None:
    for regime in ("frozen", "finetune"):
        _run_regime(regime)
    if torch.cuda.is_available():
        print("AMP path is covered by train.py with CUDA fp16; run cloud smoke test with --mixed-precision fp16.")
    print("All Stage-10 LDTF smoke tests PASSED. Official test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
