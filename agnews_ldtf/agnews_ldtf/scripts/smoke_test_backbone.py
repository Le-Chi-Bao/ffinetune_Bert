"""Terminal smoke tests for the Stage-4 BERT multi-layer backbone."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Direct execution (python scripts/smoke_test_backbone.py) sets sys.path to the
# scripts directory, whereas module execution starts at the project root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import MAX_LENGTH, MODEL_NAME  # noqa: E402
from models.bert_backbone import BertMultiLayerBackbone  # noqa: E402


def _move_optional_tensor(
    encoded: dict[str, torch.Tensor], key: str, device: torch.device
) -> torch.Tensor | None:
    """Move an optional tokenizer field to the same model device."""
    value = encoded.get(key)
    return value.to(device) if value is not None else None


def _assert_output_contract(
    outputs: dict[str, torch.Tensor],
    model: BertMultiLayerBackbone,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> None:
    """Assert the project-wide ``[B,L,T,D]`` backbone output convention."""
    hidden_states = outputs["hidden_states"]
    last_hidden_state = outputs["last_hidden_state"]
    last_cls = outputs["last_cls"]

    assert hidden_states.ndim == 4
    assert hidden_states.shape == (
        batch_size,
        model.num_hidden_layers,
        sequence_length,
        model.hidden_size,
    )
    assert last_hidden_state.shape == (batch_size, sequence_length, model.hidden_size)
    assert last_cls.shape == (batch_size, model.hidden_size)
    assert hidden_states.device == device
    assert last_hidden_state.device == device
    assert last_cls.device == device
    assert torch.isfinite(hidden_states).all()
    assert torch.isfinite(last_hidden_state).all()
    assert torch.isfinite(last_cls).all()


def main() -> None:
    """Load pretrained BERT, validate all-layer outputs, then test gradients."""
    print("=" * 60)
    print("BERT MULTI-LAYER BACKBONE SMOKE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice:\n{device}")
    if device.type == "cuda":
        print(f"\nGPU:\n{torch.cuda.get_device_name(torch.cuda.current_device())}")
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = BertMultiLayerBackbone(model_name=MODEL_NAME).to(device)
    print(f"\nModel:\n{model.model_name}")
    print(f"\nHidden size:\n{model.hidden_size}")
    print(f"\nTransformer layers:\n{model.num_hidden_layers}")

    texts = [
        "United Nations leaders meet to discuss international security.",
        "Manchester United wins the championship match.",
        "Stocks rise after strong quarterly earnings.",
        "Google announces a new artificial intelligence system.",
    ]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = _move_optional_tensor(encoded, "token_type_ids", device)
    batch_size, sequence_length = input_ids.shape

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    _assert_output_contract(outputs, model, batch_size, sequence_length, device)

    hidden_states = outputs["hidden_states"]
    last_hidden_state = outputs["last_hidden_state"]
    last_cls = outputs["last_cls"]
    print(f"\nInput shape:\n{input_ids.shape}")
    print(f"\nHidden stack:\n{hidden_states.shape}")
    print(f"\nLast hidden state:\n{last_hidden_state.shape}")
    print(f"\nLast CLS:\n{last_cls.shape}")

    assert torch.allclose(hidden_states[:, -1], last_hidden_state, atol=1e-6)
    print("\nLast-layer equality test:\nPASS")

    layer_difference = (hidden_states[:, 0] - hidden_states[:, -1]).abs().mean()
    print(f"\nH1 vs H12 mean absolute difference:\n{layer_difference.item():.8f}")
    assert layer_difference.item() > 0
    print("\nDifferent-layer sanity test:\nPASS")
    print("\nFinite-value test:\nPASS")

    # torch.stack must preserve a path from both early and late layers to BERT.
    model.train()
    model.unfreeze_encoder()
    model.zero_grad(set_to_none=True)
    gradient_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    gradient_hidden_states = gradient_outputs["hidden_states"]
    dummy_objective = (
        gradient_hidden_states[:, 0].pow(2).mean()
        + gradient_hidden_states[:, -1].pow(2).mean()
    )
    dummy_objective.backward()
    finite_gradients = [
        parameter.grad
        for parameter in model.bert.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert finite_gradients
    assert all(torch.isfinite(gradient).all() for gradient in finite_gradients)
    model.zero_grad(set_to_none=True)
    print("\nGradient-flow test:\nPASS")

    initial_stats = model.count_parameters()
    assert initial_stats["total"] > 0
    assert initial_stats["total"] == initial_stats["trainable"] + initial_stats["frozen"]
    print("\nParameter counts before freeze:")
    for name, count in initial_stats.items():
        print(f"{name}: {count:,}")

    model.freeze_encoder()
    frozen_stats = model.count_parameters()
    assert model.is_encoder_frozen()
    assert all(not parameter.requires_grad for parameter in model.bert.parameters())
    assert frozen_stats["trainable"] == 0
    assert frozen_stats["frozen"] == frozen_stats["total"]
    print("\nFreeze test:\nPASS")

    model.eval()
    with torch.no_grad():
        frozen_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    _assert_output_contract(frozen_outputs, model, batch_size, sequence_length, device)
    print("\nFrozen forward test:\nPASS")

    model.unfreeze_encoder()
    unfrozen_stats = model.count_parameters()
    assert not model.is_encoder_frozen()
    assert all(parameter.requires_grad for parameter in model.bert.parameters())
    assert unfrozen_stats["trainable"] == unfrozen_stats["total"]
    assert unfrozen_stats["frozen"] == 0
    print("\nUnfreeze test:\nPASS")

    if device.type == "cuda":
        allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"\nPeak allocated GPU memory:\n{allocated_gb:.3f} GB")
        print(f"\nPeak reserved GPU memory:\n{reserved_gb:.3f} GB")

    print("\nSTAGE 4 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
