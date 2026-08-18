"""Terminal smoke tests for the Stage-6 LDTF-BERT label token router."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

# Direct execution (python scripts/...) sets sys.path to scripts/, while module
# execution starts at the project root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.token_router import LabelTokenRouter  # noqa: E402


NUM_CLASSES = 4
HIDDEN_SIZE = 768
ROUTER_DIM = 256


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load state safely across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _assert_output_contract(
    outputs: dict[str, torch.Tensor],
    batch_size: int,
    num_classes: int,
    num_layers: int,
    sequence_length: int,
    hidden_size: int,
    device: torch.device,
) -> None:
    """Check the fixed Stage-6 output shapes, device, and finite values."""
    token_attention = outputs["token_attention"]
    token_features = outputs["token_features"]
    assert token_attention.shape == (
        batch_size,
        num_classes,
        num_layers,
        sequence_length,
    )
    assert token_features.shape == (
        batch_size,
        num_classes,
        num_layers,
        hidden_size,
    )
    assert token_attention.device == device
    assert token_features.device == device
    assert torch.isfinite(token_attention).all()
    assert torch.isfinite(token_features).all()


def _make_router(
    device: torch.device,
    hidden_size: int = HIDDEN_SIZE,
    num_classes: int = NUM_CLASSES,
    router_dim: int = ROUTER_DIM,
) -> LabelTokenRouter:
    """Construct an eval-mode V1 router for deterministic synthetic checks."""
    return LabelTokenRouter(
        hidden_size=hidden_size,
        num_classes=num_classes,
        router_dim=router_dim,
    ).to(device).eval()


def _run_synthetic_shape_and_gradient_tests(device: torch.device) -> None:
    """Validate shapes, token normalization, and all required gradient paths."""
    batch_size, num_layers, sequence_length = 2, 12, 16
    router = _make_router(device)
    hidden_states = torch.randn(
        batch_size,
        num_layers,
        sequence_length,
        HIDDEN_SIZE,
        device=device,
        requires_grad=True,
    )
    label_queries = torch.randn(
        NUM_CLASSES,
        HIDDEN_SIZE,
        device=device,
        requires_grad=True,
    )
    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long,
        device=device,
    )
    outputs = router(hidden_states, label_queries, attention_mask)
    _assert_output_contract(
        outputs,
        batch_size,
        NUM_CLASSES,
        num_layers,
        sequence_length,
        HIDDEN_SIZE,
        device,
    )
    print(f"\nSynthetic hidden states:\n{hidden_states.shape}")
    print(f"\nLabel queries:\n{label_queries.shape}")
    print(f"\nToken attention:\n{outputs['token_attention'].shape}")
    print(f"\nToken features:\n{outputs['token_features'].shape}")
    print("\nShape test:\nPASS")

    attention_sum = outputs["token_attention"].float().sum(dim=-1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)
    print("\nAttention normalization test:\nPASS")

    router.zero_grad(set_to_none=True)
    gradient_objective = outputs["token_features"].pow(2).mean()
    gradient_objective.backward()
    required_gradients = (
        hidden_states.grad,
        label_queries.grad,
        router.query_projection.weight.grad,
        router.key_projection.weight.grad,
    )
    assert all(gradient is not None for gradient in required_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in required_gradients)
    print("\nGradient-flow test:\nPASS")


def _run_padding_and_masked_gradient_tests(device: torch.device) -> None:
    """Verify padded positions cannot receive attention or gradient signal."""
    batch_size, num_layers, sequence_length, hidden_size = 2, 3, 8, 16
    router = _make_router(
        device,
        hidden_size=hidden_size,
        num_classes=NUM_CLASSES,
        router_dim=8,
    )
    hidden_states = torch.randn(
        batch_size,
        num_layers,
        sequence_length,
        hidden_size,
        device=device,
        requires_grad=True,
    )
    label_queries = torch.randn(
        NUM_CLASSES,
        hidden_size,
        device=device,
        requires_grad=True,
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]],
        device=device,
    )
    outputs = router(hidden_states, label_queries, attention_mask)
    padding_mask = ~attention_mask[:, None, None, :].bool()
    padded_attention = outputs["token_attention"].masked_select(padding_mask)
    assert torch.allclose(
        padded_attention.float(),
        torch.zeros_like(padded_attention.float()),
        atol=1e-7,
    )
    print("\nPadding mask test:\nPASS")

    router.zero_grad(set_to_none=True)
    outputs["token_features"].pow(2).mean().backward()
    assert hidden_states.grad is not None
    padding_gradient_mask = (~attention_mask.bool())[:, None, :, None].expand_as(
        hidden_states
    )
    padding_gradients = hidden_states.grad.masked_select(padding_gradient_mask)
    assert torch.allclose(
        padding_gradients,
        torch.zeros_like(padding_gradients),
        atol=1e-7,
    )
    print("\nMasked-token gradient test:\nPASS")


def _run_single_valid_token_test(device: torch.device) -> None:
    """Check that one valid token receives all attention and all value weight."""
    batch_size, num_layers, sequence_length, hidden_size = 2, 3, 4, 8
    router = _make_router(
        device,
        hidden_size=hidden_size,
        num_classes=NUM_CLASSES,
        router_dim=4,
    )
    hidden_states = torch.randn(
        batch_size,
        num_layers,
        sequence_length,
        hidden_size,
        device=device,
    )
    label_queries = torch.randn(NUM_CLASSES, hidden_size, device=device)
    attention_mask = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], device=device)
    outputs = router(hidden_states, label_queries, attention_mask)

    assert torch.allclose(
        outputs["token_attention"][..., 0].float(),
        torch.ones_like(outputs["token_attention"][..., 0].float()),
        atol=1e-6,
    )
    assert torch.allclose(
        outputs["token_attention"][..., 1:].float(),
        torch.zeros_like(outputs["token_attention"][..., 1:].float()),
        atol=1e-7,
    )
    expected_features = hidden_states[:, :, 0, :].unsqueeze(1).expand(
        -1,
        NUM_CLASSES,
        -1,
        -1,
    )
    assert torch.allclose(outputs["token_features"], expected_features, atol=1e-6)
    print("\nSingle valid token test:\nPASS")


def _run_uniform_token_test(device: torch.device) -> None:
    """Identical valid token features must yield a uniform attention distribution."""
    batch_size, num_layers, sequence_length, hidden_size = 2, 3, 5, 8
    router = _make_router(
        device,
        hidden_size=hidden_size,
        num_classes=NUM_CLASSES,
        router_dim=4,
    )
    base_feature = torch.randn(batch_size, num_layers, 1, hidden_size, device=device)
    hidden_states = base_feature.expand(-1, -1, sequence_length, -1).clone()
    label_queries = torch.randn(NUM_CLASSES, hidden_size, device=device)
    attention_mask = torch.ones(batch_size, sequence_length, device=device)
    outputs = router(hidden_states, label_queries, attention_mask)
    expected_attention = torch.full_like(
        outputs["token_attention"].float(),
        1.0 / sequence_length,
    )
    assert torch.allclose(
        outputs["token_attention"].float(), expected_attention, atol=1e-6
    )
    expected_features = base_feature.squeeze(2).unsqueeze(1).expand(
        -1,
        NUM_CLASSES,
        -1,
        -1,
    )
    assert torch.allclose(outputs["token_features"], expected_features, atol=1e-6)
    print("\nUniform token test:\nPASS")


def _set_identity_projections(router: LabelTokenRouter) -> None:
    """Configure a small router so expected routing choices are explicit."""
    with torch.no_grad():
        identity = torch.eye(router.hidden_size, device=router.query_projection.weight.device)
        router.query_projection.weight.copy_(identity)
        router.key_projection.weight.copy_(identity)


def _run_class_conditioning_and_layer_tests(device: torch.device) -> None:
    """Show that query identity and layer identity both affect token routing."""
    router = _make_router(device, hidden_size=4, num_classes=2, router_dim=4)
    _set_identity_projections(router)
    label_queries = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        device=device,
    )
    hidden_states = torch.tensor(
        [[[[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
        device=device,
    )
    attention_mask = torch.ones(1, 3, device=device)
    outputs = router(hidden_states, label_queries, attention_mask)
    assert outputs["token_attention"][0, 0, 0].argmax().item() == 0
    assert outputs["token_attention"][0, 1, 0].argmax().item() == 1
    print("\nClass-conditioning test:\nPASS")

    layer_hidden_states = torch.zeros(1, 2, 3, 4, device=device)
    layer_hidden_states[0, 0, 0, 0] = 10.0
    layer_hidden_states[0, 1, 1, 0] = 10.0
    layer_outputs = router(layer_hidden_states, label_queries, attention_mask)
    class_zero_attention = layer_outputs["token_attention"][0, 0]
    assert class_zero_attention[0].argmax().item() == 0
    assert class_zero_attention[1].argmax().item() == 1
    print("\nLayer-specific test:\nPASS")


def _run_invalid_mask_tests(device: torch.device) -> None:
    """Reject all-masked samples and validate special-token exclusion."""
    router = _make_router(device, hidden_size=8, num_classes=NUM_CLASSES, router_dim=4)
    hidden_states = torch.randn(2, 2, 5, 8, device=device)
    label_queries = torch.randn(NUM_CLASSES, 8, device=device)
    all_masked = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], device=device)
    try:
        router(hidden_states, label_queries, all_masked)
    except ValueError as error:
        assert "invalid batch indices: [1]" in str(error)
    else:
        raise AssertionError("Expected an all-masked input to raise ValueError.")
    print("\nAll-masked input test:\nPASS")

    special_attention_mask = torch.tensor([[1, 1, 1, 1, 0]], device=device)
    special_tokens_mask = torch.tensor([[1, 0, 0, 1, 0]], device=device)
    special_outputs = router(
        hidden_states[:1],
        label_queries,
        special_attention_mask,
        special_tokens_mask=special_tokens_mask,
    )
    excluded_positions = torch.tensor([0, 3, 4], device=device)
    excluded_attention = special_outputs["token_attention"].index_select(
        dim=-1,
        index=excluded_positions,
    )
    assert torch.allclose(
        excluded_attention.float(),
        torch.zeros_like(excluded_attention.float()),
        atol=1e-7,
    )
    assert torch.allclose(
        special_outputs["token_attention"].float().sum(dim=-1),
        torch.ones_like(special_outputs["token_attention"].float().sum(dim=-1)),
        atol=1e-5,
    )
    print("\nSpecial-token mask test:\nPASS")


def _run_amp_test(device: torch.device) -> None:
    """Exercise FP32-softmax routing within CUDA autocast when available."""
    if device.type != "cuda":
        print("\nAMP test:\nSKIPPED (CUDA is unavailable)")
        return

    router = _make_router(device, hidden_size=32, num_classes=NUM_CLASSES, router_dim=16)
    hidden_states = torch.randn(2, 2, 5, 32, device=device, requires_grad=True)
    label_queries = torch.randn(NUM_CLASSES, 32, device=device, requires_grad=True)
    attention_mask = torch.ones(2, 5, dtype=torch.long, device=device)
    router.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = router(hidden_states, label_queries, attention_mask)
        amp_objective = outputs["token_features"].float().pow(2).mean()
    assert torch.isfinite(outputs["token_attention"]).all()
    assert torch.isfinite(outputs["token_features"]).all()
    assert torch.allclose(
        outputs["token_attention"].float().sum(dim=-1),
        torch.ones_like(outputs["token_attention"].float().sum(dim=-1)),
        atol=1e-4,
    )
    amp_objective.backward()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert label_queries.grad is not None and torch.isfinite(label_queries.grad).all()
    assert router.query_projection.weight.grad is not None
    assert router.key_projection.weight.grad is not None
    print("\nAMP test:\nPASS")


def _run_save_load_test(device: torch.device) -> None:
    """Ensure router state can round-trip without changing eval outputs."""
    router = _make_router(device, hidden_size=16, num_classes=NUM_CLASSES, router_dim=8)
    hidden_states = torch.randn(2, 2, 5, 16, device=device)
    label_queries = torch.randn(NUM_CLASSES, 16, device=device)
    attention_mask = torch.ones(2, 5, device=device)
    original_outputs = router(hidden_states, label_queries, attention_mask)

    with tempfile.TemporaryDirectory(prefix="ldtf_token_router_") as temporary_directory:
        state_path = Path(temporary_directory) / "token_router_state.pt"
        torch.save(router.state_dict(), state_path)
        reloaded_router = _make_router(
            device,
            hidden_size=16,
            num_classes=NUM_CLASSES,
            router_dim=8,
        )
        reloaded_router.load_state_dict(_load_state_dict(state_path, device))
        reloaded_outputs = reloaded_router(hidden_states, label_queries, attention_mask)
    assert torch.allclose(
        original_outputs["token_attention"],
        reloaded_outputs["token_attention"],
        atol=1e-6,
    )
    assert torch.allclose(
        original_outputs["token_features"],
        reloaded_outputs["token_features"],
        atol=1e-6,
    )
    print("\nSave/load test:\nPASS")


def _run_backbone_and_query_bank_integration(device: torch.device) -> None:
    """Run the real Stage-4/5/6 path, including one backward pass."""
    from transformers import AutoTokenizer

    from config import MAX_LENGTH, MODEL_NAME
    from models.bert_backbone import BertMultiLayerBackbone
    from models.label_query_bank import LabelQueryBank

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    backbone = BertMultiLayerBackbone(model_name=MODEL_NAME).to(device)
    query_bank = LabelQueryBank(
        num_classes=NUM_CLASSES,
        hidden_size=backbone.hidden_size,
    ).to(device)
    router = LabelTokenRouter(
        hidden_size=backbone.hidden_size,
        num_classes=NUM_CLASSES,
        router_dim=ROUTER_DIM,
    ).to(device)
    backbone.eval()
    query_bank.train()
    router.train()

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
        max_length=min(MAX_LENGTH, 32),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    backbone.zero_grad(set_to_none=True)
    query_bank.zero_grad(set_to_none=True)
    router.zero_grad(set_to_none=True)
    backbone_outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    hidden_states = backbone_outputs["hidden_states"]
    label_queries = query_bank()
    router_outputs = router(hidden_states, label_queries, attention_mask)
    batch_size, num_layers, sequence_length, hidden_size = hidden_states.shape
    _assert_output_contract(
        router_outputs,
        batch_size,
        NUM_CLASSES,
        num_layers,
        sequence_length,
        hidden_size,
        device,
    )
    assert label_queries.shape == (NUM_CLASSES, hidden_size)
    print(f"\nIntegrated hidden states:\n{hidden_states.shape}")
    print(f"\nIntegrated label queries:\n{label_queries.shape}")
    print(f"\nIntegrated token attention:\n{router_outputs['token_attention'].shape}")
    print(f"\nIntegrated token features:\n{router_outputs['token_features'].shape}")

    integration_objective = router_outputs["token_features"].pow(2).mean()
    integration_objective.backward()
    bert_gradients = [
        parameter.grad
        for parameter in backbone.bert.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert bert_gradients and all(torch.isfinite(gradient).all() for gradient in bert_gradients)
    assert query_bank.queries.grad is not None
    assert torch.isfinite(query_bank.queries.grad).all()
    assert router.query_projection.weight.grad is not None
    assert router.key_projection.weight.grad is not None
    assert torch.isfinite(router.query_projection.weight.grad).all()
    assert torch.isfinite(router.key_projection.weight.grad).all()
    print("\nBackbone integration test:\nPASS")
    print("\nQuery Bank integration test:\nPASS")


def _parse_args() -> argparse.Namespace:
    """Parse the opt-in checkpoint-backed Stage-4/5 integration test flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-backbone",
        action="store_true",
        help="Load bert-base-uncased and run the Stage-4/5 integration test.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all checkpoint-free synthetic tests and optional real integration."""
    args = _parse_args()
    print("=" * 60)
    print("LABEL TOKEN ROUTER SMOKE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice:\n{device}")
    if device.type == "cuda":
        print(f"\nGPU:\n{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"\nHidden size:\n{HIDDEN_SIZE}")
    print(f"\nRouter dimension:\n{ROUTER_DIM}")
    print(f"\nNumber of classes:\n{NUM_CLASSES}")

    parameter_router = _make_router(device)
    parameter_stats = parameter_router.count_parameters()
    assert parameter_stats == {
        "total": 2 * HIDDEN_SIZE * ROUTER_DIM,
        "trainable": 2 * HIDDEN_SIZE * ROUTER_DIM,
        "frozen": 0,
    }
    print(f"\nRouter parameters:\n{parameter_stats['total']}")

    _run_synthetic_shape_and_gradient_tests(device)
    _run_padding_and_masked_gradient_tests(device)
    _run_single_valid_token_test(device)
    _run_uniform_token_test(device)
    _run_class_conditioning_and_layer_tests(device)
    _run_invalid_mask_tests(device)
    _run_amp_test(device)
    _run_save_load_test(device)

    if args.with_backbone:
        _run_backbone_and_query_bank_integration(device)
    else:
        print(
            "\nBackbone integration test:\n"
            "SKIPPED (run with --with-backbone after caching bert-base-uncased)"
        )
        print("\nQuery Bank integration test:\nSKIPPED (requires --with-backbone)")
    print("\nSTAGE 6 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
