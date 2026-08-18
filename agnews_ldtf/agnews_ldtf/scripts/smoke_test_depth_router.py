"""Terminal smoke tests for the Stage-7 LDTF-BERT label depth router."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch

# Direct execution (python scripts/...) sets sys.path to scripts/, while module
# execution starts at the project root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.depth_router import LabelDepthRouter  # noqa: E402


NUM_CLASSES = 4
HIDDEN_SIZE = 768
NUM_LAYERS = 12
ROUTER_DIM = 256


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a state dictionary across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _make_router(
    device: torch.device,
    hidden_size: int = HIDDEN_SIZE,
    num_classes: int = NUM_CLASSES,
    num_layers: int = NUM_LAYERS,
    router_dim: int = ROUTER_DIM,
) -> LabelDepthRouter:
    """Build an eval-mode depth router for deterministic tests."""
    return LabelDepthRouter(
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        router_dim=router_dim,
    ).to(device).eval()


def _assert_output_contract(
    outputs: dict[str, torch.Tensor],
    batch_size: int,
    num_classes: int,
    num_layers: int,
    hidden_size: int,
    device: torch.device,
) -> None:
    """Validate Stage-7 output shape, device, and finite-value contracts."""
    depth_attention = outputs["depth_attention"]
    fused_features = outputs["fused_features"]
    assert depth_attention.shape == (batch_size, num_classes, num_layers)
    assert fused_features.shape == (batch_size, num_classes, hidden_size)
    assert depth_attention.device == device
    assert fused_features.device == device
    assert torch.isfinite(depth_attention).all()
    assert torch.isfinite(fused_features).all()


def _run_shape_weighted_sum_and_gradient_tests(device: torch.device) -> None:
    """Test output contracts, layer softmax, weighted sum, and gradients."""
    batch_size = 2
    router = _make_router(device)
    token_features = torch.randn(
        batch_size,
        NUM_CLASSES,
        NUM_LAYERS,
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
    outputs = router(token_features, label_queries)
    _assert_output_contract(
        outputs,
        batch_size,
        NUM_CLASSES,
        NUM_LAYERS,
        HIDDEN_SIZE,
        device,
    )
    print(f"\nSynthetic token features:\n{token_features.shape}")
    print(f"\nLabel queries:\n{label_queries.shape}")
    print(f"\nDepth attention:\n{outputs['depth_attention'].shape}")
    print(f"\nFused features:\n{outputs['fused_features'].shape}")
    print("\nShape test:\nPASS")

    attention_sum = outputs["depth_attention"].float().sum(dim=-1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)
    print("\nAttention normalization test:\nPASS")
    print("\nFinite-value test:\nPASS")

    manual_fused_features = torch.einsum(
        "bcl,bcld->bcd",
        outputs["depth_attention"],
        token_features,
    )
    assert torch.allclose(
        outputs["fused_features"], manual_fused_features, atol=1e-6
    )
    print("\nWeighted-sum test:\nPASS")

    router.zero_grad(set_to_none=True)
    gradient_objective = outputs["fused_features"].pow(2).mean()
    gradient_objective.backward()
    required_gradients = (
        token_features.grad,
        label_queries.grad,
        router.query_projection.weight.grad,
        router.depth_key_projection.weight.grad,
    )
    assert all(gradient is not None for gradient in required_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in required_gradients)
    print("\nGradient-flow test:\nPASS")


def _run_single_and_identical_layer_tests(device: torch.device) -> None:
    """Validate exact behavior for one layer and identical layer features."""
    batch_size, hidden_size = 2, 16
    single_layer_router = _make_router(
        device,
        hidden_size=hidden_size,
        num_layers=1,
        router_dim=8,
    )
    single_features = torch.randn(
        batch_size,
        NUM_CLASSES,
        1,
        hidden_size,
        device=device,
    )
    label_queries = torch.randn(NUM_CLASSES, hidden_size, device=device)
    single_outputs = single_layer_router(single_features, label_queries)
    assert torch.allclose(
        single_outputs["depth_attention"].float(),
        torch.ones_like(single_outputs["depth_attention"].float()),
        atol=1e-6,
    )
    assert torch.allclose(
        single_outputs["fused_features"], single_features[:, :, 0, :], atol=1e-6
    )
    print("\nSingle-layer test:\nPASS")

    num_layers = 5
    identical_router = _make_router(
        device,
        hidden_size=hidden_size,
        num_layers=num_layers,
        router_dim=8,
    )
    base_feature = torch.randn(
        batch_size,
        NUM_CLASSES,
        1,
        hidden_size,
        device=device,
    )
    identical_features = base_feature.expand(-1, -1, num_layers, -1).clone()
    identical_outputs = identical_router(identical_features, label_queries)
    expected_attention = torch.full_like(
        identical_outputs["depth_attention"].float(),
        1.0 / num_layers,
    )
    assert torch.allclose(
        identical_outputs["depth_attention"].float(),
        expected_attention,
        atol=1e-6,
    )
    assert torch.allclose(
        identical_outputs["fused_features"], base_feature.squeeze(2), atol=1e-6
    )
    print("\nIdentical-layer test:\nPASS")


def _set_identity_projections(router: LabelDepthRouter) -> None:
    """Make compact tests analytically predictable without adding a new layer."""
    with torch.no_grad():
        identity = torch.eye(router.hidden_size, device=router.query_projection.weight.device)
        router.query_projection.weight.copy_(identity)
        router.depth_key_projection.weight.copy_(identity)


def _run_class_and_sample_conditioning_tests(device: torch.device) -> None:
    """Show attention changes with both class query and sample evidence."""
    router = _make_router(
        device,
        hidden_size=4,
        num_classes=2,
        num_layers=3,
        router_dim=4,
    )
    _set_identity_projections(router)
    label_queries = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        device=device,
    )
    token_features = torch.zeros(1, 2, 3, 4, device=device)
    token_features[0, 0, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0], device=device)
    token_features[0, 1, 1] = torch.tensor([0.0, 10.0, 0.0, 0.0], device=device)
    outputs = router(token_features, label_queries)
    assert outputs["depth_attention"][0, 0].argmax().item() == 0
    assert outputs["depth_attention"][0, 1].argmax().item() == 1
    print("\nClass-conditioning test:\nPASS")

    sample_router = _make_router(
        device,
        hidden_size=4,
        num_classes=2,
        num_layers=8,
        router_dim=4,
    )
    _set_identity_projections(sample_router)
    sample_features = torch.zeros(2, 2, 8, 4, device=device)
    sample_features[0, 0, 2] = torch.tensor([10.0, 0.0, 0.0, 0.0], device=device)
    sample_features[1, 0, 7] = torch.tensor([10.0, 0.0, 0.0, 0.0], device=device)
    sample_outputs = sample_router(sample_features, label_queries)
    assert sample_outputs["depth_attention"][0, 0].argmax().item() == 2
    assert sample_outputs["depth_attention"][1, 0].argmax().item() == 7
    print("\nSample-conditioning test:\nPASS")


def _assert_value_error(action: Callable[[], object], expected_message: str) -> None:
    """Require a specific validation failure without masking unexpected errors."""
    try:
        action()
    except ValueError as error:
        assert expected_message in str(error)
    else:
        raise AssertionError(f"Expected ValueError containing {expected_message!r}.")


def _run_input_validation_tests(device: torch.device) -> None:
    """Exercise every documented shape and device validation branch."""
    router = _make_router(
        device,
        hidden_size=8,
        num_classes=2,
        num_layers=3,
        router_dim=4,
    )
    features = torch.randn(1, 2, 3, 8, device=device)
    queries = torch.randn(2, 8, device=device)
    _assert_value_error(
        lambda: router(torch.randn(1, 2, 8, device=device), queries),
        "token_features must have shape",
    )
    _assert_value_error(
        lambda: router(features, torch.randn(1, 2, 8, device=device)),
        "label_queries must have shape",
    )
    _assert_value_error(
        lambda: router(torch.randn(1, 3, 3, 8, device=device), queries),
        "2 classes",
    )
    _assert_value_error(
        lambda: router(torch.randn(1, 2, 2, 8, device=device), queries),
        "3 layers",
    )
    _assert_value_error(
        lambda: router(torch.randn(1, 2, 3, 7, device=device), queries),
        "hidden size 8",
    )
    _assert_value_error(
        lambda: router(features, torch.randn(2, 7, device=device)),
        "hidden size 8",
    )
    _assert_value_error(
        lambda: router(features, torch.empty(2, 8, device="meta")),
        "same device",
    )
    print("\nInput-validation tests:\nPASS")


def _run_amp_test(device: torch.device) -> None:
    """Validate FP32 softmax behavior under CUDA AMP when available."""
    if device.type != "cuda":
        print("\nAMP test:\nSKIPPED (CUDA is unavailable)")
        return

    router = _make_router(
        device,
        hidden_size=32,
        num_classes=NUM_CLASSES,
        num_layers=3,
        router_dim=16,
    )
    token_features = torch.randn(2, NUM_CLASSES, 3, 32, device=device, requires_grad=True)
    label_queries = torch.randn(NUM_CLASSES, 32, device=device, requires_grad=True)
    router.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = router(token_features, label_queries)
        amp_objective = outputs["fused_features"].float().pow(2).mean()
    assert torch.isfinite(outputs["depth_attention"]).all()
    assert torch.isfinite(outputs["fused_features"]).all()
    assert torch.allclose(
        outputs["depth_attention"].float().sum(dim=-1),
        torch.ones_like(outputs["depth_attention"].float().sum(dim=-1)),
        atol=1e-4,
    )
    amp_objective.backward()
    assert token_features.grad is not None and torch.isfinite(token_features.grad).all()
    assert label_queries.grad is not None and torch.isfinite(label_queries.grad).all()
    assert router.query_projection.weight.grad is not None
    assert router.depth_key_projection.weight.grad is not None
    print("\nAMP test:\nPASS")


def _run_save_load_test(device: torch.device) -> None:
    """Round-trip router state and compare deterministic evaluation outputs."""
    router = _make_router(
        device,
        hidden_size=16,
        num_classes=NUM_CLASSES,
        num_layers=3,
        router_dim=8,
    )
    token_features = torch.randn(2, NUM_CLASSES, 3, 16, device=device)
    label_queries = torch.randn(NUM_CLASSES, 16, device=device)
    original_outputs = router(token_features, label_queries)

    with tempfile.TemporaryDirectory(prefix="ldtf_depth_router_") as temporary_directory:
        state_path = Path(temporary_directory) / "depth_router_state.pt"
        torch.save(router.state_dict(), state_path)
        reloaded_router = _make_router(
            device,
            hidden_size=16,
            num_classes=NUM_CLASSES,
            num_layers=3,
            router_dim=8,
        )
        reloaded_router.load_state_dict(_load_state_dict(state_path, device))
        reloaded_outputs = reloaded_router(token_features, label_queries)
    assert torch.allclose(
        original_outputs["depth_attention"],
        reloaded_outputs["depth_attention"],
        atol=1e-6,
    )
    assert torch.allclose(
        original_outputs["fused_features"],
        reloaded_outputs["fused_features"],
        atol=1e-6,
    )
    print("\nSave/load test:\nPASS")


def _run_full_integration(device: torch.device) -> None:
    """Run Stage 4–7 end to end and verify every requested gradient path."""
    from transformers import AutoTokenizer

    from config import MAX_LENGTH, MODEL_NAME
    from models.bert_backbone import BertMultiLayerBackbone
    from models.label_query_bank import LabelQueryBank
    from models.token_router import LabelTokenRouter

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    backbone = BertMultiLayerBackbone(model_name=MODEL_NAME).to(device)
    query_bank = LabelQueryBank(
        num_classes=NUM_CLASSES,
        hidden_size=backbone.hidden_size,
    ).to(device)
    token_router = LabelTokenRouter(
        hidden_size=backbone.hidden_size,
        num_classes=NUM_CLASSES,
        router_dim=ROUTER_DIM,
    ).to(device)
    depth_router = LabelDepthRouter(
        hidden_size=backbone.hidden_size,
        num_classes=NUM_CLASSES,
        num_layers=backbone.num_hidden_layers,
        router_dim=ROUTER_DIM,
    ).to(device)
    backbone.eval()
    query_bank.train()
    token_router.train()
    depth_router.train()

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
    token_router.zero_grad(set_to_none=True)
    depth_router.zero_grad(set_to_none=True)
    backbone_outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    hidden_states = backbone_outputs["hidden_states"]
    label_queries = query_bank()
    token_outputs = token_router(hidden_states, label_queries, attention_mask)
    token_features = token_outputs["token_features"]
    depth_outputs = depth_router(token_features, label_queries)
    batch_size, _, sequence_length, hidden_size = hidden_states.shape
    _assert_output_contract(
        depth_outputs,
        batch_size,
        NUM_CLASSES,
        backbone.num_hidden_layers,
        hidden_size,
        device,
    )
    assert label_queries.shape == (NUM_CLASSES, hidden_size)
    assert token_features.shape == (
        batch_size,
        NUM_CLASSES,
        backbone.num_hidden_layers,
        hidden_size,
    )
    assert token_outputs["token_attention"].shape == (
        batch_size,
        NUM_CLASSES,
        backbone.num_hidden_layers,
        sequence_length,
    )
    print(f"\nIntegrated hidden states:\n{hidden_states.shape}")
    print(f"\nIntegrated token features:\n{token_features.shape}")
    print(f"\nIntegrated depth attention:\n{depth_outputs['depth_attention'].shape}")
    print(f"\nIntegrated fused features:\n{depth_outputs['fused_features'].shape}")

    integration_objective = depth_outputs["fused_features"].pow(2).mean()
    integration_objective.backward()
    bert_gradients = [
        parameter.grad
        for parameter in backbone.bert.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert bert_gradients and all(torch.isfinite(gradient).all() for gradient in bert_gradients)
    assert query_bank.queries.grad is not None
    assert torch.isfinite(query_bank.queries.grad).all()
    required_router_gradients = (
        token_router.query_projection.weight.grad,
        token_router.key_projection.weight.grad,
        depth_router.query_projection.weight.grad,
        depth_router.depth_key_projection.weight.grad,
    )
    assert all(gradient is not None for gradient in required_router_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in required_router_gradients)
    print("\nBackbone integration:\nPASS")
    print("\nToken Router integration:\nPASS")
    print("\nQuery Bank integration:\nPASS")
    print("\nFull gradient integration:\nPASS")


def _parse_args() -> argparse.Namespace:
    """Parse opt-in full integration because it requires a BERT checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-backbone",
        action="store_true",
        help="Load bert-base-uncased and run the full Stage-4–7 integration test.",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint-free unit tests and optional real backbone integration."""
    args = _parse_args()
    print("=" * 60)
    print("LABEL DEPTH ROUTER SMOKE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice:\n{device}")
    if device.type == "cuda":
        print(f"\nGPU:\n{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"\nHidden size:\n{HIDDEN_SIZE}")
    print(f"\nNumber of classes:\n{NUM_CLASSES}")
    print(f"\nNumber of layers:\n{NUM_LAYERS}")
    print(f"\nRouter dimension:\n{ROUTER_DIM}")

    parameter_router = _make_router(device)
    parameter_stats = parameter_router.count_parameters()
    assert parameter_stats == {
        "total": 2 * HIDDEN_SIZE * ROUTER_DIM,
        "trainable": 2 * HIDDEN_SIZE * ROUTER_DIM,
        "frozen": 0,
    }
    print(f"\nRouter parameters:\n{parameter_stats['total']}")

    _run_shape_weighted_sum_and_gradient_tests(device)
    _run_single_and_identical_layer_tests(device)
    _run_class_and_sample_conditioning_tests(device)
    _run_input_validation_tests(device)
    _run_amp_test(device)
    _run_save_load_test(device)

    if args.with_backbone:
        _run_full_integration(device)
    else:
        print(
            "\nBackbone integration:\n"
            "SKIPPED (run with --with-backbone after caching bert-base-uncased)"
        )
        print("\nToken Router integration:\nSKIPPED (requires --with-backbone)")
        print("\nQuery Bank integration:\nSKIPPED (requires --with-backbone)")
        print("\nFull gradient integration:\nSKIPPED (requires --with-backbone)")
    print("\nSTAGE 7 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
