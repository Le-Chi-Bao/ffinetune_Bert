"""Terminal smoke tests for the Stage-8 LDTF-BERT shared class scorer."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

# Direct execution (python scripts/...) sets sys.path to scripts/, while module
# execution starts at the project root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.class_scorer import SharedClassScorer  # noqa: E402


NUM_CLASSES = 4
HIDDEN_SIZE = 768
DROPOUT = 0.1
MAX_LENGTH = 64


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a state dictionary across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _make_scorer(
    device: torch.device,
    hidden_size: int = HIDDEN_SIZE,
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT,
    use_bias: bool = True,
) -> SharedClassScorer:
    """Build a shared class scorer on the requested device."""
    return SharedClassScorer(
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout,
        use_bias=use_bias,
    ).to(device)


def _run_shape_and_parameter_tests(device: torch.device) -> torch.Tensor:
    """Validate shapes, device, finiteness, and shared parameter geometry."""
    batch_size = 3
    scorer = _make_scorer(device)
    fused_features = torch.randn(
        batch_size,
        NUM_CLASSES,
        HIDDEN_SIZE,
        device=device,
        requires_grad=True,
    )

    scorer.eval()
    logits = scorer(fused_features)
    assert logits.shape == (batch_size, NUM_CLASSES)
    print(f"\nSynthetic fused features:\n{fused_features.shape}")
    print(f"\nLogits:\n{logits.shape}")
    print("\nShape test:\nPASS")

    batch_one = torch.randn(1, NUM_CLASSES, HIDDEN_SIZE, device=device)
    assert scorer(batch_one).shape == (1, NUM_CLASSES)
    print("\nBatch-size-one test:\nPASS")

    assert torch.isfinite(logits).all()
    print("\nFinite-value test:\nPASS")

    assert logits.device == device
    print("\nDevice test:\nPASS")

    assert scorer.scorer.weight.shape == (1, HIDDEN_SIZE)
    assert scorer.scorer.bias is not None
    assert scorer.scorer.bias.shape == (1,)
    print(f"\nScorer weight shape:\n{scorer.scorer.weight.shape}")
    print(f"\nScorer bias shape:\n{scorer.scorer.bias.shape}")

    stats = scorer.count_parameters()
    assert stats["total"] == 769
    assert stats["trainable"] == 769
    assert stats["frozen"] == 0
    print(f"\nTotal parameters:\n{stats['total']}")
    print("\nParameter-count test:\nPASS")

    no_bias = _make_scorer(device, use_bias=False)
    no_bias_stats = no_bias.count_parameters()
    assert no_bias_stats["total"] == 768
    assert no_bias.scorer.bias is None
    return fused_features


def _run_manual_equivalence_test(device: torch.device) -> None:
    """Compare module output against an explicit shared linear formula."""
    batch_size = 3
    scorer = _make_scorer(device, dropout=0.0).eval()
    fused_features = torch.randn(batch_size, NUM_CLASSES, HIDDEN_SIZE, device=device)
    logits = scorer(fused_features)
    manual_logits = torch.einsum(
        "bcd,od->bco",
        fused_features,
        scorer.scorer.weight,
    ).squeeze(-1)
    if scorer.scorer.bias is not None:
        manual_logits = manual_logits + scorer.scorer.bias
    assert torch.allclose(logits, manual_logits, atol=1e-6)
    print("\nManual linear equivalence test:\nPASS")


def _run_equivariance_and_identical_tests(device: torch.device) -> None:
    """Shared weights must permute with classes and score identical vectors equally."""
    batch_size = 3
    scorer = _make_scorer(device, dropout=0.0).eval()
    fused_features = torch.randn(batch_size, NUM_CLASSES, HIDDEN_SIZE, device=device)

    original_logits = scorer(fused_features)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    permuted_features = fused_features[:, permutation, :]
    permuted_logits = scorer(permuted_features)
    assert torch.allclose(
        permuted_logits,
        original_logits[:, permutation],
        atol=1e-6,
    )
    print("\nClass permutation equivariance test:\nPASS")

    base = torch.randn(batch_size, 1, HIDDEN_SIZE, device=device)
    identical_features = base.expand(batch_size, NUM_CLASSES, HIDDEN_SIZE).clone()
    identical_logits = scorer(identical_features)
    reference = identical_logits[:, :1]
    assert torch.allclose(
        identical_logits,
        reference.expand_as(identical_logits),
        atol=1e-6,
    )
    print("\nIdentical representation test:\nPASS")


def _run_dropout_determinism_test(device: torch.device) -> None:
    """Eval mode must disable dropout stochasticity."""
    batch_size = 3
    scorer = _make_scorer(device, dropout=0.5).eval()
    fused_features = torch.randn(batch_size, NUM_CLASSES, HIDDEN_SIZE, device=device)
    logits_1 = scorer(fused_features)
    logits_2 = scorer(fused_features)
    assert torch.allclose(logits_1, logits_2, atol=1e-6)
    print("\nDropout eval determinism test:\nPASS")


def _run_loss_gradient_optimizer_tests(device: torch.device) -> None:
    """CrossEntropyLoss, gradients, and one optimizer step must all work."""
    batch_size = 3
    scorer = _make_scorer(device)
    fused_features = torch.randn(
        batch_size,
        NUM_CLASSES,
        HIDDEN_SIZE,
        device=device,
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    scorer.eval()
    logits = scorer(fused_features)
    loss = F.cross_entropy(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    print("\nCrossEntropyLoss compatibility test:\nPASS")

    scorer.train()
    fused_features = torch.randn(
        batch_size,
        NUM_CLASSES,
        HIDDEN_SIZE,
        device=device,
        requires_grad=True,
    )
    scorer.zero_grad(set_to_none=True)
    logits = scorer(fused_features)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert fused_features.grad is not None
    assert scorer.scorer.weight.grad is not None
    assert scorer.scorer.bias is not None
    assert scorer.scorer.bias.grad is not None
    assert torch.isfinite(fused_features.grad).all()
    assert torch.isfinite(scorer.scorer.weight.grad).all()
    assert torch.isfinite(scorer.scorer.bias.grad).all()
    print("\nGradient-flow test:\nPASS")

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-2)
    before = scorer.scorer.weight.detach().clone()
    scorer.zero_grad(set_to_none=True)
    logits = scorer(fused_features.detach().requires_grad_(True))
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    after = scorer.scorer.weight.detach().clone()
    assert not torch.allclose(before, after)
    print("\nOptimizer update test:\nPASS")


def _run_input_validation_tests(device: torch.device) -> None:
    """Invalid constructor settings and tensor contracts must raise ValueError."""
    try:
        SharedClassScorer(hidden_size=0)
        raise AssertionError("Expected ValueError for hidden_size=0.")
    except ValueError:
        pass
    try:
        SharedClassScorer(num_classes=1)
        raise AssertionError("Expected ValueError for num_classes=1.")
    except ValueError:
        pass
    try:
        SharedClassScorer(dropout=1.0)
        raise AssertionError("Expected ValueError for dropout=1.0.")
    except ValueError as error:
        assert "dropout must be in the range [0, 1)" in str(error)

    scorer = _make_scorer(device).eval()
    invalid_inputs = (
        torch.randn(3, HIDDEN_SIZE, device=device),
        torch.randn(3, NUM_CLASSES, 2, HIDDEN_SIZE, device=device),
        torch.randn(3, 3, HIDDEN_SIZE, device=device),
        torch.randn(3, NUM_CLASSES, 512, device=device),
        torch.randn(0, NUM_CLASSES, HIDDEN_SIZE, device=device),
    )
    for invalid in invalid_inputs:
        try:
            scorer(invalid)
            raise AssertionError(
                f"Expected ValueError for input shape {tuple(invalid.shape)}."
            )
        except ValueError:
            pass
    print("\nInput-validation tests:\nPASS")


def _run_amp_test(device: torch.device) -> None:
    """Optional CUDA autocast path must stay finite and differentiable."""
    if device.type != "cuda":
        print("\nAMP test:\nPASS (skipped on CPU)")
        return

    batch_size = 3
    scorer = _make_scorer(device).train()
    fused_features = torch.randn(
        batch_size,
        NUM_CLASSES,
        HIDDEN_SIZE,
        device=device,
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    scorer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = scorer(fused_features)
        loss = F.cross_entropy(logits, labels)
    assert torch.isfinite(logits.float()).all()
    assert torch.isfinite(loss.float())
    loss.backward()
    assert fused_features.grad is not None
    assert scorer.scorer.weight.grad is not None
    print("\nAMP test:\nPASS")


def _run_save_load_test(device: torch.device) -> None:
    """State-dict round-trip must reproduce eval logits."""
    batch_size = 3
    scorer = _make_scorer(device).eval()
    reloaded = _make_scorer(device).eval()
    fused_features = torch.randn(batch_size, NUM_CLASSES, HIDDEN_SIZE, device=device)

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "shared_class_scorer.pt"
        torch.save(scorer.state_dict(), path)
        reloaded.load_state_dict(_load_state_dict(path, device))

    assert torch.allclose(scorer(fused_features), reloaded(fused_features), atol=1e-6)
    print("\nSave/load test:\nPASS")


def _run_full_integration(device: torch.device) -> None:
    """End-to-end graph: backbone → routers → shared scorer → CE loss."""
    from models.bert_backbone import BertMultiLayerBackbone
    from models.depth_router import LabelDepthRouter
    from models.label_query_bank import LabelQueryBank
    from models.token_router import LabelTokenRouter
    from transformers import AutoTokenizer

    backbone = BertMultiLayerBackbone(model_name="bert-base-uncased").to(device)
    query_bank = LabelQueryBank(
        num_classes=NUM_CLASSES,
        hidden_size=backbone.hidden_size,
    ).to(device)
    token_router = LabelTokenRouter(
        hidden_size=backbone.hidden_size,
        num_classes=NUM_CLASSES,
        router_dim=256,
    ).to(device)
    depth_router = LabelDepthRouter(
        hidden_size=backbone.hidden_size,
        num_classes=NUM_CLASSES,
        num_layers=backbone.num_hidden_layers,
        router_dim=256,
    ).to(device)
    class_scorer = SharedClassScorer(
        hidden_size=backbone.hidden_size,
        num_classes=NUM_CLASSES,
        dropout=0.1,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = [
        "United Nations leaders discuss international security.",
        "Manchester United wins the championship match.",
        "Stocks rise after strong quarterly earnings.",
        "Google introduces a new artificial intelligence system.",
    ]
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
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
    class_scorer.zero_grad(set_to_none=True)

    backbone_outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    hidden_states = backbone_outputs["hidden_states"]
    label_queries = query_bank()
    token_outputs = token_router(
        hidden_states=hidden_states,
        label_queries=label_queries,
        attention_mask=attention_mask,
    )
    token_features = token_outputs["token_features"]
    depth_outputs = depth_router(
        token_features=token_features,
        label_queries=label_queries,
    )
    fused_features = depth_outputs["fused_features"]
    logits = class_scorer(fused_features=fused_features)

    batch_size, num_layers, _sequence_length, hidden_size = hidden_states.shape
    assert hidden_states.shape[0] == batch_size
    assert num_layers == backbone.num_hidden_layers
    assert label_queries.shape == (NUM_CLASSES, hidden_size)
    assert token_features.shape == (
        batch_size,
        NUM_CLASSES,
        backbone.num_hidden_layers,
        hidden_size,
    )
    assert fused_features.shape == (batch_size, NUM_CLASSES, hidden_size)
    assert logits.shape == (batch_size, NUM_CLASSES)
    assert torch.isfinite(logits).all()

    print(f"\nIntegrated hidden states:\n{hidden_states.shape}")
    print(f"\nIntegrated label queries:\n{label_queries.shape}")
    print(f"\nIntegrated token features:\n{token_features.shape}")
    print(f"\nIntegrated fused features:\n{fused_features.shape}")
    print(f"\nIntegrated logits:\n{logits.shape}")

    loss = F.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()

    bert_gradients = [
        parameter.grad
        for parameter in backbone.bert.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert bert_gradients and all(
        torch.isfinite(gradient).all() for gradient in bert_gradients
    )
    assert query_bank.queries.grad is not None
    assert torch.isfinite(query_bank.queries.grad).all()
    required_gradients = (
        token_router.query_projection.weight.grad,
        token_router.key_projection.weight.grad,
        depth_router.query_projection.weight.grad,
        depth_router.depth_key_projection.weight.grad,
        class_scorer.scorer.weight.grad,
        class_scorer.scorer.bias.grad,
    )
    assert all(gradient is not None for gradient in required_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in required_gradients)

    print("\nBackbone integration:\nPASS")
    print("\nQuery Bank integration:\nPASS")
    print("\nToken Router integration:\nPASS")
    print("\nDepth Router integration:\nPASS")
    print("\nFull end-to-end gradient test:\nPASS")


def _parse_args() -> argparse.Namespace:
    """Parse opt-in full integration because it requires a BERT checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-backbone",
        action="store_true",
        help="Load bert-base-uncased and run the full Stage-4–8 integration test.",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint-free unit tests and optional real backbone integration."""
    args = _parse_args()
    print("=" * 60)
    print("SHARED CLASS SCORER SMOKE TEST")
    print("=" * 60)

    print(f"\nPyTorch version:\n{torch.__version__}")
    print(f"\nCUDA availability:\n{torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice:\n{device}")
    if device.type == "cuda":
        print(f"\nGPU:\n{torch.cuda.get_device_name(torch.cuda.current_device())}")

    print(f"\nHidden size:\n{HIDDEN_SIZE}")
    print(f"\nNumber of classes:\n{NUM_CLASSES}")
    print(f"\nDropout:\n{DROPOUT}")
    print("\nUse bias:\nTrue")

    _run_shape_and_parameter_tests(device)
    _run_manual_equivalence_test(device)
    _run_equivariance_and_identical_tests(device)
    _run_dropout_determinism_test(device)
    _run_loss_gradient_optimizer_tests(device)
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
        print("\nQuery Bank integration:\nSKIPPED (requires --with-backbone)")
        print("\nToken Router integration:\nSKIPPED (requires --with-backbone)")
        print("\nDepth Router integration:\nSKIPPED (requires --with-backbone)")
        print("\nFull end-to-end gradient test:\nSKIPPED (requires --with-backbone)")

    print("\nSTAGE 8 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
