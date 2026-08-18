"""Terminal smoke tests for the Stage-5 LDTF-BERT label query bank."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as functional

# Direct execution (python scripts/...) sets sys.path to scripts/, while module
# execution starts at the project root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.label_query_bank import LabelQueryBank  # noqa: E402


CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
NUM_CLASSES = len(CLASS_NAMES)
HIDDEN_SIZE = 768


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a query-bank state dictionary across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _run_backbone_compatibility_test(device: torch.device) -> None:
    """Check only the Stage-4/Stage-5 hidden-size interface contract."""
    # Keep the default Stage-5 unit smoke test checkpoint-free. This optional
    # import and model load are only needed for the real Stage-4 integration test.
    from config import MODEL_NAME
    from models.bert_backbone import BertMultiLayerBackbone

    backbone = BertMultiLayerBackbone(model_name=MODEL_NAME).to(device)
    query_bank = LabelQueryBank(
        num_classes=NUM_CLASSES,
        hidden_size=backbone.hidden_size,
    ).to(device)
    queries = query_bank()

    assert query_bank.hidden_size == backbone.hidden_size
    assert queries.shape == (NUM_CLASSES, backbone.hidden_size)
    assert queries.device == device
    print("\nBackbone compatibility test:\nPASS")


def _parse_args() -> argparse.Namespace:
    """Parse the opt-in pretrained-backbone integration test flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-backbone",
        action="store_true",
        help="Load bert-base-uncased and run the Stage-4 compatibility test.",
    )
    return parser.parse_args()


def main() -> None:
    """Run isolated query-bank tests and a lightweight backbone interface test."""
    args = _parse_args()
    print("=" * 60)
    print("LABEL QUERY BANK SMOKE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice:\n{device}")
    if device.type == "cuda":
        print(f"\nGPU:\n{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"\nPyTorch version:\n{torch.__version__}")

    query_bank = LabelQueryBank(
        num_classes=NUM_CLASSES,
        hidden_size=HIDDEN_SIZE,
        init_std=0.02,
        class_names=CLASS_NAMES,
    ).to(device)
    queries = query_bank()

    assert queries.ndim == 2
    assert queries.shape == (NUM_CLASSES, HIDDEN_SIZE)
    assert queries.device == device
    assert isinstance(query_bank.queries, torch.nn.Parameter)
    assert query_bank.queries.requires_grad
    assert len(list(query_bank.parameters())) == 1
    print(f"\nNumber of classes:\n{query_bank.num_classes}")
    print(f"\nHidden size:\n{query_bank.hidden_size}")
    print(f"\nQuery shape:\n{queries.shape}")
    print("\nClass names:")
    for class_id, class_name in enumerate(query_bank.class_names):
        print(f"{class_id}: {class_name}")

    stats = query_bank.count_parameters()
    assert stats == {"total": 4 * 768, "trainable": 4 * 768, "frozen": 0}
    print(f"\nTotal parameters:\n{stats['total']}")
    print(f"\nTrainable parameters:\n{stats['trainable']}")
    print(f"\nFrozen parameters:\n{stats['frozen']}")

    query_mean = queries.mean().item()
    query_std = queries.std().item()
    assert abs(query_mean) < 0.01
    assert 0.01 < query_std < 0.03
    print(f"\nInitialization mean:\n{query_mean:.6f}")
    print(f"\nInitialization std:\n{query_std:.6f}")

    assert not torch.allclose(queries[0], queries[1])
    assert not torch.allclose(queries[1], queries[2])
    assert not torch.allclose(queries[2], queries[3])
    similarities = functional.normalize(queries.detach(), dim=-1)
    similarities = similarities @ similarities.T
    print("\nPairwise cosine-similarity diagnostic:")
    print(similarities.detach().cpu())
    print("\nQuery uniqueness test:\nPASS")

    expanded_queries = query_bank.expand_for_batch(batch_size=8)
    assert expanded_queries.shape == (8, NUM_CLASSES, HIDDEN_SIZE)
    assert expanded_queries.device == device
    assert torch.allclose(expanded_queries[0], expanded_queries[7])
    print(f"\nBatch expansion shape:\n{expanded_queries.shape}")
    print("\nBatch expansion test:\nPASS")

    query_bank.zero_grad(set_to_none=True)
    direct_loss = query_bank().pow(2).mean()
    direct_loss.backward()
    assert query_bank.queries.grad is not None
    assert query_bank.queries.grad.shape == (NUM_CLASSES, HIDDEN_SIZE)
    assert torch.isfinite(query_bank.queries.grad).all()
    print("\nDirect gradient test:\nPASS")

    query_bank.zero_grad(set_to_none=True)
    expanded_loss = query_bank.expand_for_batch(batch_size=8).pow(2).mean()
    expanded_loss.backward()
    assert query_bank.queries.grad is not None
    assert torch.isfinite(query_bank.queries.grad).all()
    print("\nExpanded gradient test:\nPASS")

    optimizer = torch.optim.AdamW(query_bank.parameters(), lr=1e-2)
    before_update = query_bank.queries.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    update_loss = query_bank().pow(2).mean()
    update_loss.backward()
    optimizer.step()
    after_update = query_bank.queries.detach().clone()
    assert not torch.allclose(before_update, after_update)
    print("\nOptimizer update test:\nPASS")

    torch.manual_seed(42)
    bank_1 = LabelQueryBank(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE)
    torch.manual_seed(42)
    bank_2 = LabelQueryBank(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE)
    torch.manual_seed(123)
    bank_3 = LabelQueryBank(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE)
    assert torch.allclose(bank_1.queries, bank_2.queries)
    assert not torch.allclose(bank_1.queries, bank_3.queries)
    print("\nReproducibility test:\nPASS")

    query_bank.freeze()
    frozen_stats = query_bank.count_parameters()
    assert not query_bank.queries.requires_grad
    assert frozen_stats == {"total": 4 * 768, "trainable": 0, "frozen": 4 * 768}
    query_bank.unfreeze()
    assert query_bank.queries.requires_grad
    assert query_bank.count_parameters() == {
        "total": 4 * 768,
        "trainable": 4 * 768,
        "frozen": 0,
    }
    print("\nFreeze/unfreeze test:\nPASS")

    with tempfile.TemporaryDirectory(prefix="ldtf_query_bank_") as temporary_directory:
        state_path = Path(temporary_directory) / "query_bank_state.pt"
        torch.save(query_bank.state_dict(), state_path)
        reloaded_bank = LabelQueryBank(
            num_classes=NUM_CLASSES,
            hidden_size=HIDDEN_SIZE,
        ).to(device)
        reloaded_bank.load_state_dict(_load_state_dict(state_path, device))
        assert torch.allclose(query_bank().detach(), reloaded_bank().detach())
    print("\nSave/load test:\nPASS")

    if args.with_backbone:
        _run_backbone_compatibility_test(device)
    else:
        print(
            "\nBackbone compatibility test:\n"
            "SKIPPED (run with --with-backbone after caching bert-base-uncased)"
        )
    print("\nSTAGE 5 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
