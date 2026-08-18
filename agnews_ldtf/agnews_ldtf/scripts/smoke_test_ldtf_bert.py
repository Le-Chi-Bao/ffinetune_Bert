"""Terminal smoke tests for full Stage-9 LDTF-BERT integration."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Direct execution (python scripts/...) sets sys.path to scripts/, while module
# execution starts at the project root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.ldtf_bert import LDTFBertClassifier  # noqa: E402


NUM_CLASSES = 4
HIDDEN_SIZE = 768
NUM_LAYERS = 12
TOKEN_ROUTER_DIM = 256
DEPTH_ROUTER_DIM = 256
CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
NON_BACKBONE_PARAMETERS = 790_273
MAX_LENGTH = 64

TITLES = [
    "World leaders discuss security",
    "Team wins championship match",
    "Stocks rise after earnings",
    "Company releases new AI system",
]
DESCRIPTIONS = [
    "United Nations officials met to discuss international security.",
    "The football team won the final after a dramatic match.",
    "Shares increased after the company reported strong quarterly revenue.",
    "The technology company introduced a new artificial intelligence platform.",
]


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a state dictionary across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _make_model(device: torch.device) -> LDTFBertClassifier:
    """Build the default AG News LDTF-BERT configuration."""
    return LDTFBertClassifier(
        model_name="bert-base-uncased",
        num_classes=NUM_CLASSES,
        token_router_dim=TOKEN_ROUTER_DIM,
        depth_router_dim=DEPTH_ROUTER_DIM,
        classifier_dropout=0.1,
        projection_bias=False,
        scorer_bias=True,
        class_names=CLASS_NAMES,
    ).to(device)


def _tokenize(
    tokenizer: AutoTokenizer,
    titles: list[str],
    descriptions: list[str],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize title/description pairs and move tensors to device."""
    encoded = tokenizer(
        titles,
        descriptions,
        truncation="only_second",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    batch: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _module_requires_grad(module: torch.nn.Module) -> bool:
    """Return True when every parameter in the module is trainable."""
    parameters = list(module.parameters())
    return bool(parameters) and all(parameter.requires_grad for parameter in parameters)


def _module_is_frozen(module: torch.nn.Module) -> bool:
    """Return True when every parameter in the module is frozen."""
    parameters = list(module.parameters())
    return bool(parameters) and all(not parameter.requires_grad for parameter in parameters)


def main() -> None:
    """Run construction, forward, gradient, freeze, AMP, and save/load checks."""
    print("=" * 60)
    print("FULL LDTF-BERT INTEGRATION SMOKE TEST")
    print("=" * 60)

    print(f"\nPyTorch version:\n{torch.__version__}")
    print(f"\nCUDA availability:\n{torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice:\n{device}")
    if device.type == "cuda":
        print(f"\nGPU:\n{torch.cuda.get_device_name(torch.cuda.current_device())}")
        torch.cuda.reset_peak_memory_stats(device)

    model = _make_model(device)
    assert model.hidden_size == HIDDEN_SIZE
    assert model.num_hidden_layers == NUM_LAYERS
    assert model.num_classes == NUM_CLASSES
    print(f"\nModel:\n{model.model_name}")
    print(f"\nNumber of classes:\n{model.num_classes}")
    print(f"\nHidden size:\n{model.hidden_size}")
    print(f"\nTransformer layers:\n{model.num_hidden_layers}")
    print(f"\nToken router dimension:\n{model.token_router_dim}")
    print(f"\nDepth router dimension:\n{model.depth_router_dim}")
    print("\nModel construction:\nPASS")

    assert model.label_query_bank.hidden_size == model.hidden_size
    assert model.token_router.hidden_size == model.hidden_size
    assert model.depth_router.hidden_size == model.hidden_size
    assert model.class_scorer.hidden_size == model.hidden_size
    assert model.label_query_bank.num_classes == NUM_CLASSES
    assert model.token_router.num_classes == NUM_CLASSES
    assert model.depth_router.num_classes == NUM_CLASSES
    assert model.class_scorer.num_classes == NUM_CLASSES
    assert model.depth_router.num_layers == model.num_hidden_layers
    print("\nModule compatibility:\nPASS")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    batch = _tokenize(tokenizer, TITLES, DESCRIPTIONS, MAX_LENGTH, device)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    batch_size, sequence_length = batch["input_ids"].shape
    print(f"\nInput IDs:\n{batch['input_ids'].shape}")

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            special_tokens_mask=batch.get("special_tokens_mask"),
            return_routing=True,
            return_features=True,
        )

    assert outputs["logits"].shape == (batch_size, NUM_CLASSES)
    assert outputs["hidden_states"].shape == (
        batch_size,
        NUM_LAYERS,
        sequence_length,
        HIDDEN_SIZE,
    )
    assert outputs["label_queries"].shape == (NUM_CLASSES, HIDDEN_SIZE)
    assert outputs["token_attention"].shape == (
        batch_size,
        NUM_CLASSES,
        NUM_LAYERS,
        sequence_length,
    )
    assert outputs["token_features"].shape == (
        batch_size,
        NUM_CLASSES,
        NUM_LAYERS,
        HIDDEN_SIZE,
    )
    assert outputs["depth_attention"].shape == (batch_size, NUM_CLASSES, NUM_LAYERS)
    assert outputs["fused_features"].shape == (batch_size, NUM_CLASSES, HIDDEN_SIZE)

    print(f"\nHidden states:\n{outputs['hidden_states'].shape}")
    print(f"\nLabel queries:\n{outputs['label_queries'].shape}")
    print(f"\nToken attention:\n{outputs['token_attention'].shape}")
    print(f"\nToken features:\n{outputs['token_features'].shape}")
    print(f"\nDepth attention:\n{outputs['depth_attention'].shape}")
    print(f"\nFused features:\n{outputs['fused_features'].shape}")
    print(f"\nLogits:\n{outputs['logits'].shape}")
    print("\nForward shape test:\nPASS")

    token_attention_sum = outputs["token_attention"].float().sum(dim=-1)
    assert token_attention_sum.shape == (batch_size, NUM_CLASSES, NUM_LAYERS)
    assert torch.allclose(
        token_attention_sum,
        torch.ones_like(token_attention_sum),
        atol=1e-5,
    )
    print("\nToken attention normalization:\nPASS")

    depth_attention_sum = outputs["depth_attention"].float().sum(dim=-1)
    assert depth_attention_sum.shape == (batch_size, NUM_CLASSES)
    assert torch.allclose(
        depth_attention_sum,
        torch.ones_like(depth_attention_sum),
        atol=1e-5,
    )
    print("\nDepth attention normalization:\nPASS")

    padding_mask = ~batch["attention_mask"].bool()
    if padding_mask.any():
        padding_attention = outputs["token_attention"][:, :, :, padding_mask.any(dim=0)]
        # Per-position: only assert padding positions for samples that pad there.
        for token_index in range(sequence_length):
            sample_pad = padding_mask[:, token_index]
            if not sample_pad.any():
                continue
            values = outputs["token_attention"][sample_pad, :, :, token_index]
            assert torch.allclose(
                values.float(),
                torch.zeros_like(values.float()),
                atol=1e-6,
            )
    special_tokens_mask = batch.get("special_tokens_mask")
    if special_tokens_mask is not None:
        special_bool = special_tokens_mask.bool()
        if special_bool.any():
            for token_index in range(sequence_length):
                sample_special = special_bool[:, token_index]
                if not sample_special.any():
                    continue
                values = outputs["token_attention"][sample_special, :, :, token_index]
                assert torch.allclose(
                    values.float(),
                    torch.zeros_like(values.float()),
                    atol=1e-6,
                )
    print("\nPadding mask test:\nPASS")

    with torch.no_grad():
        minimal_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            special_tokens_mask=batch.get("special_tokens_mask"),
            return_routing=False,
            return_features=False,
        )
    assert set(minimal_outputs.keys()) == {"logits"}
    assert "token_attention" not in minimal_outputs
    assert "depth_attention" not in minimal_outputs
    assert "hidden_states" not in minimal_outputs
    assert "token_features" not in minimal_outputs
    assert "fused_features" not in minimal_outputs
    assert "label_queries" not in minimal_outputs
    assert torch.allclose(minimal_outputs["logits"], outputs["logits"], atol=1e-6)
    print("\nReturn flags test:\nPASS")

    single_batch = _tokenize(
        tokenizer,
        [TITLES[0]],
        [DESCRIPTIONS[0]],
        MAX_LENGTH,
        device,
    )
    with torch.no_grad():
        single_outputs = model(
            input_ids=single_batch["input_ids"],
            attention_mask=single_batch["attention_mask"],
            token_type_ids=single_batch.get("token_type_ids"),
            special_tokens_mask=single_batch.get("special_tokens_mask"),
            return_routing=True,
        )
    single_t = single_batch["input_ids"].shape[1]
    assert single_outputs["logits"].shape == (1, NUM_CLASSES)
    assert single_outputs["token_attention"].shape == (1, NUM_CLASSES, NUM_LAYERS, single_t)
    assert single_outputs["depth_attention"].shape == (1, NUM_CLASSES, NUM_LAYERS)
    print("\nBatch-size-one test:\nPASS")

    short_batch = _tokenize(
        tokenizer,
        TITLES[:2],
        DESCRIPTIONS[:2],
        max_length=16,
        device=device,
    )
    long_batch = _tokenize(
        tokenizer,
        TITLES[:2],
        DESCRIPTIONS[:2],
        max_length=48,
        device=device,
    )
    with torch.no_grad():
        short_outputs = model(
            input_ids=short_batch["input_ids"],
            attention_mask=short_batch["attention_mask"],
            token_type_ids=short_batch.get("token_type_ids"),
            special_tokens_mask=short_batch.get("special_tokens_mask"),
            return_routing=True,
        )
        long_outputs = model(
            input_ids=long_batch["input_ids"],
            attention_mask=long_batch["attention_mask"],
            token_type_ids=long_batch.get("token_type_ids"),
            special_tokens_mask=long_batch.get("special_tokens_mask"),
            return_routing=True,
        )
    assert short_outputs["logits"].shape == (2, NUM_CLASSES)
    assert long_outputs["logits"].shape == (2, NUM_CLASSES)
    assert short_outputs["token_attention"].shape[-1] == short_batch["input_ids"].shape[1]
    assert long_outputs["token_attention"].shape[-1] == long_batch["input_ids"].shape[1]
    assert short_batch["input_ids"].shape[1] != long_batch["input_ids"].shape[1]
    print("\nVariable sequence-length test:\nPASS")

    for key, value in outputs.items():
        assert torch.isfinite(value).all(), key
    print("\nFinite-value test:\nPASS")

    model.train()
    model.zero_grad(set_to_none=True)
    train_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        special_tokens_mask=batch.get("special_tokens_mask"),
    )
    loss = F.cross_entropy(train_outputs["logits"], labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    print("\nCrossEntropyLoss compatibility:\nPASS")

    loss.backward()
    backbone_grad = next(
        (
            parameter.grad
            for parameter in model.backbone.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ),
        None,
    )
    assert backbone_grad is not None
    assert torch.isfinite(backbone_grad).all()
    assert not torch.allclose(backbone_grad, torch.zeros_like(backbone_grad))
    assert model.label_query_bank.queries.grad is not None
    assert torch.isfinite(model.label_query_bank.queries.grad).all()
    assert not torch.allclose(
        model.label_query_bank.queries.grad,
        torch.zeros_like(model.label_query_bank.queries.grad),
    )
    required_gradients = (
        model.token_router.query_projection.weight.grad,
        model.token_router.key_projection.weight.grad,
        model.depth_router.query_projection.weight.grad,
        model.depth_router.depth_key_projection.weight.grad,
        model.class_scorer.scorer.weight.grad,
        model.class_scorer.scorer.bias.grad,
    )
    assert all(gradient is not None for gradient in required_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in required_gradients)
    print("\nEnd-to-end gradient:\nPASS")

    model.freeze_backbone()
    assert _module_is_frozen(model.backbone)
    assert _module_requires_grad(model.label_query_bank)
    assert _module_requires_grad(model.token_router)
    assert _module_requires_grad(model.depth_router)
    assert _module_requires_grad(model.class_scorer)
    model.zero_grad(set_to_none=True)
    frozen_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        special_tokens_mask=batch.get("special_tokens_mask"),
    )
    frozen_loss = F.cross_entropy(frozen_outputs["logits"], labels)
    frozen_loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.label_query_bank.queries.grad is not None
    assert model.token_router.query_projection.weight.grad is not None
    assert model.depth_router.query_projection.weight.grad is not None
    assert model.class_scorer.scorer.weight.grad is not None
    print("\nFrozen-backbone gradient:\nPASS")

    model.unfreeze_backbone()
    assert _module_requires_grad(model.backbone)
    model.zero_grad(set_to_none=True)
    unfrozen_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        special_tokens_mask=batch.get("special_tokens_mask"),
    )
    unfrozen_loss = F.cross_entropy(unfrozen_outputs["logits"], labels)
    unfrozen_loss.backward()
    unfrozen_backbone_grad = next(
        (
            parameter.grad
            for parameter in model.backbone.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ),
        None,
    )
    assert unfrozen_backbone_grad is not None
    assert torch.isfinite(unfrozen_backbone_grad).all()
    print("\nUnfreeze-backbone gradient:\nPASS")

    stats = model.count_parameters()
    assert stats["total"] == stats["trainable"] + stats["frozen"]
    by_module = stats["by_module"]
    assert isinstance(by_module, dict)
    module_total = sum(module_stats["total"] for module_stats in by_module.values())
    assert module_total == stats["total"]
    non_backbone = (
        by_module["label_query_bank"]["total"]
        + by_module["token_router"]["total"]
        + by_module["depth_router"]["total"]
        + by_module["class_scorer"]["total"]
    )
    assert non_backbone == NON_BACKBONE_PARAMETERS
    print(f"\nTotal parameters:\n{stats['total']}")
    print(f"\nNon-backbone parameters:\n{non_backbone}")
    print("\nParameter-count test:\nPASS")

    assert not hasattr(model, "token_query_bank")
    assert not hasattr(model, "depth_query_bank")
    state_keys = list(model.state_dict().keys())
    assert any(key == "label_query_bank.queries" for key in state_keys)
    assert sum(1 for key in state_keys if key.endswith("queries")) == 1
    print("\nShared Query Bank test:\nPASS")

    model.eval()
    with torch.no_grad():
        logits_1 = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            special_tokens_mask=batch.get("special_tokens_mask"),
        )["logits"]
        logits_2 = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            special_tokens_mask=batch.get("special_tokens_mask"),
        )["logits"]
    assert torch.allclose(logits_1, logits_2, atol=1e-6)
    print("\nEval determinism:\nPASS")

    model.train()
    train_logits_1 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        special_tokens_mask=batch.get("special_tokens_mask"),
    )["logits"]
    train_logits_2 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        special_tokens_mask=batch.get("special_tokens_mask"),
    )["logits"]
    train_diff = (train_logits_1 - train_logits_2).abs().mean().item()
    print(f"\nTrain-mode dropout mean abs diff:\n{train_diff}")

    if device.type == "cuda":
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                special_tokens_mask=batch.get("special_tokens_mask"),
            )
            amp_loss = F.cross_entropy(amp_outputs["logits"], labels)
        assert torch.isfinite(amp_outputs["logits"].float()).all()
        assert torch.isfinite(amp_loss.float())
        amp_loss.backward()
        assert model.class_scorer.scorer.weight.grad is not None
        assert torch.isfinite(model.class_scorer.scorer.weight.grad).all()
        print("\nAMP test:\nPASS")
        print(
            f"\nPeak CUDA memory (bytes):\n{torch.cuda.max_memory_allocated(device)}"
        )
    else:
        print("\nAMP test:\nPASS (skipped on CPU)")

    model.eval()
    reloaded = _make_model(device).eval()
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "ldtf_bert.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
        }
        torch.save(checkpoint, path)
        loaded = _load_state_dict(path, device)
        reloaded.load_state_dict(loaded["model_state_dict"])
        assert loaded["model_config"]["num_classes"] == NUM_CLASSES

    with torch.no_grad():
        original_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            special_tokens_mask=batch.get("special_tokens_mask"),
        )["logits"]
        reloaded_logits = reloaded(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            special_tokens_mask=batch.get("special_tokens_mask"),
        )["logits"]
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-6)
    print("\nSave/load test:\nPASS")

    prefixes = (
        "backbone.",
        "label_query_bank.",
        "token_router.",
        "depth_router.",
        "class_scorer.",
    )
    for prefix in prefixes:
        assert any(key.startswith(prefix) for key in model.state_dict())
    print("\nState-dict completeness:\nPASS")

    model.unfreeze_backbone()
    print("\nSTAGE 9 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
