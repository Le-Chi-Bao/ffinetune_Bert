"""Kaggle-ready smoke tests for the Stage 2 BERT baseline.

Run this file only to validate the Stage 1 data contract and Stage 2 model.
The dummy scalar objectives below exist solely to test gradient connectivity;
this file contains no optimizer, scheduler, training loop, or evaluation loop.
"""

from __future__ import annotations

import torch

from data import prepare_data
from models.bert_baseline import BertBaselineClassifier


def _assert_finite_gradients(parameters: list[torch.nn.Parameter], name: str) -> None:
    """Verify that collected gradients exist and are finite."""
    if not parameters:
        raise AssertionError(f"{name} should have at least one gradient.")
    if not all(parameter.grad is not None for parameter in parameters):
        raise AssertionError(f"{name} has an unexpected missing gradient.")
    if not all(torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise AssertionError(f"{name} has a non-finite gradient.")


def run_smoke_tests() -> None:
    """Run every required Stage 2 smoke test using the real Stage 1 loader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = prepare_data()
    batch = next(iter(data["train_loader"]))

    model = BertBaselineClassifier(
        model_name="bert-base-uncased",
        num_classes=4,
        dropout=0.1,
    ).to(device)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # DataLoader-forward smoke test.
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    assert input_ids.ndim == 2
    assert attention_mask.ndim == 2
    assert labels.ndim == 1
    assert input_ids.shape == attention_mask.shape
    assert labels.shape[0] == input_ids.shape[0]
    assert logits.ndim == 2
    assert logits.shape == (input_ids.shape[0], 4)
    assert logits.device == device
    assert torch.isfinite(logits).all()

    print("\nDataLoader smoke test")
    print("Input IDs shape:", input_ids.shape)
    print("Attention mask shape:", attention_mask.shape)
    print("Labels shape:", labels.shape)
    print("Logits shape:", logits.shape)
    print("BERT baseline smoke test PASSED")

    # Independent tokenizer smoke test using topic-specific sample sentences.
    texts = [
        "United Nations leaders meet to discuss international security.",
        "Manchester United wins the championship match.",
        "Stocks rise after the company reports strong quarterly earnings.",
        "Google announces a new artificial intelligence research system.",
    ]
    encoded = data["tokenizer"](
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        sample_logits = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    assert sample_logits.shape == (4, 4)
    assert torch.isfinite(sample_logits).all()
    print("\nFour-sentence logits shape:", sample_logits.shape)
    print("Four-sentence test PASSED")
    print("Predictions are not meaningful before classifier-head training.")

    # Parameter-count and classifier-shape checks.
    initial_stats = model.count_parameters()
    assert initial_stats["total"] > 0
    assert initial_stats["trainable"] > 0
    assert initial_stats["frozen"] >= 0
    assert initial_stats["total"] == initial_stats["trainable"] + initial_stats["frozen"]
    assert model.classifier.in_features == model.hidden_size
    assert model.classifier.out_features == 4
    assert model.hidden_size == 768
    print("\nInitial parameter stats:", initial_stats)
    print("Parameter-count test PASSED")

    # Freeze/unfreeze checks.
    model.freeze_encoder()
    frozen_stats = model.count_parameters()
    assert model.is_encoder_frozen()
    assert all(not parameter.requires_grad for parameter in model.bert.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert frozen_stats["trainable"] < initial_stats["trainable"]
    print("\nAfter encoder freeze:", frozen_stats)
    print("Freeze test PASSED")

    model.unfreeze_encoder()
    assert not model.is_encoder_frozen()
    assert all(parameter.requires_grad for parameter in model.bert.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    print("Unfreeze test PASSED")

    # Fine-tuning gradient connectivity. This is not a training step.
    model.train()
    model.unfreeze_encoder()
    model.zero_grad(set_to_none=True)
    fine_tuning_logits = model(input_ids=input_ids, attention_mask=attention_mask)
    fine_tuning_dummy_objective = fine_tuning_logits.pow(2).mean()
    fine_tuning_dummy_objective.backward()

    bert_grad_parameters = [
        parameter
        for parameter in model.bert.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    classifier_grad_parameters = [
        parameter for parameter in model.classifier.parameters() if parameter.grad is not None
    ]
    _assert_finite_gradients(bert_grad_parameters, "Fine-tuned BERT")
    _assert_finite_gradients(classifier_grad_parameters, "Classifier")
    model.zero_grad(set_to_none=True)
    print("Fine-tuning gradient-flow test PASSED")

    # Frozen encoder gradient connectivity. The classifier must still receive gradients.
    model.train()
    model.freeze_encoder()
    model.zero_grad(set_to_none=True)
    frozen_logits = model(input_ids=input_ids, attention_mask=attention_mask)
    frozen_dummy_objective = frozen_logits.pow(2).mean()
    frozen_dummy_objective.backward()

    bert_has_grad = any(parameter.grad is not None for parameter in model.bert.parameters())
    classifier_grad_parameters = [
        parameter for parameter in model.classifier.parameters() if parameter.grad is not None
    ]
    assert not bert_has_grad
    _assert_finite_gradients(classifier_grad_parameters, "Frozen-mode classifier")
    model.zero_grad(set_to_none=True)
    model.unfreeze_encoder()
    print("Frozen gradient-flow test PASSED")

    # Dropout state sanity check: deterministic in evaluation mode.
    model.eval()
    with torch.no_grad():
        logits_1 = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_2 = model(input_ids=input_ids, attention_mask=attention_mask)
    assert torch.allclose(logits_1, logits_2, atol=1e-6)
    print("Eval-mode dropout sanity test PASSED")

    print("\nStage 2 BERT baseline: ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    run_smoke_tests()
