"""Tiny in-memory forward/backward tests for all Stage-11 ablation variants.

Uses a 2-layer, 32-hidden randomly initialized BERT config -- no pretrained
download, no dataset, no checkpoint on disk. Verifies every variant is
constructible, differentiable, and honours its training regime.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_factory import (  # noqa: E402
    build_optimizer,
    count_model_parameters,
    describe_optimizer_groups,
    extract_logits,
    validate_optimizer_groups,
    verify_training_regime,
)
from models.ldtf_ablation import ABLATION_VARIANTS, LDTFAblationClassifier  # noqa: E402

CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
BATCH, LENGTH = 2, 12


def build_variant(variant: str, **overrides):
    """Construct an ablation model, letting genuine configuration errors surface."""
    kwargs = {
        "num_classes": 4, "variant": variant, "token_router_dim": 16,
        "depth_router_dim": 16, "class_names": CLASS_NAMES,
    }
    kwargs.update(overrides)
    return LDTFAblationClassifier(model_name="bert-base-uncased", **kwargs)


class TestAblationVariantsForwardBackward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_ids = torch.randint(0, 1000, (BATCH, LENGTH))
        cls.attention_mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
        cls.labels = torch.randint(0, 4, (BATCH,))

    def _forward_backward(self, model: nn.Module) -> torch.Tensor:
        outputs = model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        logits = extract_logits(outputs, 4)
        self.assertEqual(tuple(logits.shape), (BATCH, 4))
        self.assertTrue(torch.isfinite(logits).all())
        loss = nn.CrossEntropyLoss()(logits, self.labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        return loss

    def test_every_variant_forwards_and_backwards(self):
        for variant in sorted(ABLATION_VARIANTS):
            with self.subTest(variant=variant):
                model = build_variant(variant)
                self._forward_backward(model)
                grads = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(grads, f"{variant} produced no gradients")

    def test_router_dimensions_are_configurable(self):
        for dim in (16, 32):
            with self.subTest(dim=dim):
                model = build_variant("A0_full", token_router_dim=dim, depth_router_dim=dim)
                self._forward_backward(model)

    def test_exclude_special_tokens_variant_runs(self):
        model = build_variant("A0_full", exclude_special_tokens=True)
        self._forward_backward(model)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            build_variant("Z9_does_not_exist")


class TestTrainingRegimes(unittest.TestCase):
    def test_frozen_regime_freezes_the_backbone(self):
        from model_factory import apply_training_regime

        model = build_variant("A0_full")
        apply_training_regime(model, "frozen")
        info = verify_training_regime(model, "frozen")
        self.assertFalse(info["backbone_trainable"])
        stats = count_model_parameters(model)
        self.assertGreater(stats["frozen"], 0)
        self.assertGreater(stats["trainable"], 0, "LDTF head must stay trainable when frozen")

    def test_finetune_regime_trains_the_backbone(self):
        from model_factory import apply_training_regime

        model = build_variant("A0_full")
        apply_training_regime(model, "finetune")
        info = verify_training_regime(model, "finetune")
        self.assertTrue(info["backbone_trainable"])
        self.assertEqual(count_model_parameters(model)["frozen"], 0)


class TestOptimizerCoverage(unittest.TestCase):
    def test_differential_learning_rates_are_applied(self):
        model = build_variant("A0_full")
        optimizer = build_optimizer(
            model, training_regime="finetune",
            backbone_learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=0.01,
        )
        rates = {group["lr"] for group in optimizer.param_groups}
        self.assertIn(2e-5, rates)
        self.assertIn(1e-4, rates)

    def test_every_trainable_parameter_is_covered(self):
        model = build_variant("A0_full")
        optimizer = build_optimizer(
            model, training_regime="finetune",
            backbone_learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=0.01,
        )
        validate_optimizer_groups(model, optimizer, training_regime="finetune")
        covered = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertIn(id(parameter), covered, "a trainable parameter was left out")

    def test_frozen_regime_excludes_backbone_from_optimizer(self):
        from model_factory import apply_training_regime

        model = build_variant("A0_full")
        apply_training_regime(model, "frozen")
        optimizer = build_optimizer(
            model, training_regime="frozen",
            backbone_learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=0.01,
        )
        validate_optimizer_groups(model, optimizer, training_regime="frozen")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertTrue(parameter.requires_grad)

    def test_optimizer_groups_are_describable(self):
        model = build_variant("A0_full")
        optimizer = build_optimizer(
            model, training_regime="finetune",
            backbone_learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=0.01,
        )
        groups = describe_optimizer_groups(optimizer)
        self.assertTrue(groups)
        for group in groups:
            self.assertIn("learning_rate", group)


if __name__ == "__main__":
    unittest.main(verbosity=2)
