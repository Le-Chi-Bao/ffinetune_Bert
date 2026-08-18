"""Stage-11 LDTF-BERT architecture ablations.

This file composes Stage 4–8 modules without copying their implementations.
Absent routers/scorers are not registered, hence cannot leak unused parameters
into an optimizer or checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .bert_backbone import BertMultiLayerBackbone
from .class_scorer import SharedClassScorer
from .depth_router import LabelDepthRouter
from .label_query_bank import LabelQueryBank
from .token_router import LabelTokenRouter

ABLATION_VARIANTS = {
    "A0_full", "A1_no_token_router", "A2_no_depth_router", "A3_final_layer",
    "A4_shared_token_query", "A5_shared_depth_query", "A6_class_specific_scorer",
}


class ClassSpecificScorer(nn.Module):
    """Independent scalar scorer for each class-specific representation."""
    def __init__(self, hidden_size: int, num_classes: int, dropout: float, use_bias: bool) -> None:
        super().__init__()
        self.hidden_size, self.num_classes = hidden_size, num_classes
        self.dropout = nn.Dropout(dropout)
        self.class_weights = nn.Parameter(torch.empty(num_classes, hidden_size))
        self.class_bias = nn.Parameter(torch.zeros(num_classes)) if use_bias else None
        nn.init.xavier_uniform_(self.class_weights)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        if fused_features.ndim != 3 or fused_features.shape[1:] != (self.num_classes, self.hidden_size):
            raise ValueError(f"Expected fused features [B,{self.num_classes},{self.hidden_size}], got {tuple(fused_features.shape)}.")
        logits = torch.einsum("bcd,cd->bc", self.dropout(fused_features), self.class_weights)
        return logits if self.class_bias is None else logits + self.class_bias

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class _SharedTokenRouter(nn.Module):
    """One learnable global token query, shared across candidate classes."""
    def __init__(self, hidden_size: int, router_dim: int, projection_bias: bool) -> None:
        super().__init__()
        self.hidden_size, self.router_dim, self.scale = hidden_size, router_dim, router_dim ** -0.5
        self.shared_token_query = nn.Parameter(torch.empty(1, hidden_size))
        self.query_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        self.key_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        nn.init.normal_(self.shared_token_query, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)

    def forward(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor, num_classes: int) -> dict[str, torch.Tensor]:
        query = self.query_projection(self.shared_token_query)
        keys = self.key_projection(hidden_states)
        scores = torch.einsum("qr,bltr->bqlt", query, keys) * self.scale
        scores = scores.float().masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        shared_attention = torch.softmax(scores, dim=-1).to(hidden_states.dtype)
        attention = shared_attention.expand(-1, num_classes, -1, -1)
        features = torch.einsum("bclt,bltd->bcld", attention, hidden_states)
        return {"token_attention": attention, "token_features": features}


class _SharedDepthRouter(nn.Module):
    """One global depth query; class-specific token features remain distinct."""
    def __init__(self, hidden_size: int, router_dim: int, projection_bias: bool) -> None:
        super().__init__()
        self.hidden_size, self.router_dim, self.scale = hidden_size, router_dim, router_dim ** -0.5
        self.shared_depth_query = nn.Parameter(torch.empty(1, hidden_size))
        self.query_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        self.depth_key_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        nn.init.normal_(self.shared_depth_query, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.depth_key_projection.weight)

    def forward(self, token_features: torch.Tensor) -> dict[str, torch.Tensor]:
        query = self.query_projection(self.shared_depth_query)
        keys = self.depth_key_projection(token_features)
        scores = torch.einsum("qr,bclr->bcl", query, keys) * self.scale
        attention = torch.softmax(scores.float(), dim=-1).to(token_features.dtype)
        return {"depth_attention": attention, "fused_features": torch.einsum("bcl,bcld->bcd", attention, token_features)}


class LDTFAblationClassifier(nn.Module):
    """LDTF variants A0–A6 with a uniform Stage-10 forward contract."""
    def __init__(
        self, model_name: str = "bert-base-uncased", num_classes: int = 4,
        variant: str = "A0_full", token_router_dim: int = 256, depth_router_dim: int = 256,
        classifier_dropout: float = 0.1, projection_bias: bool = False, scorer_bias: bool = True,
        exclude_special_tokens: bool = False, class_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if variant not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant {variant!r}; expected one of {sorted(ABLATION_VARIANTS)}.")
        if num_classes < 2 or token_router_dim <= 0 or depth_router_dim <= 0:
            raise ValueError("num_classes must be >=2 and router dimensions must be positive.")
        self.model_name, self.num_classes, self.variant = model_name, num_classes, variant
        self.token_router_dim, self.depth_router_dim = token_router_dim, depth_router_dim
        self.classifier_dropout, self.projection_bias = float(classifier_dropout), projection_bias
        self.scorer_bias, self.exclude_special_tokens = scorer_bias, exclude_special_tokens
        self.backbone = BertMultiLayerBackbone(model_name)
        self.hidden_size, self.num_hidden_layers = self.backbone.hidden_size, self.backbone.num_hidden_layers
        self.label_query_bank = LabelQueryBank(
            num_classes=num_classes,
            hidden_size=self.hidden_size,
            class_names=class_names,
        )
        self.class_names = self.label_query_bank.class_names

        if variant not in {"A1_no_token_router"}:
            if variant == "A4_shared_token_query":
                self.shared_token_router = _SharedTokenRouter(self.hidden_size, token_router_dim, projection_bias)
            else:
                self.token_router = LabelTokenRouter(self.hidden_size, num_classes, token_router_dim, projection_bias)
        if variant not in {"A2_no_depth_router", "A3_final_layer"}:
            if variant == "A5_shared_depth_query":
                self.shared_depth_router = _SharedDepthRouter(self.hidden_size, depth_router_dim, projection_bias)
            else:
                self.depth_router = LabelDepthRouter(self.hidden_size, num_classes, self.num_hidden_layers, depth_router_dim, projection_bias)
        if variant == "A6_class_specific_scorer":
            self.class_specific_scorer = ClassSpecificScorer(self.hidden_size, num_classes, classifier_dropout, scorer_bias)
        else:
            self.class_scorer = SharedClassScorer(self.hidden_size, num_classes, classifier_dropout, scorer_bias)

    def _valid_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor | None) -> torch.Tensor:
        valid = attention_mask.bool()
        if self.exclude_special_tokens:
            inferred_special = (input_ids == 101) | (input_ids == 102)
            special = inferred_special if special_tokens_mask is None else special_tokens_mask.bool()
            valid = valid & ~special
        if not valid.any(dim=-1).all():
            raise ValueError("Special-token exclusion left a sample without a content token.")
        return valid

    @staticmethod
    def _uniform_token_features(hidden_states: torch.Tensor, valid_mask: torch.Tensor, num_classes: int) -> dict[str, torch.Tensor]:
        weights = valid_mask.to(hidden_states.dtype)
        attention_base = weights / weights.sum(dim=-1, keepdim=True)
        b, layers, tokens, _ = hidden_states.shape
        attention = attention_base[:, None, None, :].expand(b, num_classes, layers, tokens)
        return {"token_attention": attention, "token_features": torch.einsum("bclt,bltd->bcld", attention, hidden_states)}

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None,
        special_tokens_mask: torch.Tensor | None = None, return_routing: bool = False,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(input_ids, attention_mask, token_type_ids)
        hidden_states = backbone_outputs["hidden_states"]
        valid_mask = self._valid_mask(input_ids, attention_mask, special_tokens_mask)
        label_queries = self.label_query_bank()
        router_hidden_states = hidden_states[:, -1:, :, :] if self.variant == "A3_final_layer" else hidden_states

        if self.variant == "A1_no_token_router":
            token_outputs = self._uniform_token_features(router_hidden_states, valid_mask, self.num_classes)
        elif self.variant == "A4_shared_token_query":
            token_outputs = self.shared_token_router(router_hidden_states, valid_mask, self.num_classes)
        else:
            token_outputs = self.token_router(router_hidden_states, label_queries, attention_mask, (~valid_mask & attention_mask.bool()))
        token_features = token_outputs["token_features"]
        token_attention = token_outputs["token_attention"]

        if self.variant == "A2_no_depth_router":
            depth_attention = torch.full(
                token_features.shape[:3], 1.0 / token_features.shape[2], device=token_features.device, dtype=token_features.dtype
            )
            fused_features = token_features.mean(dim=2)
        elif self.variant == "A3_final_layer":
            depth_attention = torch.ones(token_features.shape[:3], device=token_features.device, dtype=token_features.dtype)
            fused_features = token_features[:, :, 0, :]
        elif self.variant == "A5_shared_depth_query":
            depth_outputs = self.shared_depth_router(token_features)
            depth_attention, fused_features = depth_outputs["depth_attention"], depth_outputs["fused_features"]
        else:
            depth_outputs = self.depth_router(token_features, label_queries)
            depth_attention, fused_features = depth_outputs["depth_attention"], depth_outputs["fused_features"]
        logits = self.class_specific_scorer(fused_features) if self.variant == "A6_class_specific_scorer" else self.class_scorer(fused_features)
        outputs: dict[str, torch.Tensor] = {"logits": logits}
        if return_routing:
            outputs.update({"token_attention": token_attention, "depth_attention": depth_attention})
        if return_features:
            outputs.update({"hidden_states": hidden_states, "label_queries": label_queries, "token_features": token_features, "fused_features": fused_features})
        return outputs

    def freeze_backbone(self) -> None:
        self.backbone.freeze_encoder()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze_encoder()

    def count_parameters(self) -> dict[str, object]:
        modules = {name: module for name, module in self.named_children()}
        by_module = {}
        for name, module in modules.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            by_module[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
        total, trainable = sum(x["total"] for x in by_module.values()), sum(x["trainable"] for x in by_module.values())
        backbone_total = by_module["backbone"]["total"]
        return {"total": total, "trainable": trainable, "frozen": total-trainable, "non_backbone": total-backbone_total, "by_module": by_module}

    def get_config(self) -> dict[str, object]:
        return {"model_name": self.model_name, "num_classes": self.num_classes, "variant": self.variant, "token_router_dim": self.token_router_dim, "depth_router_dim": self.depth_router_dim, "classifier_dropout": self.classifier_dropout, "projection_bias": self.projection_bias, "scorer_bias": self.scorer_bias, "exclude_special_tokens": self.exclude_special_tokens, "class_names": self.class_names}
