"""End-to-end LDTF-BERT model that wires the backbone and three custom routers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .. import config
from .bert_backbone import BertBackbone
from .class_scorer import ClassScorer
from .depth_router import DepthRouter
from .label_query_bank import LabelQueryBank
from .token_router import TokenRouter


class LdtfBert(nn.Module):
    """BERT + Label Query Bank + Token Router + Depth Router + Class Scorer."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        router_dim: int = config.ROUTER_DIM,
        scorer_dropout: float = 0.1,
        freeze_encoder: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Instantiate all five modules; optionally freeze the BERT backbone."""
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, but received {num_classes!r}.")

        self.num_classes = num_classes
        self.router_dim = router_dim
        self.scorer_dropout = scorer_dropout

        self.backbone = BertBackbone(
            model_name=model_name,
            cache_dir=cache_dir,
            freeze=freeze_encoder,
        )
        hidden_size = self.backbone.hidden_size
        num_layers = self.backbone.num_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.label_queries = LabelQueryBank(num_classes=num_classes, hidden_size=hidden_size)
        self.token_router = TokenRouter(
            hidden_size=hidden_size,
            num_classes=num_classes,
            router_dim=router_dim,
        )
        self.depth_router = DepthRouter(
            num_classes=num_classes,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        self.class_scorer = ClassScorer(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=scorer_dropout,
        )

        self._init_cross_module_weights()

    def _init_cross_module_weights(self) -> None:
        """Apply final Xavier init to keep all custom modules on the same scale."""
        for module in (
            self.label_queries,
            self.token_router,
            self.depth_router,
            self.class_scorer,
        ):
            module.apply(self._xavier_init_module)

    @staticmethod
    def _xavier_init_module(module: nn.Module) -> None:
        """Apply Xavier init to Linear layers; leave other modules alone."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def freeze_encoder(self) -> None:
        """Freeze the BERT backbone (use before calling forward)."""
        self.backbone.freeze()

    def unfreeze_encoder(self) -> None:
        """Unfreeze the BERT backbone (use to switch from frozen to full fine-tune)."""
        self.backbone.unfreeze()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning logits, attention maps, and per-class features.

        Returns
        -------
        dict containing:
            - ``logits``: ``[B, C]`` classification scores.
            - ``token_attention``: ``[B, C, L, T]`` softmaxed token attention.
            - ``depth_attention``: ``[B, C, L]`` softmaxed layer attention.
            - ``class_features``: ``[B, C, D]`` fused per-class representations.
        """
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Skip the embedding layer (index 0) so we route over transformer layers only.
        transformer_states = hidden_states[:, 1:, :, :]  # [B, L, T, D]

        label_queries = self.label_queries()  # [C, D]
        token_out = self.token_router(
            hidden_states=transformer_states,
            label_queries=label_queries,
            attention_mask=attention_mask,
        )
        depth_out = self.depth_router(token_features=token_out["token_features"])
        logits = self.class_scorer(depth_out["class_features"])

        return {
            "logits": logits,
            "token_attention": token_out["token_attention"],
            "depth_attention": depth_out["depth_attention"],
            "class_features": depth_out["class_features"],
        }

    def count_parameters(self) -> dict[str, dict[str, int]]:
        """Return parameter counts broken down by module name."""
        modules = {
            "backbone": self.backbone,
            "label_queries": self.label_queries,
            "token_router": self.token_router,
            "depth_router": self.depth_router,
            "class_scorer": self.class_scorer,
        }
        counts: dict[str, dict[str, int]] = {}
        total = 0
        trainable = 0
        for name, module in modules.items():
            counts[name] = module.count_parameters()
            total += counts[name]["total"]
            trainable += counts[name]["trainable"]
        counts["total"] = {"total": total, "trainable": trainable, "frozen": total - trainable}
        return counts

    @staticmethod
    def build_tokenizer(model_name: str = config.MODEL_NAME, cache_dir: Optional[str] = None):
        """Convenience wrapper that loads the matching tokenizer for the backbone."""
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)