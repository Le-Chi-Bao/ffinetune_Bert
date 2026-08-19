"""Thin wrapper around HuggingFace BertModel exposing all hidden layers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from .. import config


class BertBackbone(nn.Module):
    """BERT encoder that returns stacked hidden states for every Transformer layer."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        *,
        cache_dir: Optional[str] = None,
        freeze: bool = False,
    ) -> None:
        """Load pretrained weights; optionally freeze every parameter."""
        super().__init__()
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(
            model_name,
            output_hidden_states=True,
            cache_dir=cache_dir,
        )
        self.encoder = BertModel.from_pretrained(
            model_name,
            config=self.config,
            cache_dir=cache_dir,
        )
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run BERT and return hidden states [B, L+1, T, D] (embeddings + layers)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        # Hidden states: tuple of (L+1) tensors, each [B, T, D]; index 0 = embeddings.
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "BertModel did not return hidden_states; ensure output_hidden_states=True."
            )
        stacked = torch.stack(hidden_states, dim=1)  # [B, L+1, T, D]
        return stacked

    def freeze(self) -> None:
        """Disable gradient updates for every encoder parameter."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        """Re-enable gradient updates for every encoder parameter."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def count_parameters(self) -> dict[str, int]:
        """Return total / trainable / frozen parameter counts for the backbone."""
        total = sum(parameter.numel() for parameter in self.encoder.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.encoder.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
