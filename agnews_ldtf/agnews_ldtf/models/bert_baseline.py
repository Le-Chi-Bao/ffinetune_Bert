"""Minimal BERT final-layer CLS baseline for AG News classification."""

from __future__ import annotations

from numbers import Real

import torch
import torch.nn as nn
from transformers import BertModel


class BertBaselineClassifier(nn.Module):
    """BERT final-layer CLS baseline that returns raw class logits.

    The model intentionally contains only a pretrained BERT encoder, dropout,
    and one linear classification layer. Loss computation belongs to Stage 3.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Load a pretrained BERT encoder and initialize a linear head."""
        super().__init__()
        self._validate_constructor_arguments(model_name, num_classes, dropout)

        self.model_name = model_name
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    @staticmethod
    def _validate_constructor_arguments(
        model_name: str, num_classes: int, dropout: float
    ) -> None:
        """Fail early for invalid model configuration values."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError(f"num_classes must be an integer greater than 1, got {num_classes!r}.")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError(f"dropout must be a number in [0, 1), got {dropout!r}.")

    @staticmethod
    def _validate_forward_inputs(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> None:
        """Validate the tensor contract without silently reshaping inputs."""
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor.")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a torch.Tensor.")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape [batch, sequence], "
                f"got {tuple(attention_mask.shape)}."
            )
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have identical shapes, got "
                f"{tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
            )
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token for CLS extraction.")
        if input_ids.device != attention_mask.device:
            raise ValueError("input_ids and attention_mask must be on the same device.")

        if token_type_ids is not None:
            if not isinstance(token_type_ids, torch.Tensor):
                raise ValueError("token_type_ids must be a torch.Tensor or None.")
            if token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    "token_type_ids must have the same shape as input_ids, got "
                    f"{tuple(token_type_ids.shape)} and {tuple(input_ids.shape)}."
                )
            if token_type_ids.device != input_ids.device:
                raise ValueError("token_type_ids and input_ids must be on the same device.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw logits with shape ``[batch_size, num_classes]``."""
        self._validate_forward_inputs(input_ids, attention_mask, token_type_ids)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        if last_hidden_state is None:
            raise RuntimeError("BERT did not return last_hidden_state.")
        if last_hidden_state.ndim != 3 or last_hidden_state.shape[:2] != input_ids.shape:
            raise RuntimeError(
                "Unexpected BERT last_hidden_state shape: "
                f"{tuple(last_hidden_state.shape)} for input shape {tuple(input_ids.shape)}."
            )

        cls_feature = last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_feature))
        expected_shape = (input_ids.shape[0], self.num_classes)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Classifier returned shape {tuple(logits.shape)}; expected {expected_shape}."
            )
        return logits

    def freeze_encoder(self) -> None:
        """Freeze pretrained BERT parameters while keeping the head trainable."""
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Make all pretrained BERT parameters trainable again."""
        for parameter in self.bert.parameters():
            parameter.requires_grad = True

    def is_encoder_frozen(self) -> bool:
        """Return True only when every BERT parameter is frozen."""
        return all(not parameter.requires_grad for parameter in self.bert.parameters())

    def count_parameters(self) -> dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
