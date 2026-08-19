"""Pretrained BERT backbone that exposes every Transformer-layer output."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class BertMultiLayerBackbone(nn.Module):
    """Return BERT Transformer features using the project convention ``[B,L,T,D]``.

    The embedding output is deliberately excluded: stacked layer index 0 is
    Transformer layer H1 and the final index is H12 for bert-base-uncased.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        """Load a pretrained BERT encoder and read dimensions from its config."""
        super().__init__()
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name
        # LDTF routes over per-layer hidden states and never reads pooler_output,
        # so the pretrained pooler is excluded. Keeping it would register
        # trainable parameters that never receive gradients.
        self.bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        if getattr(self.bert, "pooler", None) is not None:
            raise RuntimeError(
                "BertModel still exposes a pooler despite add_pooling_layer=False."
            )
        self.hidden_size = self.bert.config.hidden_size
        self.num_hidden_layers = self.bert.config.num_hidden_layers

    @staticmethod
    def _validate_forward_inputs(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> None:
        """Validate tensor shapes without silently changing caller inputs."""
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor.")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a torch.Tensor.")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B,T], got {tuple(input_ids.shape)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape [B,T], got {tuple(attention_mask.shape)}."
            )
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have identical shapes, got "
                f"{tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
            )
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must have at least one token for last_cls.")
        if input_ids.device != attention_mask.device:
            raise ValueError("input_ids and attention_mask must be on the same device.")
        if token_type_ids is not None:
            if not isinstance(token_type_ids, torch.Tensor):
                raise ValueError("token_type_ids must be a torch.Tensor or None.")
            if token_type_ids.ndim != 2 or token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    "token_type_ids must have shape identical to input_ids, got "
                    f"{tuple(token_type_ids.shape)} and {tuple(input_ids.shape)}."
                )
            if token_type_ids.device != input_ids.device:
                raise ValueError("token_type_ids and input_ids must be on the same device.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return all Transformer-layer outputs plus the final hidden state and CLS."""
        self._validate_forward_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden_states = outputs.hidden_states
        expected_hidden_state_count = self.num_hidden_layers + 1
        if all_hidden_states is None:
            raise RuntimeError("BERT returned no hidden states despite output_hidden_states=True.")
        if len(all_hidden_states) != expected_hidden_state_count:
            raise RuntimeError(
                f"Expected {expected_hidden_state_count} hidden-state tensors: "
                f"1 embedding output + {self.num_hidden_layers} Transformer layers, "
                f"but received {len(all_hidden_states)}."
            )
        if outputs.last_hidden_state is None:
            raise RuntimeError("BERT did not return last_hidden_state.")

        # hidden_states[0] is embeddings; H1..HL are stacked on dim=1.
        hidden_stack = torch.stack(all_hidden_states[1:], dim=1)
        batch_size, sequence_length = input_ids.shape
        expected_stack_shape = (
            batch_size,
            self.num_hidden_layers,
            sequence_length,
            self.hidden_size,
        )
        if tuple(hidden_stack.shape) != expected_stack_shape:
            raise RuntimeError(
                f"Unexpected stacked hidden-state shape {tuple(hidden_stack.shape)}; "
                f"expected {expected_stack_shape}."
            )
        expected_last_shape = (batch_size, sequence_length, self.hidden_size)
        if tuple(outputs.last_hidden_state.shape) != expected_last_shape:
            raise RuntimeError(
                f"Unexpected last_hidden_state shape {tuple(outputs.last_hidden_state.shape)}; "
                f"expected {expected_last_shape}."
            )

        return {
            "hidden_states": hidden_stack,
            "last_hidden_state": outputs.last_hidden_state,
            "last_cls": outputs.last_hidden_state[:, 0, :],
        }

    def freeze_encoder(self) -> None:
        """Freeze BERT parameters without changing the module train/eval state."""
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Make every BERT parameter trainable again."""
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
