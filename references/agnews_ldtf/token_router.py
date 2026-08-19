"""Class-conditioned token routing over every BERT Transformer layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class LabelTokenRouter(nn.Module):
    """Select token evidence for each candidate class independently per layer.

    Given BERT hidden states ``[B, L, T, D]`` and label queries ``[C, D]``,
    the router returns token attention ``[B, C, L, T]`` and the corresponding
    weighted original BERT features ``[B, C, L, D]``. It intentionally leaves
    the layer dimension untouched for the Stage-7 depth router.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 4,
        router_dim: int = 256,
        projection_bias: bool = False,
    ) -> None:
        """Create shared query/key projections for scaled token attention."""
        super().__init__()
        self._validate_constructor_arguments(
            hidden_size=hidden_size,
            num_classes=num_classes,
            router_dim=router_dim,
            projection_bias=projection_bias,
        )

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.router_dim = router_dim
        self.projection_bias = projection_bias
        self.scale = router_dim ** -0.5

        self.query_projection = nn.Linear(
            hidden_size,
            router_dim,
            bias=projection_bias,
        )
        self.key_projection = nn.Linear(
            hidden_size,
            router_dim,
            bias=projection_bias,
        )
        self._reset_parameters()

    @staticmethod
    def _validate_constructor_arguments(
        hidden_size: int,
        num_classes: int,
        router_dim: int,
        projection_bias: bool,
    ) -> None:
        """Validate router configuration without silently correcting values."""
        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
        ):
            raise ValueError(
                f"hidden_size must be greater than zero, but received {hidden_size!r}."
            )
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes < 2
        ):
            raise ValueError(
                f"num_classes must be greater than or equal to 2, but received {num_classes!r}."
            )
        if (
            not isinstance(router_dim, int)
            or isinstance(router_dim, bool)
            or router_dim <= 0
        ):
            raise ValueError(
                f"router_dim must be greater than zero, but received {router_dim!r}."
            )
        if not isinstance(projection_bias, bool):
            raise ValueError("projection_bias must be a boolean.")

    def _reset_parameters(self) -> None:
        """Use non-degenerate projection initialization for token routing."""
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        if self.query_projection.bias is not None:
            nn.init.zeros_(self.query_projection.bias)
        if self.key_projection.bias is not None:
            nn.init.zeros_(self.key_projection.bias)

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        label_queries: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None,
    ) -> tuple[int, int, int, int]:
        """Check all tensor contracts before calculating attention scores."""
        for name, tensor in (
            ("hidden_states", hidden_states),
            ("label_queries", label_queries),
            ("attention_mask", attention_mask),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor.")
        if hidden_states.ndim != 4:
            raise ValueError(
                "hidden_states must have shape [B,L,T,D], "
                f"got {tuple(hidden_states.shape)}."
            )
        if label_queries.ndim != 2:
            raise ValueError(
                "label_queries must have shape [C,D], "
                f"got {tuple(label_queries.shape)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape [B,T], "
                f"got {tuple(attention_mask.shape)}."
            )

        batch_size, num_layers, sequence_length, hidden_size = hidden_states.shape
        num_classes, query_hidden_size = label_queries.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {self.hidden_size}, "
                f"but received {hidden_size}."
            )
        if query_hidden_size != self.hidden_size:
            raise ValueError(
                f"label_queries last dimension must be {self.hidden_size}, "
                f"but received {query_hidden_size}."
            )
        if num_classes != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} label queries, but received {num_classes}."
            )
        if tuple(attention_mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must have shape [B,T] matching hidden_states, got "
                f"{tuple(attention_mask.shape)} for hidden_states "
                f"{tuple(hidden_states.shape)}."
            )
        if hidden_states.device != label_queries.device:
            raise ValueError("hidden_states and label_queries must be on the same device.")
        if hidden_states.device != attention_mask.device:
            raise ValueError("hidden_states and attention_mask must be on the same device.")
        if self.query_projection.weight.device != hidden_states.device:
            raise ValueError(
                "Router parameters and input tensors must be on the same device. "
                "Move the router with router.to(device)."
            )

        if special_tokens_mask is not None:
            if not isinstance(special_tokens_mask, torch.Tensor):
                raise ValueError("special_tokens_mask must be a torch.Tensor or None.")
            if special_tokens_mask.ndim != 2 or tuple(special_tokens_mask.shape) != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "special_tokens_mask must have shape [B,T] matching hidden_states, got "
                    f"{tuple(special_tokens_mask.shape)}."
                )
            if special_tokens_mask.device != hidden_states.device:
                raise ValueError(
                    "special_tokens_mask and hidden_states must be on the same device."
                )
        return batch_size, num_layers, sequence_length, hidden_size

    @staticmethod
    def _build_valid_mask(
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the non-padding, optionally non-special, routing positions."""
        valid_mask = attention_mask.bool()
        if special_tokens_mask is not None:
            valid_mask = valid_mask & ~special_tokens_mask.bool()
        invalid_sample_indices = torch.nonzero(
            valid_mask.sum(dim=-1) == 0,
            as_tuple=False,
        ).flatten()
        if invalid_sample_indices.numel() > 0:
            invalid_indices = [index.item() for index in invalid_sample_indices]
            raise ValueError(
                "Each sample must contain at least one valid routing token; "
                f"invalid batch indices: {invalid_indices}."
            )
        return valid_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        label_queries: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Route each class query across tokens independently at every layer."""
        self._validate_forward_inputs(
            hidden_states=hidden_states,
            label_queries=label_queries,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )
        valid_mask = self._build_valid_mask(attention_mask, special_tokens_mask)

        projected_queries = self.query_projection(label_queries)
        projected_keys = self.key_projection(hidden_states)
        scores = torch.einsum("cr,bltr->bclt", projected_queries, projected_keys)
        scores = scores * self.scale

        routing_mask = valid_mask[:, None, None, :]
        masked_scores = scores.float().masked_fill(~routing_mask, float("-inf"))
        token_attention = torch.softmax(masked_scores, dim=-1).to(
            dtype=hidden_states.dtype
        )
        token_features = torch.einsum(
            "bclt,bltd->bcld",
            token_attention,
            hidden_states,
        )
        return {
            "token_features": token_features,
            "token_attention": token_attention,
        }

    def count_parameters(self) -> dict[str, int]:
        """Return this router's projection parameter counts only."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def extra_repr(self) -> str:
        """Show router configuration in ``print(module)`` output."""
        return (
            f"hidden_size={self.hidden_size}, num_classes={self.num_classes}, "
            f"router_dim={self.router_dim}, projection_bias={self.projection_bias}"
        )
