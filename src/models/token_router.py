"""Class-conditioned token routing across every BERT hidden layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


class TokenRouter(nn.Module):
    """Compute class-conditioned token attention and weighted features per layer."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = config.NUM_CLASSES,
        router_dim: int = config.ROUTER_DIM,
        projection_bias: bool = config.PROJECTION_BIAS,
    ) -> None:
        """Create shared query/key projections for scaled token attention."""
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, but received {hidden_size!r}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, but received {num_classes!r}.")
        if router_dim <= 0:
            raise ValueError(f"router_dim must be > 0, but received {router_dim!r}.")
        if not isinstance(projection_bias, bool):
            raise ValueError(f"projection_bias must be bool, got {type(projection_bias).__name__}.")

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.router_dim = router_dim
        self.projection_bias = projection_bias
        self.scale = router_dim ** -0.5

        self.query_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        self.key_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        if self.query_projection.bias is not None:
            nn.init.zeros_(self.query_projection.bias)
        if self.key_projection.bias is not None:
            nn.init.zeros_(self.key_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        label_queries: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Route each class query across tokens independently at every layer.

        Parameters
        ----------
        hidden_states:
            Tensor of shape ``[B, L, T, D]`` from the BERT backbone (skip embedding).
        label_queries:
            Tensor of shape ``[C, D]`` from :class:`LabelQueryBank`.
        attention_mask:
            Tensor of shape ``[B, T]`` where 1 marks real tokens and 0 marks padding.

        Returns
        -------
        dict with:
            - ``token_attention``: ``[B, C, L, T]`` softmaxed attention weights.
            - ``token_features``: ``[B, C, L, D]`` class-conditioned features per layer.
        """
        if hidden_states.ndim != 4:
            raise ValueError(
                f"hidden_states must have shape [B, L, T, D], got {tuple(hidden_states.shape)}."
            )
        if label_queries.ndim != 2:
            raise ValueError(
                f"label_queries must have shape [C, D], got {tuple(label_queries.shape)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape [B, T], got {tuple(attention_mask.shape)}."
            )

        projected_queries = self.query_projection(label_queries)  # [C, R]
        projected_keys = self.key_projection(hidden_states)        # [B, L, T, R]
        # Score: einsum over R, result [B, C, L, T]
        scores = torch.einsum("cr,bltr->bclt", projected_queries, projected_keys)
        scores = scores * self.scale

        valid_mask = attention_mask.bool()                            # [B, T]
        routing_mask = valid_mask[:, None, None, :]                   # [B, 1, 1, T]
        masked_scores = scores.float().masked_fill(~routing_mask, float("-inf"))
        token_attention = torch.softmax(masked_scores, dim=-1).to(dtype=hidden_states.dtype)

        # Weighted sum over tokens: [B, C, L, T] x [B, L, T, D] -> [B, C, L, D]
        token_features = torch.einsum("bclt,bltd->bcld", token_attention, hidden_states)

        return {
            "token_attention": token_attention,
            "token_features": token_features,
        }

    def count_parameters(self) -> dict[str, int]:
        """Return total / trainable / frozen parameter counts for the router."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def extra_repr(self) -> str:
        """Show router configuration in ``print(module)`` output."""
        return (
            f"hidden_size={self.hidden_size}, num_classes={self.num_classes}, "
            f"router_dim={self.router_dim}, projection_bias={self.projection_bias}"
        )
