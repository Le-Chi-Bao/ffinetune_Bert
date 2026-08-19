"""Layer-wise routing that fuses token features across BERT depths."""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthRouter(nn.Module):
    """Compute class-conditioned depth attention and combine features across layers."""

    def __init__(
        self,
        num_classes: int = 4,
        num_layers: int = 12,
        hidden_size: int = 768,
    ) -> None:
        """Create per-class, per-layer projection used for depth scoring."""
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, but received {num_classes!r}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, but received {num_layers!r}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, but received {hidden_size!r}.")

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.scale = num_layers ** -0.5

        self.depth_projection = nn.Linear(hidden_size, num_layers, bias=False)
        nn.init.xavier_uniform_(self.depth_projection.weight)

    def forward(self, token_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute depth attention and a fused feature vector per (batch, class).

        Parameters
        ----------
        token_features:
            Tensor of shape ``[B, C, L, D]`` produced by :class:`TokenRouter`.

        Returns
        -------
        dict with:
            - ``depth_attention``: ``[B, C, L]`` softmaxed layer attention.
            - ``class_features``: ``[B, C, D]`` fused feature vector per class.
        """
        if token_features.ndim != 4:
            raise ValueError(
                f"token_features must have shape [B, C, L, D], got {tuple(token_features.shape)}."
            )
        # Project class-conditioned features to one score per layer: [B, C, L]
        layer_scores = self.depth_projection(token_features) * self.scale
        depth_attention = torch.softmax(layer_scores, dim=-1).to(dtype=token_features.dtype)

        # Weighted sum across layers: [B, C, L] x [B, C, L, D] -> [B, C, D]
        class_features = torch.einsum("bcl,bcld->bcd", depth_attention, token_features)

        return {
            "depth_attention": depth_attention,
            "class_features": class_features,
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
            f"num_classes={self.num_classes}, num_layers={self.num_layers}, "
            f"hidden_size={self.hidden_size}"
        )