"""Convert fused class features into classification logits."""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


class ClassScorer(nn.Module):
    """Tiny MLP head that maps class-conditioned features to per-class logits."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> None:
        """Build a single hidden-layer projection from ``hidden_size`` -> ``hidden_size`` -> 1."""
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, but received {num_classes!r}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, but received {hidden_size!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), but received {dropout!r}.")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        for module in self.layer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, class_features: torch.Tensor) -> torch.Tensor:
        """Map ``[B, C, D]`` features into ``[B, C]`` logits (one per class)."""
        if class_features.ndim != 3:
            raise ValueError(
                f"class_features must have shape [B, C, D], got {tuple(class_features.shape)}."
            )
        # Project each class vector independently to a scalar -> [B, C]
        logits = self.layer(class_features).squeeze(-1)
        return logits

    def count_parameters(self) -> dict[str, int]:
        """Return total / trainable / frozen parameter counts for the scorer."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def extra_repr(self) -> str:
        """Show scorer configuration in ``print(module)`` output."""
        return (
            f"num_classes={self.num_classes}, hidden_size={self.hidden_size}, "
            f"dropout={self.dropout}"
        )