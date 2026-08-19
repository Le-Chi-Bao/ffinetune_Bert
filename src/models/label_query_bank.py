"""Learnable per-class query vectors used by both routers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


class LabelQueryBank(nn.Module):
    """Holds one learnable query vector per class for class-conditioned attention."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_size: int = 768,
    ) -> None:
        """Allocate and initialize C query vectors of dimension *hidden_size*."""
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, but received {num_classes!r}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, but received {hidden_size!r}.")
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.queries = nn.Parameter(torch.empty(num_classes, hidden_size))
        nn.init.xavier_uniform_(self.queries)

    def forward(self) -> torch.Tensor:
        """Return the [C, D] tensor of class queries (with gradient flow)."""
        return self.queries

    def count_parameters(self) -> dict[str, int]:
        """Return total / trainable / frozen parameter counts for the bank."""
        total = self.queries.numel()
        trainable = self.queries.numel() if self.queries.requires_grad else 0
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def extra_repr(self) -> str:
        """Show bank configuration in ``print(module)`` output."""
        return f"num_classes={self.num_classes}, hidden_size={self.hidden_size}"
