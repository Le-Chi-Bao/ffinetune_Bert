"""Shared scalar scorer over class-specific fused representations."""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedClassScorer(nn.Module):
    """Map each class-specific vector in ``[B, C, D]`` to one shared logit.

    A single ``Linear(D → 1)`` is applied independently on every class
    representation. Class identity must come from upstream routers, not from
    class-specific classifier weights.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 4,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        """Create dropout and the shared ``D → 1`` linear scorer."""
        super().__init__()
        self._validate_constructor_arguments(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            use_bias=use_bias,
        )

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.use_bias = use_bias

        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(hidden_size, 1, bias=use_bias)
        self._reset_parameters()

    @staticmethod
    def _validate_constructor_arguments(
        hidden_size: int,
        num_classes: int,
        dropout: float,
        use_bias: bool,
    ) -> None:
        """Reject invalid scorer settings without silently correcting them."""
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
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError(
                f"dropout must be a real number in [0, 1), but received {dropout!r}."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(
                f"dropout must be in the range [0, 1), but received {dropout}."
            )
        if not isinstance(use_bias, bool):
            raise ValueError("use_bias must be a boolean.")

    def _reset_parameters(self) -> None:
        """Initialize the shared scorer weight and optional scalar bias."""
        nn.init.xavier_uniform_(self.scorer.weight)
        if self.scorer.bias is not None:
            nn.init.zeros_(self.scorer.bias)

    def _validate_forward_inputs(self, fused_features: torch.Tensor) -> None:
        """Validate the Stage-7 fused feature contract ``[B, C, D]``."""
        if not isinstance(fused_features, torch.Tensor):
            raise ValueError("fused_features must be a torch.Tensor.")
        if fused_features.ndim != 3:
            raise ValueError(
                "fused_features must have shape [B,C,D], "
                f"got {tuple(fused_features.shape)}."
            )

        batch_size, num_classes, hidden_size = fused_features.shape
        if batch_size <= 0:
            raise ValueError(
                f"batch size must be greater than zero, but received {batch_size}."
            )
        if num_classes != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class representations, "
                f"but received {num_classes}."
            )
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, but received {hidden_size}."
            )
        if self.scorer.weight.device != fused_features.device:
            raise ValueError(
                "Scorer parameters and input tensors must be on the same device. "
                "Move the scorer with scorer.to(device)."
            )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Score each class representation with the shared linear head."""
        self._validate_forward_inputs(fused_features)
        dropped_features = self.dropout(fused_features)
        scores = self.scorer(dropped_features)
        logits = scores.squeeze(-1)
        return logits

    def count_parameters(self) -> dict[str, int]:
        """Return this scorer's own parameter counts."""
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
        """Show immutable configuration in the module representation."""
        return (
            f"hidden_size={self.hidden_size}, num_classes={self.num_classes}, "
            f"dropout={self.dropout_p}, use_bias={self.use_bias}"
        )
