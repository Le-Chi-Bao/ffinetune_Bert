"""Learnable, class-specific query vectors for the LDTF-BERT architecture."""

from __future__ import annotations

from numbers import Real

import torch
import torch.nn as nn


DEFAULT_CLASS_NAMES = (
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
)


class LabelQueryBank(nn.Module):
    """Store one raw learnable query vector for every candidate class.

    This Stage-5 component intentionally performs no routing, projection,
    normalization, scoring, or classification. Later routing modules consume
    the returned query tensor with shape ``[C, D]``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        hidden_size: int = 768,
        init_std: float = 0.02,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize a normally distributed shared query bank.

        Args:
            num_classes: Number of candidate labels, at least two.
            hidden_size: Width of each query vector.
            init_std: Standard deviation of normal parameter initialization.
            class_names: Optional label metadata. It never affects computation.
        """
        super().__init__()
        self._validate_constructor_arguments(
            num_classes=num_classes,
            hidden_size=hidden_size,
            init_std=init_std,
            class_names=class_names,
        )

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.init_std = float(init_std)
        self.class_names = self._resolve_class_names(num_classes, class_names)

        self.queries = nn.Parameter(torch.empty(num_classes, hidden_size))
        nn.init.normal_(self.queries, mean=0.0, std=self.init_std)

    @staticmethod
    def _validate_constructor_arguments(
        num_classes: int,
        hidden_size: int,
        init_std: float,
        class_names: tuple[str, ...] | None,
    ) -> None:
        """Raise clear ``ValueError`` exceptions for invalid configuration."""
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes < 2
        ):
            raise ValueError("num_classes must be an integer greater than or equal to 2.")
        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
        ):
            raise ValueError("hidden_size must be a positive integer.")
        if (
            not isinstance(init_std, Real)
            or isinstance(init_std, bool)
            or init_std <= 0
        ):
            raise ValueError("init_std must be a positive number.")
        if class_names is not None:
            try:
                received_count = len(class_names)
            except TypeError as error:
                raise ValueError("class_names must be a tuple of class names or None.") from error
            if received_count != num_classes:
                raise ValueError(
                    f"Expected {num_classes} class names, but received {received_count}."
                )

    @staticmethod
    def _resolve_class_names(
        num_classes: int, class_names: tuple[str, ...] | None
    ) -> tuple[str, ...]:
        """Return metadata only; the labels never participate in computation."""
        if class_names is not None:
            return tuple(class_names)
        if num_classes == len(DEFAULT_CLASS_NAMES):
            return DEFAULT_CLASS_NAMES
        return tuple(f"Class {index}" for index in range(num_classes))

    def forward(self) -> torch.Tensor:
        """Return raw class queries with shape ``[C, D]``."""
        return self.queries

    def expand_for_batch(self, batch_size: int) -> torch.Tensor:
        """View the shared query bank as ``[B, C, D]`` without copying it."""
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        return self.queries.unsqueeze(0).expand(batch_size, -1, -1)

    def freeze(self) -> None:
        """Disable gradients for the query bank without changing module mode."""
        self.queries.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradients for the query bank."""
        self.queries.requires_grad_(True)

    def count_parameters(self) -> dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""
        total = self.queries.numel()
        trainable = total if self.queries.requires_grad else 0
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def extra_repr(self) -> str:
        """Summarize immutable bank configuration in ``print(module)`` output."""
        return (
            f"num_classes={self.num_classes}, hidden_size={self.hidden_size}, "
            f"init_std={self.init_std}"
        )
