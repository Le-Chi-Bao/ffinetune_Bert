"""Class-conditioned routing across BERT Transformer depth."""

from __future__ import annotations

import torch
import torch.nn as nn


class LabelDepthRouter(nn.Module):
    """Fuse token-routed features across layers separately for every class.

    Inputs follow the Stage-6 convention: token features ``[B, C, L, D]`` and
    shared label queries ``[C, D]``. The module produces layer attention
    ``[B, C, L]`` and fused original feature vectors ``[B, C, D]``.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 4,
        num_layers: int = 12,
        router_dim: int = 256,
        projection_bias: bool = False,
    ) -> None:
        """Create the V1 query and depth-key projections."""
        super().__init__()
        self._validate_constructor_arguments(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            router_dim=router_dim,
            projection_bias=projection_bias,
        )

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.router_dim = router_dim
        self.projection_bias = projection_bias
        self.scale = router_dim ** -0.5

        self.query_projection = nn.Linear(
            hidden_size,
            router_dim,
            bias=projection_bias,
        )
        self.depth_key_projection = nn.Linear(
            hidden_size,
            router_dim,
            bias=projection_bias,
        )
        self._reset_parameters()

    @staticmethod
    def _validate_constructor_arguments(
        hidden_size: int,
        num_classes: int,
        num_layers: int,
        router_dim: int,
        projection_bias: bool,
    ) -> None:
        """Reject invalid architecture settings with clear configuration errors."""
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
            not isinstance(num_layers, int)
            or isinstance(num_layers, bool)
            or num_layers <= 0
        ):
            raise ValueError(
                f"num_layers must be greater than zero, but received {num_layers!r}."
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
        """Initialize both routing projections without score symmetry."""
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.depth_key_projection.weight)
        if self.query_projection.bias is not None:
            nn.init.zeros_(self.query_projection.bias)
        if self.depth_key_projection.bias is not None:
            nn.init.zeros_(self.depth_key_projection.bias)

    def _validate_forward_inputs(
        self,
        token_features: torch.Tensor,
        label_queries: torch.Tensor,
    ) -> None:
        """Validate the strict Stage-6 feature and Stage-5 query contracts."""
        if not isinstance(token_features, torch.Tensor):
            raise ValueError("token_features must be a torch.Tensor.")
        if not isinstance(label_queries, torch.Tensor):
            raise ValueError("label_queries must be a torch.Tensor.")
        if token_features.ndim != 4:
            raise ValueError(
                "token_features must have shape [B,C,L,D], "
                f"got {tuple(token_features.shape)}."
            )
        if label_queries.ndim != 2:
            raise ValueError(
                "label_queries must have shape [C,D], "
                f"got {tuple(label_queries.shape)}."
            )

        _, feature_classes, feature_layers, feature_hidden_size = token_features.shape
        query_classes, query_hidden_size = label_queries.shape
        if feature_classes != self.num_classes:
            raise ValueError(
                f"Expected token features with {self.num_classes} classes, "
                f"but received {feature_classes} classes."
            )
        if feature_layers != self.num_layers:
            raise ValueError(
                f"Expected token features with {self.num_layers} layers, "
                f"but received {feature_layers} layers."
            )
        if feature_hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected token features with hidden size {self.hidden_size}, "
                f"but received {feature_hidden_size}."
            )
        if query_classes != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} label queries, but received {query_classes}."
            )
        if query_hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected label queries with hidden size {self.hidden_size}, "
                f"but received {query_hidden_size}."
            )
        if token_features.device != label_queries.device:
            raise ValueError("token_features and label_queries must be on the same device.")
        if self.query_projection.weight.device != token_features.device:
            raise ValueError(
                "Router parameters and input tensors must be on the same device. "
                "Move the router with router.to(device)."
            )

    def forward(
        self,
        token_features: torch.Tensor,
        label_queries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute class-conditioned layer attention and fused original features."""
        self._validate_forward_inputs(token_features, label_queries)

        projected_queries = self.query_projection(label_queries)
        projected_depth_keys = self.depth_key_projection(token_features)
        depth_scores = torch.einsum(
            "cr,bclr->bcl",
            projected_queries,
            projected_depth_keys,
        )
        depth_scores = depth_scores * self.scale
        depth_attention = torch.softmax(depth_scores.float(), dim=-1).to(
            dtype=token_features.dtype
        )
        fused_features = torch.einsum(
            "bcl,bcld->bcd",
            depth_attention,
            token_features,
        )
        return {
            "fused_features": fused_features,
            "depth_attention": depth_attention,
        }

    def count_parameters(self) -> dict[str, int]:
        """Return this router's own projection parameter counts."""
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
            f"num_layers={self.num_layers}, router_dim={self.router_dim}, "
            f"projection_bias={self.projection_bias}"
        )
