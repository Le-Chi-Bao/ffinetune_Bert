"""Non-causal descriptive routing diagnostics for Stage-11 validation analysis."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F


def _entropy(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return -(safe * safe.log()).sum(dim=dim)


def compute_routing_diagnostics(
    token_attention: torch.Tensor,
    depth_attention: torch.Tensor,
    label_queries: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    special_token_ids: tuple[int, ...] = (101, 102),
) -> dict[str, Any]:
    """Return descriptive attention/query statistics; no causal claims are implied."""
    if token_attention.ndim != 4 or depth_attention.ndim != 3 or label_queries.ndim != 2:
        raise ValueError("Expected token_attention=[B,C,L,T], depth_attention=[B,C,L], label_queries=[C,D].")
    if token_attention.shape[:3] != depth_attention.shape:
        raise ValueError("Token/depth attention dimensions [B,C,L] must agree.")
    token_attention = token_attention.detach()
    depth_attention = depth_attention.detach()
    label_queries = label_queries.detach()
    token_entropy = _entropy(token_attention.float(), -1)
    depth_entropy = _entropy(depth_attention.float(), -1)
    mean_depth = depth_attention.float().mean(dim=(0, 1))
    depth_argmax = depth_attention.argmax(dim=-1).reshape(-1)
    histogram = torch.bincount(depth_argmax, minlength=depth_attention.shape[-1])
    normalized_queries = F.normalize(label_queries.float(), dim=-1)
    special_fraction = 0.0
    if input_ids is not None:
        if attention_mask is None or input_ids.shape != attention_mask.shape or input_ids.shape != token_attention.shape[:1] + token_attention.shape[-1:]:
            raise ValueError("input_ids/attention_mask must be [B,T] matching token attention.")
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_token_ids:
            special |= input_ids == token_id
        special &= attention_mask.bool()
        special_fraction = float(
            token_attention.float().masked_select(special[:, None, None, :].expand_as(token_attention)).sum()
            / token_attention.shape[0] / token_attention.shape[1] / token_attention.shape[2]
        )
    return {
        "mean_token_attention_entropy": float(token_entropy.mean()),
        "std_token_attention_entropy": float(token_entropy.std(unbiased=False)),
        "mean_depth_attention_entropy": float(depth_entropy.mean()),
        "std_depth_attention_entropy": float(depth_entropy.std(unbiased=False)),
        "mean_depth_weight_by_layer": mean_depth.tolist(),
        "depth_argmax_histogram": histogram.tolist(),
        "query_cosine_similarity": (normalized_queries @ normalized_queries.T).tolist(),
        "special_token_attention_fraction": special_fraction,
        "interpretation": "Descriptive routing statistics only; attention is not causal evidence.",
    }
