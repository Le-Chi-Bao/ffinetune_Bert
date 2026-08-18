"""Full LDTF-BERT classifier assembling Stages 4–8."""

from __future__ import annotations

import torch
import torch.nn as nn

from .bert_backbone import BertMultiLayerBackbone
from .class_scorer import SharedClassScorer
from .depth_router import LabelDepthRouter
from .label_query_bank import LabelQueryBank
from .token_router import LabelTokenRouter


class LDTFBertClassifier(nn.Module):
    """Label-conditioned Depth–Token Fusion BERT for multi-class text topics.

    Pipeline:
        backbone → shared label queries → token router → depth router → scorer
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        token_router_dim: int = 256,
        depth_router_dim: int = 256,
        classifier_dropout: float = 0.1,
        projection_bias: bool = False,
        scorer_bias: bool = True,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        """Wire Stage-4–8 modules and verify dimensional compatibility."""
        super().__init__()
        self._validate_constructor_arguments(
            model_name=model_name,
            num_classes=num_classes,
            token_router_dim=token_router_dim,
            depth_router_dim=depth_router_dim,
            classifier_dropout=classifier_dropout,
            projection_bias=projection_bias,
            scorer_bias=scorer_bias,
            class_names=class_names,
        )

        self.model_name = model_name
        self.num_classes = num_classes
        self.token_router_dim = token_router_dim
        self.depth_router_dim = depth_router_dim
        self.classifier_dropout = float(classifier_dropout)
        self.projection_bias = projection_bias
        self.scorer_bias = scorer_bias

        self.backbone = BertMultiLayerBackbone(model_name=model_name)
        self.hidden_size = self.backbone.hidden_size
        self.num_hidden_layers = self.backbone.num_hidden_layers

        self.label_query_bank = LabelQueryBank(
            num_classes=num_classes,
            hidden_size=self.hidden_size,
            class_names=class_names,
        )
        self.class_names = self.label_query_bank.class_names

        self.token_router = LabelTokenRouter(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            router_dim=token_router_dim,
            projection_bias=projection_bias,
        )
        self.depth_router = LabelDepthRouter(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            num_layers=self.num_hidden_layers,
            router_dim=depth_router_dim,
            projection_bias=projection_bias,
        )
        self.class_scorer = SharedClassScorer(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            dropout=classifier_dropout,
            use_bias=scorer_bias,
        )
        self._validate_module_compatibility()

    @staticmethod
    def _validate_constructor_arguments(
        model_name: str,
        num_classes: int,
        token_router_dim: int,
        depth_router_dim: int,
        classifier_dropout: float,
        projection_bias: bool,
        scorer_bias: bool,
        class_names: tuple[str, ...] | None,
    ) -> None:
        """Reject invalid full-model configuration without silent fixes."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes < 2
        ):
            raise ValueError(
                f"num_classes must be greater than or equal to 2, but received {num_classes!r}."
            )
        if (
            not isinstance(token_router_dim, int)
            or isinstance(token_router_dim, bool)
            or token_router_dim <= 0
        ):
            raise ValueError(
                f"token_router_dim must be greater than zero, but received {token_router_dim!r}."
            )
        if (
            not isinstance(depth_router_dim, int)
            or isinstance(depth_router_dim, bool)
            or depth_router_dim <= 0
        ):
            raise ValueError(
                f"depth_router_dim must be greater than zero, but received {depth_router_dim!r}."
            )
        if isinstance(classifier_dropout, bool) or not isinstance(
            classifier_dropout, (int, float)
        ):
            raise ValueError(
                "classifier_dropout must be a real number in [0, 1), "
                f"but received {classifier_dropout!r}."
            )
        if not 0.0 <= float(classifier_dropout) < 1.0:
            raise ValueError(
                "classifier_dropout must be in the range [0, 1), "
                f"but received {classifier_dropout}."
            )
        if not isinstance(projection_bias, bool):
            raise ValueError("projection_bias must be a boolean.")
        if not isinstance(scorer_bias, bool):
            raise ValueError("scorer_bias must be a boolean.")
        if class_names is not None:
            try:
                received_count = len(class_names)
            except TypeError as error:
                raise ValueError(
                    "class_names must be a tuple of class names or None."
                ) from error
            if received_count != num_classes:
                raise ValueError(
                    f"Expected {num_classes} class names, but received {received_count}."
                )

    def _validate_module_compatibility(self) -> None:
        """Ensure Stage-4–8 submodules agree on width, classes, and depth."""
        modules = (
            ("label_query_bank", self.label_query_bank),
            ("token_router", self.token_router),
            ("depth_router", self.depth_router),
            ("class_scorer", self.class_scorer),
        )
        for name, module in modules:
            if module.hidden_size != self.hidden_size:
                raise ValueError(
                    f"{name} hidden_size {module.hidden_size} does not match "
                    f"backbone hidden_size {self.hidden_size}."
                )
            if module.num_classes != self.num_classes:
                raise ValueError(
                    f"{name} num_classes {module.num_classes} does not match "
                    f"model num_classes {self.num_classes}."
                )
        if self.depth_router.num_layers != self.num_hidden_layers:
            raise ValueError(
                f"depth_router num_layers {self.depth_router.num_layers} does not match "
                f"backbone num_hidden_layers {self.num_hidden_layers}."
            )

    def _validate_forward_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        special_tokens_mask: torch.Tensor | None,
    ) -> None:
        """Validate batch tensors without moving or reshaping them."""
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
        batch_size, sequence_length = input_ids.shape
        if batch_size <= 0:
            raise ValueError(
                f"batch size must be greater than zero, but received {batch_size}."
            )
        if sequence_length <= 0:
            raise ValueError(
                f"sequence length must be greater than zero, but received {sequence_length}."
            )
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

        if special_tokens_mask is not None:
            if not isinstance(special_tokens_mask, torch.Tensor):
                raise ValueError("special_tokens_mask must be a torch.Tensor or None.")
            if (
                special_tokens_mask.ndim != 2
                or special_tokens_mask.shape != input_ids.shape
            ):
                raise ValueError(
                    "special_tokens_mask must have shape identical to input_ids, got "
                    f"{tuple(special_tokens_mask.shape)} and {tuple(input_ids.shape)}."
                )
            if special_tokens_mask.device != input_ids.device:
                raise ValueError(
                    "special_tokens_mask and input_ids must be on the same device."
                )

        mask_values = attention_mask.detach()
        if mask_values.dtype == torch.bool:
            valid_values = True
        else:
            unique_values = torch.unique(mask_values)
            valid_values = all(
                float(value.item()) in (0.0, 1.0) for value in unique_values
            )
        if not valid_values:
            raise ValueError("attention_mask must contain only boolean or 0/1 values.")
        if not attention_mask.bool().any(dim=-1).all():
            raise ValueError("Each sample must contain at least one valid token.")

        if next(self.parameters()).device != input_ids.device:
            raise ValueError(
                "Model parameters and input tensors must be on the same device. "
                "Move the model with model.to(device)."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        special_tokens_mask: torch.Tensor | None = None,
        return_routing: bool = False,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the full LDTF pipeline and return logits plus optional diagnostics."""
        self._validate_forward_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            special_tokens_mask=special_tokens_mask,
        )

        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = backbone_outputs["hidden_states"]
        label_queries = self.label_query_bank()
        token_outputs = self.token_router(
            hidden_states=hidden_states,
            label_queries=label_queries,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )
        token_features = token_outputs["token_features"]
        token_attention = token_outputs["token_attention"]
        depth_outputs = self.depth_router(
            token_features=token_features,
            label_queries=label_queries,
        )
        fused_features = depth_outputs["fused_features"]
        depth_attention = depth_outputs["depth_attention"]
        logits = self.class_scorer(fused_features=fused_features)

        outputs: dict[str, torch.Tensor] = {"logits": logits}
        if return_routing:
            outputs["token_attention"] = token_attention
            outputs["depth_attention"] = depth_attention
        if return_features:
            outputs["hidden_states"] = hidden_states
            outputs["token_features"] = token_features
            outputs["fused_features"] = fused_features
            outputs["label_queries"] = label_queries
        return outputs

    def freeze_backbone(self) -> None:
        """Freeze BERT parameters without changing train/eval mode."""
        self.backbone.freeze_encoder()

    def unfreeze_backbone(self) -> None:
        """Restore trainable BERT parameters."""
        self.backbone.unfreeze_encoder()

    def count_parameters(self) -> dict[str, object]:
        """Return global and per-module parameter statistics."""
        by_module = {
            "backbone": self.backbone.count_parameters(),
            "label_query_bank": self.label_query_bank.count_parameters(),
            "token_router": self.token_router.count_parameters(),
            "depth_router": self.depth_router.count_parameters(),
            "class_scorer": self.class_scorer.count_parameters(),
        }
        total = sum(stats["total"] for stats in by_module.values())
        trainable = sum(stats["trainable"] for stats in by_module.values())
        frozen = sum(stats["frozen"] for stats in by_module.values())
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "by_module": by_module,
        }

    def get_config(self) -> dict[str, object]:
        """Return serializable architecture metadata for later checkpoints."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "token_router_dim": self.token_router_dim,
            "depth_router_dim": self.depth_router_dim,
            "classifier_dropout": self.classifier_dropout,
            "projection_bias": self.projection_bias,
            "scorer_bias": self.scorer_bias,
            "class_names": self.class_names,
        }

    def extra_repr(self) -> str:
        """Summarize high-level architecture settings."""
        return (
            f"model_name={self.model_name!r}, num_classes={self.num_classes}, "
            f"hidden_size={self.hidden_size}, num_hidden_layers={self.num_hidden_layers}, "
            f"token_router_dim={self.token_router_dim}, "
            f"depth_router_dim={self.depth_router_dim}"
        )
