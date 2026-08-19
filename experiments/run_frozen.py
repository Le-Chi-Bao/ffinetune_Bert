"""Frozen-encoder experiment: only train the routers, scorer, and label queries."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                # noqa: E402
from src.dataset import build_all_dataloaders         # noqa: E402
from src.evaluate import build_model_from_checkpoint, evaluate_checkpoint  # noqa: E402
from src.models import LdtfBert                       # noqa: E402
from src.train import TrainConfig, train_model        # noqa: E402
from src.utils import ensure_output_dirs, set_seed    # noqa: E402


def main() -> None:
    """Train only the custom heads (3 routers + scorer + queries); BERT stays frozen."""
    ensure_output_dirs()
    set_seed(config.SEED)

    output_dir = config.FROZEN_OUTPUT
    print(f"[run_frozen] writing outputs to {output_dir}")

    tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)
    loaders = build_all_dataloaders(tokenizer=tokenizer)

    print(
        f"[run_frozen] dataset sizes: train={loaders['train_size']}, "
        f"val={loaders['val_size']}, test={loaders['test_size']}"
    )

    model = LdtfBert(model_name=config.MODEL_NAME, freeze_encoder=True)
    counts = model.count_parameters()
    print("[run_frozen] parameter counts:")
    for name, summary in counts.items():
        print(f"  - {name}: {summary}")

    train_config = TrainConfig(output_dir=output_dir, freeze_encoder=True)
    result = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        train_config=train_config,
        progress_callback=lambda epoch, record: print(
            f"[run_frozen] epoch={epoch}: {record}"
        ),
    )
    print(
        f"[run_frozen] best val accuracy={result.best_val_accuracy:.4f} "
        f"@ epoch={result.best_epoch}"
    )

    eval_model = build_model_from_checkpoint(output_dir / "best_model.pt")
    test_metrics = evaluate_checkpoint(
        model=eval_model,
        dataloader=loaders["test"],
        checkpoint_path=output_dir / "best_model.pt",
        output_path=output_dir / "test_metrics.json",
    )
    print(
        f"[run_frozen] test accuracy={test_metrics['accuracy']:.4f}, "
        f"f1_macro={test_metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()