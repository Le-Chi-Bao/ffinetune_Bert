"""Stage-13 official AG News test access, guarded so it can happen exactly once.

Every earlier stage is forbidden from touching the official test split. Stage 13
is the single point where it may be read, and only under a locked manifest whose
checksums already pin the exact checkpoints to be evaluated.

The guard is a persistent ``official_test_access_log.json``. Once the official
test has been evaluated for a given manifest hash, a second evaluation of that
same manifest is refused: results would otherwise be silently re-rollable until a
favourable number appeared, which is exactly the failure mode the lock prevents.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from config import ID2LABEL, LABEL2ID
from data import (
    REQUIRED_BATCH_KEYS,
    clean_text,
    create_single_dataloader,
    create_train_val_test,
    get_tokenizer,
    load_ag_news,
)

ACCESS_LOG_NAME = "official_test_access_log.json"


class OfficialTestAccessError(RuntimeError):
    """Raised when the official test split is accessed outside the locked protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_access_log(path: str | Path) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.is_file():
        return []
    with log_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise OfficialTestAccessError(f"Corrupt official test access log: {log_path}")
    return payload


def assert_first_official_test_access(
    access_log_path: str | Path,
    manifest_sha256: str,
    *,
    allow_repeat: bool = False,
) -> list[dict[str, Any]]:
    """Refuse a second official-test evaluation of an already-evaluated manifest."""
    entries = read_access_log(access_log_path)
    previous = [entry for entry in entries if entry.get("manifest_sha256") == manifest_sha256]
    if previous and not allow_repeat:
        raise OfficialTestAccessError(
            "The official test split has already been evaluated for manifest "
            f"{manifest_sha256}. Recorded at: {[entry.get('timestamp_utc') for entry in previous]}. "
            "Re-running the official test on a locked manifest is prohibited because it "
            "permits implicit selection on test results. Create a new protocol version "
            "with a new manifest instead."
        )
    return entries


def record_official_test_access(
    access_log_path: str | Path,
    manifest_sha256: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append an immutable audit entry after the official test has been read."""
    log_path = Path(access_log_path)
    entries = read_access_log(log_path)
    entries.append(
        {
            "manifest_sha256": manifest_sha256,
            "timestamp_utc": _now(),
            "detail": dict(detail or {}),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_official_test_data(
    *,
    max_length: int = 128,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Build the official AG News test DataLoader.

    This function is intentionally the only official-test loader in the project and
    must be called solely from the guarded Stage-13 runner.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}.")
    raw_dataset = load_ag_news()
    _train, _validation, test_dataset = create_train_val_test(raw_dataset)
    tokenizer = tokenizer or get_tokenizer()

    def _tokenize(batch: dict[str, list[str]]) -> Any:
        return tokenizer(
            [clean_text(text) for text in batch["text"]],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = test_dataset.map(_tokenize, batched=True, desc="Tokenizing official_test")
    columns_to_remove = [
        column
        for column in tokenized.column_names
        if column not in {"input_ids", "attention_mask", "label"}
    ]
    if columns_to_remove:
        tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized = tokenized.with_format("torch", columns=sorted(REQUIRED_BATCH_KEYS))

    loader = create_single_dataloader(
        tokenized, tokenizer, eval_batch_size, False, num_workers, None
    )
    return {
        "test_loader": loader,
        "test_dataset": tokenized,
        "test_samples": len(tokenized),
        "tokenizer": tokenizer,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "max_length": max_length,
        "official_test_loaded": True,
    }


def paired_prediction_contingency(
    labels: list[int],
    baseline_predictions: list[int],
    candidate_predictions: list[int],
) -> dict[str, Any]:
    """Descriptive prediction-level contingency table for a paired comparison.

    Returns the 2x2 counts required for a later McNemar-style analysis. No
    statistical significance is computed or claimed here; the counts are reported
    so the comparison stays auditable and paired on identical examples.
    """
    if not (len(labels) == len(baseline_predictions) == len(candidate_predictions)):
        raise ValueError(
            "Paired prediction comparison requires equal-length label and prediction "
            f"sequences, got {len(labels)}, {len(baseline_predictions)}, {len(candidate_predictions)}."
        )
    if not labels:
        raise ValueError("Cannot compare empty prediction sequences.")
    both_correct = only_baseline = only_candidate = both_wrong = 0
    for truth, baseline, candidate in zip(labels, baseline_predictions, candidate_predictions):
        baseline_ok = baseline == truth
        candidate_ok = candidate == truth
        if baseline_ok and candidate_ok:
            both_correct += 1
        elif baseline_ok and not candidate_ok:
            only_baseline += 1
        elif candidate_ok and not baseline_ok:
            only_candidate += 1
        else:
            both_wrong += 1
    total = len(labels)
    discordant = only_baseline + only_candidate
    return {
        "n_examples": total,
        "both_correct": both_correct,
        "only_baseline_correct": only_baseline,
        "only_candidate_correct": only_candidate,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "agreement_rate": (both_correct + both_wrong) / total,
        "baseline_accuracy": (both_correct + only_baseline) / total,
        "candidate_accuracy": (both_correct + only_candidate) / total,
        "accuracy_delta": (only_candidate - only_baseline) / total,
        "interpretation": (
            "Counts are paired on identical test examples. delta = candidate - baseline."
        ),
        "statistical_claim": (
            "Descriptive contingency counts only; no significance test is performed or claimed."
        ),
    }


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, list[int]]:
    """Run inference and return predictions and labels in deterministic order."""
    from model_factory import extract_logits
    from training_utils import move_batch_to_device

    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids, attention_mask, batch_labels, token_type_ids = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = extract_logits(outputs)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())
    if not predictions:
        raise RuntimeError("Official test inference produced zero predictions.")
    return {"predictions": predictions, "labels": labels}
