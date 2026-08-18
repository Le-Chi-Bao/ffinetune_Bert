"""Stage 1 data pipeline for AG News topic classification.

This module deliberately prepares data only.  It does not define a model,
loss, optimizer, scheduler, or training/evaluation loop.
"""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from config import (
    DEBUG,
    DEBUG_TEST_SIZE,
    DEBUG_TRAIN_SIZE,
    DEBUG_VAL_SIZE,
    EVAL_BATCH_SIZE,
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_INSPECTION_EXAMPLES,
    NUM_WORKERS,
    SEED,
    TOKEN_LENGTH_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
    VALIDATION_SIZE,
)

REQUIRED_RAW_COLUMNS = {"text", "label"}
REQUIRED_BATCH_KEYS = {"input_ids", "attention_mask", "labels"}

# Identifiers recorded in the Stage-10 data signature. Bump the cleaning version
# whenever clean_text or the split protocol changes.
DATA_CLEANING_VERSION = "v1-whitespace-normalization"
DATASET_PROTOCOL = "ag_news_stratified_90_10_research_split"


def set_seed(seed: int) -> None:
    """Set random seeds used by data preparation and later experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _label_is_valid(label: Any) -> bool:
    """Return whether a raw label is an integer in the configured class range."""
    if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)):
        return False
    return 0 <= int(label) < NUM_CLASSES


def _get_label_column(dataset: Dataset) -> str:
    """Return the supported label column name, or fail with a clear error."""
    if "label" in dataset.column_names:
        return "label"
    if "labels" in dataset.column_names:
        return "labels"
    raise ValueError(
        "Dataset is missing a label column. Expected 'label' before tokenization "
        "or 'labels' after tokenization."
    )


def validate_dataset_schema(dataset: DatasetDict) -> None:
    """Validate required AG News splits, columns, and non-empty datasets."""
    required_splits = {"train", "test"}
    missing_splits = required_splits.difference(dataset.keys())
    if missing_splits:
        raise ValueError(
            "AG News is missing required split(s): "
            f"{sorted(missing_splits)}. Available splits: {list(dataset.keys())}."
        )

    for split_name in ("train", "test"):
        split = dataset[split_name]
        missing_columns = REQUIRED_RAW_COLUMNS.difference(split.column_names)
        if missing_columns:
            raise ValueError(
                f"AG News '{split_name}' split is missing required column(s): "
                f"{sorted(missing_columns)}. Found: {split.column_names}."
            )
        if len(split) == 0:
            raise ValueError(f"AG News '{split_name}' split is empty.")

    _validate_label_metadata(dataset["train"])


def _validate_label_metadata(train_dataset: Dataset) -> None:
    """Compare dataset ClassLabel metadata with the project label mapping."""
    label_feature = train_dataset.features.get("label")
    actual_names = getattr(label_feature, "names", None)
    expected_names = [ID2LABEL[index] for index in range(NUM_CLASSES)]

    print("\nExpected label mapping:")
    print(ID2LABEL)
    print("Dataset label metadata:")
    print(actual_names)

    if actual_names is None:
        warnings.warn(
            "AG News exposes no label-name metadata. The pipeline will use the "
            f"explicit project mapping: {ID2LABEL}.",
            stacklevel=2,
        )
        return

    actual_names = list(actual_names)
    actual_mapping = {index: name for index, name in enumerate(actual_names)}
    print("Actual label mapping:")
    print(actual_mapping)

    if actual_names != expected_names:
        raise ValueError(
            "AG News label metadata does not match the required mapping. "
            f"Expected {ID2LABEL}, but dataset reports {actual_mapping}. "
            "Do not continue until this mismatch is understood."
        )


def _validate_label_values(dataset: Dataset, split_name: str) -> None:
    """Raise if a split contains missing or out-of-range labels."""
    labels = dataset["label"]
    missing_count = sum(label is None for label in labels)
    invalid_labels = [label for label in labels if not _label_is_valid(label)]

    if missing_count or invalid_labels:
        unique_invalid = sorted({repr(label) for label in invalid_labels})[:10]
        raise ValueError(
            f"Invalid labels in '{split_name}': missing={missing_count}, "
            f"invalid={len(invalid_labels)}, examples={unique_invalid}. "
            f"Expected integer labels in [0, {NUM_CLASSES - 1}]."
        )


def _print_raw_dataset_inspection(dataset: DatasetDict) -> None:
    """Print a compact, explicit inspection of the downloaded raw dataset."""
    print("\nAG News dataset structure")
    print("-" * 48)
    print(dataset)
    print("\nTrain column names:")
    print(dataset["train"].column_names)
    print("\nTrain feature schema:")
    print(dataset["train"].features)
    print("\nFirst raw train record:")
    print(dataset["train"][0])

    for split_name in ("train", "test"):
        split = dataset[split_name]
        print(f"\n{split_name.title()} samples: {len(split)}")
        for index in range(min(NUM_INSPECTION_EXAMPLES, len(split))):
            example = split[index]
            label_id = int(example["label"])
            print(f"\nExample {index + 1} ({split_name})")
            print("-" * 32)
            print("Text:")
            print(example["text"])
            print("\nLabel ID:")
            print(label_id)
            print("\nLabel Name:")
            print(ID2LABEL[label_id])


def load_ag_news() -> DatasetDict:
    """Download AG News from Hugging Face and validate its raw structure."""
    try:
        dataset = load_dataset("fancyzhx/ag_news")
    except Exception as error:
        raise RuntimeError(
            "Could not load 'fancyzhx/ag_news'. On Kaggle, enable Internet for "
            "the first download or make the dataset available in the Hugging Face cache."
        ) from error

    validate_dataset_schema(dataset)
    _validate_label_values(dataset["train"], "official_train")
    _validate_label_values(dataset["test"], "official_test")
    _print_raw_dataset_inspection(dataset)
    return dataset


def clean_text(text: str) -> str:
    """Apply only conservative whitespace normalization for BERT input."""
    return " ".join(str(text).strip().split())


def create_train_val_test(dataset: DatasetDict) -> tuple[Dataset, Dataset, Dataset]:
    """Make a deterministic 90/10 stratified train/validation split.

    The official test split is assigned directly and is never split or merged.
    """
    if not 0.0 < VALIDATION_SIZE < 1.0:
        raise ValueError(f"VALIDATION_SIZE must be in (0, 1), got {VALIDATION_SIZE}.")

    official_train = dataset["train"]
    official_test = dataset["test"]
    split = official_train.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=SEED,
        stratify_by_column="label",
    )
    train_dataset = split["train"]
    val_dataset = split["test"]
    test_dataset = official_test

    if not all((len(train_dataset), len(val_dataset), len(test_dataset))):
        raise ValueError("Train, validation, and test datasets must all be non-empty.")
    if len(train_dataset) + len(val_dataset) != len(official_train):
        raise RuntimeError("Train/validation split does not reconstruct official train.")
    if len(test_dataset) != len(official_test):
        raise RuntimeError("Official test split was unexpectedly changed.")

    actual_validation_size = len(val_dataset) / len(official_train)
    allowed_rounding_error = 1 / len(official_train)
    if abs(actual_validation_size - VALIDATION_SIZE) > allowed_rounding_error:
        raise RuntimeError(
            "Validation split size violates policy: "
            f"expected {VALIDATION_SIZE:.2%}, got {actual_validation_size:.2%}."
        )

    print("\nOfficial split policy")
    print("-" * 48)
    print(f"Official train: {len(official_train)}")
    print(f"Train (90% of official train): {len(train_dataset)}")
    print(f"Validation (10% of official train): {len(val_dataset)}")
    print(f"Official test (unchanged): {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def _select_debug_subset(dataset: Dataset, requested_size: int, seed: int) -> Dataset:
    """Select a reproducible small subset without changing the official split policy."""
    if requested_size <= 0:
        raise ValueError(f"Debug subset size must be positive, got {requested_size}.")
    if requested_size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(requested_size))


def _apply_debug_mode(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> tuple[Dataset, Dataset, Dataset]:
    """Reduce already-split datasets only when DEBUG is explicitly enabled."""
    if not DEBUG:
        return train_dataset, val_dataset, test_dataset

    print("\nDEBUG MODE ENABLED")
    print("Using reduced datasets after the official train/validation split.")
    return (
        _select_debug_subset(train_dataset, DEBUG_TRAIN_SIZE, SEED + 1),
        _select_debug_subset(val_dataset, DEBUG_VAL_SIZE, SEED + 2),
        _select_debug_subset(test_dataset, DEBUG_TEST_SIZE, SEED + 3),
    )


def check_dataset_quality(dataset: Dataset, split_name: str) -> dict[str, Any]:
    """Report raw-text and label quality issues without modifying examples."""
    texts = dataset["text"]
    label_column = _get_label_column(dataset)
    labels = dataset[label_column]

    missing_text = 0
    empty_text = 0
    whitespace_only_text = 0
    duplicate_texts = 0
    seen_texts: set[str] = set()
    text_lengths: list[int] = []

    for text in texts:
        if text is None:
            missing_text += 1
            continue

        text_as_string = str(text)
        text_lengths.append(len(text_as_string))
        if text_as_string == "":
            empty_text += 1
        elif text_as_string.strip() == "":
            whitespace_only_text += 1

        normalized_text = clean_text(text_as_string)
        if normalized_text in seen_texts:
            duplicate_texts += 1
        else:
            seen_texts.add(normalized_text)

    missing_labels = sum(label is None for label in labels)
    invalid_labels = sum(not _label_is_valid(label) for label in labels)
    report: dict[str, Any] = {
        "split": split_name,
        "num_samples": len(dataset),
        "missing_text": missing_text,
        "empty_text": empty_text,
        "whitespace_only_text": whitespace_only_text,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "duplicate_texts": duplicate_texts,
        "min_text_length": min(text_lengths, default=0),
        "max_text_length": max(text_lengths, default=0),
    }

    print(f"\n{split_name.title()} quality report")
    print("-" * 48)
    for key, value in report.items():
        if key != "split":
            print(f"{key}: {value}")
    return report


def _raise_for_critical_quality_issues(report: Mapping[str, Any]) -> None:
    """Do not silently tokenize examples with missing text or invalid labels."""
    critical_keys = ("missing_text", "missing_labels", "invalid_labels")
    issues = {key: report[key] for key in critical_keys if report[key]}
    if issues:
        raise ValueError(
            f"Cannot tokenize '{report['split']}' because of critical data issues: {issues}."
        )


def get_class_distribution(dataset: Dataset, split_name: str) -> dict[str, Any]:
    """Compute and print the class distribution in the fixed project label order."""
    label_column = _get_label_column(dataset)
    labels = dataset[label_column]
    counts = Counter(int(label) for label in labels)
    total = len(dataset)

    distribution = {
        "split": split_name,
        "total": total,
        "counts": {ID2LABEL[label_id]: counts[label_id] for label_id in range(NUM_CLASSES)},
        "percentages": {
            ID2LABEL[label_id]: (100 * counts[label_id] / total) for label_id in range(NUM_CLASSES)
        },
    }

    print(f"\n{split_name.title()} distribution")
    print("-" * 48)
    for label_id in range(NUM_CLASSES):
        label_name = ID2LABEL[label_id]
        print(
            f"{label_name:<10} {counts[label_id]:>7} "
            f"{distribution['percentages'][label_name]:>7.2f}%"
        )
    print(f"Total      {total:>7}")
    return distribution


def _report_train_val_distribution_gap(
    train_distribution: Mapping[str, Any], val_distribution: Mapping[str, Any]
) -> dict[str, float]:
    """Print absolute class-percentage gaps as a stratified-split sanity check."""
    gaps = {
        label_name: abs(
            train_distribution["percentages"][label_name]
            - val_distribution["percentages"][label_name]
        )
        for label_name in ID2LABEL.values()
    }
    print("\nTrain/validation distribution gaps (percentage points)")
    print("-" * 48)
    for label_name, gap in gaps.items():
        print(f"{label_name:<10} {gap:.3f}")
    return gaps


def _cleaned_text_set(dataset: Dataset) -> set[str]:
    """Build a set used only for duplicate-overlap reporting."""
    return {clean_text(text) for text in dataset["text"] if text is not None}


def check_research_split_overlap(train_dataset: Dataset, val_dataset: Dataset) -> dict[str, int]:
    """Report train/validation overlap without loading or inspecting official test."""
    report = {"train_val_overlap": len(_cleaned_text_set(train_dataset).intersection(_cleaned_text_set(val_dataset)))}
    print("\nResearch train/validation overlap report")
    print("-" * 48)
    print(f"train_val_overlap: {report['train_val_overlap']}")
    return report


def check_split_overlap(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> dict[str, int]:
    """Report exact cleaned-text overlap without deleting any records."""
    train_texts = _cleaned_text_set(train_dataset)
    val_texts = _cleaned_text_set(val_dataset)
    test_texts = _cleaned_text_set(test_dataset)

    report = {
        "train_val_overlap": len(train_texts.intersection(val_texts)),
        "train_test_overlap": len(train_texts.intersection(test_texts)),
        "val_test_overlap": len(val_texts.intersection(test_texts)),
    }
    print("\nCleaned-text overlap report")
    print("-" * 48)
    for key, value in report.items():
        print(f"{key}: {value}")
    return report


def get_tokenizer() -> PreTrainedTokenizerBase:
    """Load and validate the tokenizer paired with the selected BERT backbone."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as error:
        raise RuntimeError(
            f"Could not load tokenizer '{MODEL_NAME}'. Ensure it is cached or Kaggle Internet is enabled."
        ) from error

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        raise ValueError(
            f"Tokenizer '{tokenizer.name_or_path}' has no pad token. "
            "Dynamic padding cannot be configured safely."
        )

    print("\nTokenizer inspection")
    print("-" * 48)
    print(f"name_or_path: {tokenizer.name_or_path}")
    print(f"pad_token: {tokenizer.pad_token}")
    print(f"cls_token: {tokenizer.cls_token}")
    print(f"sep_token: {tokenizer.sep_token}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    return tokenizer


def _percentile(values: np.ndarray, percentile: float) -> float:
    """Use NumPy's current percentile API while supporting older Kaggle images."""
    try:
        return float(np.percentile(values, percentile, method="linear"))
    except TypeError:
        return float(np.percentile(values, percentile, interpolation="linear"))


def analyze_token_lengths(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Measure unpadded BERT token lengths in chunks before dataset tokenization."""
    if "text" not in dataset.column_names:
        raise ValueError("Token-length analysis requires a raw dataset with a 'text' column.")
    if len(dataset) == 0:
        raise ValueError("Cannot analyze token lengths for an empty dataset.")
    if sample_size is not None and sample_size <= 0:
        raise ValueError(f"sample_size must be positive or None, got {sample_size}.")

    analysis_dataset = dataset
    if sample_size is not None and sample_size < len(dataset):
        generator = np.random.default_rng(SEED)
        sampled_indices = np.sort(generator.choice(len(dataset), size=sample_size, replace=False))
        analysis_dataset = dataset.select([int(index) for index in sampled_indices])

    lengths: list[int] = []
    for start_index in range(0, len(analysis_dataset), TOKEN_LENGTH_BATCH_SIZE):
        end_index = min(start_index + TOKEN_LENGTH_BATCH_SIZE, len(analysis_dataset))
        texts = analysis_dataset[start_index:end_index]["text"]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])

    lengths_array = np.asarray(lengths, dtype=np.int64)
    report = {
        "num_examples": int(len(lengths_array)),
        "sample_size_requested": sample_size,
        "max_length": MAX_LENGTH,
        "mean": float(lengths_array.mean()),
        "median": float(np.median(lengths_array)),
        "p90": _percentile(lengths_array, 90),
        "p95": _percentile(lengths_array, 95),
        "p99": _percentile(lengths_array, 99),
        "max": int(lengths_array.max()),
        "percentage_over_max_length": float(100 * np.mean(lengths_array > MAX_LENGTH)),
    }

    print("\nToken-length analysis")
    print("-" * 48)
    print(f"Examples analyzed: {report['num_examples']}")
    print(f"MAX_LENGTH = {MAX_LENGTH}")
    for key in ("mean", "median", "p90", "p95", "p99", "max"):
        print(f"{key}: {report[key]:.2f}" if key != "max" else f"{key}: {report[key]}")
    print(f"Percentage truncated: {report['percentage_over_max_length']:.2f}%")
    return report


def tokenize_batch(
    batch: Mapping[str, list[str]], tokenizer: PreTrainedTokenizerBase
) -> Mapping[str, list[list[int]]]:
    """Tokenize one Hugging Face batch without static padding."""
    if "text" not in batch:
        raise ValueError("Tokenization batch is missing the required 'text' field.")
    cleaned_texts = [clean_text(text) for text in batch["text"]]
    return tokenizer(
        cleaned_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


def _tokenize_one_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    split_name: str,
) -> Dataset:
    """Map batched tokenization and retain only fields needed by a DataLoader."""
    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        desc=f"Tokenizing {split_name}",
    )

    allowed_columns = {"input_ids", "attention_mask", "label"}
    columns_to_remove = [
        column_name
        for column_name in tokenized.column_names
        if column_name not in allowed_columns
    ]
    if columns_to_remove:
        tokenized = tokenized.remove_columns(columns_to_remove)

    if "label" not in tokenized.column_names:
        raise ValueError(f"Tokenized '{split_name}' dataset is missing its 'label' column.")
    if "input_ids" not in tokenized.column_names or "attention_mask" not in tokenized.column_names:
        raise ValueError(
            f"Tokenized '{split_name}' dataset is missing input_ids or attention_mask."
        )

    tokenized = tokenized.rename_column("label", "labels")
    tokenized = tokenized.with_format("torch", columns=sorted(REQUIRED_BATCH_KEYS))
    return tokenized


def tokenize_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset, Dataset]:
    """Tokenize each split with Dataset.map and defer padding to the collator."""
    return (
        _tokenize_one_dataset(train_dataset, tokenizer, "train"),
        _tokenize_one_dataset(val_dataset, tokenizer, "validation"),
        _tokenize_one_dataset(test_dataset, tokenizer, "test"),
    )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create CPU DataLoaders with dynamic batch-level padding."""
    if not all((len(train_dataset), len(val_dataset), len(test_dataset))):
        raise ValueError("Cannot create a DataLoader for an empty dataset.")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    loader_common = {
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": data_collator,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        **loader_common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        **loader_common,
    )

    if not all((len(train_loader), len(val_loader), len(test_loader))):
        raise RuntimeError("One or more DataLoaders are empty.")
    return train_loader, val_loader, test_loader


def assert_batch_is_valid(batch: Mapping[str, torch.Tensor], split_name: str) -> None:
    """Validate the batch contract expected by later training stages."""
    missing_keys = REQUIRED_BATCH_KEYS.difference(batch.keys())
    if missing_keys:
        raise KeyError(
            f"{split_name} batch is missing required key(s): {sorted(missing_keys)}. "
            f"Found: {sorted(batch.keys())}."
        )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    if input_ids.ndim != 2 or attention_mask.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            f"{split_name} batch has invalid dimensions: input_ids={tuple(input_ids.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}, labels={tuple(labels.shape)}."
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"{split_name} input_ids and attention_mask shapes differ.")
    if input_ids.shape[0] != labels.shape[0]:
        raise ValueError(f"{split_name} batch size does not match number of labels.")
    if input_ids.shape[1] > MAX_LENGTH:
        raise ValueError(
            f"{split_name} batch sequence length {input_ids.shape[1]} exceeds MAX_LENGTH={MAX_LENGTH}."
        )
    if labels.numel() == 0:
        raise ValueError(f"{split_name} batch has no labels.")
    if labels.min().item() < 0 or labels.max().item() >= NUM_CLASSES:
        raise ValueError(f"{split_name} batch has labels outside [0, {NUM_CLASSES - 1}].")
    if not torch.isfinite(input_ids.float()).all():
        raise ValueError(f"{split_name} batch contains non-finite input_ids.")
    if not torch.isfinite(attention_mask.float()).all():
        raise ValueError(f"{split_name} batch contains non-finite attention_mask values.")
    if not torch.isfinite(labels.float()).all():
        raise ValueError(f"{split_name} batch contains non-finite labels.")

    unique_mask_values = set(torch.unique(attention_mask).cpu().tolist())
    if not unique_mask_values.issubset({0, 1}):
        raise ValueError(
            f"{split_name} attention_mask must contain only 0 or 1, found {unique_mask_values}."
        )
    if not torch.all(attention_mask.sum(dim=-1) > 0):
        raise ValueError(f"{split_name} contains a sample without a valid token.")


def run_dataloader_sanity_checks(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Check one batch from every split before returning the prepared pipeline."""
    for split_name, loader in (
        ("train", train_loader),
        ("validation", val_loader),
        ("test", test_loader),
    ):
        batch = next(iter(loader))
        assert_batch_is_valid(batch, split_name)
        print(
            f"{split_name.title()} loader batch OK: "
            f"input_ids={tuple(batch['input_ids'].shape)}, "
            f"attention_mask={tuple(batch['attention_mask'].shape)}, "
            f"labels={tuple(batch['labels'].shape)}"
        )


RESEARCH_TRAIN_PATH = "data/processed/research_train.parquet"
RESEARCH_VALIDATION_PATH = "data/processed/research_validation.parquet"
FINAL_DATA_REPORT_PATH = "data/reports/final_data_report.json"


def file_checksum(path: str | Path) -> str:
    """Return a streaming SHA-256 checksum used inside Stage-10 data signatures."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_validation_against_train(
    train_dataset: Dataset, val_dataset: Dataset
) -> tuple[Dataset, dict[str, Any]]:
    """Drop validation rows whose cleaned text also occurs in train.

    AG News contains a small number of exact duplicate articles. Leaving them in
    both splits would leak training data into validation-based model selection.
    Rows are removed from validation only, so the training split stays intact,
    and the operation is deterministic given a fixed upstream split.
    """
    train_texts = _cleaned_text_set(train_dataset)
    keep_indices = [
        index
        for index, text in enumerate(val_dataset["text"])
        if clean_text(text) not in train_texts
    ]
    removed = len(val_dataset) - len(keep_indices)
    deduplicated = val_dataset.select(keep_indices)
    report = {
        "validation_rows_before": len(val_dataset),
        "validation_rows_after": len(deduplicated),
        "validation_rows_removed": removed,
        "policy": "remove_validation_rows_duplicated_in_train",
    }
    print("\nValidation deduplication against train")
    print("-" * 48)
    for key, value in report.items():
        print(f"{key}: {value}")
    return deduplicated, report


def export_research_splits(
    train_path: str | Path = RESEARCH_TRAIN_PATH,
    validation_path: str | Path = RESEARCH_VALIDATION_PATH,
    report_path: str | Path = FINAL_DATA_REPORT_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize the frozen research train/validation splits and the quality report.

    The official test split is validated by Stage 1 but is deliberately never
    written here, so no later stage can read it by accident.
    """
    train_file = Path(train_path)
    validation_file = Path(validation_path)
    report_file = Path(report_path)
    if train_file.is_file() and validation_file.is_file() and report_file.is_file() and not overwrite:
        with report_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    raw_dataset = load_ag_news()
    train_dataset, val_dataset, _official_test_dataset = create_train_val_test(raw_dataset)
    val_dataset, deduplication_report = deduplicate_validation_against_train(
        train_dataset, val_dataset
    )
    quality_reports = {
        "train": check_dataset_quality(train_dataset, "train"),
        "validation": check_dataset_quality(val_dataset, "validation"),
    }
    for report in quality_reports.values():
        _raise_for_critical_quality_issues(report)
    class_distributions = {
        "train": get_class_distribution(train_dataset, "train"),
        "validation": get_class_distribution(val_dataset, "validation"),
    }
    overlap_report = check_research_split_overlap(train_dataset, val_dataset)

    train_file.parent.mkdir(parents=True, exist_ok=True)
    validation_file.parent.mkdir(parents=True, exist_ok=True)
    train_dataset.to_parquet(str(train_file))
    val_dataset.to_parquet(str(validation_file))

    train_val_overlap = int(overlap_report.get("train_val_overlap", 0))
    critical_failures: list[str] = []
    if train_val_overlap != 0:
        critical_failures.append(f"train_validation_overlap={train_val_overlap}")
    if not len(train_dataset) or not len(val_dataset):
        critical_failures.append("empty_research_split")

    report_payload = {
        "overall_status": "FAIL" if critical_failures else "PASS",
        "READY_FOR_OFFICIAL_TRAINING": not critical_failures,
        "critical_failures": critical_failures,
        "data_cleaning_version": DATA_CLEANING_VERSION,
        "dataset_protocol": DATASET_PROTOCOL,
        "source_dataset": "fancyzhx/ag_news",
        "split_seed": SEED,
        "validation_size": VALIDATION_SIZE,
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(val_dataset),
        "label_mapping": {str(key): value for key, value in ID2LABEL.items()},
        "train_manifest_checksum": file_checksum(train_file),
        "validation_manifest_checksum": file_checksum(validation_file),
        "train_path": str(train_file),
        "validation_path": str(validation_file),
        "quality_reports": quality_reports,
        "class_distributions": class_distributions,
        "overlap_report": overlap_report,
        "deduplication_report": deduplication_report,
        "official_test_exported": False,
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    return report_payload


def load_research_split(path: str | Path, split_name: str) -> Dataset:
    """Load one research parquet split and re-validate its Stage-1 contract."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Research {split_name} split not found at {file_path}. "
            "Run `python -m data --export-research-splits` first."
        )
    dataset = Dataset.from_parquet(str(file_path))
    missing_columns = REQUIRED_RAW_COLUMNS.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Research {split_name} split is missing column(s): {sorted(missing_columns)}."
        )
    _validate_label_values(dataset, split_name)
    if len(dataset) == 0:
        raise ValueError(f"Research {split_name} split is empty.")
    return dataset


def _seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a deterministic, run-seed-derived state."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_single_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create one dynamically padded DataLoader with explicit Stage-10 settings."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
    if len(dataset) == 0:
        raise ValueError("Cannot create a DataLoader for an empty dataset.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"),
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )
    if len(loader) == 0:
        raise RuntimeError("Created an empty DataLoader.")
    return loader


def prepare_research_data(
    train_path: str | Path = RESEARCH_TRAIN_PATH,
    validation_path: str | Path = RESEARCH_VALIDATION_PATH,
    max_length: int = MAX_LENGTH,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Build Stage-10 train/validation loaders only; the test split is never touched."""
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}.")

    train_dataset = load_research_split(train_path, "research_train")
    val_dataset = load_research_split(validation_path, "research_validation")
    tokenizer = get_tokenizer()

    def _tokenize(batch: Mapping[str, list[str]]) -> Mapping[str, list[list[int]]]:
        cleaned_texts = [clean_text(text) for text in batch["text"]]
        return tokenizer(cleaned_texts, truncation=True, max_length=max_length, padding=False)

    prepared: dict[str, Dataset] = {}
    for split_name, dataset in (("train", train_dataset), ("validation", val_dataset)):
        tokenized = dataset.map(_tokenize, batched=True, desc=f"Tokenizing {split_name}")
        columns_to_remove = [
            column
            for column in tokenized.column_names
            if column not in {"input_ids", "attention_mask", "label"}
        ]
        if columns_to_remove:
            tokenized = tokenized.remove_columns(columns_to_remove)
        tokenized = tokenized.rename_column("label", "labels")
        prepared[split_name] = tokenized.with_format("torch", columns=sorted(REQUIRED_BATCH_KEYS))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = create_single_dataloader(
        prepared["train"], tokenizer, train_batch_size, True, num_workers, generator
    )
    val_loader = create_single_dataloader(
        prepared["validation"], tokenizer, eval_batch_size, False, num_workers, None
    )
    for split_name, loader in (("train", train_loader), ("validation", val_loader)):
        assert_batch_is_valid(next(iter(loader)), split_name)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "tokenizer": tokenizer,
        "train_dataset": prepared["train"],
        "val_dataset": prepared["validation"],
        "train_samples": len(prepared["train"]),
        "validation_samples": len(prepared["validation"]),
        "train_manifest_checksum": file_checksum(train_path),
        "validation_manifest_checksum": file_checksum(validation_path),
        "dataloader_generator": generator,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "max_length": max_length,
        "official_test_loaded": False,
    }


def prepare_data() -> dict[str, Any]:
    """Prepare validated AG News datasets, loaders, tokenizer, and diagnostics."""
    set_seed(SEED)
    raw_dataset = load_ag_news()
    train_dataset, val_dataset, test_dataset = create_train_val_test(raw_dataset)
    train_dataset, val_dataset, test_dataset = _apply_debug_mode(
        train_dataset, val_dataset, test_dataset
    )

    quality_reports = {
        "train": check_dataset_quality(train_dataset, "train"),
        "validation": check_dataset_quality(val_dataset, "validation"),
        "test": check_dataset_quality(test_dataset, "test"),
    }
    for report in quality_reports.values():
        _raise_for_critical_quality_issues(report)

    class_distributions = {
        "train": get_class_distribution(train_dataset, "train"),
        "validation": get_class_distribution(val_dataset, "validation"),
        "test": get_class_distribution(test_dataset, "test"),
    }
    class_distributions["train_validation_percentage_gaps"] = _report_train_val_distribution_gap(
        class_distributions["train"], class_distributions["validation"]
    )
    overlap_report = check_split_overlap(train_dataset, val_dataset, test_dataset)

    tokenizer = get_tokenizer()
    token_length_report = analyze_token_lengths(train_dataset, tokenizer)
    tokenized_train, tokenized_val, tokenized_test = tokenize_datasets(
        train_dataset, val_dataset, test_dataset, tokenizer
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenized_train, tokenized_val, tokenized_test, tokenizer
    )
    run_dataloader_sanity_checks(train_loader, val_loader, test_loader)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "tokenized_train": tokenized_train,
        "tokenized_val": tokenized_val,
        "tokenized_test": tokenized_test,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tokenizer": tokenizer,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "quality_reports": quality_reports,
        "class_distributions": class_distributions,
        "overlap_report": overlap_report,
        "token_length_report": token_length_report,
    }


def _parse_args() -> "argparse.Namespace":
    import argparse

    parser = argparse.ArgumentParser(description="Export frozen AG News research splits.")
    parser.add_argument("--export-research-splits", action="store_true", required=True)
    parser.add_argument("--train-path", default=RESEARCH_TRAIN_PATH)
    parser.add_argument("--validation-path", default=RESEARCH_VALIDATION_PATH)
    parser.add_argument("--report-path", default=FINAL_DATA_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = export_research_splits(
        train_path=args.train_path,
        validation_path=args.validation_path,
        report_path=args.report_path,
        overwrite=args.overwrite,
    )
    print(f"overall_status: {report['overall_status']}")
    print(f"READY_FOR_OFFICIAL_TRAINING: {report['READY_FOR_OFFICIAL_TRAINING']}")
    print(f"train_sample_count: {report['train_sample_count']}")
    print(f"validation_sample_count: {report['validation_sample_count']}")
    print(f"Report written to: {args.report_path}")
