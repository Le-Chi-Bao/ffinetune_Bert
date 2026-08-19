"""Dataset loaders for AG News classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .config import (
    EVAL_BATCH_SIZE,
    LABEL_COLUMN,
    MAX_LENGTH,
    TEXT_COLUMN,
    ensure_output_dirs,
)


@dataclass(frozen=True)
class TokenizedSample:
    """A single tokenized example with all tensors required by the model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    label: torch.Tensor


class AgNewsDataset(Dataset):
    """In-memory dataset that lazily tokenizes pandas rows on first access."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = MAX_LENGTH,
    ) -> None:
        """Validate the dataframe and store tokenizer + length configuration."""
        super().__init__()
        if TEXT_COLUMN not in dataframe.columns:
            raise ValueError(
                f"Dataframe must contain a '{TEXT_COLUMN}' column; got {list(dataframe.columns)}."
            )
        if LABEL_COLUMN not in dataframe.columns:
            raise ValueError(
                f"Dataframe must contain a '{LABEL_COLUMN}' column; got {list(dataframe.columns)}."
            )
        self.texts: Sequence[str] = dataframe[TEXT_COLUMN].astype(str).tolist()
        self.labels: Sequence[int] = dataframe[LABEL_COLUMN].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.texts)

    def __getitem__(self, index: int) -> TokenizedSample:
        """Tokenize a single sample and return a TokenizedSample dataclass."""
        text = self.texts[index]
        label = self.labels[index]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return TokenizedSample(
            input_ids=encoded["input_ids"].squeeze(0),
            attention_mask=encoded["attention_mask"].squeeze(0),
            token_type_ids=encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).squeeze(0),
            label=torch.tensor(int(label), dtype=torch.long),
        )


def _collate_batch(batch: list[TokenizedSample]) -> dict[str, torch.Tensor]:
    """Stack a list of TokenizedSample into a single batch dictionary."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "input_ids": torch.stack([sample.input_ids for sample in batch], dim=0),
        "attention_mask": torch.stack([sample.attention_mask for sample in batch], dim=0),
        "token_type_ids": torch.stack([sample.token_type_ids for sample in batch], dim=0),
        "labels": torch.stack([sample.label for sample in batch], dim=0),
    }


def load_split(path: str | Path) -> pd.DataFrame:
    """Read a parquet file and return a DataFrame with the expected columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected parquet file at {path}, but it does not exist.")
    dataframe = pd.read_parquet(path)
    return dataframe.reset_index(drop=True)


def build_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap *dataframe* in a DataLoader with the project's collate function."""
    dataset = AgNewsDataset(dataframe=dataframe, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_batch,
        drop_last=False,
    )


def make_train_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    batch_size: int | None = None,
) -> DataLoader:
    """Create a shuffled training DataLoader using the default batch size."""
    return build_dataloader(
        dataframe=dataframe,
        tokenizer=tokenizer,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=True,
    )


def make_eval_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    batch_size: int | None = None,
) -> DataLoader:
    """Create a non-shuffled evaluation DataLoader using the eval batch size."""
    return build_dataloader(
        dataframe=dataframe,
        tokenizer=tokenizer,
        batch_size=batch_size or EVAL_BATCH_SIZE,
        shuffle=False,
    )


def build_all_dataloaders(
    tokenizer,
    *,
    batch_size: int | None = None,
) -> dict[str, DataLoader]:
    """Return train/val/test DataLoaders, ensuring output directories exist."""
    ensure_output_dirs()
    train_df = load_split(config.PROCESSED_TRAIN)
    val_df = load_split(config.PROCESSED_VAL)
    test_df = load_split(config.PROCESSED_TEST)
    return {
        "train": make_train_dataloader(train_df, tokenizer, batch_size=batch_size),
        "validation": make_eval_dataloader(val_df, tokenizer, batch_size=batch_size),
        "test": make_eval_dataloader(test_df, tokenizer, batch_size=batch_size),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
