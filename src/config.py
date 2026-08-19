"""Central configuration for AG News classification with LDTF-BERT."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REFERENCES_DIR = ROOT_DIR / "references"

PROCESSED_TRAIN = PROCESSED_DIR / "research_train.parquet"
PROCESSED_VAL = PROCESSED_DIR / "research_validation.parquet"
PROCESSED_TEST = PROCESSED_DIR / "research_test.parquet"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUM_CLASSES = 4
LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")

# ---------------------------------------------------------------------------
# Tokenizer / Model
# ---------------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
PAD_TOKEN_ID = 0

# Custom LDTF module sizes
ROUTER_DIM = 256
PROJECTION_BIAS = False

# ---------------------------------------------------------------------------
# Training (fine-tune baseline)
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
GRAD_ACCUM_STEPS = 1
LABEL_SMOOTHING = 0.0

# ---------------------------------------------------------------------------
# Output sub-directories
# ---------------------------------------------------------------------------
FINETUNE_OUTPUT = OUTPUTS_DIR / "finetune"
FROZEN_OUTPUT = OUTPUTS_DIR / "frozen"


def ensure_output_dirs() -> None:
    """Create every on-disk directory used during training and reporting."""
    for directory in (
        FINETUNE_OUTPUT,
        FROZEN_OUTPUT,
        FIGURES_DIR,
    ):
        os.makedirs(directory, exist_ok=True)
