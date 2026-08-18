"""Configuration for Stage 1 of the AG News LDTF-BERT project."""

MODEL_NAME = "bert-base-uncased"

NUM_CLASSES = 4
ID2LABEL = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}
LABEL2ID = {label: label_id for label_id, label in ID2LABEL.items()}

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 2

VALIDATION_SIZE = 0.10
SEED = 42

DEBUG = False
DEBUG_TRAIN_SIZE = 4000
DEBUG_VAL_SIZE = 1000
DEBUG_TEST_SIZE = 1000

# Stage-1 inspection settings. They are deliberately small and do not alter data.
NUM_INSPECTION_EXAMPLES = 3
TOKEN_LENGTH_BATCH_SIZE = 1_000

# Stage-2 baseline configuration.
BASELINE_DROPOUT = 0.1

# Stage-3 training and evaluation defaults. CLI arguments in train.py override
# these values for an individual Linux GPU Cloud run.
NUM_EPOCHS = 3
ENCODER_LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
GRAD_ACCUMULATION_STEPS = 1
USE_AMP = True
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.0
MONITOR_METRIC = "f1_macro"

OUTPUT_ROOT = "outputs"
LOG_ROOT = "logs"
SAVE_LAST_CHECKPOINT = True

# Test evaluation remains opt-in at the CLI (--run-test) so the official test
# split cannot accidentally influence architecture or hyperparameter choices.
RUN_TEST_AFTER_TRAINING = False

MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None
MAX_TEST_BATCHES = None
