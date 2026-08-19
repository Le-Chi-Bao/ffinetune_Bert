# LDTF-BERT on AG News — Fine-tune vs Frozen Encoder

A small reproduction study of the **class-conditioned attention** idea for
news-topic classification. We add three custom modules on top of
`bert-base-uncased`:

| Module | Role |
|---|---|
| `LabelQueryBank`     | One learnable vector per class. |
| `TokenRouter`        | Class-conditioned token attention over each hidden layer. |
| `DepthRouter`        | Layer attention that fuses token features across BERT depths. |
| `ClassScorer`        | MLP head producing one logit per class. |

Two experiments are run side-by-side:

* **Fine-tune** — the entire BERT backbone plus the custom modules receive
  gradient updates.
* **Frozen encoder** — BERT weights are frozen; only the routers and scorer
  train.

## Repository layout

```
.
├── src/
│   ├── config.py                # Paths, hyper-parameters, constants
│   ├── dataset.py               # parquet → DataLoader
│   ├── utils.py                 # seed, JSON I/O, timer, device
│   ├── train.py                 # training loop with freeze toggle
│   ├── evaluate.py              # checkpoint load + metric computation
│   └── models/
│       ├── bert_backbone.py     # Pretrained BERT, returns all hidden states
│       ├── label_query_bank.py  # [C, D] learnable queries
│       ├── token_router.py      # class-conditioned token attention
│       ├── depth_router.py      # class-conditioned layer attention
│       ├── class_scorer.py      # [B, C, D] -> [B, C] logits
│       └── ldtf_bert.py         # end-to-end model
├── experiments/
│   ├── run_finetune.py          # full fine-tune experiment
│   ├── run_frozen.py            # frozen-encoder experiment
│   └── compare_results.py       # build comparison CSV/MD/PNGs
├── notebooks/
│   └── colab_finetune_vs_frozen.ipynb    # 10-cell Colab notebook
├── references/
│   └── agnews_ldtf/             # read-only copies of the original paper code
├── reports/                       # populated by the comparison script
│   ├── comparison.csv
│   ├── comparison.md
│   └── figures/
├── outputs/                       # checkpoints + JSON logs
├── data/processed/                # parquet splits (gitignored)
├── requirements.txt
└── README.md
```

## Quick start (local)

```bash
pip install -r requirements.txt

# Train both experiments end-to-end
python experiments/run_finetune.py
python experiments/run_frozen.py

# Aggregate results into reports/
python experiments/compare_results.py
```

Each `run_*.py` script writes its own `outputs/<experiment>/best_model.pt`
plus `train_log.json`, `val_metrics.json`, `test_metrics.json`.

## Quick start (Google Colab Pro)

1. Copy this repository to `MyDrive/HocSau_LDTF_BERT/` on your Drive.
2. Open `notebooks/colab_finetune_vs_frozen.ipynb` in Colab.
3. Set the runtime to **GPU → T4** (Colab Pro).
4. Run cells top-to-bottom. Each cell is idempotent and writes its outputs
   back to Drive.

## Outputs

After a successful run you will have:

* `outputs/finetune/` — best checkpoint and per-epoch history (fine-tune).
* `outputs/frozen/`   — best checkpoint and per-epoch history (frozen).
* `reports/comparison.csv` — single-row-per-experiment summary.
* `reports/comparison.md`  — Markdown version for the paper.
* `reports/figures/loss_curve.png` — train loss + val accuracy.
* `reports/figures/confusion.png`  — confusion matrices side by side.

## Configuration

Hyperparameters live in `src/config.py`. To re-tune the model:

```python
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
SEED = 42
```

## Reference code

The `references/agnews_ldtf/` folder contains four files copied verbatim
from the original `agnews_ldtf/` project (kept at the repo root for
historical context). Use them as a read-only side-by-side comparison while
inspecting our own implementations in `src/models/`.