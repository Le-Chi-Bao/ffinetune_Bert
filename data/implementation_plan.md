# Implementation Plan - AG News Data Preparation Pipeline

Implement the complete AG News dataset processing pipeline described in `data/dataset.pdf` (Steps 0 through 18) by creating a unified Python script `scripts/prepare_agnews.py`.

## User Review Required

> [!IMPORTANT]
> - Raw data files (`train.csv` and `test.csv`) are located in `data/raw/` and will **never** be edited directly.
> - `scripts/prepare_agnews.py` will serve as the single executable script supporting both `--mode audit-only` and `--mode build`.
> - All intermediate and final output files will be written to `data/processed/`, `data/manifests/`, and `data/reports/`.

## Proposed Changes

### Data Pipeline Script

#### [NEW] [prepare_agnews.py](file:///e:/Desktop/C%C3%A1c%20m%C3%B4n%20h%E1%BB%8Dc%20n%C4%83m%202/H%E1%BB%8Dc%20S%C3%A2u/data/scripts/prepare_agnews.py)

Implement the single unified script containing the required functions:
- `compute_raw_checksums()`: Generate SHA-256 hashes in `data/manifests/raw_checksums.txt`.
- `load_raw_csv()` & `validate_schema()`: Validate 3-column CSV structure, check missing fields/malformed rows, assign `row_id`. Save `data/reports/schema_report.json`.
- `remap_labels()`: Map `1..4` to `0..3` (`0: World`, `1: Sports`, `2: Business`, `3: Sci/Tech`).
- `clean_text()`: Standardize text (NFKC unicode, fix numeric HTML entities missing `&`, HTML unescape, strip HTML tags while keeping tickers like `<MSFT.O>`, clean backslash artifacts `\$`, `\"`, `\'`, `\`, whitespace collapse).
- `generate_cleaning_samples()`: Export 50 samples per class to `data/reports/cleaning_samples.csv` and `data/reports/cleaning_report.json`.
- `create_text_hash()` & `audit_exact_duplicates()`: Compute `text_hash = sha256(canonical_title + "\n" + canonical_description)`. Audit exact duplicates (same-label deduplication vs conflicting-label removal). Export `exact_duplicates.csv`, `conflicting_duplicates.csv`, and `removed_rows.csv`.
- `save_processed_data()`: Export Protocol A (`standard_train.parquet`, `standard_test.parquet`) and Protocol B pool (`research_clean_pool.parquet`).
- `build_near_duplicate_groups()`: MinHash / character n-gram LSH near-duplicate clustering at threshold `0.90`. Export `near_duplicate_pairs.csv` and `near_duplicate_groups.parquet`.
- `audit_train_test_overlap()`: Check train vs official test exact and near-duplicate contamination. Export `decontaminated_test_ids.json` and `split_overlap_report.json`.
- `create_group_stratified_split()`: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` grouping by near-duplicate cluster and stratifying by label. Export `research_train.parquet`, `research_validation.parquet`, `research_test.parquet`, and IDs JSON manifests + `split_metadata.json`.
- `analyze_token_lengths()`: `bert-base-uncased` pair tokenization length distribution analysis. Export `token_length_report.json` and `token_length_distribution.csv`.
- `run_batch_sanity_checks()`: PyTorch DataLoader & DataCollatorWithPadding sanity check (`truncation="only_second"`, `max_length=128`, dynamic padding).
- `save_reports()` & PASS Gate: Export `class_distribution.csv`, `final_data_report.json`, and check 15 PASS conditions.

## Verification Plan

### Automated Verification
1. Run audit mode:
   ```bash
   python scripts/prepare_agnews.py --train data/raw/train.csv --test data/raw/test.csv --mode audit-only
   ```
2. Run build mode:
   ```bash
   python scripts/prepare_agnews.py --train data/raw/train.csv --test data/raw/test.csv --mode build --validation-ratio 0.10 --seed 42 --max-length 128 --near-duplicate-threshold 0.90
   ```
3. Verify all output files in `data/processed/`, `data/manifests/`, and `data/reports/`.
