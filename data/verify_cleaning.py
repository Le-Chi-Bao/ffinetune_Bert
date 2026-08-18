#!/usr/bin/env python3
"""Verify actual HTML artifacts in cleaned data."""
import pandas as pd
import re

# Load processed data
train = pd.read_parquet('processed/research_train.parquet')
test = pd.read_parquet('processed/research_test.parquet')

print(f"Train: {len(train)}, Test: {len(test)}")

# Pattern for actual concatenated artifacts
# Should be: br + capital letter (e.g., brBold)
# NOT: br appearing naturally in words (e.g., "brought", "bribe")

ARTIFACT_PATTERNS = [
    r"br[A-Z][a-z]+",   # brBold, brFirst
    r"strong[A-Z][a-z]+",  # strongOpinion
    r"font[A-Z][a-z]+",    # fontColor
]

# Count actual artifacts
total_artifacts = 0
for pattern in ARTIFACT_PATTERNS:
    train_matches = train["title_clean"].str.contains(pattern, regex=True, na=False).sum()
    train_matches += train["description_clean"].str.contains(pattern, regex=True, na=False).sum()
    test_matches = test["title_clean"].str.contains(pattern, regex=True, na=False).sum()
    test_matches += test["description_clean"].str.contains(pattern, regex=True, na=False).sum()
    total = train_matches + test_matches
    print(f"Pattern '{pattern}': train={train_matches}, test={test_matches}, total={total}")
    total_artifacts += total

print(f"\nTotal actual artifacts: {total_artifacts}")

# Show some examples
print("\n--- Examples of 'br[A-Z][a-z]+' matches ---")
for idx, row in train[train["title_clean"].str.contains(r"br[A-Z][a-z]+", regex=True, na=False)].head(5).iterrows():
    print(f"  {row['title_clean'][:80]}...")

# Check if there's any < or > remaining
has_angle_brackets = train["title_clean"].str.contains(r"<[^>]+>", regex=True, na=False).sum()
has_angle_brackets += train["description_clean"].str.contains(r"<[^>]+>", regex=True, na=False).sum()
print(f"\nRemaining <...> tags in train: {has_angle_brackets}")
