#!/usr/bin/env python3
"""Check for actual HTML tag name artifacts in cleaned data."""
import pandas as pd
import re

# Load cleaned data
train = pd.read_parquet('processed/research_train.parquet')
print(f"Total rows: {len(train)}")

# Check for actual concatenated artifacts like:
# "strongOpinion", "brB", "brFirst", "fontText", etc.

# Pattern: HTML tag name directly followed by a word character (no space)
artifacts = []

for idx, row in train.head(1000).iterrows():
    text = str(row.get("title_clean", "")) + " " + str(row.get("description_clean", ""))

    # Check for common patterns
    patterns = [
        r"strong[A-Za-z]",   # strongOpinion
        r"br[A-Za-z]",       # brFirst, brB
        r"font[A-Za-z]",     # fontColor
        r"href[A-Za-z]",     # hrefSomething
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            artifacts.append({
                "row_id": row.get("row_id", idx),
                "pattern": pattern,
                "match": matches[0],
                "text": text[:200]
            })

print(f"Found {len(artifacts)} potential artifacts in 1000 rows")
for a in artifacts[:5]:
    print(f"  Pattern: {a['pattern']}, Match: {a['match']}")
    print(f"  Text: {a['text'][:100]}...")
    print()

# Check raw data to see if there were HTML tags
print("\n--- Checking raw data for HTML ---")
raw_train = pd.read_csv('raw/train.csv')
has_html = raw_train["Title"].str.contains("&lt;|&gt;|<[^>]+>", regex=True, na=False).sum()
print(f"Rows with HTML in raw Title: {has_html}")
