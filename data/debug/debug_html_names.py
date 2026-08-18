#!/usr/bin/env python3
"""Debug HTML tag names detection."""
import pandas as pd
import re

# Suspicious tokens
SUSPICIOUS_HTML_TOKENS = [
    "strong", "em", "b", "i", "p", "br", "font", "href", "color",
    "verdana", "helvetica", "strongopinion", "brb", "brB"
]

# Load cleaned data
train = pd.read_parquet('processed/research_train.parquet')
print(f"Total rows: {len(train)}")

# Check for each token
for token in SUSPICIOUS_HTML_TOKENS[:5]:
    count_title = train["title_clean"].str.contains(token, case=False, na=False).sum()
    count_desc = train["description_clean"].str.contains(token, case=False, na=False).sum()
    print(f"'{token}': title={count_title}, desc={count_desc}")

# Check what "strong" actually matches
print("\n--- Checking 'strong' matches in title ---")
strong_matches = train[train["title_clean"].str.contains("strong", case=False, na=False)]
for idx, row in strong_matches.head(3).iterrows():
    print(f"  {row['title_clean'][:100]}...")

# Check for actual HTML tag artifacts like "strongOpinion"
print("\n--- Checking 'strongopinion' (joined) ---")
joined = train["title_clean"].str.contains("strongopinion", case=False, na=False).sum()
print(f"Rows with 'strongopinion' joined: {joined}")

# Check the pattern "strong" followed by capital letter
print("\n--- Checking 'strong[A-Z]' pattern ---")
pattern = r"strong[A-Z]"
matches = train["title_clean"].str.contains(pattern, regex=True, na=False).sum()
print(f"Rows with 'strong' followed by capital letter: {matches}")
