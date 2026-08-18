#!/usr/bin/env python3
"""Debug what patterns are matching."""
import pandas as pd
import re

# Load processed data
train = pd.read_parquet('processed/research_train.parquet')

# Patterns that were counted
patterns = [
    r"strong[A-Za-z]",      # strongOpinion, strongB
    r"emb[A-Za-z]",         # embolden (not an artifact)
    r"br[A-Z][a-z]*",       # brBold, brFirst
    r"brB",                 # literal brB
    r"font[A-Za-z]",        # fontColor
    r"href[A-Za-z]",       # hrefLink
    r"color[A-Za-z]",      # colorText
    r"verdana[A-Za-z]",     # verdanaText
    r"helvetica[A-Za-z]",   # helveticaText
]

for pattern in patterns:
    matches = train["title_clean"].str.contains(pattern, regex=True, na=False).sum()
    matches += train["description_clean"].str.contains(pattern, regex=True, na=False).sum()
    if matches > 0:
        print(f"Pattern '{pattern}': {matches} matches")

        # Show examples
        for idx, row in train[train["title_clean"].str.contains(pattern, regex=True, na=False)].head(2).iterrows():
            print(f"  Title: {row['title_clean'][:80]}...")

# The issue is: "br[A-Z][a-z]*" matches "Br" at start of sentences!
# And "color" is in "colorful", "colored", etc.
# These are false positives!
