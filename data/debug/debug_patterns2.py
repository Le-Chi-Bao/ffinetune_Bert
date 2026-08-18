#!/usr/bin/env python3
"""Debug pattern matching with more detail."""
import pandas as pd
import re

# Load processed data
train = pd.read_parquet('processed/research_train.parquet')

# Patterns
patterns = [
    r"strong[A-Za-z][a-z]+",      # strongOpinion
    r"emb[A-Za-z][a-z]+",         # embTest
    r"br[A-Z][A-Za-z]+",          # brBold
    r"font[A-Za-z][a-z]+",         # fontTest
    r"href[A-Za-z][a-z]+",        # hrefTest
    r"color[A-Za-z][a-z]+",       # colorTest
    r"verdana[A-Za-z][a-z]+",     # verdanaTest
    r"helvetica[A-Za-z][a-z]+",   # helveticaTest
]

for pattern in patterns:
    title_matches = train["title_clean"].str.contains(pattern, regex=True, na=False)
    desc_matches = train["description_clean"].str.contains(pattern, regex=True, na=False)
    total = title_matches.sum() + desc_matches.sum()
    if total > 0:
        print(f"Pattern '{pattern}': {total} matches")
        # Show examples
        for idx, row in train[title_matches].head(2).iterrows():
            # Find the match
            match = re.search(pattern, row['title_clean'], re.IGNORECASE)
            if match:
                print(f"  Title: ...{row['title_clean'][max(0, match.start()-20):match.end()+20]}...")

# Check for lowercase concatenated patterns
lowercase_patterns = ["strongopinion", "brembold", "fontcolor", "brfirst"]
for p in lowercase_patterns:
    matches = train["title_clean"].str.contains(p, case=False, na=False).sum()
    matches += train["description_clean"].str.contains(p, case=False, na=False).sum()
    print(f"Lowercase '{p}': {matches} matches")
