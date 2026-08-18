#!/usr/bin/env python3
"""Test script for clean_text() function to verify bugs."""
import pandas as pd
import re
import html
import unicodedata

def clean_text_current(text):
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<!&#)#(\d+);", r"&#\1;", text)
    text = html.unescape(text)
    # BUG: This regex extracts tag content and removes <>
    # <strong> -> strong, </strong> -> /strong, <br> -> br
    text = re.sub(r"<([A-Za-z0-9_\-\.]+)>", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace(r"\$", "$").replace(r"\"", "\"").replace(r"\'", "'")
    text = text.replace("\\", " ")
    text = " ".join(text.split())
    return text

# Test cases from the spec
test_cases = [
    ("&lt;strong&gt;Opinion&lt;/strong&gt;", "Opinion"),
    ("First&lt;br&gt;Second", "First Second"),
    ("Microsoft Corp. &lt;MSFT.O&gt;", "Microsoft Corp. MSFT.O"),
    ("The company earned #36;10 million", "The company earned $10 million"),
    ("The firm doesn #39;t expect losses", "The firm doesn't expect losses"),
]

print("=== Testing current clean_text() ===")
for raw, expected in test_cases:
    result = clean_text_current(raw)
    status = "PASS" if result == expected else "FAIL"
    print(f"{status}: \"{raw}\" -> \"{result}\" (expected: \"{expected}\")")

print("\n=== Additional tests for edge cases ===")
print(f"clean_text('dwindling\\\\band'): '{clean_text_current('dwindling\\band')}'")
print(f"clean_text('record \\$55.8bn'): '{clean_text_current('record $55.8bn')}'")
