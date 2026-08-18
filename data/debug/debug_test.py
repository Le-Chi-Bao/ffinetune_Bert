#!/usr/bin/env python3
import re
import html
import unicodedata

def clean_text_v2_debug(text):
    print(f"Input: {repr(text)}")
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    print(f"After NFKC: {repr(text)}")

    # Fix missing &
    text = re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text)
    print(f"After entity fix: {repr(text)}")

    # HTML unescape
    text = html.unescape(text)
    print(f"After unescape: {repr(text)}")

    # br handling
    text = re.sub(r"<br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    print(f"After br fix: {repr(text)}")

    # Remove tags
    text = re.sub(r"<[^>]+>", "", text)
    print(f"After tag removal: {repr(text)}")

    # Backslash
    text = text.replace("\\$", "$").replace('\\"', '"').replace("\\'", "'")
    text = text.replace("\\", " ")
    print(f"After backslash: {repr(text)}")

    # Whitespace collapse
    text = " ".join(text.split())
    print(f"Final: {repr(text)}")
    return text

# Test the failing case
result = clean_text_v2_debug("The firm doesn #39;t expect losses")
