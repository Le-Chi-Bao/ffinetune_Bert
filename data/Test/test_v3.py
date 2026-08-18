#!/usr/bin/env python3
import re
import html

# The issue: "doesn #39;t" should become "doesn't"
# But my current code produces "doesn 't"
# The space before #39; should be removed ONLY when followed by a letter (apostrophe case)

def clean_text_v3(text):
    import unicodedata
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    text = unicodedata.normalize("NFKC", text)

    # Special case: #39; (apostrophe) with space before and letter after
    # "word #39;letter" -> "word'letter" (remove space)
    # But #36; (dollar) with space before and number after needs space preserved
    # "earned #36;10" -> "earned $10" (keep space)

    # Strategy: Handle #39; specially
    # Pattern: (letter) #39; -> (letter)'
    text = re.sub(r"(\w) #39;", r"\1'", text)

    # For other numeric entities, just add &
    text = re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text)

    # HTML unescape
    text = html.unescape(text)

    # br handling
    text = re.sub(r"<br\s*/?\s*>", " ", text, flags=re.IGNORECASE)

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Backslash handling
    text = text.replace("\\$", "$").replace('\\"', '"').replace("\\'", "'")
    text = text.replace("\\", " ")

    # Whitespace collapse
    text = " ".join(text.split())

    return text

# Test cases
tests = [
    ("doesn #39;t", "doesn't"),
    ("earned #36;10 million", "$10 million"),
    ("The company earned #36;10 million", "The company earned $10 million"),
    ("AT #38;T", "AT & T"),
    ("&lt;strong&gt;Opinion&lt;/strong&gt;", "Opinion"),
    ("First&lt;br&gt;Second", "First Second"),
    ("Microsoft Corp. &lt;MSFT.O&gt;", "Microsoft Corp. MSFT.O"),
    ("dwindling\\band", "dwindling band"),
]

print("Testing v3:")
for input_text, expected in tests:
    result = clean_text_v3(input_text)
    status = "PASS" if result == expected else "FAIL"
    print(f"{status}: {repr(input_text)} -> {repr(result)} (expected: {repr(expected)})")
