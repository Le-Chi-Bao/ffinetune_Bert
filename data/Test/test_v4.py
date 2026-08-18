#!/usr/bin/env python3
import re
import html

def clean_text_v4(text):
    import unicodedata
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    text = unicodedata.normalize("NFKC", text)

    # First: handle #39; (apostrophe) specially
    # "doesn #39;t" -> "doesn't" (no space before or after)
    # The space before #39; becomes the apostrophe position
    text = re.sub(r"(?<=\S) #39;", "'", text)  # word followed by space+#39; -> word'

    # For other numeric entities, add &
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
    ("earned #36;10 million", "earned $10 million"),
    ("The company earned #36;10 million", "The company earned $10 million"),
    ("AT #38;T", "AT & T"),
    ("&lt;strong&gt;Opinion&lt;/strong&gt;", "Opinion"),
    ("First&lt;br&gt;Second", "First Second"),
    ("Microsoft Corp. &lt;MSFT.O&gt;", "Microsoft Corp. MSFT.O"),
    ("dwindling\\band", "dwindling band"),
    ("&lt;br&gt;Bold&lt;/br&gt;", "Bold"),
]

print("Testing v4:")
for input_text, expected in tests:
    result = clean_text_v4(input_text)
    status = "PASS" if result == expected else "FAIL"
    print(f"{status}: {repr(input_text)} -> {repr(result)} (expected: {repr(expected)})")
