#!/usr/bin/env python3
import re
import html

def clean_text_v6(text):
    import unicodedata
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    text = unicodedata.normalize("NFKC", text)

    # Step 1: Fix numeric HTML entities missing &
    # #39; -> &#39; (then -> '), #36; -> &#36; (then -> $)
    text = re.sub(r"(?<=\S) #39;", "'", text)  # Special case: apostrophe
    text = re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text)

    # Step 2: HTML unescape - this converts &lt; to <, &gt; to >, and &#39; to '
    text = html.unescape(text)

    # Step 3: NOW protect stock tickers (after unescape, we have real < >)
    # Pattern: <ALLCAPS.ticker> like <MSFT.O>, <AAPL.O>
    ticker_pattern = r"<([A-Z][A-Z0-9]*[.-][A-Z]{1,5})>"
    tickers = {}

    def save_ticker(match):
        ticker = match.group(1)
        placeholder = f"__TICKER_{len(tickers)}__{ticker}__"
        tickers[placeholder] = ticker
        return placeholder

    text = re.sub(ticker_pattern, save_ticker, text)

    # Step 4: br handling - convert to space
    text = re.sub(r"<br\s*/?\s*>", " ", text, flags=re.IGNORECASE)

    # Step 5: Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Step 6: Backslash handling
    text = text.replace("\\$", "$").replace('\\"', '"').replace("\\'", "'")
    text = text.replace("\\", " ")

    # Step 7: Whitespace collapse
    text = " ".join(text.split())

    # Step 8: Restore stock tickers
    for placeholder, ticker in tickers.items():
        text = text.replace(placeholder, ticker)

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
    ("&lt;br/&gt;", ""),
    ("&lt;font&gt;Text&lt;/font&gt;", "Text"),
    ("&lt;b&gt;Bold&lt;/b&gt;", "Bold"),
    ("&lt;em&gt;Italic&lt;/em&gt;", "Italic"),
    ("&lt;p&gt;Para&lt;/p&gt;", "Para"),
    ("&lt;MSFT.O&gt;", "MSFT.O"),
    ("&lt;AAPL.O&gt;", "AAPL.O"),
]

print("Testing v6:")
all_pass = True
for input_text, expected in tests:
    result = clean_text_v6(input_text)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"{status}: {repr(input_text)} -> {repr(result)} (expected: {repr(expected)})")

print(f"\n{'ALL TESTS PASSED!' if all_pass else 'SOME TESTS FAILED!'}")
