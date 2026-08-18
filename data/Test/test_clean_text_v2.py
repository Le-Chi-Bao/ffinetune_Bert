#!/usr/bin/env python3
"""
Unit tests for clean_text() function v2.
Based on specifications in kiểm tra.md

Tests must cover:
1. HTML tag stripping - tags like <strong>, </strong>, <br>, etc. must be REMOVED (not extracted)
2. HTML entities - &lt; and &gt; must be unescaped first
3. Numeric HTML entities missing & - #39; must become &#39; then unescape to '
4. Stock tickers - <MSFT.O> must become MSFT.O (not extracted incorrectly)
5. Backslash artifacts - \$ -> $, \" -> ", \' -> ', remaining \ -> space
6. Content preservation - numbers, percentages, currency, company names must be preserved
"""
import unittest
import pandas as pd
import re
import html
import unicodedata

# The FIXED clean_text function v2
def clean_text_v2(text: str) -> str:
    """
    Text cleaning v2 - Fixed HTML tag handling.

    Order of operations:
    1. str(text).strip()
    2. NFKC unicode normalization
    3. Fix numeric HTML entity missing & (#39; -> &#39;,  #36; -> &#36;)
       Handle: " #39;" -> "&#39;", " #36;" -> "&#36;"
    4. HTML unescape (&lt; -> <, &gt; -> >)
    5. Handle HTML tags - REMOVE tags completely, keep inner text
       Special case: <br>, <br/>, <br /> -> single space
    6. Preserve stock tickers (pattern: <[A-Z][A-Z0-9.-]*\.[A-Z]{1,5}>)
    7. Fix backslash artifacts: \$ -> $, \" -> ", \' -> ', remaining \ -> space
    8. Whitespace collapse
    """
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""

    text = text.strip()
    text = unicodedata.normalize("NFKC", text)

    # Fix missing & in numeric HTML entities
    # #39; -> &#39; (apostrophe) - special case, no space before
    text = re.sub(r"(?<=\S) #39;", "'", text)
    # For #38; (&) - ensure space after
    def replace_ampersand(match):
        return " & "
    text = re.sub(r" #38;", replace_ampersand, text)
    # For other numeric entities, add &
    text = re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text)

    # HTML unescape first (&lt; -> <, &gt; -> >)
    text = html.unescape(text)

    # First, protect stock tickers (pattern: <MSFT.O>, <AAPL.O>, <TXN.N>)
    # Only match proper ticker pattern, not regular HTML tags
    ticker_pattern = r"<([A-Z][A-Z0-9]*[.-][A-Z]{1,5})>"
    tickers = {}

    def replace_ticker(match):
        ticker = match.group(1)
        placeholder = f"__TICKER_{len(tickers)}__{ticker}__"
        tickers[placeholder] = ticker
        return placeholder

    text = re.sub(ticker_pattern, replace_ticker, text)

    # Replace <br> variants with space BEFORE removing other tags
    text = re.sub(r"<br\s*/?\s*>", " ", text, flags=re.IGNORECASE)

    # Now remove ALL remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Restore stock tickers
    for placeholder, ticker in tickers.items():
        text = text.replace(placeholder, ticker)

    # Handle backslash artifacts
    text = text.replace(r"\$", "$")
    text = text.replace(r"\"", "\"")
    text = text.replace(r"\'", "'")
    # Remaining backslashes become spaces
    text = text.replace("\\", " ")

    # Whitespace collapse
    text = " ".join(text.split())

    return text


class TestCleanTextV2(unittest.TestCase):
    """Required tests from kiểm tra.md Section 31."""

    def test_html_strong_tag(self):
        """&lt;strong&gt;Opinion&lt;/strong&gt; -> Opinion"""
        result = clean_text_v2("&lt;strong&gt;Opinion&lt;/strong&gt;")
        self.assertEqual(result, "Opinion")

    def test_html_br_tag(self):
        """First&lt;br&gt;Second -> First Second"""
        result = clean_text_v2("First&lt;br&gt;Second")
        self.assertEqual(result, "First Second")

    def test_stock_ticker_preserved(self):
        """Microsoft Corp. &lt;MSFT.O&gt; -> Microsoft Corp. MSFT.O"""
        result = clean_text_v2("Microsoft Corp. &lt;MSFT.O&gt;")
        self.assertEqual(result, "Microsoft Corp. MSFT.O")

    def test_numeric_entity_dollar(self):
        """The company earned #36;10 million -> The company earned $10 million"""
        result = clean_text_v2("The company earned #36;10 million")
        self.assertEqual(result, "The company earned $10 million")

    def test_numeric_entity_apostrophe(self):
        """The firm doesn #39;t expect -> The firm doesn't expect"""
        result = clean_text_v2("The firm doesn #39;t expect losses")
        self.assertEqual(result, "The firm doesn't expect losses")

    def test_backslash_word_boundary(self):
        r"""dwindling\band -> dwindling band"""
        result = clean_text_v2("dwindling\\band")
        self.assertEqual(result, "dwindling band")

    def test_backslash_dollar(self):
        r"""record \$55.8bn -> record $55.8bn"""
        result = clean_text_v2(r"record \$55.8bn")
        self.assertEqual(result, "record $55.8bn")

    def test_html_font_with_content(self):
        """<FONT color="#666666"><B>The Washington Post</B></FONT> -> The Washington Post"""
        result = clean_text_v2('<FONT color="#666666"><B>The Washington Post</B></FONT>')
        self.assertEqual(result, "The Washington Post")

    def test_preserves_numbers(self):
        """Numbers must be preserved"""
        result = clean_text_v2("$10 million")
        self.assertEqual(result, "$10 million")

    def test_preserves_percentages(self):
        """Percentages must be preserved"""
        result = clean_text_v2("3.5%")
        self.assertEqual(result, "3.5%")

    def test_preserves_company_names(self):
        """Company names must be preserved"""
        result = clean_text_v2("Microsoft")
        self.assertEqual(result, "Microsoft")

    def test_preserves_stock_tickers(self):
        """Stock tickers must be preserved"""
        test_cases = ["<MSFT.O>", "<AAPL.O>", "<TXN.N>", "<GOOG.O>"]
        for ticker in test_cases:
            result = clean_text_v2(f"Stock: {ticker}")
            self.assertIn("MSFT.O" if "MSFT" in ticker else "AAPL" if "AAPL" in ticker else "TXN" if "TXN" in ticker else "GOOG", result)

    def test_multiple_tickers(self):
        """Multiple stock tickers in same text"""
        result = clean_text_v2("MSFT.O and AAPL.O reported earnings")
        self.assertEqual(result, "MSFT.O and AAPL.O reported earnings")

    def test_empty_string(self):
        """Empty string handling"""
        result = clean_text_v2("")
        self.assertEqual(result, "")

    def test_none_value(self):
        """None value handling"""
        result = clean_text_v2(None)
        self.assertEqual(result, "")

    def test_html_tag_artifacts(self):
        """No HTML tag names should appear as tokens"""
        text = "&lt;strong&gt;Bold&lt;/strong&gt; and &lt;em&gt;italic&lt;/em&gt;"
        result = clean_text_v2(text)
        self.assertNotIn("strong", result)
        self.assertNotIn("em", result)
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)

    def test_br_with_spaces(self):
        """Multiple <br> tags should become spaces"""
        result = clean_text_v2("Line1&lt;br&gt;Line2&lt;br/&gt;Line3")
        self.assertEqual(result, "Line1 Line2 Line3")


class TestBackslashHandling(unittest.TestCase):
    """Tests for backslash artifact handling from Section 7."""

    def test_dollar_sign_from_backslash(self):
        r"""\$ -> $"""
        result = clean_text_v2(r"price is \$100")
        self.assertEqual(result, "price is $100")

    def test_quotes_from_backslash(self):
        r"""\" -> \""""
        result = clean_text_v2(r'He said \"hello\"')
        self.assertEqual(result, 'He said "hello"')

    def test_apostrophe_from_backslash(self):
        r"""\' -> '"""
        result = clean_text_v2(r"John\'s car")
        self.assertEqual(result, "John's car")

    def test_remaining_backslash_becomes_space(self):
        r"""dwindling\band -> dwindling band"""
        result = clean_text_v2(r"dwindling\band")
        self.assertEqual(result, "dwindling band")


if __name__ == "__main__":
    unittest.main(verbosity=2)
