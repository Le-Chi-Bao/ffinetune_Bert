import pandas as pd
import re
import html
import unicodedata

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    # Fix missing & in numeric HTML entities like #39; #36; #151;
    text = re.sub(r"(?<!&)#(\d+);", r"&#\1;", text)
    # Unescape HTML entities (e.g. &#39; -> ', &amp; -> &)
    text = html.unescape(text)
    # Preserve stock tickers inside angle brackets like <MSFT.O> -> MSFT.O
    text = re.sub(r"<([A-Za-z0-9_\-\.]+)>", r"\1", text)
    # Remove remaining HTML tags while retaining inner text (e.g. <strong>Opinion</strong> -> Opinion)
    text = re.sub(r"<[^>]+>", "", text)
    # Handle backslash artifacts: \$ -> $, \" -> ", \' -> '
    text = text.replace(r"\$", "$").replace(r'\"', '"').replace(r"\'", "'")
    # Replace remaining backslashes with spaces (e.g. dwindling\band -> dwindling band)
    text = text.replace("\\", " ")
    # Whitespace normalization
    text = " ".join(text.split())
    return text

if __name__ == "__main__":
    train_df = pd.read_csv("raw/train.csv")
    print(f"Loaded {len(train_df)} rows from raw/train.csv")
    for idx, row in train_df.iloc[:5].iterrows():
        print(f"--- SAMPLE {idx} ---")
        print("RAW TITLE:       ", row["Title"])
        print("CLEAN TITLE:     ", clean_text(row["Title"]))
        print("RAW DESCRIPTION: ", row["Description"])
        print("CLEAN DESCRIPTION:", clean_text(row["Description"]))
