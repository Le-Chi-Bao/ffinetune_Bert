"""
Thêm cột 'text' vào 3 file research_*.parquet để tương thích với agnews_ldtf.

Công thức:
    text = clean_text(title_clean + " " + description_clean)

Trong đó clean_text = " ".join(str(text).strip().split())
     (giống hệt agnews_ldtf/data.py:clean_text)

Input : data/processed/research_{train,validation,test}.parquet
Output: data/processed/research_{train,validation,test}.parquet (ghi đè)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["research_train", "research_validation", "research_test"]


def clean_text(text: str) -> str:
    """Giống hệt agnews_ldtf/data.py:clean_text (whitespace normalization)."""
    return " ".join(str(text).strip().split())


def add_text_column(parquet_path: Path) -> None:
    if not parquet_path.exists():
        print(f"[skip] File không tồn tại: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    n_before = len(df.columns)

    # Lấy title_clean / description_clean làm nguồn chính,
    # fallback về Title / Description nếu thiếu.
    title = df["title_clean"] if "title_clean" in df.columns else df["Title"]
    desc = df["description_clean"] if "description_clean" in df.columns else df["Description"]

    text_series = (title.astype(str) + " " + desc.astype(str)).map(clean_text)

    # Tránh ghi đè nếu cột 'text' đã tồn tại với giá trị giống
    if "text" in df.columns:
        if df["text"].tolist() == text_series.tolist():
            print(f"[ok ] {parquet_path.name} | text đã tồn tại, khớp nội dung")
            return
        print(f"[upd] {parquet_path.name} | cột 'text' ĐÃ CÓ nhưng khác -> ghi đè")

    # Đặt 'text' ngay sau 'label_name' để dễ nhìn
    df["text"] = text_series

    # Rearrange: label_name, text, ...
    preferred_order = [
        "label", "label_name", "text",
        "Title", "Description",
        "title_raw", "description_raw",
        "title_clean", "description_clean",
        "row_id", "text_hash", "near_duplicate_group_id",
        "Class Index", "original_class_index",
    ]
    cols = [c for c in preferred_order if c in df.columns] + \
           [c for c in df.columns if c not in preferred_order]
    df = df[cols]

    df.to_parquet(parquet_path, index=False)
    print(f"[done] {parquet_path.name} | cols {n_before} -> {len(df.columns)} | "
          f"text[0][:80] = {text_series.iloc[0][:80]!r}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        add_text_column(DATA_DIR / f"{split}.parquet")


if __name__ == "__main__":
    main()
