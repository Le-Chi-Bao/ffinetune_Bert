"""
Verify rằng data/processed/research_*.parquet map đúng với agnews_ldtf contract.

Không import agnews_ldtf.data (vì cần 'datasets' chưa cài).
Chỉ check cột 'text' + 'label' đúng theo REQUIRED_RAW_COLUMNS = {'text', 'label'}.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED = {"text", "label"}
EXPECTED_ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def check(name: str) -> bool:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        print(f"[FAIL] {name}: thieu file {p}")
        return False

    df = pd.read_parquet(p)
    cols = set(df.columns)
    missing = REQUIRED - cols
    if missing:
        print(f"[FAIL] {name}: thieu cot {missing}")
        return False

    label_vals = sorted(df["label"].unique())
    if not all(0 <= int(v) <= 3 for v in label_vals):
        print(f"[FAIL] {name}: label ngoai [0,3] -> {label_vals}")
        return False

    # Check text column
    sample = df.iloc[0]
    text = sample["text"]
    if not isinstance(text, str) or len(text) < 10:
        print(f"[FAIL] {name}: text khong hop le -> {text!r}")
        return False

    # Check label_id -> label_name dinh dang (optional)
    if "label_name" in df.columns:
        label_names = set(df["label_name"].unique())
        expected = set(EXPECTED_ID2LABEL.values())
        if not label_names.issubset(expected):
            print(f"[WARN] {name}: label_name chua gia tri ngoai {expected}: {label_names}")

    print(f"[OK ] {name:22} | rows={len(df):>6} | label={label_vals} | "
          f"text[0][:80]={text[:80]!r}")
    return True


def main() -> int:
    ok = all(check(n) for n in ("research_train", "research_validation", "research_test"))
    print()
    if ok:
        print(">>> TAT CA 3 FILE mapping DUNG voi agnews_ldtf contract.")
        print(">>> Ban co the chay: cd agnews_ldtf/agnews_ldtf && python scripts/...")
    else:
        print(">>> CO FILE chua mapping. Sua lai!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
