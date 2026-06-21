"""Task 1: EDA and preprocessing for CFPB complaint narratives.

Usage:
    python src/preprocess.py --input data/raw/complaints.csv.zip --output data/processed/filtered_complaints.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

TARGET_PRODUCTS: Dict[str, List[str]] = {
    "Credit Card": ["credit card", "credit card or prepaid card"],
    "Personal Loan": ["personal loan", "consumer loan"],
    "Savings Account": ["bank account or service", "checking or savings account", "savings account"],
    "Money Transfer": ["money transfer", "money transfers", "money transfer, virtual currency, or money service"],
}

NARRATIVE_COL_CANDIDATES = [
    "Consumer complaint narrative",
    "consumer_complaint_narrative",
    "narrative",
    "complaint_narrative",
]
PRODUCT_COL_CANDIDATES = ["Product", "product"]
ID_COL_CANDIDATES = ["Complaint ID", "complaint_id", "complaint_id_original"]

BOILERPLATE_PATTERNS = [
    r"i am writing to file a complaint",
    r"i am filing this complaint",
    r"this complaint is regarding",
    r"to whom it may concern",
    r"please investigate",
    r"thank you for your assistance",
    r"xxxx+",
]


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first matching column name from candidates, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"None of the expected columns were found: {candidates}")


def normalize_product(product: str) -> str | None:
    """Map raw CFPB product names into the four target product categories."""
    if pd.isna(product):
        return None
    product_l = str(product).strip().lower()
    for category, aliases in TARGET_PRODUCTS.items():
        if product_l in aliases or any(alias in product_l for alias in aliases):
            return category
    return None


def clean_text(text: str) -> str:
    """Clean and normalize complaint narrative text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(str(text).split()) if pd.notna(text) and str(text).strip() else 0


def load_complaints(input_path: str | Path) -> pd.DataFrame:
    """Load CFPB complaints from CSV or zipped CSV."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return pd.read_csv(input_path, low_memory=False)


def run_eda(df: pd.DataFrame, figures_dir: str | Path = "figures") -> pd.DataFrame:
    """Generate EDA summary and save plots."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    product_col = find_column(df, PRODUCT_COL_CANDIDATES)
    narrative_col = find_column(df, NARRATIVE_COL_CANDIDATES)

    df = df.copy()
    df["has_narrative"] = df[narrative_col].notna() & (df[narrative_col].astype(str).str.strip() != "")
    df["narrative_word_count"] = df[narrative_col].fillna("").apply(word_count)

    product_counts = df[product_col].value_counts().head(15)
    plt.figure(figsize=(11, 6))
    product_counts.sort_values().plot(kind="barh")
    plt.title("Top 15 CFPB Complaint Products")
    plt.xlabel("Number of Complaints")
    plt.tight_layout()
    plt.savefig(figures_dir / "product_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    df.loc[df["narrative_word_count"] > 0, "narrative_word_count"].clip(upper=1000).hist(bins=50)
    plt.title("Consumer Narrative Word Count Distribution (clipped at 1000)")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Complaints")
    plt.tight_layout()
    plt.savefig(figures_dir / "narrative_word_count_distribution.png", dpi=160)
    plt.close()

    narrative_counts = df["has_narrative"].value_counts().rename(index={True: "With narrative", False: "Without narrative"})
    plt.figure(figsize=(6, 4))
    narrative_counts.plot(kind="bar")
    plt.title("Complaints With vs Without Consumer Narratives")
    plt.ylabel("Number of Complaints")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "narrative_presence_counts.png", dpi=160)
    plt.close()

    summary = pd.DataFrame({
        "metric": [
            "total_records",
            "records_with_narratives",
            "records_without_narratives",
            "median_word_count_with_narrative",
            "mean_word_count_with_narrative",
        ],
        "value": [
            len(df),
            int(df["has_narrative"].sum()),
            int((~df["has_narrative"]).sum()),
            float(df.loc[df["narrative_word_count"] > 0, "narrative_word_count"].median()),
            float(df.loc[df["narrative_word_count"] > 0, "narrative_word_count"].mean()),
        ],
    })
    summary.to_csv(figures_dir / "eda_summary.csv", index=False)
    return df


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Retain target products, remove empty narratives, and clean narrative text."""
    product_col = find_column(df, PRODUCT_COL_CANDIDATES)
    narrative_col = find_column(df, NARRATIVE_COL_CANDIDATES)

    id_col = None
    try:
        id_col = find_column(df, ID_COL_CANDIDATES)
    except KeyError:
        pass

    df = df.copy()
    df["product_category"] = df[product_col].apply(normalize_product)
    df = df[df["product_category"].notna()].copy()
    df = df[df[narrative_col].notna() & (df[narrative_col].astype(str).str.strip() != "")].copy()
    df["cleaned_narrative"] = df[narrative_col].apply(clean_text)
    df = df[df["cleaned_narrative"].str.split().str.len() >= 5].copy()
    df["narrative_word_count"] = df["cleaned_narrative"].apply(word_count)

    if id_col is None:
        df["complaint_id"] = range(1, len(df) + 1)
    elif id_col != "complaint_id":
        df["complaint_id"] = df[id_col]

    keep_cols = ["complaint_id", "product_category", product_col, narrative_col, "cleaned_narrative", "narrative_word_count"]
    optional_cols = ["Issue", "Sub-issue", "Company", "State", "Date received", "Submitted via"]
    for col in optional_cols:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    return df[keep_cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/complaints.csv.zip")
    parser.add_argument("--output", default="data/processed/filtered_complaints.csv")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    df = load_complaints(args.input)
    run_eda(df, args.figures_dir)
    cleaned = filter_and_clean(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset: {output_path} ({len(cleaned):,} rows)")
    print(cleaned["product_category"].value_counts())


if __name__ == "__main__":
    main()
