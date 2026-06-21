"""Task 2: Stratified sampling, chunking, embeddings, and FAISS vector indexing.

Usage:
    python src/vector_indexing.py --input data/processed/filtered_complaints.csv --sample-size 12000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None


def load_cleaned_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned file not found: {path}. Run src/preprocess.py first.")
    df = pd.read_csv(path)
    required = {"complaint_id", "product_category", "cleaned_narrative"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[df["cleaned_narrative"].notna()].copy()


def stratified_sample(df: pd.DataFrame, sample_size: int = 12000, random_state: int = 42) -> pd.DataFrame:
    """Sample complaints proportionally by product_category."""
    if len(df) <= sample_size:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    fractions = df["product_category"].value_counts(normalize=True)
    samples = []
    for product, frac in fractions.items():
        n = max(1, round(sample_size * frac))
        group = df[df["product_category"] == product]
        samples.append(group.sample(n=min(n, len(group)), random_state=random_state))
    sampled = pd.concat(samples).sample(frac=1, random_state=random_state).reset_index(drop=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return sampled


def chunk_texts(df: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 50) -> pd.DataFrame:
    """Split complaint narratives into overlapping chunks with traceable metadata."""
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_fn = splitter.split_text
    else:
        def split_fn(text: str) -> List[str]:
            step = max(1, chunk_size - chunk_overlap)
            return [text[i:i + chunk_size] for i in range(0, len(text), step)]

    rows: List[Dict] = []
    for _, row in df.iterrows():
        chunks = [c.strip() for c in split_fn(str(row["cleaned_narrative"])) if c.strip()]
        for i, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{row['complaint_id']}_{i}",
                "complaint_id": row["complaint_id"],
                "product_category": row["product_category"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
            })
    if not rows:
        raise ValueError("No chunks were created. Check narrative text and chunk settings.")
    return pd.DataFrame(rows)


def build_faiss_index(chunks: pd.DataFrame, model_name: str, output_dir: str | Path, batch_size: int = 128) -> None:
    """Generate sentence-transformer embeddings and persist FAISS index plus metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        chunks["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "complaints_faiss.index"))

    chunks.to_parquet(output_dir / "chunk_metadata.parquet", index=False)
    config = {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "num_chunks": int(len(chunks)),
        "index_type": "faiss.IndexFlatIP",
        "similarity": "cosine via normalized inner product",
    }
    with open(output_dir / "index_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved FAISS index and metadata to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/filtered_complaints.csv")
    parser.add_argument("--output-dir", default="vector_store")
    parser.add_argument("--sample-output", default="data/processed/stratified_sample.csv")
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    df = load_cleaned_data(args.input)
    sample = stratified_sample(df, sample_size=args.sample_size)
    Path(args.sample_output).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.sample_output, index=False)
    print("Stratified sample distribution:")
    print(sample["product_category"].value_counts(normalize=True).round(3))

    chunks = chunk_texts(sample, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    build_faiss_index(chunks, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
