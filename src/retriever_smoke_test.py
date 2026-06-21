"""Small smoke test for the persisted FAISS vector store."""
from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


def search(query: str, vector_dir: str = "vector_store", k: int = 5) -> pd.DataFrame:
    vector_dir = Path(vector_dir)
    index = faiss.read_index(str(vector_dir / "complaints_faiss.index"))
    metadata = pd.read_parquet(vector_dir / "chunk_metadata.parquet")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    results = metadata.iloc[ids[0]].copy()
    results["score"] = scores[0]
    return results[["score", "complaint_id", "product_category", "chunk_index", "text"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Why are customers unhappy with credit cards?")
    args = parser.parse_args()
    print(search(args.query).to_string(index=False))
