# Intelligent Complaint Analysis for Financial Services

Week 7 Interim Submission

## Completed Tasks

- Task 1: EDA and Data Preprocessing
- Task 2: Chunking, Embedding, and Vector Store Indexing

## Technologies

- Python
- Pandas
- LangChain
- Sentence Transformers
- FAISS


This repository contains the interim submission for **B9W7: Intelligent Complaint Analysis for Financial Services**. The project builds the first two stages of a Retrieval-Augmented Generation (RAG) system for CrediTrust Financial: complaint data exploration/preprocessing and vector-store indexing.

## Business Context

CrediTrust Financial receives large volumes of customer complaints across credit cards, personal loans, savings accounts, and money transfers. The goal is to convert raw complaint narratives into a searchable knowledge base that product, support, compliance, and leadership teams can query using natural language.

## Repository Structure

```text
rag-complaint-chatbot/
├── .github/workflows/       # CI workflow
├── data/
│   ├── raw/                 # Raw CFPB dataset, not committed
│   └── processed/           # Cleaned outputs, not committed if large
├── figures/                 # EDA charts and summaries
├── notebooks/               # EDA notebook
├── reports/                 # Interim report
├── src/                     # Reusable Python modules
├── tests/                   # Basic tests
├── vector_store/            # Persisted FAISS/Chroma index, not committed if large
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the CFPB complaints file in:

```text
data/raw/complaints.csv.zip
```

## Run Task 1

```bash
python src/preprocess.py \
  --input data/raw/complaints.csv.zip \
  --output data/processed/filtered_complaints.csv \
  --figures-dir figures
```

Expected outputs:

```text
data/processed/filtered_complaints.csv
figures/product_distribution.png
figures/narrative_word_count_distribution.png
figures/narrative_presence_counts.png
figures/eda_summary.csv
```

## Run Task 2

```bash
python src/vector_indexing.py \
  --input data/processed/filtered_complaints.csv \
  --sample-size 12000 \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --output-dir vector_store
```

Expected outputs:

```text
data/processed/stratified_sample.csv
vector_store/complaints_faiss.index
vector_store/chunk_metadata.parquet
vector_store/index_config.json
```

## Smoke Test Retrieval

```bash
python src/retriever_smoke_test.py "Why are customers unhappy with credit cards?"
```

## Technical Choices

- **Cleaning:** Lowercasing, special-character removal, boilerplate phrase removal, whitespace normalization.
- **Sampling:** Stratified proportional sampling to preserve representation across the four product categories.
- **Chunking:** 500-character chunks with 50-character overlap to balance semantic context and retrieval precision.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`, selected because it is lightweight, fast, open-source, and produces 384-dimensional sentence embeddings suitable for semantic search.
- **Vector store:** FAISS `IndexFlatIP` with normalized embeddings to support cosine-style similarity search.

## Data Note

Raw data and generated vector indexes can be large, so they are excluded from Git by default. The repository includes reproducible scripts to regenerate processed files and vector stores locally.
