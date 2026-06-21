# B9W7 Interim Report: Intelligent Complaint Analysis for Financial Services

**Project:** CrediTrust Financial Complaint Analysis RAG System  
**Submission:** Interim Submission — Task 1 and Task 2  
**Author:** Mariamawit Ewnetu Alemu  

## 1. Introduction

CrediTrust Financial needs a faster way to understand customer pain points across its main digital finance products. Customer complaints contain valuable evidence about billing issues, transaction delays, loan servicing problems, savings account access concerns, and money transfer failures. However, these insights are difficult to use when they remain buried inside thousands of unstructured narratives.

The purpose of this interim work is to prepare the foundation for a Retrieval-Augmented Generation system. Task 1 focuses on understanding and cleaning the CFPB complaint data. Task 2 converts cleaned complaint narratives into searchable text chunks and stores their embeddings in a vector database so that later stages of the project can retrieve relevant evidence for user questions.

## 2. Task 1: EDA and Data Preprocessing

### 2.1 EDA Approach

The preprocessing pipeline loads the CFPB complaints dataset from `data/raw/complaints.csv.zip`. The first EDA step checks the distribution of complaint records across product categories. This helps identify whether the dataset is balanced or dominated by specific financial products. The second EDA step calculates the narrative word count for each complaint narrative and visualizes the distribution. This is important because very short narratives may contain limited context, while very long narratives may need chunking before embedding. The third EDA step counts records with and without consumer narratives, because records without narrative text cannot support semantic retrieval.

The EDA outputs are saved in the `figures/` folder:

- `product_distribution.png`
- `narrative_word_count_distribution.png`
- `narrative_presence_counts.png`
- `eda_summary.csv`

### 2.2 Filtering Logic

The dataset is filtered to retain only the four target product categories required by the challenge:

- Credit Card
- Personal Loan
- Savings Account
- Money Transfer

Raw CFPB product labels are mapped into these target categories using a normalization function. For example, `Credit card or prepaid card` is mapped to `Credit Card`, and `Checking or savings account` is mapped to `Savings Account`. Records with missing or empty consumer complaint narratives are removed because they cannot contribute useful text to the RAG system.

### 2.3 Text Cleaning

The narrative cleaning function applies the following steps:

- Converts all text to lowercase.
- Removes common boilerplate phrases such as complaint-introduction wording.
- Removes anonymization placeholders such as repeated `XXXX` text.
- Removes URLs and unnecessary special characters.
- Normalizes repeated whitespace.
- Drops extremely short cleaned narratives.

The final cleaned dataset is saved as:

```text
data/processed/filtered_complaints.csv
```

This file contains the complaint ID, normalized product category, original product field, original narrative, cleaned narrative, and narrative word count. Optional metadata columns such as issue, company, state, and date received are preserved when available.

## 3. Task 2: Chunking, Embedding, and Vector Store Indexing

### 3.1 Stratified Sampling Strategy

For the interim vector store, the pipeline creates a stratified sample of 12,000 complaints from the cleaned dataset. The sampling is proportional by `product_category`, meaning each of the four target product categories keeps approximately the same share it had in the cleaned data. This prevents the sample from overrepresenting one product and underrepresenting another.

The sampled dataset is saved as:

```text
data/processed/stratified_sample.csv
```

### 3.2 Chunking Strategy

Complaint narratives vary in length, and long narratives are not ideal when embedded as one large text block. The pipeline uses overlapping chunks with:

- `chunk_size = 500` characters
- `chunk_overlap = 50` characters

This choice keeps each chunk focused enough for semantic search while preserving context across chunk boundaries. Each chunk stores traceable metadata including the original complaint ID, product category, chunk index, and total number of chunks for that complaint.

### 3.3 Embedding Model Choice

The selected embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is appropriate for the interim submission because it is lightweight, fast, widely used for semantic similarity tasks, and produces compact 384-dimensional embeddings. It is efficient enough for local development while still strong enough to support meaningful retrieval over complaint narratives.

### 3.4 Vector Store

The vector store is implemented using FAISS. Embeddings are normalized and stored in a `faiss.IndexFlatIP` index, which supports cosine-style similarity search through inner product over normalized vectors.

The vector store outputs are saved in:

```text
vector_store/
```

Expected files:

```text
vector_store/complaints_faiss.index
vector_store/chunk_metadata.parquet
vector_store/index_config.json
```

The metadata file allows each retrieved chunk to be traced back to its original complaint and product category, which is essential for evidence-backed RAG responses in the final system.

## 4. Repository Best Practices

The repository follows the expected project structure:

```text
rag-complaint-chatbot/
├── data/raw/
├── data/processed/
├── vector_store/
├── notebooks/
├── src/
├── tests/
├── reports/
├── figures/
├── requirements.txt
├── README.md
└── .gitignore
```

The `.gitignore` excludes raw data, large processed files, vector-store files, virtual environments, cache files, and operating-system artifacts. The `requirements.txt` file lists the main dependencies required for EDA, preprocessing, embeddings, chunking, FAISS indexing, testing, and notebook work.

## 5. Code Quality

The implementation is modular. Task 1 logic is placed in `src/preprocess.py`, with separate functions for loading data, identifying columns, product normalization, text cleaning, EDA, and filtering. Task 2 logic is placed in `src/vector_indexing.py`, with separate functions for loading cleaned data, stratified sampling, chunking, embedding generation, and vector-store persistence.

Basic error handling is included for missing input files, missing required columns, and cases where no chunks are created. Unit tests are included for key preprocessing functions, and a GitHub Actions workflow is provided for running tests on push and pull request events.

## 6. Next Steps for Final Submission

The next phase will build the core RAG pipeline and interactive application. This will include loading the full-scale vector store, retrieving top-k relevant complaint chunks for user questions, using a prompt template to ground LLM responses in retrieved evidence, evaluating the system with representative questions, and building a Gradio or Streamlit interface that displays both answers and source chunks.
