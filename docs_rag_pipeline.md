# RAG Pipeline Internal Design

## Retriever

The retriever embeds the user question using the same sentence-transformer model used during indexing:

`sentence-transformers/all-MiniLM-L6-v2`

It then performs FAISS similarity search and returns the top-k most relevant complaint chunks.

Returned metadata includes:

- complaint_id
- product_category
- chunk_index
- similarity score
- source text

## Prompt Template

The prompt instructs the model to act as a financial analyst assistant for CrediTrust Financial. It requires the model to use only retrieved context and avoid inventing unsupported information.

## Generator

The generator combines:

1. User question
2. Retrieved complaint chunks
3. Financial analyst prompt template

It returns a concise answer grounded in source complaints.

## Error Handling

The pipeline handles:

- empty user questions
- missing vector store files
- empty retrieval results
- generator/LLM loading failures
- missing metadata fields

## Interface Evidence

The app.py Gradio interface includes:

- question text input
- Ask button
- generated answer display
- retrieved source chunks
- Clear button
- product filter
