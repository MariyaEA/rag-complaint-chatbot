"""
End-to-end RAG pipeline for CrediTrust complaint analysis.
"""

from src.retriever import retrieve_context
from src.generator import generate_answer


def answer_question(question, k=5, product_filter=None):
    """
    Retrieve relevant complaint chunks and generate an answer.

    Returns:
        dict: {
            "answer": generated answer,
            "sources": retrieved complaint chunks
        }
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a valid question.",
            "sources": [],
        }

    try:
        retrieved_chunks = retrieve_context(
            query=question,
            k=k,
            product_filter=product_filter,
        )

        answer = generate_answer(
            question=question,
            chunks=retrieved_chunks,
        )

        return {
            "answer": answer,
            "sources": retrieved_chunks,
        }

    except Exception as exc:
        return {
            "answer": (
                "The RAG pipeline could not complete the request. "
                "Please confirm that the vector store and metadata files exist."
            ),
            "sources": [
                {
                    "product_category": "System",
                    "complaint_id": "N/A",
                    "score": 0,
                    "text": str(exc),
                }
            ],
        }


if __name__ == "__main__":
    demo_question = "Why are customers unhappy with Credit Cards?"
    result = answer_question(demo_question)

    print("Question:", demo_question)
    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"][:3]:
        print("-", source.get("product_category"), source.get("complaint_id"))
        print(source.get("text", "")[:300])
