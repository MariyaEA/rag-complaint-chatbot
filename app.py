"""
Interactive Gradio chat interface for the CrediTrust complaint RAG system.

Run:
    python app.py

Features required by Task 4:
- Text input box
- Ask/Submit button
- AI-generated answer display
- Retrieved source chunks shown below the answer
- Clear button
"""

import gradio as gr

from src.rag_pipeline import answer_question


def respond(question, product_filter, top_k):
    """Run the RAG pipeline and return answer plus retrieved sources."""
    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        result = answer_question(
            question=question,
            k=int(top_k),
            product_filter=None if product_filter == "All" else product_filter,
        )

        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])

        if not sources:
            source_text = "No retrieved sources found."
        else:
            formatted_sources = []
            for i, source in enumerate(sources, start=1):
                formatted_sources.append(
                    f"Source {i}\n"
                    f"Product: {source.get('product_category', 'N/A')}\n"
                    f"Complaint ID: {source.get('complaint_id', 'N/A')}\n"
                    f"Score: {source.get('score', 0):.4f}\n"
                    f"Text: {source.get('text', '')}\n"
                )
            source_text = "\n\n" + ("-" * 80 + "\n\n").join(formatted_sources)

        return answer, source_text

    except Exception as exc:
        return (
            "The app could not complete the request. "
            "Please confirm the vector_store files exist and the RAG pipeline is configured correctly.",
            f"Error details: {exc}",
        )


with gr.Blocks(title="CrediTrust Complaint Analyst") as demo:
    gr.Markdown(
        """
        # CrediTrust Complaint Analyst

        This interactive app allows non-technical users to ask questions about customer complaint narratives.
        It retrieves relevant complaint chunks from the vector store and generates an evidence-based answer.
        """
    )

    with gr.Row():
        question = gr.Textbox(
            label="Ask a complaint analysis question",
            placeholder="Example: Why are customers unhappy with Credit Cards?",
            lines=3,
        )

    with gr.Row():
        product_filter = gr.Dropdown(
            choices=["All", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"],
            value="All",
            label="Product filter",
        )
        top_k = gr.Slider(
            minimum=3,
            maximum=10,
            value=5,
            step=1,
            label="Number of retrieved sources",
        )

    with gr.Row():
        ask_button = gr.Button("Ask", variant="primary")
        clear_button = gr.ClearButton()

    answer_output = gr.Textbox(label="AI-generated answer", lines=8)
    sources_output = gr.Textbox(label="Retrieved source chunks", lines=14)

    ask_button.click(
        fn=respond,
        inputs=[question, product_filter, top_k],
        outputs=[answer_output, sources_output],
    )

    clear_button.add([question, answer_output, sources_output])


if __name__ == "__main__":
    demo.launch()
