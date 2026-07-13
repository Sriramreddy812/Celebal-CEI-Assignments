from transformers import pipeline


def get_llm_pipeline(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load and return a text-generation pipeline for a Qwen instruct model.
    Qwen2.5-0.5B-Instruct is small enough to run free/local on CPU, but
    noticeably better than flan-t5-base at pulling exact facts from context
    instead of defaulting to a broad summary.
    """
    llm_pipeline = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
    )
    return llm_pipeline


def generate_answer(llm_pipeline, query: str, retrieved_results):
    """
    Build a context-grounded chat prompt from retrieved chunks and generate an answer.
    Qwen is an instruct/chat model, so we use its chat message format instead of
    a plain text2text prompt string.
    """
    context_text = "\n\n".join([chunk.page_content for chunk, score in retrieved_results])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions using only the "
                "provided context. Give a clear, complete answer using multiple sentences "
                "if needed. If the answer is not in the context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]

    result = llm_pipeline(messages, max_new_tokens=300, do_sample=False)
    answer = result[0]["generated_text"][-1]["content"].strip()

    return answer
