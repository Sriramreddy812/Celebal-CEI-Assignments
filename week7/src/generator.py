from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def get_llm_pipeline(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load a Qwen2.5 instruct model wrapped as a LangChain-compatible pipeline.
    return_full_text=False means the pipeline only returns the newly generated
    text, not the prompt echoed back - simplifies parsing the answer.
    """
    hf_pipe = pipeline(
        task="text-generation",
        model=model_name,
        max_new_tokens=150,
        max_length=None,
        temperature=0.2,
        do_sample=True,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=hf_pipe)


def generate_answer(llm, query: str, retrieved_results):
    """
    Build a chat-structured prompt using Qwen's expected chat format tags,
    to prevent the model confusing instructions with context.
    """
    context_text = "\n\n".join(chunk.page_content for chunk, score in retrieved_results)

    prompt = (
        "<|im_start|>system\n"
        "You are a factual assistant. Read the provided context carefully and answer the user's question with a short, direct answer. "
        "Pay close attention to document metadata: distinguish between the author of a book being discussed and the person who submitted/wrote the document itself. "
        "Extract the exact answer from the context if possible. If the answer is not in the context, say you don't know.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return llm.invoke(prompt).strip()
