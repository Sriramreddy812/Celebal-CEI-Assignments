from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def get_llm_pipeline(model_name: str = MODEL_NAME):
    """
    Load Qwen2.5 with its own tokenizer, so we can use apply_chat_template()
    instead of hand-typing chat tags - this guarantees the prompt format
    exactly matches what the model was trained on.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hf_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,        # deterministic output - more reliable for factual Q&A than sampling
        repetition_penalty=1.2, # discourages rambling/looping
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    return llm, tokenizer


def generate_answer(llm_and_tokenizer, query: str, retrieved_results):
    """
    Build a grounding prompt using the tokenizer's official chat template,
    then generate an answer.
    """
    llm, tokenizer = llm_and_tokenizer

    if not retrieved_results:
        return "I don't have enough information in the document to answer that."

    context_text = "\n\n".join(chunk.page_content for chunk, score in retrieved_results)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual assistant. Answer the user's question using only the "
                "provided context. Give a short, direct, complete answer. If the question "
                "has multiple parts, answer all of them. Distinguish between the author of "
                "a book being discussed and the person who submitted/wrote the document itself. "
                "If the answer is not in the context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer = llm.invoke(prompt)
    return answer.strip()
