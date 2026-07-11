# Document Question Answering System (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions from an uploaded PDF. It retrieves relevant chunks of the document and uses them as context for the language model, so answers stay grounded in the actual content instead of the model's own knowledge.

**Live Demo:** [https://document-rag-week-7-fko3ety8p8q4r8zfx6who7.streamlit.app/](https://document-rag-week-7-fko3ety8p8q4r8zfx6who7.streamlit.app/)

## Tech Stack

- **Document Loading**: LangChain `PyPDFLoader` / `TextLoader`
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **Language Model**: `google/flan-t5-base`
- **Interface**: Streamlit

## Project Structure
document-rag/
├── streamlit_app.py    # Main Streamlit app
├── requirements.txt
├── app.py          # for testing only
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectorstore.py
│   ├── retriever.py
│   └── generator.py
└── README.md

## How It Works

1. Upload a PDF
2. Text is extracted and split into chunks
3. Chunks are embedded and stored in FAISS
4. Your question is embedded and matched against the chunks (Top-K = 2)
5. The matched chunks are given to the LLM to generate a grounded answer

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- Top-K is set to 2 — testing showed this gave more accurate answers than a higher Top-K for fact-based questions.
- `transformers` and `torch` are pinned in `requirements.txt` since the newer `transformers` (v5) removed the pipeline task this project relies on.
