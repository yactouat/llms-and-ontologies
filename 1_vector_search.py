#!/usr/bin/env python3
"""
Demo: vectorize PDFs in data/, persist Chroma DB, run vector search with Gemini embeddings.
Requires GOOGLE_API_KEY in the environment.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.messages import HumanMessage, SystemMessage

DATA_DIR = Path("data")
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "renesas_docs"

# Gemini embedding model (stable)
EMBEDDING_MODEL = "gemini-embedding-001"
# LLM for final answer (context-only, no internal knowledge)
LLM_MODEL = "gemini-3-flash-preview"

# DEMO QUESTIONS (Vector Search Failures):
# 1. "How did the N3 Building fire affect Kentucky Truck Plant inventory?"
# 2. "Link plating equipment failure to Ford's supplier-related constraints."
# 3. "Why did 23 machines lost cost billions in revenue?"

# Load .env from project root (optional; env vars can also be set in the shell)
load_dotenv()


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY in the environment.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm() -> ChatGoogleGenerativeAI:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY in the environment.")
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


CONTEXT_ONLY_SYSTEM = """You must answer the user's question using ONLY the text from the "Retrieved documents" section below. Do not use your internal knowledge or any information outside these documents.

Rules:
- Base your answer strictly on the provided document excerpts.
- If the documents do not contain enough information to answer the question, say so clearly (e.g. "The provided documents do not contain information that answers this question.").
- Do not add facts, interpretations, or details that are not supported by the retrieved text.
- You may summarize or rephrase only what is in the documents."""


def answer_from_documents(question: str, documents: list, llm: ChatGoogleGenerativeAI) -> str:
    """Formulate a response using only the retrieved document chunks."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "?")
        context_parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        SystemMessage(content=CONTEXT_ONLY_SYSTEM),
        HumanMessage(
            content=f"Retrieved documents:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer using only the retrieved documents above:"
        ),
    ]
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    # Normalize: content may be a list of blocks (e.g. [{"type": "text", "text": "..."}])
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return content if isinstance(content, str) else str(content)


def load_pdfs(data_dir: Path):
    """Load all PDFs from data_dir into a single list of documents."""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        return []
    docs = []
    for path in sorted(data_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())
    return docs


def get_or_build_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Load existing Chroma DB or build from PDFs in data/."""
    persist_path = PERSIST_DIR.resolve()
    persist_path.mkdir(parents=True, exist_ok=True)

    # Try to use existing persisted collection
    try:
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )
        # Quick check that it has data (Chroma will open empty collection too)
        if existing._collection.count() > 0:
            return existing
    except Exception:
        pass

    # Build from PDFs
    raw_docs = load_pdfs(DATA_DIR)
    if not raw_docs:
        raise SystemExit(
            f"No PDFs found in {DATA_DIR}. Add .pdf files there and run again."
        )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50,
    )
    chunks = splitter.split_documents(raw_docs)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_path),
    )
    print(f"Indexed {len(chunks)} chunks from {len(raw_docs)} page(s).")
    return vector_store


def _short_source(source: str) -> str:
    """Show only filename for cleaner demo output."""
    return Path(source).name if source else "?"


def main():
    parser = argparse.ArgumentParser(description="Vector search over PDFs in data/")
    parser.add_argument(
        "question",
        nargs="?",
        default="What is this document about?",
        help="Question to run vector search for",
    )
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to return")
    args = parser.parse_args()

    embeddings = get_embeddings()
    vector_store = get_or_build_vector_store(embeddings)
    llm = get_llm()

    results = vector_store.similarity_search(args.question, k=args.k)

    # --- Demo-friendly display ---
    width = 72
    line = "═" * width
    thin = "─" * width

    print()
    print(line)
    print("  QUESTION")
    print(thin)
    print(f"  {args.question}")
    print(line)
    print()
    print("  TOP CHUNKS (retrieved by similarity)")
    print(thin)
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "?")
        short = _short_source(source)
        excerpt = doc.page_content[:500].strip() + ("..." if len(doc.page_content) > 500 else "")
        print(f"\n  [{i}] {short}")
        for paragraph in excerpt.split("\n"):
            print(f"      {paragraph.strip()}")
    print()
    print(thin)

    answer = answer_from_documents(args.question, results, llm)
    print()
    print("  FINAL RESPONSE (from retrieved documents only)")
    print(thin)
    for paragraph in answer.strip().split("\n"):
        print(f"  {paragraph}")
    print()
    print(line)
    print()


if __name__ == "__main__":
    main()
