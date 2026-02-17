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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

DATA_DIR = Path("data")
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "renesas_docs"

# Gemini embedding model (stable)
EMBEDDING_MODEL = "gemini-embedding-001"

# Load .env from project root (optional; env vars can also be set in the shell)
load_dotenv()


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY in the environment.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


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

    results = vector_store.similarity_search(args.question, k=args.k)
    print(f"\nQuestion: {args.question}\n")
    print("--- Top chunks ---")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] (source: {doc.metadata.get('source', '?')})")
        print(doc.page_content[:500].strip() + ("..." if len(doc.page_content) > 500 else ""))


if __name__ == "__main__":
    main()
