#!/usr/bin/env python3
"""
Shared utilities for LLM and ontologies demos.
Extracted from 1_vector_search.py to avoid code duplication.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# Load environment variables from .env file if present
# This allows using a .env file instead of exporting vars in the shell
load_dotenv()


def check_google_api_key() -> None:
    """
    Verify that GOOGLE_API_KEY is set in the environment.
    Exits with an error message if not found.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit(
            "ERROR: GOOGLE_API_KEY not found in environment.\n"
            "Set it in your .env file or export it in your shell:\n"
            "  export GOOGLE_API_KEY='your-key-here'\n"
            "Get a key at: https://aistudio.google.com/apikey"
        )


def get_embeddings(model: str = "gemini-embedding-001") -> GoogleGenerativeAIEmbeddings:
    """
    Create a Google Generative AI embeddings instance.

    Args:
        model: The embedding model to use (default: gemini-embedding-001)

    Returns:
        GoogleGenerativeAIEmbeddings: Configured embeddings instance
    """
    check_google_api_key()
    return GoogleGenerativeAIEmbeddings(model=model)


def get_llm(
    model: str = "gemini-3-flash-preview",
    temperature: float = 0.0
) -> ChatGoogleGenerativeAI:
    """
    Create a Google Generative AI chat model instance.

    Args:
        model: The LLM model to use (default: gemini-3-flash-preview)
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)

    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance
    """
    check_google_api_key()
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def load_pdf(pdf_path: Path) -> list:
    """
    Load a single PDF file and extract its text into documents.

    Args:
        pdf_path: Path to the PDF file to load

    Returns:
        list: List of Document objects containing the PDF text and metadata
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # PyPDFLoader splits the PDF into one document per page
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def load_pdfs_from_directory(data_dir: Path) -> list:
    """
    Load all PDF files from a directory.

    Args:
        data_dir: Directory containing PDF files

    Returns:
        list: Combined list of Document objects from all PDFs
    """
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        return []

    docs = []
    # Sort ensures consistent ordering across runs
    for path in sorted(data_dir.glob("*.pdf")):
        docs.extend(load_pdf(path))

    return docs


def extract_text_from_pdf(
    pdf_path: Path,
    start_page: int = None,
    end_page: int = None
) -> str:
    """
    Extract raw text from a PDF file, optionally limiting to a page range.

    This is useful when you need plain text rather than Document objects,
    for example when feeding text directly to an LLM for analysis.

    Args:
        pdf_path: Path to the PDF file
        start_page: First page to extract (0-indexed). If None, starts from first page.
        end_page: Last page to extract (0-indexed, inclusive). If None, extracts all pages.

    Returns:
        str: The extracted text as a single string
    """
    # Load all pages as documents
    docs = load_pdf(pdf_path)

    # Determine page range
    total_pages = len(docs)
    if start_page is None:
        start_page = 0
    if end_page is None:
        end_page = total_pages - 1

    # Validate and clamp page range
    start_page = max(0, min(start_page, total_pages - 1))
    end_page = max(start_page, min(end_page, total_pages - 1))

    # Extract text from specified page range
    text_parts = []
    for i in range(start_page, end_page + 1):
        text_parts.append(docs[i].page_content)

    return "\n".join(text_parts)
