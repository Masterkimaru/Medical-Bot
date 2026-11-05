"""Load PDFs, split into chunks, and expose the embedding adapter.

The embedding factory now returns the LangChain `Embeddings` adapter backed by
the EmbeddingService (OpenRouter with local Sentence-Transformers fallback).
"""
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vector_service import ServiceEmbeddings, get_vector_service


def load_pdf_file(data: str) -> List[Document]:
    """Extract data from PDF files in the specified directory."""
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def text_split(extracted_data: List[Document]) -> List[Document]:
    """Split documents into smaller text chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def get_embeddings():
    """Return a LangChain-compatible Embeddings instance.

    Backed by the EmbeddingService: OpenRouter (`nvidia/nemotron-3-embed-1b:free`)
    with a local Sentence-Transformers fallback. The vector service guarantees
    the fallback only ever reads/writes its own index.
    """
    return ServiceEmbeddings(get_vector_service().service)
