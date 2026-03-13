"""
Document Ingestion Pipeline - Load, split, embed, and store documents
"""
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load documents from a list of file paths."""
    all_docs = []

    for file_path in file_paths:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"⚠️  Unsupported file type: {ext} — skipping {path.name}")
            continue

        try:
            loader_cls = SUPPORTED_EXTENSIONS[ext]
            loader = loader_cls(str(path))
            docs = loader.load()

            # Attach filename metadata
            for doc in docs:
                doc.metadata["source"] = path.name
                doc.metadata["file_path"] = str(path)

            all_docs.extend(docs)
            print(f"✅ Loaded: {path.name} ({len(docs)} chunks)")

        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

    return all_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"📄 Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_vectorstore(
    chunks: List[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Chroma:
    """Create and persist a Chroma vectorstore from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"🗄️  Vectorstore created with {len(chunks)} embeddings → {persist_directory}")
    return vectorstore


def load_vectorstore(
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Chroma:
    """Load an existing Chroma vectorstore from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    print(f"🗄️  Loaded vectorstore: {count} documents from {persist_directory}")
    return vectorstore


def ingest_documents(
    file_paths: List[str],
    persist_directory: str = "./chroma_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Chroma:
    """Full ingestion pipeline: load → split → embed → store."""
    print("\n🚀 Starting document ingestion pipeline...")

    docs = load_documents(file_paths)
    if not docs:
        raise ValueError("No documents were successfully loaded.")

    chunks = split_documents(docs, chunk_size, chunk_overlap)
    vectorstore = create_vectorstore(chunks, persist_directory)

    print("✅ Ingestion complete!\n")
    return vectorstore
