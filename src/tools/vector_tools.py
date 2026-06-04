"""
vector_tools.py — Chunking, embedding and ChromaDB index management.

build_index() is called once by scripts/build_index.py.
The resulting vectorstore/ directory is mounted as a Docker volume
so it persists across container restarts.

Chunking rationale:
  - chunk_size=500 tokens: large enough to capture a full legal paragraph
    (which often carries the full semantic meaning of a clause) but small
    enough that the retriever does not inject irrelevant surrounding text.
  - chunk_overlap=50 tokens: prevents a clause boundary from being cut
    exactly at a sentence that carries the normative weight. 50 is ~10%
    of chunk size — enough overlap without doubling storage.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.tools.pdf_tools import CorpusDocument

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legal_corpus"
CHUNK_SIZE = 500        # tokens (approximate via character proxy; ~4 chars/token)
CHUNK_OVERLAP = 50
CHARS_PER_TOKEN = 4     # rough conversion; good enough for splitter sizing


def _get_client(persist_directory: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_directory)


def build_index(
    documents: list[CorpusDocument],
    persist_directory: str = "vectorstore",
) -> None:
    """
    Chunk, embed and store all corpus documents in ChromaDB.

    Uses OpenAI text-embedding-3-small via ChromaDB's built-in embedding
    function — no manual embedding loop needed.

    Metadata stored per chunk:
      - source: citation key (relative path)
      - doc_type: gdpr | lege | contract | uncitral | anpc | other
      - title: human-readable document title
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = _get_client(persist_directory)

    # Abort if collection already has data to prevent duplicates
    existing = client.list_collections()
    if any(c.name == COLLECTION_NAME for c in existing):
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            logger.warning(
                "Collection '%s' already has %d documents. "
                "Delete vectorstore/ and re-run to rebuild.",
                COLLECTION_NAME,
                collection.count(),
            )
            return

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * CHARS_PER_TOKEN,
        chunk_overlap=CHUNK_OVERLAP * CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for doc in documents:
        chunks = splitter.split_text(doc.text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.source}::chunk_{i:04d}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                }
            )

    if not all_texts:
        logger.warning("No chunks to index — corpus is empty.")
        return

    # ChromaDB add() in batches of 500 to stay within API limits
    batch_size = 500
    for start in range(0, len(all_texts), batch_size):
        end = start + batch_size
        collection.add(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            metadatas=all_metadatas[start:end],
        )
        logger.info("Indexed chunks %d–%d / %d", start, min(end, len(all_texts)), len(all_texts))

    logger.info(
        "Index built: %d chunks from %d documents in '%s'",
        len(all_texts),
        len(documents),
        persist_directory,
    )


def get_collection(
    persist_directory: str = "vectorstore",
) -> chromadb.Collection:
    """
    Load the existing ChromaDB collection at runtime.
    Raises if the collection does not exist (build_index must run first).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = _get_client(persist_directory)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
