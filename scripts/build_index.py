"""
build_index.py — Run ONCE before launching the application.

Usage:
    python scripts/build_index.py

What it does:
  1. Loads all PDFs from corpus/ via load_corpus()
  2. Chunks, embeds and stores them in vectorstore/ via build_index()

Safety checks:
  - If vectorstore/ already contains data, the script aborts with a warning.
    Delete vectorstore/ manually to rebuild from scratch.
  - If the corpus is empty, the script aborts with an error.

Environment:
  Requires OPENAI_API_KEY in .env (loaded automatically).
"""

import logging
import os
import sys
from pathlib import Path

# Allow running from project root: python scripts/build_index.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.tools.pdf_tools import load_corpus
from src.tools.vector_tools import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_index")

CORPUS_DIR = "corpus"
VECTORSTORE_DIR = "vectorstore"


def main() -> None:
    # Guard: abort if vectorstore already has content
    vs_path = Path(VECTORSTORE_DIR)
    if vs_path.exists() and any(vs_path.iterdir()):
        logger.warning(
            "vectorstore/ already exists and is not empty. "
            "Delete it manually and re-run to rebuild the index."
        )
        sys.exit(0)

    logger.info("Loading corpus from '%s'...", CORPUS_DIR)
    documents = load_corpus(CORPUS_DIR)

    if not documents:
        logger.error(
            "Corpus is empty — no PDFs found in '%s'. "
            "Add at least 15 legal documents before indexing.",
            CORPUS_DIR,
        )
        sys.exit(1)

    logger.info("Loaded %d documents. Building index...", len(documents))
    build_index(documents, persist_directory=VECTORSTORE_DIR)
    logger.info("Index built successfully in '%s'.", VECTORSTORE_DIR)


if __name__ == "__main__":
    main()
