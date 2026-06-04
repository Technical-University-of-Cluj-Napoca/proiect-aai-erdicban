"""
pdf_tools.py — Corpus loading utilities.

load_corpus() walks corpus/ recursively and extracts text + metadata
from every PDF it finds. The result feeds directly into build_index().
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class CorpusDocument:
    """Raw extracted document before chunking and embedding."""
    text: str
    source: str          # relative path used as the citation key
    title: str
    doc_type: str        # gdpr | lege | contract | uncitral | anpc | other
    page_count: int
    file_path: str


def _infer_doc_type(rel_path: str) -> str:
    """
    Infer document category from the corpus sub-directory name.
    E.g. corpus/gdpr/regulation.pdf -> 'gdpr'
    """
    parts = Path(rel_path).parts
    # The first path component after 'corpus/' is the category folder
    if len(parts) >= 2:
        return parts[0].lower()
    return "other"


def _extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """
    Extract plain text and page count from a PDF using pdfplumber.
    Returns (text, page_count). Falls back to empty string on error.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            return "\n\n".join(pages), len(pdf.pages)
    except Exception as exc:
        logger.warning("Could not extract text from %s: %s", pdf_path, exc)
        return "", 0


def load_corpus(corpus_dir: str | Path) -> list[CorpusDocument]:
    """
    Walk corpus_dir recursively and return one CorpusDocument per PDF.

    Documents with empty text (scanned-only PDFs without OCR) are skipped
    with a warning — the caller should decide how to handle gaps.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    documents: list[CorpusDocument] = []

    for pdf_path in sorted(corpus_dir.rglob("*.pdf")):
        rel_path = pdf_path.relative_to(corpus_dir)
        source = str(rel_path)
        doc_type = _infer_doc_type(str(rel_path))

        text, page_count = _extract_text_from_pdf(pdf_path)

        if not text.strip():
            logger.warning(
                "Skipping %s — no text extracted (possibly scanned; consider OCR)", source
            )
            continue

        title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

        documents.append(
            CorpusDocument(
                text=text,
                source=source,
                title=title,
                doc_type=doc_type,
                page_count=page_count,
                file_path=str(pdf_path),
            )
        )
        logger.info("Loaded %s (%d pages, %d chars)", source, page_count, len(text))

    logger.info("Total corpus documents loaded: %d", len(documents))
    return documents
