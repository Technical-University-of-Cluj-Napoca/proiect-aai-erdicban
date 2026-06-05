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


def extract_text_with_ocr_fallback(pdf_path: Path | str) -> list[str]:
    """
    Extract text page-by-page from a PDF. If the extracted text is empty
    or too short (scanned PDF), falls back to OCR via pypdfium2 and tesseract.
    """
    import pypdfium2 as pdfium
    import tempfile
    import subprocess

    pages_text: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
    except Exception as exc:
        logger.warning("Normal text extraction failed for %s: %s. Trying direct OCR fallback.", pdf_path, exc)
        pages_text = []

    total_len = sum(len(p.strip()) for p in pages_text)
    if total_len < 50:
        logger.info("Extracted text is empty or very short (%d chars). Falling back to OCR using pypdfium2 + tesseract...", total_len)
        pages_text = []
        try:
            doc = pdfium.PdfDocument(str(pdf_path))
            for i, page in enumerate(doc):
                image = page.render(scale=2).to_pil()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    tmp_img_path = tmp_file.name
                try:
                    image.save(tmp_img_path)
                    tess_cmd = "/opt/homebrew/bin/tesseract" if os.path.exists("/opt/homebrew/bin/tesseract") else "tesseract"
                    res = subprocess.run(
                        [tess_cmd, tmp_img_path, "stdout", "-l", "eng"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    ocr_text = res.stdout or ""
                    pages_text.append(ocr_text)
                    logger.info("OCR completed for page %d: %d chars extracted", i + 1, len(ocr_text))
                finally:
                    if os.path.exists(tmp_img_path):
                        os.remove(tmp_img_path)
        except Exception as ocr_exc:
            logger.error("OCR fallback failed for %s: %s", pdf_path, ocr_exc)

    return pages_text


def _extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """
    Extract plain text and page count from a PDF using OCR fallback.
    Returns (text, page_count).
    """
    pages = extract_text_with_ocr_fallback(pdf_path)
    return "\n\n".join(pages), len(pages)


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
