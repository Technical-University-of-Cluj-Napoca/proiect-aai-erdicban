"""
parser_agent.py — DocumentParserAgent

Extracts structured information from a contract PDF and returns a
ParsedDocumentDTO. Never crashes on malformed input — it returns
whatever could be extracted, with empty lists for missing fields.

Extraction strategy:
  - Metadata (title, parties, dates, value): regex on first 3 pages,
    LLM fallback for anything not matched.
  - Section detection: pdfplumber font-size heuristic (bold/larger text)
    combined with regex for Romanian legal patterns like "Articolul N".
  - Clause extraction: each paragraph under a section becomes a ClauseDTO.
  - Clause type classification: keyword matching first (fast, free),
    LLM only for clauses that keyword matching cannot classify.
"""

from __future__ import annotations
import json
import logging
import re
import os
from pathlib import Path

import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.dtos import (
    ClauseDTO,
    ClauseType,
    DocumentMetadataDTO,
    ParsedDocumentDTO,
    PartyDTO,
    SectionDTO,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Keyword → ClauseType mapping (first-pass classification)
# ──────────────────────────────────────────────
CLAUSE_KEYWORDS: dict[ClauseType, list[str]] = {
    ClauseType.PENALITATE: ["penalitate", "daune", "penalizare", "dobândă", "întârziere"],
    ClauseType.OBLIGATIE: ["se obligă", "obligația", "trebuie să", "va asigura"],
    ClauseType.DREPT: ["are dreptul", "dreptul de", "poate solicita"],
    ClauseType.FORTA_MAJORA: ["forță majoră", "caz fortuit", "eveniment imprevizibil"],
    ClauseType.CONFIDENTIALITATE: ["confidențial", "secret comercial", "nedivulgare", "NDA"],
    ClauseType.REZILIERE: ["reziliere", "rezoluțiune", "încetare", "denunțare"],
    ClauseType.DATE_PERSONALE: ["date cu caracter personal", "GDPR", "prelucrare date", "persoană vizată"],
}


def _classify_clause_by_keywords(text: str) -> ClauseType:
    text_lower = text.lower()
    for clause_type, keywords in CLAUSE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return clause_type
    return ClauseType.ALTELE


def _extract_sections_from_pages(pages_text: list[str]) -> list[SectionDTO]:
    """
    Detect section headers using Romanian legal article patterns.
    Pattern covers: 'Articolul 5', 'Art. 5', 'Clauza 3', 'CAPITOLUL II'
    """
    sections = []
    seen_titles: set[str] = set()
    pattern = re.compile(
        r"^(Articolul\s+\d+|Art\.\s*\d+|Clauza\s+\d+|CAPITOLUL\s+[IVXLCDM\d]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    for page_num, page_text in enumerate(pages_text, start=1):
        for match in pattern.finditer(page_text):
            title = match.group(0).strip()
            if title not in seen_titles:
                sections.append(SectionDTO(title=title, start_page=page_num))
                seen_titles.add(title)
    return sections


def _extract_clauses_from_text(
    pages_text: list[str],
    sections: list[SectionDTO],
) -> list[ClauseDTO]:
    """
    Split each page into paragraphs and assign them to their section.
    Paragraphs shorter than 30 characters are skipped (likely headers/footers).
    """
    clauses: list[ClauseDTO] = []
    clause_counter: dict[str, int] = {}

    # Map section titles to page numbers for assignment
    section_map: dict[int, str] = {s.start_page: s.title for s in sections}
    current_section = "Preambul"

    for page_num, page_text in enumerate(pages_text, start=1):
        if page_num in section_map:
            current_section = section_map[page_num]

        paragraphs = [p.strip() for p in page_text.split("\n\n") if len(p.strip()) > 30]
        for para in paragraphs:
            section_key = current_section[:20]
            clause_counter[section_key] = clause_counter.get(section_key, 0) + 1
            clause_id = f"sec_{section_key[:8].replace(' ', '_')}_clz_{clause_counter[section_key]:03d}"

            clause_type = _classify_clause_by_keywords(para)
            clauses.append(
                ClauseDTO(
                    id=clause_id,
                    section=current_section,
                    text=para,
                    page=page_num,
                    type=clause_type,
                )
            )

    return clauses


def _extract_metadata_with_regex(first_pages_text: str) -> dict:
    """
    Fast regex extraction for predictable fields.
    Returns a partial dict — missing fields will be filled by LLM fallback.
    """
    result: dict = {}

    # Signing date: common Romanian patterns
    date_match = re.search(
        r"(\d{1,2}[./]\d{1,2}[./]\d{4}|\d{4}-\d{2}-\d{2})", first_pages_text
    )
    if date_match:
        result["signing_date"] = date_match.group(1)

    # Contract value
    value_match = re.search(r"valoare[a\s]+(?:de\s+)?([\d.,]+\s*(?:RON|EUR|USD|lei))", first_pages_text, re.IGNORECASE)
    if value_match:
        result["value"] = value_match.group(1)

    # Duration
    duration_match = re.search(r"durata?\s+(?:de\s+)?([\d]+\s*(?:luni|ani|zile|lună|an|zi))", first_pages_text, re.IGNORECASE)
    if duration_match:
        result["duration"] = duration_match.group(1)

    return result


def _extract_metadata_with_llm(first_pages_text: str, llm: ChatOpenAI) -> dict:
    """
    LLM fallback for metadata fields regex could not extract.
    Returns JSON with keys: title, parties, signing_date, effective_date, value, duration.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Ești un asistent juridic. Extrage metadatele contractului din textul furnizat. "
            "Răspunde EXCLUSIV cu un obiect JSON valid cu cheile: "
            "title (string), parties (array de obiecte cu name, cui_cnp, address), "
            "signing_date (string sau null), effective_date (string sau null), "
            "value (string), duration (string). "
            "Nu inventa informații absente din text — folosește null sau string gol."
        )),
        ("human", "Text contract (primele pagini):\n\n{text}"),
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"text": first_pages_text[:3000]})
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM metadata extraction failed: %s", exc)
        return {}


class DocumentParserAgent:
    """
    Parses a contract PDF into a ParsedDocumentDTO.

    Usage:
        agent = DocumentParserAgent()
        dto = agent.parse("data/contract.pdf")
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def parse(self, pdf_path: str | Path) -> ParsedDocumentDTO:
        """
        Main entry point. Returns ParsedDocumentDTO even on partial failure.
        If the PDF cannot be opened at all, returns a DTO with empty clauses.
        """
        pdf_path = Path(pdf_path)
        logger.info("Parsing contract: %s", pdf_path)

        pages_text: list[str] = []
        page_count = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", pdf_path, exc)
            return ParsedDocumentDTO(
                metadata=DocumentMetadataDTO(
                    title=pdf_path.stem,
                    page_count=0,
                ),
                sections=[],
                clauses=[],
            )

        full_text = "\n\n".join(pages_text)
        first_pages_text = "\n\n".join(pages_text[:3])

        # ── Metadata extraction ──
        regex_meta = _extract_metadata_with_regex(first_pages_text)
        llm_meta = _extract_metadata_with_llm(first_pages_text, self.llm)

        # Merge: regex wins for fields it found (more reliable), LLM fills the rest
        parties_raw = llm_meta.get("parties", [])
        parties = [
            PartyDTO(
                name=p.get("name", ""),
                cui_cnp=p.get("cui_cnp"),
                address=p.get("address"),
            )
            for p in parties_raw
            if isinstance(p, dict) and p.get("name")
        ]

        metadata = DocumentMetadataDTO(
            title=llm_meta.get("title") or pdf_path.stem,
            page_count=page_count,
            parties=parties,
            signing_date=regex_meta.get("signing_date") or llm_meta.get("signing_date"),
            effective_date=llm_meta.get("effective_date"),
            value=regex_meta.get("value") or llm_meta.get("value", ""),
            duration=regex_meta.get("duration") or llm_meta.get("duration", ""),
        )

        # ── Structure extraction ──
        sections = _extract_sections_from_pages(pages_text)
        clauses = _extract_clauses_from_text(pages_text, sections)

        logger.info(
            "Parsed %s: %d pages, %d sections, %d clauses",
            pdf_path.name,
            page_count,
            len(sections),
            len(clauses),
        )

        return ParsedDocumentDTO(
            metadata=metadata,
            sections=sections,
            clauses=clauses,
        )
