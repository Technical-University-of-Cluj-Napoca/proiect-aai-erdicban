"""
parser_agent.py — DocumentParserAgent

Extracts structured information from a contract PDF and returns a
ParsedDocumentDTO. Never crashes on malformed input — it returns
whatever could be extracted, with empty lists for missing fields.

Extraction strategy:
  - Metadata (title, parties, dates, value): regex on first 3 pages,
    LLM fallback for anything not matched.
  - Section detection: regex for Romanian legal patterns and Roman numerals (e.g. "I. PARTI").
  - Clause extraction: splits by paragraph and bullet points.
  - Clause type classification: keyword matching first (fast, free),
    LLM fallback for unclassified clauses.
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
    ClauseType.PENALITATE: ["penalitate", "daune", "penalizare", "dobândă", "întârziere", "majorări"],
    ClauseType.OBLIGATIE: ["se obligă", "obligația", "trebuie să", "va asigura", "îndatoriri"],
    ClauseType.DREPT: ["are dreptul", "dreptul de", "poate solicita", "este îndreptățit"],
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


def _classify_clause_with_llm(text: str, llm: ChatOpenAI) -> ClauseType:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Ești un asistent juridic. Clasifică clauza contractuală furnizată în una dintre categoriile: "
            "penalitate, obligatie, drept, forta_majora, confidentialitate, reziliere, date_personale, altele. "
            "Răspunde EXCLUSIV cu numele categoriei exact cum este scrisă mai sus, fără alte cuvinte sau semne."
        )),
        ("human", "Clauza: {text}"),
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"text": text[:1000]})
        val = response.content.strip().lower()
        for ct in ClauseType:
            if ct.value == val:
                return ct
    except Exception as exc:
        logger.warning("LLM clause classification failed: %s", exc)
    return ClauseType.ALTELE


def _extract_sections_from_pages(pages_text: list[str]) -> list[SectionDTO]:
    """
    Detect section headers using Romanian legal article patterns and Roman numerals.
    Matches: 'Articolul 5', 'Art. 5', 'Clauza 3', 'CAPITOLUL II', 'I. PARTILE CONTRACTANTE'
    """
    sections = []
    seen_titles: set[str] = set()
    # Matches Roman numerals like I. II. III. IV. V. VI. at start of line
    pattern = re.compile(
        r"^(Articolul\s+\d+|Art\.\s*\d+|Clauza\s+\d+|CAPITOLUL\s+[IVXLCDM\d]+|^[IVXLCDM]+\.\s+[A-ZĂÂÎȘȚa-z\t ]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    for page_num, page_text in enumerate(pages_text, start=1):
        for match in pattern.finditer(page_text):
            title = match.group(0).strip()
            # Clean up title
            title = re.sub(r"\s+", " ", title)
            if title not in seen_titles and len(title) > 3:
                sections.append(SectionDTO(title=title, start_page=page_num))
                seen_titles.add(title)
    return sections


def _extract_clauses_from_text(
    pages_text: list[str],
    sections: list[SectionDTO],
    llm: ChatOpenAI,
) -> list[ClauseDTO]:
    """
    Split text into clauses, accounting for newlines, bullet points and sections.
    """
    clauses: list[ClauseDTO] = []
    clause_counter: dict[str, int] = {}
    current_section = "Preambul"

    for page_num, page_text in enumerate(pages_text, start=1):
        # Normalize line endings
        text = page_text.replace("\r\n", "\n")

        # Smart splitting: split on double newlines
        raw_paragraphs = text.split("\n\n")
        
        # If there are no double newlines, split by single newlines
        if len(raw_paragraphs) <= 1:
            raw_paragraphs = text.split("\n")

        processed_paragraphs = []
        for p in raw_paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            
            # If paragraph contains bullet points, split by bullet points
            if "•" in p_clean:
                bullets = [b.strip() for b in p_clean.split("•") if len(b.strip()) > 10]
                processed_paragraphs.extend(bullets)
            elif "\n•" in p_clean or "\n-" in p_clean:
                bullets = [b.strip() for b in re.split(r"\n[•-]", p_clean) if len(b.strip()) > 10]
                processed_paragraphs.extend(bullets)
            else:
                if len(p_clean) > 20:
                    processed_paragraphs.append(p_clean)

        for para in processed_paragraphs:
            # Check if this paragraph is a section header (fuzzy match to account for extraction noise)
            matched_section = None
            for s in sections:
                if para.lower() in s.title.lower() or s.title.lower() in para.lower():
                    matched_section = s.title
                    break
            
            if matched_section:
                current_section = matched_section
                # Skip adding the section title itself as a clause
                continue

            section_key = current_section[:20]
            clause_counter[section_key] = clause_counter.get(section_key, 0) + 1
            
            clean_sec_key = re.sub(r"[^a-zA-Z0-9_]", "", section_key[:12].replace(" ", "_"))
            if not clean_sec_key:
                clean_sec_key = "clz"
            clause_id = f"sec_{clean_sec_key}_clz_{clause_counter[section_key]:03d}"

            # Keyword classification
            clause_type = _classify_clause_by_keywords(para)
            
            # LLM fallback if keyword classification returns ALTELE
            if clause_type == ClauseType.ALTELE:
                clause_type = _classify_clause_with_llm(para, llm)

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
    """
    result: dict = {}

    # Signing date
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

        first_pages_text = "\n\n".join(pages_text[:3])

        # ── Metadata extraction ──
        regex_meta = _extract_metadata_with_regex(first_pages_text)
        llm_meta = _extract_metadata_with_llm(first_pages_text, self.llm)

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
        clauses = _extract_clauses_from_text(pages_text, sections, self.llm)

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
