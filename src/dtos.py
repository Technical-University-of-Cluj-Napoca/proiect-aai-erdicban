"""
dtos.py — Central data transfer objects for the legal contract analyzer.

All agents communicate exclusively through these validated Pydantic models.
A ValidationError at a boundary means the system is working as intended.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class ClauseType(str, Enum):
    """Contractual clause categories used throughout the pipeline."""
    PENALITATE = "penalitate"
    OBLIGATIE = "obligatie"
    DREPT = "drept"
    FORTA_MAJORA = "forta_majora"
    CONFIDENTIALITATE = "confidentialitate"
    REZILIERE = "reziliere"
    DATE_PERSONALE = "date_personale"
    ALTELE = "altele"


class RiskLevel(str, Enum):
    """Risk classification levels, ordered from highest to lowest severity."""
    RIDICAT = "RIDICAT"
    MEDIU = "MEDIU"
    SCAZUT = "SCAZUT"
    CONFORM = "CONFORM"
    NECUNOSCUT = "NECUNOSCUT"


# ──────────────────────────────────────────────
# Parsing DTOs (simple → composite)
# ──────────────────────────────────────────────

class PartyDTO(BaseModel):
    """One contracting party (company or individual)."""
    name: str = Field(..., description="Full legal name of the party")
    cui_cnp: Optional[str] = Field(None, description="Tax ID or personal ID number")
    address: Optional[str] = Field(None, description="Registered address")


class SectionDTO(BaseModel):
    """A top-level section or article in the contract."""
    title: str = Field(..., description="Section title, e.g. 'Articolul 5'")
    start_page: int = Field(..., description="Page where this section begins (1-indexed)")


class ClauseDTO(BaseModel):
    """
    A single contractual clause extracted from the document.

    The id follows the pattern 'art_<N>_clz_<M>' to make it traceable
    back to the original contract without storing personal data.
    """
    id: str = Field(..., description="Unique clause identifier, e.g. 'art_5_clz_2'")
    section: str = Field(..., description="Parent section title")
    text: str = Field(..., description="Full clause text as extracted from PDF")
    page: int = Field(..., description="Page number (1-indexed)")
    type: ClauseType = Field(ClauseType.ALTELE, description="Clause category")


class DocumentMetadataDTO(BaseModel):
    """Contract-level metadata extracted from the first pages."""
    title: str = Field(..., description="Contract title")
    page_count: int = Field(..., description="Total number of pages")
    parties: list[PartyDTO] = Field(default_factory=list, description="Contracting parties")
    signing_date: Optional[str] = Field(None, description="Date of signing (ISO or free text)")
    effective_date: Optional[str] = Field(None, description="Date contract enters into force")
    value: str = Field("", description="Contract value if stated")
    duration: str = Field("", description="Contract duration if stated")


class ParsedDocumentDTO(BaseModel):
    """
    Full structured representation of a parsed contract PDF.
    Output of DocumentParserAgent.parse().
    """
    metadata: DocumentMetadataDTO
    sections: list[SectionDTO] = Field(default_factory=list)
    clauses: list[ClauseDTO] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Pipeline DTOs
# ──────────────────────────────────────────────

class RetrievalResultDTO(BaseModel):
    """One chunk retrieved from the vector store for a given clause."""
    text: str = Field(..., description="Raw chunk text from the corpus")
    source: str = Field(..., description="Source document filename or URL")
    score: float = Field(..., description="Cosine similarity score (higher = more similar)")


class RiskAssessmentDTO(BaseModel):
    """
    Risk evaluation for a single clause produced by RiskAssessmentAgent.
    context_was_empty=True means the LLM was never called — treat result as unreliable.
    """
    clause_id: str
    risk_level: RiskLevel = Field(RiskLevel.NECUNOSCUT)
    issues: list[str] = Field(
        default_factory=list,
        description="List of identified legal issues, each anchored in a cited source"
    )
    references: list[str] = Field(
        default_factory=list,
        description="Legislation or corpus sources cited (must match actual corpus files)"
    )
    context_was_empty: bool = Field(
        False,
        description="True when no relevant chunks were retrieved — LLM was not called"
    )


class RecommendationDTO(BaseModel):
    """
    Reformulation proposal for a risky clause, produced by RecommendationAgent.
    candidates is populated only for RIDICAT clauses (self-consistency: 3 drafts, 1 chosen).
    """
    clause_id: str
    original_text: str
    reformulated_text: str = Field(
        "",
        description="Proposed rewrite in formal legal language. Empty for SCAZUT/CONFORM."
    )
    explanation: str = Field(
        "",
        description="Why this reformulation addresses the identified risk"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Corpus sources used to anchor the reformulation"
    )
    candidates: Optional[list[str]] = Field(
        None,
        description="The 3 draft candidates generated by self-consistency (RIDICAT only)"
    )
