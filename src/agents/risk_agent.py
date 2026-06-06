"""
risk_agent.py — RiskAssessmentAgent

Evaluates the legal risk of a single clause using the corpus chunks
retrieved by RAGRetrievalAgent as grounding context.

If context_chunks is empty, the agent returns NECUNOSCUT immediately
without calling the LLM — a short-circuit that is explicit and logged.
"""

from __future__ import annotations
import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.output_parsers import JsonOutputParser

from src.dtos import ClauseDTO, RetrievalResultDTO, RiskAssessmentDTO, RiskLevel

logger = logging.getLogger(__name__)

# Enable SQLite cache for development
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

SYSTEM_PROMPT = """\
Ești un expert juridic specializat în legislația română și europeană.
Evaluează clauza contractuală furnizată pe baza EXCLUSIVĂ a fragmentelor din corpus-ul juridic de mai jos.

Reguli stricte:
1. Citează NUMAI surse care apar explicit în contextul furnizat.
2. Nu inventa articole de lege, regulamente sau hotărâri judecătorești.
3. Dacă contextul nu acoperă clauza, returnează risk_level: "NECUNOSCUT".

Formatul de răspuns trebuie să fie un obiect JSON valid, conform instrucțiunilor de mai jos.
{format_instructions}
"""

HUMAN_PROMPT = """\
Corpus juridic relevant:
{legal_context}

---
Clauza de evaluat (ID: {clause_id}):
{clause_text}
"""


class RiskAssessmentAgent:
    """
    Assesses the legal risk of a clause given retrieved context chunks.

    Usage:
        agent = RiskAssessmentAgent()
        dto = agent.assess(clause_dto, context_chunks)
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.parser = JsonOutputParser(pydantic_object=RiskAssessmentDTO)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.chain = self.prompt | self.llm | self.parser

    def assess(
        self,
        clause: ClauseDTO,
        context_chunks: list[RetrievalResultDTO],
    ) -> RiskAssessmentDTO:
        """
        Evaluate clause risk. Returns NECUNOSCUT if context is empty.
        Never raises — logs errors and returns a safe fallback DTO.
        """
        if not context_chunks:
            logger.info("Clause %s: no context chunks — returning NECUNOSCUT", clause.id)
            return RiskAssessmentDTO(
                clause_id=clause.id,
                risk_level=RiskLevel.NECUNOSCUT,
                issues=["Contextul juridic recuperat este gol — evaluare imposibilă."],
                references=[],
                context_was_empty=True,
            )

        # Build the legal context string, tagging each chunk with its source
        # so the model can cite it correctly in the references field.
        legal_context = "\n\n".join(
            f"[Sursa: {chunk.source}]\n{chunk.text}"
            for chunk in context_chunks
        )

        try:
            parsed = self.chain.invoke(
                {
                    "legal_context": legal_context,
                    "clause_id": clause.id,
                    "clause_text": clause.text[:2000],
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )

            # JsonOutputParser returns parsed dictionary matching DTO fields
            return RiskAssessmentDTO(
                clause_id=clause.id,
                risk_level=RiskLevel(parsed.get("risk_level", "NECUNOSCUT")),
                issues=parsed.get("issues", []),
                references=parsed.get("references", []),
                context_was_empty=False,
            )

        except Exception as exc:
            logger.error("Risk assessment failed or JSON parsing error for clause %s: %s", clause.id, exc)

        # Safe fallback — never crash
        return RiskAssessmentDTO(
            clause_id=clause.id,
            risk_level=RiskLevel.NECUNOSCUT,
            issues=["Eroare internă la evaluarea riscului."],
            references=[],
            context_was_empty=False,
        )
