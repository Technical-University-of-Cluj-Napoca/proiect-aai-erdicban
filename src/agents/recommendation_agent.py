"""
recommendation_agent.py — RecommendationAgent

Generates clause reformulations for RIDICAT and MEDIU risk clauses.
SCAZUT / CONFORM / NECUNOSCUT clauses are returned with empty reformulation
(no LLM call — saves cost and latency).

Self-consistency for RIDICAT clauses:
  Three independent reformulations are requested in separate LLM calls,
  then a fourth call selects the best candidate. This increases reliability
  for the highest-risk clauses at the cost of ~4× API calls.
  The three candidates are stored in RecommendationDTO.candidates for audit.

Report generation:
  generate_report() produces a Markdown file readable by a jurist without
  any AI background — plain language, source citations, risk color codes.
"""

from __future__ import annotations
import logging
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.dtos import (
    ClauseDTO,
    RecommendationDTO,
    RetrievalResultDTO,
    RiskAssessmentDTO,
    RiskLevel,
)

logger = logging.getLogger(__name__)

REFORMULATION_SYSTEM = """\
Ești un avocat specializat în drept contractual român.
Reformulează clauza contractuală furnizată astfel încât:
1. Să fie conformă cu legislația citată în context.
2. Să asigure echilibrul contractual între părți.
3. Să folosească limbaj juridic formal, clar și precis.
4. Să citeze explicit baza legală în textul reformulat (ex: "conform art. X din Legea Y").

Răspunde NUMAI cu textul reformulat al clauzei, fără explicații introductive.
"""

REFORMULATION_HUMAN = """\
Context juridic de referință:
{legal_context}

Clauza originală cu risc {risk_level}:
{clause_text}

Problemele identificate:
{issues}
"""

SELECTION_SYSTEM = """\
Ești un avocat senior. Alege dintre cele trei variante de reformulare
pe cea mai bună din punct de vedere juridic și al echilibrului contractual.
Răspunde NUMAI cu textul variantei alese, fără explicații.
"""

SELECTION_HUMAN = """\
Clauza originală:
{original}

Varianta 1:
{v1}

Varianta 2:
{v2}

Varianta 3:
{v3}
"""


class RecommendationAgent:
    """
    Generates reformulations for risky clauses and compiles the final report.

    Usage:
        agent = RecommendationAgent()
        dto = agent.recommend(clause, risk_assessment, context_chunks)
        agent.generate_report(results, "output/report.md")
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.3,   # slight creativity for legal language variation
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system", REFORMULATION_SYSTEM),
            ("human", REFORMULATION_HUMAN),
        ])
        self.selection_prompt = ChatPromptTemplate.from_messages([
            ("system", SELECTION_SYSTEM),
            ("human", SELECTION_HUMAN),
        ])

    def _build_legal_context(self, chunks: list[RetrievalResultDTO]) -> str:
        return "\n\n".join(
            f"[{chunk.source}]\n{chunk.text}" for chunk in chunks
        )

    def _single_reformulation(
        self,
        clause_text: str,
        risk_level: str,
        issues: list[str],
        legal_context: str,
    ) -> str:
        chain = self.reformulation_prompt | self.llm
        try:
            response = chain.invoke(
                {
                    "legal_context": legal_context,
                    "clause_text": clause_text[:2000],
                    "risk_level": risk_level,
                    "issues": "\n".join(f"- {i}" for i in issues),
                }
            )
            return response.content.strip()
        except Exception as exc:
            logger.error("Reformulation LLM call failed: %s", exc)
            return ""

    def _select_best(self, original: str, candidates: list[str]) -> str:
        """Use a separate LLM call to pick the best of 3 candidates."""
        if not any(candidates):
            return candidates[0] if candidates else ""
        chain = self.selection_prompt | self.llm
        try:
            response = chain.invoke(
                {
                    "original": original,
                    "v1": candidates[0],
                    "v2": candidates[1] if len(candidates) > 1 else "",
                    "v3": candidates[2] if len(candidates) > 2 else "",
                }
            )
            return response.content.strip()
        except Exception as exc:
            logger.error("Candidate selection failed: %s", exc)
            return candidates[0]

    def recommend(
        self,
        clause: ClauseDTO,
        risk_assessment: RiskAssessmentDTO,
        context_chunks: list[RetrievalResultDTO],
    ) -> RecommendationDTO:
        """
        Generate a reformulation proposal for the clause.

        RIDICAT  → self-consistency (3 drafts + 1 selection call)
        MEDIU    → single reformulation call
        Others   → no LLM call, empty reformulated_text
        """
        base = RecommendationDTO(
            clause_id=clause.id,
            original_text=clause.text,
            sources=[c.source for c in context_chunks],
        )

        if risk_assessment.risk_level not in (RiskLevel.RIDICAT, RiskLevel.MEDIU):
            return base

        legal_context = self._build_legal_context(context_chunks)
        issues = risk_assessment.issues
        risk_label = risk_assessment.risk_level.value

        if risk_assessment.risk_level == RiskLevel.RIDICAT:
            # Self-consistency: 3 independent drafts
            candidates = [
                self._single_reformulation(clause.text, risk_label, issues, legal_context)
                for _ in range(3)
            ]
            best = self._select_best(clause.text, candidates)
            explanation = (
                f"Clauza prezintă risc RIDICAT. "
                f"Au fost generate 3 variante; cea mai bună a fost selectată "
                f"pe criterii de conformitate și echilibru contractual."
            )
            return RecommendationDTO(
                clause_id=clause.id,
                original_text=clause.text,
                reformulated_text=best,
                explanation=explanation,
                sources=[c.source for c in context_chunks],
                candidates=candidates,
            )

        # MEDIU — single call
        reformulated = self._single_reformulation(clause.text, risk_label, issues, legal_context)
        return RecommendationDTO(
            clause_id=clause.id,
            original_text=clause.text,
            reformulated_text=reformulated,
            explanation=f"Clauza prezintă risc MEDIU și a fost reformulată conform surselor citate.",
            sources=[c.source for c in context_chunks],
            candidates=None,
        )

    # ──────────────────────────────────────────────
    # Report generation
    # ──────────────────────────────────────────────

    def generate_report(
        self,
        results: list[tuple[ClauseDTO, RiskAssessmentDTO, RecommendationDTO]],
        output_path: str,
    ) -> None:
        """
        Write a Markdown report readable by a jurist without AI knowledge.
        Colour-coded risk table + per-clause sections with reformulations.
        """
        lines: list[str] = []
        lines.append("# Raport de Analiză Contract\n")
        lines.append(
            "> **Notă**: Acest raport a fost generat automat de un sistem AI. "
            "Concluziile trebuie validate de un jurist înainte de orice acțiune legală.\n"
        )

        # Summary table
        lines.append("## Sumar Riscuri\n")
        lines.append("| ID Clauză | Tip | Nivel Risc | Probleme identificate |")
        lines.append("|-----------|-----|------------|----------------------|")

        risk_order = {
            RiskLevel.RIDICAT: 0,
            RiskLevel.MEDIU: 1,
            RiskLevel.SCAZUT: 2,
            RiskLevel.CONFORM: 3,
            RiskLevel.NECUNOSCUT: 4,
        }
        sorted_results = sorted(results, key=lambda x: risk_order.get(x[1].risk_level, 99))

        for clause, risk, _ in sorted_results:
            risk_emoji = {
                RiskLevel.RIDICAT: "🔴",
                RiskLevel.MEDIU: "🟡",
                RiskLevel.SCAZUT: "🟢",
                RiskLevel.CONFORM: "✅",
                RiskLevel.NECUNOSCUT: "⚪",
            }.get(risk.risk_level, "⚪")

            issues_short = "; ".join(risk.issues[:2])[:80] + ("..." if len(risk.issues) > 2 else "")
            lines.append(
                f"| {clause.id} | {clause.type.value} | {risk_emoji} {risk.risk_level.value} | {issues_short} |"
            )

        lines.append("")

        # Detailed sections for RIDICAT and MEDIU only
        lines.append("## Clauze cu Risc Ridicat și Mediu — Detalii\n")
        for clause, risk, rec in sorted_results:
            if risk.risk_level not in (RiskLevel.RIDICAT, RiskLevel.MEDIU):
                continue

            lines.append(f"### {clause.id} — {risk.risk_level.value}")
            lines.append(f"**Pagina**: {clause.page} | **Secțiune**: {clause.section}\n")
            lines.append("**Text original:**")
            lines.append(f"> {clause.text[:500]}{'...' if len(clause.text) > 500 else ''}\n")

            if risk.issues:
                lines.append("**Probleme identificate:**")
                for issue in risk.issues:
                    lines.append(f"- {issue}")
                lines.append("")

            if risk.references:
                lines.append("**Referințe legislative:**")
                for ref in risk.references:
                    lines.append(f"- `{ref}`")
                lines.append("")

            if rec.reformulated_text:
                lines.append("**Reformulare propusă:**")
                lines.append(f"> {rec.reformulated_text}\n")
                if rec.explanation:
                    lines.append(f"*{rec.explanation}*\n")

            lines.append("---\n")

        report_text = "\n".join(lines)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Report written to %s", output_path)
