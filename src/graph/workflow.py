"""
workflow.py — LangGraph StateGraph orchestrating the full pipeline.

Node sequence:
  parse_document
    → retrieve_context
    → assess_risk
    → quality_check          ← feedback loop if too many NECUNOSCUT
    → flag_high_risk
    → generate_recommendations
    → compile_report

Feedback loop:
  If > 40% of clauses are NECUNOSCUT and iteration < MAX_ITER,
  quality_check routes back to retrieve_context with k+3 and a lower
  threshold. This gives the RAG agent a second chance with more permissive
  parameters before declaring defeat.

Logging:
  Each node appends a record to WorkflowState["node_log"], which is
  serialized to logs/run_<timestamp>.json by compile_report.
"""

from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END
from langchain_community.callbacks.manager import get_openai_callback

from src.agents.parser_agent import DocumentParserAgent
from src.agents.retrieval_agent import RAGRetrievalAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.dtos import (
    ClauseDTO,
    ParsedDocumentDTO,
    RecommendationDTO,
    RetrievalResultDTO,
    RiskAssessmentDTO,
    RiskLevel,
)

logger = logging.getLogger(__name__)

MAX_ITER = 2
# fraction of NECUNOSCUT clauses that triggers retry
NECUNOSCUT_THRESHOLD = 0.40   
PERSIST_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class WorkflowState(TypedDict):
    pdf_path: str
    parsed_doc: Optional[ParsedDocumentDTO]
    context_map: dict[str, list[RetrievalResultDTO]]   # clause_id → chunks
    risk_map: dict[str, RiskAssessmentDTO]              # clause_id → assessment
    high_risk_alert: bool
    recommendations: list[RecommendationDTO]
    report_path: str
    iteration: int
    retrieval_k: int
    retrieval_threshold: float
    node_log: list[dict[str, Any]]


def _initial_state(pdf_path: str) -> WorkflowState:
    return WorkflowState(
        pdf_path=pdf_path,
        parsed_doc=None,
        context_map={},
        risk_map={},
        high_risk_alert=False,
        recommendations=[],
        report_path="",
        iteration=0,
        retrieval_k=5,
        retrieval_threshold=float(os.getenv("RETRIEVAL_THRESHOLD", "0.30")),
        node_log=[],
    )


# ──────────────────────────────────────────────
# Node helpers
# ──────────────────────────────────────────────

def _log_node(state: WorkflowState, node: str, start: float, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, **extra) -> None:
    state["node_log"].append(
        {
            "node": node,
            "duration_s": round(time.time() - start, 2),
            "iteration": state["iteration"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            **extra,
        }
    )


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

def parse_document(state: WorkflowState) -> WorkflowState:
    start = time.time()
    agent = DocumentParserAgent()
    
    with get_openai_callback() as cb:
        parsed = agent.parse(state["pdf_path"])
        p_tok = cb.prompt_tokens
        c_tok = cb.completion_tokens
        t_tok = cb.total_tokens
        
    state["parsed_doc"] = parsed
    _log_node(
        state, "parse_document", start,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        sections=len(parsed.sections),
        clauses=len(parsed.clauses),
    )
    logger.info("[parse_document] %d clauses extracted", len(parsed.clauses))
    return state


def retrieve_context(state: WorkflowState) -> WorkflowState:
    start = time.time()
    agent = RAGRetrievalAgent(
        persist_directory=PERSIST_DIR,
        threshold=state["retrieval_threshold"],
    )
    parsed = state["parsed_doc"]

    # Parallel retrieval
    context_map = agent.retrieve_many(parsed.clauses, k=state["retrieval_k"])

    state["context_map"] = context_map
    empty_count = sum(1 for v in context_map.values() if not v)
    _log_node(
        state, "retrieve_context", start,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        total_clauses=len(parsed.clauses),
        empty_retrievals=empty_count,
        k=state["retrieval_k"],
        threshold=state["retrieval_threshold"],
    )
    logger.info(
        "[retrieve_context] iter=%d, k=%d, empty=%d/%d",
        state["iteration"],
        state["retrieval_k"],
        empty_count,
        len(parsed.clauses),
    )
    return state


def assess_risk(state: WorkflowState) -> WorkflowState:
    start = time.time()
    agent = RiskAssessmentAgent()
    risk_map: dict[str, RiskAssessmentDTO] = {}

    with get_openai_callback() as cb:
        for clause in state["parsed_doc"].clauses:
            chunks = state["context_map"].get(clause.id, [])
            assessment = agent.assess(clause, chunks)
            risk_map[clause.id] = assessment
        p_tok = cb.prompt_tokens
        c_tok = cb.completion_tokens
        t_tok = cb.total_tokens

    state["risk_map"] = risk_map
    counts = {level.value: 0 for level in RiskLevel}
    for a in risk_map.values():
        counts[a.risk_level.value] += 1

    _log_node(
        state, "assess_risk", start,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        risk_distribution=counts
    )
    logger.info("[assess_risk] distribution: %s", counts)
    return state


def quality_check(state: WorkflowState) -> str:
    """
    Conditional node — returns the next node name.
    Retries retrieve_context if too many NECUNOSCUT and we have budget.
    """
    risk_map = state["risk_map"]
    if not risk_map:
        return "flag_high_risk"

    necunoscut_frac = sum(
        1 for a in risk_map.values() if a.risk_level == RiskLevel.NECUNOSCUT
    ) / len(risk_map)

    if necunoscut_frac > NECUNOSCUT_THRESHOLD and state["iteration"] < MAX_ITER:
        state["iteration"] += 1
        state["retrieval_k"] += 3
        state["retrieval_threshold"] = max(0.10, state["retrieval_threshold"] - 0.10)
        logger.info(
            "[quality_check] %.0f%% NECUNOSCUT → retry (iter=%d, k=%d, thr=%.2f)",
            necunoscut_frac * 100,
            state["iteration"],
            state["retrieval_k"],
            state["retrieval_threshold"],
        )
        return "retrieve_context"

    logger.info(
        "[quality_check] %.0f%% NECUNOSCUT — proceeding (iter=%d)",
        necunoscut_frac * 100,
        state["iteration"],
    )
    return "flag_high_risk"


def flag_high_risk(state: WorkflowState) -> WorkflowState:
    start = time.time()
    high_risk_count = sum(
        1 for a in state["risk_map"].values() if a.risk_level == RiskLevel.RIDICAT
    )
    state["high_risk_alert"] = high_risk_count > 0
    _log_node(
        state, "flag_high_risk", start,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        high_risk_clauses=high_risk_count
    )
    if state["high_risk_alert"]:
        logger.warning("[flag_high_risk] %d RIDICAT clause(s) detected!", high_risk_count)
    return state


def generate_recommendations(state: WorkflowState) -> WorkflowState:
    start = time.time()
    agent = RecommendationAgent()
    recommendations: list[RecommendationDTO] = []

    clause_map = {c.id: c for c in state["parsed_doc"].clauses}
    with get_openai_callback() as cb:
        for clause_id, risk in state["risk_map"].items():
            clause = clause_map.get(clause_id)
            if not clause:
                continue
            chunks = state["context_map"].get(clause_id, [])
            rec = agent.recommend(clause, risk, chunks)
            recommendations.append(rec)
        p_tok = cb.prompt_tokens
        c_tok = cb.completion_tokens
        t_tok = cb.total_tokens

    state["recommendations"] = recommendations
    reformulated = sum(1 for r in recommendations if r.reformulated_text)
    _log_node(
        state, "generate_recommendations", start,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        total=len(recommendations),
        reformulated=reformulated,
    )
    logger.info(
        "[generate_recommendations] %d clauses, %d reformulated",
        len(recommendations),
        reformulated,
    )
    return state


def compile_report(state: WorkflowState) -> WorkflowState:
    start = time.time()
    agent = RecommendationAgent()

    # Build tuples for report generation
    clause_map = {c.id: c for c in state["parsed_doc"].clauses}
    rec_map = {r.clause_id: r for r in state["recommendations"]}

    results = []
    for clause_id, risk in state["risk_map"].items():
        clause = clause_map.get(clause_id)
        rec = rec_map.get(clause_id, RecommendationDTO(
            clause_id=clause_id,
            original_text=clause.text if clause else "",
        ))
        if clause:
            results.append((clause, risk, rec))

    # Write Markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/report_{timestamp}.md"
    os.makedirs("data", exist_ok=True)
    agent.generate_report(results, report_path)
    state["report_path"] = report_path

    # Write run log
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/run_{timestamp}.json"
    _log_node(
        state, "compile_report", start,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        report_path=report_path
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(state["node_log"], f, ensure_ascii=False, indent=2)

    logger.info("[compile_report] Report: %s | Log: %s", report_path, log_path)
    return state


# ──────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("parse_document", parse_document)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("flag_high_risk", flag_high_risk)
    graph.add_node("generate_recommendations", generate_recommendations)
    graph.add_node("compile_report", compile_report)

    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "retrieve_context")
    graph.add_edge("retrieve_context", "assess_risk")

    # Conditional edge: quality_check decides next node
    graph.add_conditional_edges(
        "assess_risk",
        quality_check,
        {
            "retrieve_context": "retrieve_context",
            "flag_high_risk": "flag_high_risk",
        },
    )

    graph.add_edge("flag_high_risk", "generate_recommendations")
    graph.add_edge("generate_recommendations", "compile_report")
    graph.add_edge("compile_report", END)

    return graph


def run_pipeline(pdf_path: str) -> WorkflowState:
    """
    Entry point for running the full pipeline on a contract PDF.
    Returns the final WorkflowState (contains report_path and all DTOs).
    """
    graph = build_graph()
    app = graph.compile()

    state = _initial_state(pdf_path)
    final_state = app.invoke(state)
    return final_state


def export_graph_png(output_path: str = "logs/workflow_graph.png") -> None:
    """Export the graph diagram as PNG for the notebook."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    graph = build_graph()
    try:
        png = graph.compile().get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png)
        logger.info("Graph diagram saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not export graph PNG: %s", exc)
