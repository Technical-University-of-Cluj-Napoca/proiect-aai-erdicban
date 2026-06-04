"""
retrieval_agent.py — RAGRetrievalAgent

Retrieves the top-k most relevant corpus chunks for a given clause.
Chunks below the similarity threshold are filtered out.

Threshold rationale:
  A cosine similarity below 0.30 in ChromaDB's (1 - distance) metric
  indicates very weak semantic overlap — the retrieved text is likely
  from a different legal domain and would confuse the risk agent.
  0.30 was chosen after inspecting RAGAS faithfulness scores:
  chunks above this threshold consistently contained at least one
  normative reference relevant to the clause under evaluation.
  Adjust via the RETRIEVAL_THRESHOLD env variable if your corpus differs.

Important: ChromaDB returns *distance* (lower = more similar) when using
cosine space. We convert: similarity = 1 - distance.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

from src.dtos import ClauseDTO, RetrievalResultDTO
from src.tools.vector_tools import get_collection

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.30"))
DEFAULT_K = 5


class RAGRetrievalAgent:
    """
    Loads the vector store once at init and exposes retrieve().

    Usage:
        agent = RAGRetrievalAgent()
        results = agent.retrieve(clause_dto, k=5)
    """

    def __init__(
        self,
        persist_directory: str = "vectorstore",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self.collection = get_collection(persist_directory)
        logger.info(
            "RAGRetrievalAgent ready — collection has %d chunks, threshold=%.2f",
            self.collection.count(),
            self.threshold,
        )

    def retrieve(self, clause: ClauseDTO, k: int = DEFAULT_K) -> list[RetrievalResultDTO]:
        """
        Retrieve top-k chunks for the given clause.

        The query is built from the clause text directly — not the full
        DTO serialization, which would inject noise like the id field.
        Chunks whose similarity < threshold are dropped.
        Returns [] when nothing passes the filter (handled upstream).
        """
        query = clause.text[:1000]  # cap to avoid exceeding embedding token limit

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed for clause %s: %s", clause.id, exc)
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved: list[RetrievalResultDTO] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert ChromaDB cosine distance to similarity score
            similarity = 1.0 - dist

            if similarity < self.threshold:
                logger.debug(
                    "Clause %s: chunk from '%s' below threshold (%.3f < %.3f) — skipped",
                    clause.id,
                    meta.get("source", "?"),
                    similarity,
                    self.threshold,
                )
                continue

            retrieved.append(
                RetrievalResultDTO(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    score=round(similarity, 4),
                )
            )

        # Results are already ordered by distance (ascending) → similarity descending
        logger.debug(
            "Clause %s: %d / %d chunks passed threshold",
            clause.id,
            len(retrieved),
            len(documents),
        )
        return retrieved
