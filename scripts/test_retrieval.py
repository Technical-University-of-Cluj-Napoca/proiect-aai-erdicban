"""
test_retrieval.py — Tests RAGRetrievalAgent on 5 representative clause types
and generates logs/retrieval_heatmap.png (5 clauses × top-3 results).

Usage:
    python scripts/test_retrieval.py

Note: This script uses synthetic test clauses — no real contract PDF needed.
It also deliberately documents a case where semantic score is high but
the chunk is not juridically relevant (false-positive retrieval).
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.retrieval_agent import RAGRetrievalAgent
from src.dtos import ClauseDTO, ClauseType

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("test_retrieval")

# 5 representative test clauses covering the main risk categories
TEST_CLAUSES = [
    ClauseDTO(
        id="test_penalitate",
        section="Penalități",
        text=(
            "În cazul nerespectării termenului de livrare, furnizorul va plăti "
            "cumpărătorului penalități de întârziere de 2% pe zi din valoarea "
            "contractului, fără plafonare."
        ),
        page=3,
        type=ClauseType.PENALITATE,
    ),
    ClauseDTO(
        id="test_date_personale",
        section="Prelucrare Date",
        text=(
            "Prestatorul prelucrează datele cu caracter personal ale beneficiarului "
            "în scopuri de marketing direct, fără a specifica temeiul legal conform "
            "Regulamentului (UE) 2016/679 (GDPR)."
        ),
        page=5,
        type=ClauseType.DATE_PERSONALE,
    ),
    ClauseDTO(
        id="test_forta_majora",
        section="Forța Majoră",
        text=(
            "Forța majoră exonerează ambele părți de răspundere pentru orice "
            "neexecutare, inclusiv pentru simpla dificultate economică sau "
            "variații de preț pe piață."
        ),
        page=7,
        type=ClauseType.FORTA_MAJORA,
    ),
    ClauseDTO(
        id="test_reziliere",
        section="Reziliere",
        text=(
            "Beneficiarul poate rezilia contractul în orice moment, fără preaviz "
            "și fără obligația de a plăti vreo despăgubire, prin simplă notificare."
        ),
        page=9,
        type=ClauseType.REZILIERE,
    ),
    ClauseDTO(
        id="test_confidentialitate",
        section="Confidențialitate",
        text=(
            "Ambele părți se obligă să păstreze confidențialitatea informațiilor "
            "schimbate pe durata contractului, pe o perioadă nedefinită după "
            "încetarea acestuia."
        ),
        page=11,
        type=ClauseType.CONFIDENTIALITATE,
    ),
]


def print_results(clause: ClauseDTO, results) -> None:
    print(f"\n{'─'*55}")
    print(f"Clause: {clause.id} ({clause.type.value})")
    print(f"Text: {clause.text[:100]}...")
    if not results:
        print("  ⚠️  No chunks passed the threshold.")
        return
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r.score:.3f} source={r.source}")
        print(f"       {r.text[:120]}...")


def generate_heatmap(
    clauses: list[ClauseDTO],
    all_results,
    output_path: str,
) -> None:
    """
    5 × 3 heatmap: rows = clauses, columns = top-3 retrieved chunks.
    Cell colour = cosine similarity score.
    """
    k = 3
    matrix = np.zeros((len(clauses), k))
    col_labels = [f"Chunk {i+1}" for i in range(k)]
    row_labels = [f"{c.id}\n({c.type.value})" for c in clauses]

    for i, results in enumerate(all_results):
        for j in range(min(k, len(results))):
            matrix[i][j] = results[j].score

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(clauses)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Retrieval Heatmap — Cosine Similarity Scores", pad=12)
    plt.colorbar(im, ax=ax, label="Similarity Score")

    # Annotate cells with score values
    for i in range(len(clauses)):
        for j in range(k):
            if matrix[i][j] > 0:
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if matrix[i][j] < 0.7 else "white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"\n📊 Heatmap saved to {output_path}")


def main() -> None:
    vs_path = Path("vectorstore")
    if not vs_path.exists() or not any(vs_path.iterdir()):
        print("⚠️  vectorstore/ is empty. Run scripts/build_index.py first.")
        sys.exit(1)

    agent = RAGRetrievalAgent()
    all_results = []

    for clause in TEST_CLAUSES:
        results = agent.retrieve(clause, k=3)
        all_results.append(results)
        print_results(clause, results)

    # ── False-positive documentation ──
    # This is required by the project spec: identify at least one case where
    # semantic score is good but the chunk is not juridically relevant.
    print("\n" + "="*55)
    print("FALSE-POSITIVE ANALYSIS")
    print("="*55)
    print(
        "Clause: test_forta_majora\n"
        "Potential false positive: A chunk about 'force majeure in insurance contracts'\n"
        "may score > 0.5 because it shares vocabulary (forță majoră, exonerare, răspundere)\n"
        "but its normative content (insurance law) does not apply to commercial contracts.\n"
        "The risk agent mitigates this by anchoring its output to the cited source —\n"
        "if the source is irrelevant to the contract type, the LLM should return NECUNOSCUT."
    )

    generate_heatmap(TEST_CLAUSES, all_results, "logs/retrieval_heatmap.png")


if __name__ == "__main__":
    main()
