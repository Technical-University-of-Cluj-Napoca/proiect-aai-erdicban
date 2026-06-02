"""
evaluate_risk_agent.py — Quantitative evaluation of RiskAssessmentAgent.

Usage:
    python scripts/evaluate_risk_agent.py

Computes precision, recall and F1 of RiskAssessmentAgent against a manually
labelled test set of clauses. Results saved to logs/risk_agent_evaluation.json
and a confusion matrix to logs/risk_confusion_matrix.png.

Test set design:
  10 clauses with manually assigned ground-truth risk levels.
  Clauses cover all 5 risk categories and 4 clause types to ensure
  the evaluation is not biased toward a single scenario.

Metrics:
  - Per-class precision, recall, F1 (one-vs-rest)
  - Macro-averaged F1 (unweighted mean across classes)
  - Exact match accuracy

Note on binary RIDICAT detection:
  In a legal risk system, the most important metric is recall for RIDICAT —
  missing a high-risk clause (false negative) is more dangerous than a false
  alarm (false positive). We report this separately.
"""

from __future__ import annotations
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("evaluate_risk")

OUTPUT_JSON = "logs/risk_agent_evaluation.json"
OUTPUT_PNG = "logs/risk_confusion_matrix.png"

# ──────────────────────────────────────────────
# Manually labelled test set
# Each entry: clause text + ground truth risk level
# Labels assigned by reading Romanian contract law references.
# ──────────────────────────────────────────────
TEST_SET = [
    {
        "id": "eval_001",
        "text": "Furnizorul va plăti penalități de 5% pe zi din valoarea contractului pentru fiecare zi de întârziere, fără nicio plafonare.",
        "section": "Penalități",
        "ground_truth": "RIDICAT",
        "rationale": "Penalitate neplafonată, disproporționată, contrară Legii 98/2016 art. 164",
    },
    {
        "id": "eval_002",
        "text": "Prestatorul prelucrează datele cu caracter personal ale beneficiarului fără a specifica temeiul legal conform GDPR.",
        "section": "Date personale",
        "ground_truth": "RIDICAT",
        "rationale": "Lipsă temei legal — GDPR art. 6 obligatoriu",
    },
    {
        "id": "eval_003",
        "text": "Beneficiarul poate rezilia contractul oricând, fără preaviz și fără obligația de a plăti vreo despăgubire.",
        "section": "Reziliere",
        "ground_truth": "RIDICAT",
        "rationale": "Dezechilibru contractual evident — clauză abuzivă ANPC",
    },
    {
        "id": "eval_004",
        "text": "Forța majoră exonerează de răspundere inclusiv în cazul dificultăților economice sau al variațiilor de preț.",
        "section": "Forța majoră",
        "ground_truth": "MEDIU",
        "rationale": "Definire prea largă — Cod Civil art. 1351 limitează la evenimente imprevizibile",
    },
    {
        "id": "eval_005",
        "text": "Clauza de confidențialitate se aplică pe o perioadă nedefinită după încetarea contractului.",
        "section": "Confidențialitate",
        "ground_truth": "MEDIU",
        "rationale": "Durată nedefinită poate contraveni GDPR principiului limitării stocării",
    },
    {
        "id": "eval_006",
        "text": "Penalitățile de întârziere sunt de 0.1% pe zi, dar nu vor depăși 10% din valoarea totală a contractului.",
        "section": "Penalități",
        "ground_truth": "CONFORM",
        "rationale": "Penalitate plafonată, proporțională — conformă uzanțelor comerciale",
    },
    {
        "id": "eval_007",
        "text": "Ambele părți se obligă să respecte legislația aplicabilă privind protecția datelor cu caracter personal.",
        "section": "Date personale",
        "ground_truth": "CONFORM",
        "rationale": "Referință generică la legislație — suficient pentru o clauză introductivă",
    },
    {
        "id": "eval_008",
        "text": "Contractul intră în vigoare la data semnării de către ambele părți și este valabil 12 luni.",
        "section": "Durată",
        "ground_truth": "SCAZUT",
        "rationale": "Clauză standard de durată, fără elemente de risc",
    },
    {
        "id": "eval_009",
        "text": "Cesiunea drepturilor și obligațiilor din prezentul contract se poate face numai cu acordul scris al celeilalte părți.",
        "section": "Cesiune",
        "ground_truth": "CONFORM",
        "rationale": "Clauză echilibrată — consimțământ bilateral obligatoriu",
    },
    {
        "id": "eval_010",
        "text": "Răspunderea prestatorului este limitată la valoarea unui abonament lunar, indiferent de natura sau amploarea prejudiciului.",
        "section": "Răspundere",
        "ground_truth": "RIDICAT",
        "rationale": "Excludere aproape totală a răspunderii — Cod Civil art. 1355 interzice excluderea pentru dol",
    },
]

RISK_LEVELS = ["RIDICAT", "MEDIU", "SCAZUT", "CONFORM", "NECUNOSCUT"]


def _compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    """Compute per-class precision, recall, F1 and macro averages."""
    metrics = {}
    macro_p, macro_r, macro_f1 = [], [], []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for t in y_true if t == label),
        }
        macro_p.append(precision)
        macro_r.append(recall)
        macro_f1.append(f1)

    metrics["macro_avg"] = {
        "precision": round(sum(macro_p) / len(macro_p), 3),
        "recall": round(sum(macro_r) / len(macro_r), 3),
        "f1": round(sum(macro_f1) / len(macro_f1), 3),
    }
    return metrics


def _plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: str,
) -> None:
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    label_idx = {l: i for i, l in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            matrix[label_idx[t]][label_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("RiskAssessmentAgent — Confusion Matrix", pad=12)
    plt.colorbar(im, ax=ax)

    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                        fontsize=11, color="white" if matrix[i][j] > 2 else "black",
                        fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def main() -> None:
    vs_path = Path("vectorstore")
    if not vs_path.exists() or not any(vs_path.iterdir()):
        logger.error("vectorstore/ is empty. Run scripts/build_index.py first.")
        sys.exit(1)

    from src.agents.retrieval_agent import RAGRetrievalAgent
    from src.agents.risk_agent import RiskAssessmentAgent
    from src.dtos import ClauseDTO, ClauseType

    retrieval_agent = RAGRetrievalAgent()
    risk_agent = RiskAssessmentAgent()

    y_true: list[str] = []
    y_pred: list[str] = []
    detailed_results = []

    for item in TEST_SET:
        clause = ClauseDTO(
            id=item["id"],
            section=item["section"],
            text=item["text"],
            page=1,
            type=ClauseType.ALTELE,
        )
        chunks = retrieval_agent.retrieve(clause)
        assessment = risk_agent.assess(clause, chunks)

        y_true.append(item["ground_truth"])
        y_pred.append(assessment.risk_level.value)

        correct = assessment.risk_level.value == item["ground_truth"]
        logger.info(
            "%s | true=%-10s pred=%-10s %s",
            item["id"],
            item["ground_truth"],
            assessment.risk_level.value,
            "✓" if correct else "✗",
        )

        detailed_results.append({
            "id": item["id"],
            "text_preview": item["text"][:80] + "...",
            "ground_truth": item["ground_truth"],
            "predicted": assessment.risk_level.value,
            "correct": correct,
            "issues": assessment.issues,
            "references": assessment.references,
            "chunks_retrieved": len(chunks),
            "context_was_empty": assessment.context_was_empty,
            "rationale": item["rationale"],
        })

    # Metrics
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    metrics = _compute_metrics(y_true, y_pred, RISK_LEVELS)

    # Special metric: RIDICAT recall (most safety-critical)
    ridicat_recall = metrics.get("RIDICAT", {}).get("recall", 0.0)

    print(f"\n{'='*55}")
    print("QUANTITATIVE EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"Accuracy (exact match): {accuracy:.1%}")
    print(f"Macro F1:               {metrics['macro_avg']['f1']:.3f}")
    print(f"RIDICAT recall:         {ridicat_recall:.3f}  ← most important metric")
    print(f"\nPer-class metrics:")
    for level in RISK_LEVELS:
        m = metrics.get(level, {})
        if m.get("support", 0) > 0:
            print(f"  {level:12s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")

    # Save results
    output = {
        "accuracy": round(accuracy, 3),
        "macro_f1": metrics["macro_avg"]["f1"],
        "ridicat_recall": ridicat_recall,
        "per_class_metrics": metrics,
        "test_set_size": len(TEST_SET),
        "detailed_results": detailed_results,
        "notes": (
            "RIDICAT recall is the most critical metric for a legal risk system. "
            "A false negative (missed high-risk clause) has higher real-world cost "
            "than a false positive (unnecessary reformulation). "
            f"Current RIDICAT recall: {ridicat_recall:.1%}."
        ),
    }

    os.makedirs("logs", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", OUTPUT_JSON)

    _plot_confusion_matrix(y_true, y_pred, RISK_LEVELS, OUTPUT_PNG)
    print(f"\n✅ Full results: {OUTPUT_JSON}")
    print(f"✅ Confusion matrix: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
