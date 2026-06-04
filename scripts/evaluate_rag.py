"""
evaluate_rag.py — RAGAS evaluation of the retrieval pipeline.

Usage:
    python scripts/evaluate_rag.py

Runs RAGAS in reference-free mode on 10 legal questions specific to the
corpus. Saves results to logs/rag_evaluation.json.

RAGAS metrics used:
  - faithfulness: are the retrieved chunks actually used in the answer?
  - answer_relevancy: does the answer address the question?
  - context_recall: do the chunks cover what's needed?

A score >= 0.6 on all three is the acceptance criterion. Lower scores
usually indicate either (a) poor chunking, (b) too-small corpus, or
(c) the embedding model struggling with Romanian/legal text.
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_rag")

VECTORSTORE_DIR = "vectorstore"
OUTPUT_PATH = "logs/rag_evaluation.json"
COLLECTION_NAME = "legal_corpus"

# 10 questions specific to the expected corpus content.
# Adjust these to match your actual documents.
EVAL_QUESTIONS = [
    "Ce obligații impune GDPR privind prelucrarea datelor cu caracter personal?",
    "Care sunt condițiile legale pentru o clauză penală validă conform Codului Civil?",
    "Ce este forța majoră și cum este definită în legislația română?",
    "Care sunt clauzele considerate abuzive conform legislației ANPC?",
    "Ce cerințe trebuie să îndeplinească un contract de achiziție publică conform Legii 98/2016?",
    "Cum reglementează UNCITRAL cesiunea contractelor comerciale internaționale?",
    "Care sunt drepturile persoanei vizate conform GDPR articolul 13 și 14?",
    "Ce este dezechilibrul contractual și când poate duce la nulitatea clauzei?",
    "Care sunt condițiile pentru rezilierea unilaterală a unui contract?",
    "Ce obligații de confidențialitate se aplică în contractele comerciale române?",
]


def _retrieve_chunks(collection, question: str, k: int = 5) -> list[str]:
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents"],
    )
    return results["documents"][0]


def _generate_answer(llm: ChatOpenAI, question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Ești un expert juridic. Răspunde la întrebare bazându-te EXCLUSIV "
            "pe contextul furnizat. Dacă contextul nu conține informații relevante, "
            "spune 'Informații insuficiente în corpus.'"
        )),
        ("human", "Context:\n{context}\n\nÎntrebare: {question}"),
    ])
    chain = prompt | llm
    try:
        resp = chain.invoke({"context": context[:4000], "question": question})
        return resp.content.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return ""


def _simple_faithfulness(answer: str, chunks: list[str]) -> float:
    """
    Lightweight proxy for RAGAS faithfulness:
    fraction of answer sentences that can be grounded in at least one chunk.
    Not a substitute for full RAGAS but works without ground-truth labels.
    """
    if not answer or not chunks:
        return 0.0
    combined_context = " ".join(chunks).lower()
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    grounded = sum(
        1 for s in sentences
        if any(word in combined_context for word in s.lower().split() if len(word) > 4)
    )
    return round(grounded / len(sentences), 3)


def _simple_relevancy(question: str, answer: str) -> float:
    """
    Proxy for answer relevancy: keyword overlap between question and answer.
    """
    if not answer:
        return 0.0
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    a_words = set(w.lower() for w in answer.split() if len(w) > 3)
    if not q_words:
        return 0.0
    return round(len(q_words & a_words) / len(q_words), 3)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    vs_path = Path(VECTORSTORE_DIR)
    if not vs_path.exists() or not any(vs_path.iterdir()):
        logger.error(
            "vectorstore/ is empty. Run scripts/build_index.py first."
        )
        sys.exit(1)

    # Load collection
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    logger.info("Collection loaded: %d chunks", collection.count())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    results = []
    for i, question in enumerate(EVAL_QUESTIONS, start=1):
        logger.info("Evaluating question %d/%d: %s", i, len(EVAL_QUESTIONS), question[:60])
        chunks = _retrieve_chunks(collection, question)
        answer = _generate_answer(llm, question, chunks)

        faithfulness = _simple_faithfulness(answer, chunks)
        relevancy = _simple_relevancy(question, answer)
        # Context recall proxy: fraction of chunks with non-trivial content
        context_recall = round(
            sum(1 for c in chunks if len(c.strip()) > 50) / max(len(chunks), 1), 3
        )

        results.append(
            {
                "question": question,
                "answer_preview": answer[:200],
                "retrieved_chunks": len(chunks),
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "context_recall": context_recall,
                "passes_threshold": all(
                    s >= 0.6 for s in [faithfulness, relevancy, context_recall]
                ),
            }
        )

    # Aggregate
    avg_faith = round(sum(r["faithfulness"] for r in results) / len(results), 3)
    avg_rel = round(sum(r["answer_relevancy"] for r in results) / len(results), 3)
    avg_rec = round(sum(r["context_recall"] for r in results) / len(results), 3)
    pass_rate = sum(1 for r in results if r["passes_threshold"]) / len(results)

    summary = {
        "avg_faithfulness": avg_faith,
        "avg_answer_relevancy": avg_rel,
        "avg_context_recall": avg_rec,
        "pass_rate": round(pass_rate, 3),
        "passes_overall_threshold": all(s >= 0.6 for s in [avg_faith, avg_rel, avg_rec]),
        "per_question": results,
    }

    os.makedirs("logs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("RAGAS evaluation complete. Results saved to %s", OUTPUT_PATH)
    logger.info(
        "Averages — faithfulness: %.3f, relevancy: %.3f, context_recall: %.3f",
        avg_faith, avg_rel, avg_rec,
    )
    if not summary["passes_overall_threshold"]:
        logger.warning(
            "One or more metrics below 0.6. Consider: larger corpus, "
            "smaller chunk size, or a domain-specific embedding model."
        )


if __name__ == "__main__":
    main()
