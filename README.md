[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tDAXz5sa)

# Legal Contract Analyzer — Agentic AI

Multi-agent system for analyzing Romanian legal contracts using RAG + LangGraph.

## Stack
- **LLM**: OpenAI gpt-4o-mini (risk/recommendations), gpt-4o-mini (parsing)
- **RAG**: ChromaDB + text-embedding-3-small
- **Orchestration**: LangGraph StateGraph
- **UI**: Streamlit
- **Infra**: Docker + docker-compose

## Quick Start

### 1. Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
# or with uv:
uv init .
uv add -r requirements.txt
```

### 3. Add corpus documents
Place at least 15 legal PDFs in `corpus/` sub-directories:
```
corpus/
  gdpr/        ← GDPR regulation (eur-lex.europa.eu)
  legi/        ← Romanian laws (legislatie.just.ro)
  contracte/   ← Sample public contracts
  uncitral/    ← UNCITRAL model laws
  anpc/        ← ANPC consumer protection guides
```

### 4. Build the index (run once)
```bash
python scripts/build_index.py
```

### 5. Run the app
```bash
# Local
streamlit run src/app.py

# Docker
docker compose up
# App available at http://localhost:8501
```

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/build_index.py` | Build ChromaDB index from corpus (run once) |
| `scripts/evaluate_rag.py` | RAGAS evaluation → `logs/rag_evaluation.json` |
| `scripts/test_parser.py <pdf>` | Test DocumentParserAgent on a PDF |
| `scripts/test_retrieval.py` | Test RAGRetrievalAgent + generate heatmap |

## Pipeline
```
PDF Upload
  → DocumentParserAgent    (extract clauses, metadata)
  → RAGRetrievalAgent      (retrieve relevant corpus chunks per clause)
  → RiskAssessmentAgent    (classify risk: RIDICAT/MEDIU/SCAZUT/CONFORM)
  → quality_check          (retry if >40% NECUNOSCUT)
  → RecommendationAgent    (reformulate risky clauses)
  → Markdown Report
```

## Security Note
Never commit `.env` to Git. The `.gitignore` excludes it.
