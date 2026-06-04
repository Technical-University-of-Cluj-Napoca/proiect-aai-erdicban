"""
app.py — Streamlit interface for the Legal Contract Analyzer.

Layout: sidebar (upload + controls) | main area (results).

Session state is used to avoid re-running the expensive pipeline
on every Streamlit interaction (slider move, expander click, etc.).
"""

from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config must be first Streamlit call ──
st.set_page_config(
    page_title="Analizor Contract Juridic",
    page_icon="⚖️",
    layout="wide",
)

from src.agents.parser_agent import DocumentParserAgent
from src.agents.retrieval_agent import RAGRetrievalAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.dtos import RiskLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")

# ── Risk colour palette ──
RISK_COLORS = {
    RiskLevel.RIDICAT:    "#ffe3e3",
    RiskLevel.MEDIU:      "#fff3bf",
    RiskLevel.SCAZUT:     "#fff9c4",
    RiskLevel.CONFORM:    "#d3f9d8",
    RiskLevel.NECUNOSCUT: "#f0f0f0",
}
RISK_EMOJI = {
    RiskLevel.RIDICAT: "🔴",
    RiskLevel.MEDIU: "🟡",
    RiskLevel.SCAZUT: "🟢",
    RiskLevel.CONFORM: "✅",
    RiskLevel.NECUNOSCUT: "⚪",
}


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ Analizor Contract")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Încarcă contract PDF",
        type=["pdf"],
        help="Suportă contracte digitale (nu scanate fără OCR).",
    )

    st.markdown("### Parametri")
    retrieval_threshold = st.slider(
        "Prag relevanță retrieval",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="Chunk-urile cu similaritate sub acest prag sunt ignorate.",
    )
    risk_alert_threshold = st.slider(
        "Prag alertă risc ridicat (nr. clauze)",
        min_value=1,
        max_value=10,
        value=1,
        help="Câte clauze RIDICAT declanșează bannerul de alertă.",
    )

    analyze_btn = st.button("🔍 Analizează Contractul", use_container_width=True)

    st.markdown("---")
    st.info(
        "**Disclaimer**: Acest sistem nu oferă consultanță juridică. "
        "Rezultatele trebuie validate de un jurist înainte de orice acțiune legală.",
        icon="ℹ️",
    )


# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────

st.title("Analizor de Contracte Juridice cu AI")

if not uploaded_file:
    st.info("Încarcă un fișier PDF în bara laterală pentru a începe analiza.")
    st.stop()

# Save uploaded file temporarily
tmp_path = Path(f"/tmp/{uploaded_file.name}")
tmp_path.write_bytes(uploaded_file.read())

# ── Run pipeline (cached in session_state) ──
cache_key = f"results_{uploaded_file.name}_{retrieval_threshold}"

if analyze_btn or (cache_key not in st.session_state):
    if analyze_btn:
        # Clear old results when user explicitly clicks Analyze
        for key in list(st.session_state.keys()):
            if key.startswith("results_"):
                del st.session_state[key]

    progress = st.progress(0, text="Inițializare...")
    status = st.empty()

    try:
        # Step 1: Parse
        status.info("📄 Parsare document... (Pasul 1/4)")
        progress.progress(10)
        parser = DocumentParserAgent()
        parsed_doc = parser.parse(str(tmp_path))
        progress.progress(25)

        # Step 2: Retrieve
        status.info("🔍 Recuperare context juridic... (Pasul 2/4)")
        retrieval_agent = RAGRetrievalAgent(
            persist_directory=VECTORSTORE_DIR,
            threshold=retrieval_threshold,
        )
        context_map = {}
        for i, clause in enumerate(parsed_doc.clauses):
            context_map[clause.id] = retrieval_agent.retrieve(clause)
            progress.progress(25 + int(25 * i / max(len(parsed_doc.clauses), 1)))

        # Step 3: Risk assessment
        status.info("⚠️ Evaluare riscuri... (Pasul 3/4)")
        risk_agent = RiskAssessmentAgent()
        risk_map = {}
        for i, clause in enumerate(parsed_doc.clauses):
            chunks = context_map.get(clause.id, [])
            risk_map[clause.id] = risk_agent.assess(clause, chunks)
            progress.progress(50 + int(25 * i / max(len(parsed_doc.clauses), 1)))

        # Step 4: Recommendations
        status.info("📝 Generare recomandări... (Pasul 4/4)")
        rec_agent = RecommendationAgent()
        recommendations = {}
        risky_clauses = [
            c for c in parsed_doc.clauses
            if risk_map[c.id].risk_level in (RiskLevel.RIDICAT, RiskLevel.MEDIU)
        ]
        for i, clause in enumerate(risky_clauses):
            risk = risk_map[clause.id]
            chunks = context_map.get(clause.id, [])
            recommendations[clause.id] = rec_agent.recommend(clause, risk, chunks)
            progress.progress(75 + int(20 * i / max(len(risky_clauses), 1)))

        # Generate and save report
        clause_map = {c.id: c for c in parsed_doc.clauses}
        results_for_report = [
            (clause_map[cid], risk, recommendations.get(cid))
            for cid, risk in risk_map.items()
            if cid in clause_map
        ]
        # Filter out None recommendations
        results_for_report = [
            (c, r, rec) for c, r, rec in results_for_report if rec is not None
        ]

        from src.dtos import RecommendationDTO
        full_results = []
        for clause in parsed_doc.clauses:
            risk = risk_map[clause.id]
            rec = recommendations.get(clause.id, RecommendationDTO(
                clause_id=clause.id,
                original_text=clause.text,
            ))
            full_results.append((clause, risk, rec))

        import os
        os.makedirs("data", exist_ok=True)
        report_path = f"data/report_{uploaded_file.name.replace('.pdf', '')}.md"
        rec_agent.generate_report(full_results, report_path)

        # Cache results
        st.session_state[cache_key] = {
            "parsed_doc": parsed_doc,
            "risk_map": risk_map,
            "recommendations": recommendations,
            "report_path": report_path,
        }

        progress.progress(100)
        status.success("✅ Analiză completă!")
        time.sleep(0.5)
        status.empty()
        progress.empty()

    except Exception as exc:
        st.error(f"Eroare la analiză: {exc}")
        logger.exception("Pipeline error")
        st.stop()

# ── Display results ──
if cache_key not in st.session_state:
    st.stop()

data = st.session_state[cache_key]
parsed_doc = data["parsed_doc"]
risk_map = data["risk_map"]
recommendations = data["recommendations"]
report_path = data["report_path"]

# High-risk banner
high_risk_count = sum(1 for r in risk_map.values() if r.risk_level == RiskLevel.RIDICAT)
if high_risk_count >= risk_alert_threshold:
    st.error(
        f"🚨 **ATENȚIE**: Au fost detectate **{high_risk_count} clauze cu risc RIDICAT**. "
        "Consultați un jurist înainte de semnare.",
        icon="🚨",
    )

# Contract metadata
with st.expander("📋 Metadate Contract", expanded=False):
    meta = parsed_doc.metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Titlu", meta.title[:30])
    col2.metric("Pagini", meta.page_count)
    col3.metric("Clauze extrase", len(parsed_doc.clauses))
    if meta.parties:
        st.write("**Părți contractante:**", ", ".join(p.name for p in meta.parties))

# Risk distribution summary
st.markdown("### 📊 Distribuție Riscuri")
risk_counts = {level: 0 for level in RiskLevel}
for r in risk_map.values():
    risk_counts[r.risk_level] += 1

cols = st.columns(len(RiskLevel))
for col, (level, count) in zip(cols, risk_counts.items()):
    col.metric(
        f"{RISK_EMOJI[level]} {level.value}",
        count,
    )

st.markdown("---")

# Clause table
st.markdown("### 📜 Clauze Analizate")

risk_order = {
    RiskLevel.RIDICAT: 0,
    RiskLevel.MEDIU: 1,
    RiskLevel.SCAZUT: 2,
    RiskLevel.CONFORM: 3,
    RiskLevel.NECUNOSCUT: 4,
}
sorted_clauses = sorted(
    parsed_doc.clauses,
    key=lambda c: risk_order.get(risk_map[c.id].risk_level, 99),
)

for clause in sorted_clauses:
    risk = risk_map[clause.id]
    bg_color = RISK_COLORS[risk.risk_level]
    emoji = RISK_EMOJI[risk.risk_level]

    label = f"{emoji} {clause.id} — {risk.risk_level.value} | {clause.type.value} | Pagina {clause.page}"

    with st.expander(label):
        st.markdown(
            f"<div style='background:{bg_color};padding:10px;border-radius:6px;'>",
            unsafe_allow_html=True,
        )

        st.markdown("**Text original:**")
        st.write(clause.text[:600] + ("..." if len(clause.text) > 600 else ""))

        if risk.issues:
            st.markdown("**Probleme identificate:**")
            for issue in risk.issues:
                st.write(f"• {issue}")

        if risk.references:
            st.markdown("**Referințe legislative:**")
            st.code(", ".join(risk.references))

        rec = recommendations.get(clause.id)
        if rec and rec.reformulated_text:
            st.markdown("**✏️ Reformulare propusă:**")
            st.info(rec.reformulated_text)
            if rec.explanation:
                st.caption(rec.explanation)

        st.markdown("</div>", unsafe_allow_html=True)

# Download button
st.markdown("---")
if Path(report_path).exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()
    st.download_button(
        label="📥 Descarcă Raport Markdown",
        data=report_content,
        file_name=Path(report_path).name,
        mime="text/markdown",
        use_container_width=True,
    )

# Persistent disclaimer at the bottom
st.markdown("---")
st.warning(
    "**⚠️ Disclaimer important**: Acest sistem este un instrument de suport, nu un înlocuitor "
    "al consultanței juridice profesionale. Toate recomandările trebuie verificate de un avocat "
    "calificat. Sistemul poate genera erori sau omisiuni.",
    icon="⚠️",
)
