"""
app.py — Streamlit interface for the Legal Contract Analyzer.

Layout: sidebar (upload + controls) | main area (results).

Session state is used to avoid re-running the expensive pipeline
on every Streamlit interaction (slider move, expander click, etc.).
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# ── Page config must be first Streamlit call ──
st.set_page_config(
    page_title="Analizor Contract Juridic",
    page_icon="⚖️",
    layout="wide",
)

from src.dtos import RiskLevel, RecommendationDTO
from src.graph.workflow import build_graph

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
        # Build LangGraph workflow
        graph = build_graph()
        app = graph.compile()

        # Set up initial state
        initial_state = {
            "pdf_path": str(tmp_path),
            "parsed_doc": None,
            "context_map": {},
            "risk_map": {},
            "high_risk_alert": False,
            "recommendations": [],
            "report_path": "",
            "iteration": 0,
            "retrieval_k": 5,
            "retrieval_threshold": retrieval_threshold,
            "node_log": [],
        }

        # Stream execution to update the UI progress progressively
        state = initial_state.copy()
        for event in app.stream(initial_state):
            for node_name, updated_values in event.items():
                state.update(updated_values)
                
                if node_name == "parse_document":
                    status.info("📄 Parsare document... (Pasul 1/5)")
                    progress.progress(20)
                elif node_name == "retrieve_context":
                    status.info("🔍 Recuperare context juridic... (Pasul 2/5)")
                    progress.progress(40)
                elif node_name == "assess_risk":
                    status.info("⚠️ Evaluare riscuri... (Pasul 3/5)")
                    progress.progress(60)
                elif node_name == "flag_high_risk":
                    status.info("🚨 Verificare alerte risc ridicat... (Pasul 4/5)")
                    progress.progress(85)
                elif node_name == "generate_recommendations":
                    status.info("📝 Generare recomandări... (Pasul 5/5)")
                    progress.progress(95)
                elif node_name == "compile_report":
                    status.info("📊 Compilare raport final...")
                    progress.progress(100)

        # Cache results in expected format
        st.session_state[cache_key] = {
            "parsed_doc": state["parsed_doc"],
            "risk_map": state["risk_map"],
            "recommendations": {r.clause_id: r for r in state["recommendations"]},
            "report_path": state["report_path"],
            "high_risk_alert": state["high_risk_alert"],
        }

        status.success("✅ Analiză completă!")
        time.sleep(0.5)
        status.empty()
        progress.empty()

    except Exception as exc:
        st.error(f"Eroare la analiză: {exc}")
        st.stop()

# ── Display results ──
if cache_key not in st.session_state:
    st.stop()

data = st.session_state[cache_key]
parsed_doc = data["parsed_doc"]
risk_map = data["risk_map"]
recommendations = data["recommendations"]
report_path = data["report_path"]
high_risk_alert = data.get("high_risk_alert", False)

# High-risk banner (Modified to warning as per guidelines)
high_risk_count = sum(1 for r in risk_map.values() if r.risk_level == RiskLevel.RIDICAT)
if high_risk_alert or high_risk_count >= risk_alert_threshold:
    st.warning(
        f"🚨 **ATENȚIE**: Au fost detectate **{high_risk_count} clauze cu risc RIDICAT**. "
        "Consultați un jurist înainte de semnare.",
        icon="⚠️",
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
