import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings, ensure_workspace
from src.state import init_session_state, inject_css, render_stepper, stage_status, STAGES

st.set_page_config(page_title="AI Data Analyst | Multi-Agent Platform", page_icon="🧠", layout="wide")
ensure_workspace()
init_session_state()
inject_css()

st.markdown(
    """
    <div class="hero">
        <h1>🧠 Multi-Agent AI Data Analyst</h1>
        <p>Upload any tabular dataset and let eight specialized AI agents clean, explore, engineer,
        model, visualize, and report on it — end to end, with zero manual coding.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_stepper()

st.subheader("Your workflow at a glance")
cols = st.columns(4)
labels_top = STAGES[:4]
labels_bottom = STAGES[4:]

def render_card(col, key, label, icon):
    status = stage_status(key)
    kind = {"done": "🟢 Complete", "active": "🟣 Up next", "locked": "⚪ Locked"}[status]
    with col:
        st.markdown(
            f"""<div class="agent-card"><h4>{icon} {label}</h4><span class="pill pill-{"done" if status=="done" else ("active" if status=="active" else "locked")}">{kind}</span></div>""",
            unsafe_allow_html=True,
        )

for col, (key, label, icon) in zip(cols, labels_top):
    render_card(col, key, label, icon)
cols2 = st.columns(4)
for col, (key, label, icon) in zip(cols2, labels_bottom):
    render_card(col, key, label, icon)

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("How it works")
    st.markdown(
        """
        1. **📤 Upload** a CSV/Parquet dataset — the system validates its structure automatically.
        2. **🧹 Cleaning & 🔍 EDA agents** remove duplicates, impute missing values, and profile the data.
        3. **🧬 Feature Engineering & 🤖 ML agents** build a leak-safe pipeline and train multiple model families.
        4. **📊 Visualization agent** produces publication-quality charts.
        5. **💡 Insights agent** (powered by Groq, with an offline fallback) writes the executive narrative.
        6. **📄 Report agent** compiles everything into a downloadable PDF business report.

        Use the tracker above, or the sidebar, to move through each stage. Every agent action is logged
        and auditable from the **System Log Explorer**.
        """
    )
with right:
    st.subheader("System status")
    st.metric("LLM Insights Provider", "Groq (connected)" if settings.llm_enabled else "Offline template mode")
    if not settings.llm_enabled:
        st.caption("Add a free `GROQ_API_KEY` to your `.env` file to unlock AI-generated narratives. The app works fully without it.")
    st.metric("Workspace directory", settings.workspace_dir)
    st.metric("Audit database", os.path.basename(settings.sqlite_db_path))

st.divider()
st.subheader("Ready to begin?")
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🚀 Start with Data Ingestion", type="primary", use_container_width=True):
        st.switch_page("pages/1_Data_Ingestion.py")
with c2:
    st.caption("You can jump back to Home anytime from the sidebar to check overall progress.")
