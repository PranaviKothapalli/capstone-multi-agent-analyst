import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import workspace_path
from src.state import init_session_state, inject_css, render_stepper, guard_prerequisite
from src.agents.report_agent import ReportGenerationAgent

st.set_page_config(page_title="Reports Hub", page_icon="📄", layout="wide")
init_session_state()
inject_css()
render_stepper("report")
guard_prerequisite("report", "⚠️ Please generate business insights first.")

st.title("📄 Reports Hub")
st.caption("Compiles diagnostics, model performance, charts and insights into one executive PDF report.")

if st.button("▶ Compile Executive Report", type="primary"):
    with st.status("📄 Report Generation Agent compiling document...", expanded=True) as status:
        df_shape = pd.read_csv(st.session_state.cleaned_path).shape
        agent = ReportGenerationAgent()
        out_path = workspace_path("reports", f"{st.session_state.dataset_name}_executive_report.pdf")
        ok, out = agent.run(
            st.session_state.dataset_name, df_shape, st.session_state.clean_delta,
            st.session_state.metrics, st.session_state.task_type,
            st.session_state.narrative, st.session_state.recommendations,
            st.session_state.chart_paths, out_path,
        )
        if ok:
            st.session_state.report_path = out["path"]
            fmt = "PDF" if out["is_pdf"] else "HTML (PDF engine unavailable — see note below)"
            status.update(label=f"Report compiled as {fmt} ✅", state="complete")
        else:
            status.update(label="Report generation failed", state="error")
            st.error(out.get("error"))

if st.session_state.report_path and os.path.exists(st.session_state.report_path):
    st.divider()
    is_pdf = st.session_state.report_path.endswith(".pdf")
    if not is_pdf:
        st.warning(
            "PDF rendering requires native system libraries (Cairo/Pango/GDK-Pixbuf) used by WeasyPrint. "
            "Your report was generated as HTML instead so nothing is blocked — see the README for one-line "
            "install instructions per OS, then re-compile to get a true PDF."
        )
    with open(st.session_state.report_path, "rb") as f:
        st.download_button(
            f"⬇️ Download {'PDF' if is_pdf else 'HTML'} Report",
            data=f, file_name=os.path.basename(st.session_state.report_path),
            mime="application/pdf" if is_pdf else "text/html",
            type="primary",
        )
    st.subheader("Report preview")
    with open(st.session_state.report_path, "r", encoding="utf-8", errors="ignore") if not is_pdf else open(st.session_state.report_path, "rb") as f:
        if is_pdf:
            st.info("PDF preview isn't rendered inline — use the download button above to view it.")
        else:
            st.components.v1.html(f.read(), height=700, scrolling=True)

    st.success("🎉 Pipeline complete! Visit the System Log Explorer to review the full audit trail of every agent action.")
