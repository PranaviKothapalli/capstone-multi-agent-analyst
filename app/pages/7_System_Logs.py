import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.state import init_session_state, inject_css, render_stepper
from src.database import fetch_logs, clear_logs

st.set_page_config(page_title="System Log Explorer", page_icon="🗂️", layout="wide")
init_session_state()
inject_css()
render_stepper()

st.title("🗂️ System Log Explorer")
st.caption("Every agent action, tool call, and status transition is written to a local SQLite audit table in real time.")

c1, c2, c3 = st.columns([1, 1, 3])
with c1:
    limit = st.selectbox("Rows to show", [50, 100, 300, 1000], index=2)
with c2:
    if st.button("🔄 Refresh"):
        st.rerun()
with c3:
    if st.button("🗑️ Clear log history"):
        clear_logs()
        st.rerun()

logs = fetch_logs(limit=limit)
if not logs:
    st.info("No agent activity logged yet. Run the pipeline from Data Ingestion onward to populate this view.")
else:
    df = pd.DataFrame(logs)

    status_counts = df["status"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total events", len(df))
    m2.metric("Successes", int(status_counts.get("success", 0)))
    m3.metric("Errors", int(status_counts.get("error", 0)))
    m4.metric("Unique agents", df["agent"].nunique())

    agent_filter = st.multiselect("Filter by agent", sorted(df["agent"].unique()))
    status_filter = st.multiselect("Filter by status", sorted(df["status"].unique()))

    filtered = df.copy()
    if agent_filter:
        filtered = filtered[filtered["agent"].isin(agent_filter)]
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]

    def _badge(status):
        color = {"success": "🟢", "error": "🔴", "started": "🟡", "info": "🔵"}.get(status, "⚪")
        return f"{color} {status}"

    filtered = filtered.copy()
    filtered["status"] = filtered["status"].apply(_badge)
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=500)
