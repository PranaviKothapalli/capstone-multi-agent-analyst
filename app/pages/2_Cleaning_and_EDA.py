import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import workspace_path
from src.state import init_session_state, inject_css, render_stepper, next_step_banner, guard_prerequisite
from src.agents.cleaning_agent import DataCleaningAgent
from src.agents.eda_agent import EDAAgent

st.set_page_config(page_title="Cleaning & EDA", page_icon="🧹", layout="wide")
init_session_state()
inject_css()
render_stepper("clean")
guard_prerequisite("clean", "⚠️ Please upload a dataset first on the Data Ingestion page.")

st.title("🧹 Data Cleaning & 🔍 Exploratory Analysis")
st.caption("Two specialized agents run in sequence: the Data Cleaning Agent, then the EDA Agent.")

run_col, info_col = st.columns([1, 3])
with run_col:
    run_clicked = st.button("▶ Run Cleaning & EDA Agents", type="primary", use_container_width=True,
                             disabled=st.session_state.cleaned_path is not None)
with info_col:
    if st.session_state.cleaned_path is not None:
        st.success("Already completed for this dataset. Re-upload a new file to run again, or continue to the next step.")

if run_clicked:
    with st.status("🧹 Data Cleaning Agent working...", expanded=True) as status:
        agent = DataCleaningAgent()
        ok, out = agent.run(st.session_state.raw_path, workspace_path("cleaned"))
        if not ok:
            status.update(label="Cleaning failed", state="error")
            st.error(out.get("error"))
        else:
            st.session_state.cleaned_path = out["cleaned_path"]
            st.session_state.clean_delta = out["delta"]
            st.write(f"Removed **{out['delta']['duplicates_removed']}** duplicate rows.")
            st.write(f"Imputed **{len(out['delta']['imputation'])}** columns with missing values.")
            status.update(label="Cleaning complete ✅", state="complete")

    if st.session_state.cleaned_path:
        with st.status("🔍 EDA Agent working...", expanded=True) as status:
            eda_agent = EDAAgent()
            ok, out = eda_agent.run(st.session_state.cleaned_path)
            if not ok:
                status.update(label="EDA failed", state="error")
                st.error(out.get("error"))
            else:
                st.session_state.eda_report = out
                status.update(label="EDA complete ✅", state="complete")

if st.session_state.clean_delta:
    st.divider()
    st.subheader("Cleaning summary")
    delta = st.session_state.clean_delta
    c1, c2, c3 = st.columns(3)
    c1.metric("Original shape", f"{delta['original_shape'][0]:,} × {delta['original_shape'][1]}")
    c2.metric("Final shape", f"{delta['final_shape'][0]:,} × {delta['final_shape'][1]}")
    c3.metric("Duplicates removed", delta["duplicates_removed"])

    if delta["imputation"]:
        st.markdown("**Imputation log**")
        imp_df = pd.DataFrame([
            {"column": k, "strategy": v["strategy"], "fill_value": v["value"], "cells_filled": v["count"]}
            for k, v in delta["imputation"].items()
        ])
        st.dataframe(imp_df, use_container_width=True)
    else:
        st.info("No missing values were found — dataset was already clean on that front.")

if st.session_state.eda_report:
    st.divider()
    st.subheader("Exploratory profile")
    report = st.session_state.eda_report

    tab1, tab2, tab3 = st.tabs(["📈 Missingness & Skew", "🔗 Correlation Matrix", "📋 Descriptive Stats"])

    with tab1:
        skew_df = pd.DataFrame(list(report["skew"].items()), columns=["column", "skewness"])
        miss_df = pd.DataFrame(list(report["missing_pct"].items()), columns=["column", "missing_%"])
        colA, colB = st.columns(2)
        with colA:
            if not skew_df.empty:
                fig = px.bar(skew_df, x="column", y="skewness", title="Feature Skewness", color_discrete_sequence=["#5B5FEF"])
                st.plotly_chart(fig, use_container_width=True)
        with colB:
            fig2 = px.bar(miss_df, x="column", y="missing_%", title="Missing Value % (pre-cleaning columns)", color_discrete_sequence=["#8B6CF1"])
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if report["correlation"]:
            corr_df = pd.DataFrame(report["correlation"])
            fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="Purples", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns to compute a correlation matrix.")

    with tab3:
        st.dataframe(pd.DataFrame(report["summary"]), use_container_width=True)

    st.divider()
    next_step_banner("eda")
    if st.button("Proceed to Feature Engineering & ML Studio →", type="primary"):
        st.switch_page("pages/3_ML_Studio.py")
