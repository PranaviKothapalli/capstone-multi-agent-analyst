import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import workspace_path
from src.state import init_session_state, inject_css, render_stepper, next_step_banner, guard_prerequisite, show_image
from src.agents.visualization_agent import VisualizationAgent

st.set_page_config(page_title="Visualization Gallery", page_icon="📊", layout="wide")
init_session_state()
inject_css()
render_stepper("visuals")
guard_prerequisite("visuals", "⚠️ Please train a model in the ML Studio first.")

st.title("📊 Visualization Gallery")
st.caption("The Visualization Agent renders high-contrast, presentation-ready charts from your cleaned data and model results.")

if st.button("▶ Generate Visuals", type="primary", disabled=bool(st.session_state.chart_paths)):
    with st.status("📊 Visualization Agent rendering charts...", expanded=True) as status:
        agent = VisualizationAgent()
        ok, paths = agent.run(
            st.session_state.cleaned_path, st.session_state.target_col,
            st.session_state.metrics, st.session_state.feature_importances,
            workspace_path("visualizations"),
        )
        if ok and paths:
            st.session_state.chart_paths = paths
            status.update(label=f"Generated {len(paths)} charts ✅", state="complete")
        else:
            status.update(label="Visualization generation failed", state="error")

if st.session_state.chart_paths:
    st.divider()
    cols = st.columns(2)
    for i, path in enumerate(st.session_state.chart_paths):
        if os.path.exists(path):
            with cols[i % 2]:
                show_image(path, caption=os.path.basename(path).replace("_", " ").replace(".png", "").title())

    st.divider()
    next_step_banner("visuals")
    if st.button("Proceed to Business Insights →", type="primary"):
        st.switch_page("pages/5_Business_Insights.py")
else:
    st.info("Click **Generate Visuals** above to render the chart gallery.")
