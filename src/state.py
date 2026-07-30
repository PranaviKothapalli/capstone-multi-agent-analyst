"""
Shared session-state initialization and reusable UI building blocks so every
page in the app looks and behaves consistently, and the user always sees a
guided "what's next" workflow instead of relying purely on the sidebar.
"""
import streamlit as st

STAGES = [
    ("upload", "Upload", "📤"),
    ("clean", "Clean", "🧹"),
    ("eda", "Explore", "🔍"),
    ("features", "Features", "🧬"),
    ("model", "Model", "🤖"),
    ("visuals", "Visuals", "📊"),
    ("insights", "Insights", "💡"),
    ("report", "Report", "📄"),
]

PAGE_FOR_STAGE = {
    "upload": "pages/1_Data_Ingestion.py",
    "clean": "pages/2_Cleaning_and_EDA.py",
    "eda": "pages/2_Cleaning_and_EDA.py",
    "features": "pages/3_ML_Studio.py",
    "model": "pages/3_ML_Studio.py",
    "visuals": "pages/4_Visualizations.py",
    "insights": "pages/5_Business_Insights.py",
    "report": "pages/6_Reports_Hub.py",
}

DEFAULTS = {
    "dataset_name": None,
    "raw_path": None,
    "raw_df": None,
    "validation_issues": [],
    "cleaned_path": None,
    "clean_delta": None,
    "eda_report": None,
    "target_col": None,
    "task_type": None,
    "feat_path": None,
    "feat_map": None,
    "model_path": None,
    "model_name": None,
    "metrics": None,
    "feature_importances": None,
    "chart_paths": None,
    "narrative": None,
    "narrative_source": None,
    "report_path": None,
    "industry": "General / Cross-Industry",
}


def init_session_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def stage_status(key: str) -> str:
    """Returns 'done', 'active', or 'locked' for a workflow stage."""
    s = st.session_state
    completion = {
        "upload": s.raw_df is not None,
        "clean": s.cleaned_path is not None,
        "eda": s.eda_report is not None,
        "features": s.feat_path is not None,
        "model": s.metrics is not None,
        "visuals": bool(s.chart_paths),
        "insights": s.narrative is not None,
        "report": s.report_path is not None,
    }
    order = [k for k, _, _ in STAGES]
    idx = order.index(key)
    if completion[key]:
        return "done"
    prereqs_met = all(completion[order[i]] for i in range(idx))
    return "active" if prereqs_met else "locked"


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 3.25rem; padding-bottom: 3rem; max-width: 1250px;}
        header[data-testid="stHeader"] {background: rgba(0,0,0,0);}
        div[data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #E7E9F5; border-radius: 14px;
            padding: 14px 16px; box-shadow: 0 1px 3px rgba(20,20,50,0.04);
        }
        .agent-card {
            background: #FFFFFF; border: 1px solid #E7E9F5; border-radius: 16px;
            padding: 20px 22px; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(20,20,50,0.05);
        }
        .agent-card h4 {margin-top: 0; margin-bottom: 6px;}
        .pill {
            display: inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; letter-spacing: .02em;
        }
        .pill-done {background: #E4F6EA; color: #1E7A46;}
        .pill-active {background: #EAEAFF; color: #4A4EDB;}
        .pill-locked {background: #F1F1F4; color: #8A8DA0;}
        .pill-warn {background: #FFF3D9; color: #9A6B00;}
        .pill-error {background: #FDE7E7; color: #B3261E;}
        .step-track {display:flex; gap:6px; margin-top: 6px; margin-bottom: 22px; flex-wrap: wrap;}
        .step-chip {
            flex: 1; min-width: 110px; text-align:center; padding: 10px 8px;
            border-radius: 12px; font-size: 0.82rem; font-weight: 600; border: 1px solid transparent;
        }
        .step-done {background:#EEF9F1; color:#1E7A46; border-color:#CFEFDA;}
        .step-active {background:#EFEFFF; color:#4A4EDB; border-color:#D6D6FF;}
        .step-locked {background:#F6F6F9; color:#A6A9BE; border-color:#EEEEF3;}
        .hero {
            background: linear-gradient(135deg, #5B5FEF 0%, #8B6CF1 100%);
            color: white; border-radius: 20px; padding: 34px 36px; margin-bottom: 24px;
        }
        .hero h1 {color:white; margin-bottom: 6px;}
        .hero p {color: #EDEBFF; font-size: 1.02rem; margin-bottom:0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(active_key: str | None = None):
    """Horizontal, always-visible workflow tracker used at the top of every page."""
    chips = []
    for key, label, icon in STAGES:
        status = stage_status(key)
        css_class = {"done": "step-done", "active": "step-active", "locked": "step-locked"}[status]
        marker = "✓ " if status == "done" else ("➤ " if key == active_key or status == "active" else "")
        chips.append(f'<div class="step-chip {css_class}">{icon} {marker}{label}</div>')
    st.markdown(f'<div class="step-track">{"".join(chips)}</div>', unsafe_allow_html=True)


def status_pill(text: str, kind: str = "active"):
    st.markdown(f'<span class="pill pill-{kind}">{text}</span>', unsafe_allow_html=True)


def next_step_banner(current_key: str):
    """Tells the user explicitly what to do next, with a jump-to button."""
    order = [k for k, _, _ in STAGES]
    idx = order.index(current_key)
    if idx + 1 < len(order):
        nxt_key, nxt_label, nxt_icon = STAGES[idx + 1]
        if stage_status(current_key) == "done":
            st.success(f"✅ This step is complete. Next up: **{nxt_icon} {nxt_label}**")
        else:
            st.info(f"👉 Finish this step, then continue to **{nxt_icon} {nxt_label}**.")
    else:
        st.success("🎉 You've completed the full pipeline!")


def guard_prerequisite(key: str, message: str):
    """Stops page execution early with a friendly notice if prerequisites aren't met."""
    if stage_status(key) == "locked":
        st.warning(message)
        st.stop()


def show_image(path: str, caption: str | None = None):
    """Version-safe wrapper around st.image().

    Different Streamlit releases have used different keyword arguments to make
    an image fill its container (`use_container_width` in newer versions,
    `use_column_width` in older ones). Calling the wrong one raises a
    TypeError and crashes the page. This helper tries the modern API first
    and gracefully falls back so the app works regardless of which Streamlit
    version is installed.
    """
    try:
        st.image(path, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(path, caption=caption, use_column_width=True)
        except TypeError:
            st.image(path, caption=caption)
