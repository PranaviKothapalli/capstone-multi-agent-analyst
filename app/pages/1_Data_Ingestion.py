import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import settings, workspace_path
from src.state import init_session_state, inject_css, render_stepper, next_step_banner
from src.tools.data_tools import load_dataset, validate_dataset
from src.database import log_event

st.set_page_config(page_title="Data Ingestion Portal", page_icon="📤", layout="wide")
init_session_state()
inject_css()
render_stepper("upload")

st.title("📤 Dataset Ingestion Portal")
st.caption("Drag and drop a CSV or Parquet file. The system validates structural integrity before anything else runs.")

industries = [
    "General / Cross-Industry", "Corporate Finance", "Healthcare Systems",
    "Retail Commerce", "Manufacturing", "Marketing Operations",
]
st.session_state.industry = st.selectbox(
    "Select the industry context for this analysis (customizes downstream business insights):",
    industries, index=industries.index(st.session_state.industry) if st.session_state.industry in industries else 0,
)

uploaded = st.file_uploader("Upload dataset", type=["csv", "parquet"], accept_multiple_files=False)

if uploaded is not None:
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > settings.max_file_mb:
        st.error(f"File is {size_mb:.1f} MB, which exceeds the {settings.max_file_mb} MB limit.")
    else:
        raw_dir = workspace_path("uploads")
        raw_path = os.path.join(raw_dir, uploaded.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.status("Validating dataset...", expanded=True) as status:
            try:
                df = load_dataset(raw_path)
                st.write(f"Loaded shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
                result = validate_dataset(df, settings.min_rows, settings.max_rows, settings.min_cols, settings.max_cols)
                for w in result["warnings"]:
                    st.warning(w)
                for e in result["errors"]:
                    st.error(e)

                if result["errors"]:
                    status.update(label="Validation failed", state="error")
                    log_event("IngestionUI", "validate_upload", "error", result["errors"])
                else:
                    st.session_state.raw_df = df
                    st.session_state.raw_path = raw_path
                    st.session_state.dataset_name = os.path.splitext(uploaded.name)[0]
                    st.session_state.validation_issues = result["warnings"]
                    # Reset any downstream state from a previous dataset
                    for k in ("cleaned_path", "clean_delta", "eda_report", "feat_path", "feat_map",
                              "model_path", "metrics", "chart_paths", "narrative", "report_path"):
                        st.session_state[k] = None
                    status.update(label="Validation passed ✅", state="complete")
                    log_event("IngestionUI", "validate_upload", "success", {"shape": list(df.shape)})
            except Exception as e:
                status.update(label="Failed to read file", state="error")
                st.error(f"Could not parse file: {e}")

if st.session_state.raw_df is not None:
    df = st.session_state.raw_df
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Column schema"):
        st.dataframe(
            df.dtypes.astype(str).reset_index().rename(columns={"index": "column", 0: "dtype"}),
            use_container_width=True,
        )

    st.divider()
    next_step_banner("upload")
    if st.button("Proceed to Cleaning & EDA →", type="primary"):
        st.switch_page("pages/2_Cleaning_and_EDA.py")
else:
    st.info("👆 Upload a dataset to get started. Sample public datasets (Kaggle churn, UCI heart disease, etc.) work great for testing.")
