import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import workspace_path
from src.state import init_session_state, inject_css, render_stepper, next_step_banner, guard_prerequisite
from src.agents.feature_engineering_agent import FeatureEngineeringAgent
from src.agents.ml_agent import MachineLearningAgent
from src.tools.data_tools import infer_task_type, validate_classification_target, validate_regression_target

st.set_page_config(page_title="ML Studio", page_icon="🤖", layout="wide")
init_session_state()
inject_css()
render_stepper("model")
guard_prerequisite("features", "⚠️ Please run Cleaning & EDA first.")

st.title("🧬 Feature Engineering & 🤖 Machine Learning Studio")

df_clean = pd.read_csv(st.session_state.cleaned_path)

st.subheader("1. Select your target column")
target_col = st.selectbox(
    "What are you predicting?", options=list(df_clean.columns),
    index=list(df_clean.columns).index(st.session_state.target_col) if st.session_state.target_col in df_clean.columns else len(df_clean.columns) - 1,
)
st.session_state.target_col = target_col

detected_task_type = infer_task_type(df_clean[target_col])

col_a, col_b, col_c = st.columns(3)
col_a.metric("Detected column dtype", str(df_clean[target_col].dtype))
col_b.metric("Unique values", df_clean[target_col].nunique())
col_c.metric("Detected task type", "Classification" if detected_task_type == "classification" else "Regression")

# --- Target validation, shown live so the user never has to hit "Train" to find out it will fail ---
if detected_task_type == "classification":
    target_validation = validate_classification_target(df_clean[target_col])
else:
    target_validation = validate_regression_target(df_clean[target_col])

for err in target_validation["errors"]:
    st.error(f"🚫 {err}")
for warn in target_validation["warnings"]:
    st.warning(f"⚠️ {warn}")

if detected_task_type == "classification" and target_validation["is_valid"]:
    with st.expander(f"Class distribution ({target_validation['n_classes']} classes, smallest has {target_validation['min_class_count']} sample(s))"):
        dist_df = pd.DataFrame(
            list(target_validation["class_counts"].items()), columns=["class", "count"]
        ).sort_values("count", ascending=False)
        fig = px.bar(dist_df, x="class", y="count", title="Samples per class", color_discrete_sequence=["#5B5FEF"])
        st.plotly_chart(fig, use_container_width=True)

target_is_valid = target_validation["is_valid"]
if not target_is_valid:
    st.info("👆 Choose a different target column, or clean up the classes above, before training.")

st.divider()
st.subheader("2. Run Feature Engineering Agent")
if st.button("▶ Build Engineered Feature Preview", disabled=st.session_state.eda_report is None):
    with st.status("🧬 Feature Engineering Agent working...", expanded=True) as status:
        agent = FeatureEngineeringAgent()
        ok, out = agent.run(st.session_state.cleaned_path, st.session_state.eda_report, workspace_path("features"))
        if not ok:
            status.update(label="Feature engineering failed", state="error")
            st.error(out.get("error"))
        else:
            st.session_state.feat_path = out["feat_path"]
            st.session_state.feat_map = out["pipeline_map"]
            status.update(label="Feature engineering complete ✅", state="complete")

if st.session_state.feat_map:
    fm = st.session_state.feat_map
    c1, c2, c3 = st.columns(3)
    c1.metric("Categorical columns encoded", len(fm["encoded_categorical_columns"]))
    c2.metric("Numeric columns scaled", len(fm["scaled_numeric_columns"]))
    c3.metric("Interaction terms created", len(fm["interaction_terms"]))
    with st.expander("Preview engineered feature table"):
        st.dataframe(pd.read_csv(st.session_state.feat_path).head(15), use_container_width=True)
    if fm["interaction_terms"]:
        with st.expander("Interaction terms detail"):
            st.dataframe(pd.DataFrame(fm["interaction_terms"]), use_container_width=True)

st.divider()
st.subheader("3. Train models with the Machine Learning Agent")
st.caption("Trains multiple candidate models under leakage-safe stratified/K-fold cross-validation, then selects the best.")

train_disabled = st.session_state.feat_path is None or not target_is_valid
if st.button("▶ Train & Evaluate Models", type="primary", disabled=train_disabled):
    with st.status("🤖 Machine Learning Agent training candidate models...", expanded=True) as status:
        agent = MachineLearningAgent()
        model_out_path = workspace_path("models", "best_production_model.pkl")
        ok, out = agent.run(st.session_state.cleaned_path, target_col, model_out_path)
        if not ok:
            status.update(label="Training failed", state="error")
            st.error(f"🚫 {out.get('error')}")
            st.caption("No changes were made — pick a different target column above and try again.")
        else:
            st.session_state.model_path = out["model_path"]
            st.session_state.metrics = out["metrics"]
            st.session_state.task_type = out["task_type"]
            st.session_state.feature_importances = out["feature_importances"]
            st.write(f"Task type detected: **{out['task_type']}**")
            st.write(f"Winning model: **{out['metrics']['model_name']}**")
            status.update(label="Training complete ✅", state="complete")

if st.session_state.metrics:
    st.divider()
    st.subheader("Model Performance Leaderboard")
    metrics = st.session_state.metrics
    lb = pd.DataFrame([
        {"Model": name, "CV Score": round(info["cv_score"], 4)} for name, info in metrics["cv_leaderboard"].items()
    ]).sort_values("CV Score", ascending=False)
    st.dataframe(lb, use_container_width=True, hide_index=True)

    st.markdown(f"#### 🏆 Best model: `{metrics['model_name']}`")
    if st.session_state.task_type == "classification":
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        m2.metric("Macro F1", f"{metrics.get('f1_macro', 0):.3f}")
        m3.metric("Precision", f"{metrics.get('precision_macro', 0):.3f}")
        m4.metric("Recall", f"{metrics.get('recall_macro', 0):.3f}")
        m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics.get("roc_auc") else "N/A")

        if metrics.get("confusion_matrix"):
            with st.expander("Confusion matrix", expanded=True):
                cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics.get("labels"), columns=metrics.get("labels"))
                fig = px.imshow(
                    cm_df, text_auto=True, color_continuous_scale="Purples",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion Matrix (held-out test set)",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
        m2.metric("MAE", f"{metrics.get('mae', 0):.3f}")
        m3.metric("R²", f"{metrics.get('r2', 0):.3f}")
        m4.metric("Adjusted R²", f"{metrics.get('adjusted_r2', 0):.3f}")

    if st.session_state.feature_importances:
        with st.expander("Feature importances (top 15)"):
            fi_df = pd.DataFrame(list(st.session_state.feature_importances.items()), columns=["feature", "importance"]).head(15)
            st.dataframe(fi_df, use_container_width=True)

    st.divider()
    next_step_banner("model")
    if st.button("Proceed to Visualizations →", type="primary"):
        st.switch_page("pages/4_Visualizations.py")
