"""
Visualization Agent tools (Section 11 of the handbook).
Enforces deterministic, professional styling: desaturated palette, fixed
10x6 in sizing, full titles/axis labels/legends, high-contrast output
suitable for executive presentation decks and the PDF report.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#5B5FEF", "#8B6CF1", "#4FB0AE", "#E4A94F", "#B3261E", "#3E4C6D"]
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1E2130",
    "text.color": "#1E2130",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_distribution_plots(df: pd.DataFrame, out_dir: str, max_cols: int = 6) -> list[str]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:max_cols]
    paths = []
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, color=PALETTE[0], ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        paths.append(_save(fig, out_dir, f"dist_{col}.png"))
    return paths


def generate_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> str | None:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(numeric_only=True).round(2), annot=True, fmt=".2f",
        cmap="mako", ax=ax, linewidths=0.5, cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Feature Correlation Matrix")
    return _save(fig, out_dir, "correlation_heatmap.png")


def generate_target_relationship_plots(df: pd.DataFrame, target_col: str, out_dir: str, max_cols: int = 4) -> list[str]:
    if target_col not in df.columns:
        return []
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col][:max_cols]
    paths = []
    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.scatter(df[col], df[target_col], alpha=0.5, color=PALETTE[1], edgecolor="none")
        ax.set_title(f"{col} vs {target_col}")
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        paths.append(_save(fig, out_dir, f"scatter_{col}_vs_target.png"))
    return paths


def generate_feature_importance_chart(importances: dict, out_dir: str, top_n: int = 15) -> str | None:
    if not importances:
        return None
    items = sorted(importances.items(), key=lambda x: abs(x[1]))[-top_n:]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))
    ax.barh(labels, values, color=PALETTE[2])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance / Weight")
    ax.set_ylabel("Feature")
    return _save(fig, out_dir, "feature_importance.png")


def generate_metrics_chart(metrics: dict, out_dir: str) -> str | None:
    leaderboard = metrics.get("cv_leaderboard") if metrics else None
    if not leaderboard:
        return None
    names = list(leaderboard.keys())
    scores = [leaderboard[n]["cv_score"] for n in names]
    fig, ax = plt.subplots()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    ax.bar(names, scores, color=colors)
    ax.set_title("Model Comparison (Cross-Validation Score)")
    ax.set_xlabel("Model")
    ax.set_ylabel("CV Score")
    return _save(fig, out_dir, "model_comparison.png")


def generate_confusion_matrix_plot(cm, labels, out_dir: str) -> str | None:
    if not cm:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return _save(fig, out_dir, "confusion_matrix.png")


def generate_all_visuals(df: pd.DataFrame, target_col: str | None, metrics: dict, importances: dict, out_dir: str) -> list[str]:
    paths = []
    paths += generate_distribution_plots(df, out_dir)
    hm = generate_correlation_heatmap(df, out_dir)
    if hm:
        paths.append(hm)
    if target_col:
        paths += generate_target_relationship_plots(df, target_col, out_dir)
    fi = generate_feature_importance_chart(importances, out_dir)
    if fi:
        paths.append(fi)
    mc = generate_metrics_chart(metrics, out_dir)
    if mc:
        paths.append(mc)
    if metrics and metrics.get("confusion_matrix"):
        cm_path = generate_confusion_matrix_plot(metrics["confusion_matrix"], metrics.get("labels", []), out_dir)
        if cm_path:
            paths.append(cm_path)
    return paths
