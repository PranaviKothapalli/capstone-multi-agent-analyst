"""
Feature engineering preview + machine learning training tools.

IMPORTANT (Section 10.2 of the handbook — zero data leakage):
The *preview* feature dataframe (encoded/scaled) built here is only for
display in the Feature Engineering page. The actual model training in
`train_and_evaluate()` rebuilds all encoding/scaling as a scikit-learn
ColumnTransformer wrapped inside a Pipeline, so every transform is fit
exclusively on each training fold and applied (not re-fit) to the
corresponding validation fold.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
)

from src.tools.data_tools import validate_classification_target, validate_regression_target


# ---------------------------------------------------------------------------
# Feature Engineering Agent (preview build, high-correlation interaction terms)
# ---------------------------------------------------------------------------
def build_feature_preview(df: pd.DataFrame, correlation: dict, corr_threshold: float = 0.75) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Encode categoricals (one-hot, capped to avoid explosion on high-cardinality cols)
    encoded_cols = []
    for col in categorical_cols:
        if df[col].nunique(dropna=True) <= 20:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoded_cols.append(col)

    # Scale numeric columns for the preview
    scaled_cols = []
    for col in numeric_cols:
        std = df[col].std()
        if std and std > 0:
            df[col] = (df[col] - df[col].mean()) / std
            scaled_cols.append(col)

    # Interaction terms for highly correlated numeric pairs
    interactions = []
    seen = set()
    for c1, row in (correlation or {}).items():
        for c2, val in row.items():
            if c1 == c2 or not isinstance(val, (int, float)):
                continue
            pair = tuple(sorted((c1, c2)))
            if pair in seen or abs(val) < corr_threshold:
                continue
            seen.add(pair)
            if c1 in df.columns and c2 in df.columns:
                new_col = f"{c1}_x_{c2}"
                df[new_col] = df[c1] * df[c2]
                interactions.append({"pair": pair, "correlation": val, "new_column": new_col})

    pipeline_map = {
        "encoded_categorical_columns": encoded_cols,
        "scaled_numeric_columns": scaled_cols,
        "interaction_terms": interactions,
        "final_shape": list(df.shape),
    }
    return df, pipeline_map


# ---------------------------------------------------------------------------
# Machine Learning Agent
# ---------------------------------------------------------------------------
CANDIDATE_MODELS = {
    "classification": {
        "RandomForest": (RandomForestClassifier(random_state=42), {"clf__n_estimators": [150, 300], "clf__max_depth": [6, None]}),
        "LogisticRegression": (LogisticRegression(max_iter=2000), {"clf__C": [0.1, 1.0, 10.0]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3]}),
    },
    "regression": {
        "RandomForest": (RandomForestRegressor(random_state=42), {"clf__n_estimators": [150, 300], "clf__max_depth": [6, None]}),
        "Ridge": (Ridge(), {"clf__alpha": [0.1, 1.0, 10.0]}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3]}),
    },
}


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def train_and_evaluate(df: pd.DataFrame, target_col: str, task_type: str, n_splits: int = 5, fast: bool = True):
    """Trains every candidate model with leakage-safe CV pipelines, returns the
    best estimator (refit on the full data) plus a full evaluation profile.

    Raises ValueError with a user-friendly message (never a raw sklearn stack
    trace) if the target column is not actually usable for the given task —
    this is a defensive re-check even though the UI already validates and
    blocks training on an invalid target before this is ever called.
    """
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if task_type == "classification":
        validation = validate_classification_target(y, max_cv_splits=n_splits)
        if not validation["is_valid"]:
            raise ValueError(" ".join(validation["errors"]))
        n_splits = validation["recommended_cv_splits"]
    else:
        validation = validate_regression_target(y)
        if not validation["is_valid"]:
            raise ValueError(" ".join(validation["errors"]))

    label_encoder = None
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target_col)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task_type == "classification" \
        else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = "f1_macro" if task_type == "classification" else "neg_root_mean_squared_error"

    leaderboard = {}
    best_name, best_estimator, best_score = None, None, -np.inf

    candidates = CANDIDATE_MODELS[task_type]
    for name, (estimator, grid) in candidates.items():
        preprocessor = _build_preprocessor(X)
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        if fast:
            grid = {k: [v[0]] for k, v in grid.items()}  # single config for speed on large/CI datasets
        search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        search.fit(X, y)
        leaderboard[name] = {"cv_score": float(search.best_score_), "best_params": search.best_params_}
        if search.best_score_ > best_score:
            best_score, best_name, best_estimator = search.best_score_, name, search.best_estimator_

    # Hold-out split purely for reporting confusion matrices / residuals (still leakage-safe:
    # the winning pipeline above was selected via CV only, this is refit fresh for reporting).
    strat = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    best_estimator.fit(X_train, y_train)
    y_pred = best_estimator.predict(X_test)

    metrics = {"model_name": best_name, "cv_leaderboard": leaderboard}
    if task_type == "classification":
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        metrics.update({
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "labels": [str(c) for c in (label_encoder.classes_ if label_encoder else sorted(y.unique()))],
        })
        try:
            if hasattr(best_estimator, "predict_proba"):
                proba = best_estimator.predict_proba(X_test)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr"))
        except (ValueError, IndexError):
            # ROC-AUC isn't always computable (e.g. a class missing from the test
            # fold) — this is genuinely optional ("where applicable"), so we
            # simply omit it rather than fail the whole training run.
            metrics["roc_auc"] = None
    else:
        metrics.update({
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "adjusted_r2": float(1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / max(len(y_test) - X_test.shape[1] - 1, 1)),
        })

    # Refit best pipeline on ALL data for the final production artifact.
    best_estimator.fit(X, y)

    feature_importances = _extract_feature_importances(best_estimator)
    return best_estimator, metrics, feature_importances, label_encoder


def _extract_feature_importances(pipeline: Pipeline) -> dict:
    try:
        prep = pipeline.named_steps["prep"]
        feature_names = prep.get_feature_names_out()
        model = pipeline.named_steps["clf"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return {}
        pairs = sorted(zip(feature_names, importances), key=lambda x: -abs(x[1]))
        return {str(name): float(val) for name, val in pairs[:20]}
    except Exception:
        return {}


def serialize_model(model, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path
