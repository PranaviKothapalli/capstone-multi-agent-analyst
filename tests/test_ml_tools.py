import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.tools.ml_tools import train_and_evaluate, build_feature_preview


def _make_classification_df(n=400):
    rng = np.random.RandomState(42)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    cat = rng.choice(["A", "B", "C"], size=n)
    y = (x1 + 0.5 * x2 + (cat == "A").astype(int) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": y})


def _make_regression_df(n=400):
    rng = np.random.RandomState(7)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = 3 * x1 - 2 * x2 + rng.randn(n) * 0.1
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_train_and_evaluate_classification():
    df = _make_classification_df()
    model, metrics, importances, encoder = train_and_evaluate(df, "target", "classification", n_splits=3)
    assert "f1_macro" in metrics
    assert metrics["f1_macro"] > 0.5
    assert "cv_leaderboard" in metrics


def test_train_and_evaluate_regression():
    df = _make_regression_df()
    model, metrics, importances, encoder = train_and_evaluate(df, "target", "regression", n_splits=3)
    assert "rmse" in metrics
    assert metrics["r2"] > 0.5


def _make_object_target_classification_df(n=400):
    rng = np.random.RandomState(11)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    cat = rng.choice(["north", "south", "east"], size=n)
    label = np.where(x1 + 0.5 * x2 + (cat == "north").astype(int) > 0, "churn", "stay")
    return pd.DataFrame({"x1": x1, "x2": x2, "region": cat, "status": label})


def test_train_and_evaluate_object_dtype_target_classification():
    """The exact bug being fixed: an object/string target column must train
    successfully as classification, not crash."""
    df = _make_object_target_classification_df()
    model, metrics, importances, encoder = train_and_evaluate(df, "status", "classification", n_splits=5)
    assert encoder is not None
    assert set(encoder.classes_) == {"churn", "stay"}
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] > 0.5
    assert metrics["confusion_matrix"] is not None


def test_train_and_evaluate_raises_friendly_error_for_singleton_class():
    """A class with only 1 sample must raise a clear, user-friendly
    ValueError — never a raw sklearn 'least populated class' stack trace."""
    df = _make_object_target_classification_df(n=100)
    # Inject one row whose class appears exactly once.
    df = pd.concat(
        [df, pd.DataFrame([{"x1": 0.1, "x2": 0.2, "region": "north", "status": "rare_case"}])],
        ignore_index=True,
    )
    with pytest.raises(ValueError) as exc_info:
        train_and_evaluate(df, "status", "classification", n_splits=5)
    msg = str(exc_info.value)
    assert "fewer than 2 samples" in msg
    assert "least populated class" not in msg.lower()


def test_train_and_evaluate_raises_friendly_error_for_high_cardinality_target():
    """An ID-like column should be rejected with a clear explanation, not a crash."""
    n = 150
    df = pd.DataFrame({
        "x1": np.random.RandomState(5).randn(n),
        "x2": np.random.RandomState(6).randn(n),
        "user_id": [f"id_{i}" for i in range(n)],
    })
    with pytest.raises(ValueError) as exc_info:
        train_and_evaluate(df, "user_id", "classification", n_splits=5)
    assert "identifier" in str(exc_info.value)


def test_build_feature_preview_creates_interactions():
    df = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 2, "cat": ["x", "y"] * 10})
    correlation = {"a": {"a": 1.0, "b": 1.0}, "b": {"a": 1.0, "b": 1.0}}
    feat_df, pipeline_map = build_feature_preview(df, correlation, corr_threshold=0.9)
    assert any("a_x_b" == it["new_column"] for it in pipeline_map["interaction_terms"])
    assert "cat" in pipeline_map["encoded_categorical_columns"]
