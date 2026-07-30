import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools.data_tools import (
    validate_dataset, remove_duplicates, impute_missing_values,
    correct_invalid_data_types, clean_dataset, compute_eda_report, infer_task_type,
    validate_classification_target, validate_regression_target,
)


def test_validate_dataset_empty():
    df = pd.DataFrame()
    result = validate_dataset(df)
    assert any("empty" in e.lower() for e in result["errors"])


def test_validate_dataset_row_warning():
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    result = validate_dataset(df, min_rows=5000)
    assert any("below the recommended minimum" in w for w in result["warnings"])


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    out, n = remove_duplicates(df)
    assert n == 1
    assert len(out) == 2


def test_impute_missing_values_numeric_and_categorical():
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["x", None, "x"]})
    out, report = impute_missing_values(df)
    assert out["num"].isna().sum() == 0
    assert out["cat"].isna().sum() == 0
    assert report["num"]["strategy"] == "median"
    assert report["cat"]["strategy"] == "mode"


def test_correct_invalid_data_types_handles_infinite():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0]})
    out = correct_invalid_data_types(df)
    assert not np.isinf(out["a"]).any()


def test_clean_dataset_zero_variance_column():
    df = pd.DataFrame({"const": [5, 5, 5, 5], "target": [1, 2, 3, 4]})
    cleaned, delta = clean_dataset(df)
    assert cleaned.shape[0] == 4
    assert "final_shape" in delta


def test_compute_eda_report_structure():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": ["x", "y", "x", "y", "x"]})
    report = compute_eda_report(df)
    assert "correlation" in report
    assert "skew" in report
    assert "outliers" in report
    assert "a" in report["numeric_columns"]


def test_infer_task_type_classification_vs_regression():
    assert infer_task_type(pd.Series([0, 1, 0, 1, 1])) == "classification"
    assert infer_task_type(pd.Series(np.random.randn(500) * 100)) == "regression"


def test_infer_task_type_object_dtype_always_classification():
    y = pd.Series(["yes", "no", "yes", "no", "maybe"] * 50)
    assert infer_task_type(y) == "classification"


def test_infer_task_type_boolean_dtype_classification():
    y = pd.Series([True, False, True, True, False])
    assert infer_task_type(y) == "classification"


def test_infer_task_type_category_dtype_classification():
    y = pd.Series(["low", "medium", "high"] * 20).astype("category")
    assert infer_task_type(y) == "classification"


def test_validate_classification_target_healthy():
    y = pd.Series((["a"] * 60) + (["b"] * 40))
    result = validate_classification_target(y)
    assert result["is_valid"] is True
    assert result["n_classes"] == 2
    assert result["min_class_count"] == 40
    assert result["recommended_cv_splits"] == 5


def test_validate_classification_target_single_class():
    y = pd.Series(["a"] * 10)
    result = validate_classification_target(y)
    assert result["is_valid"] is False
    assert any("at least 2 distinct classes" in e for e in result["errors"])


def test_validate_classification_target_rare_class_singleton():
    # This is the exact real-world scenario that used to crash with
    # sklearn's "least populated class... has only 1 member" error.
    y = pd.Series((["common"] * 50) + (["rare"] * 1))
    result = validate_classification_target(y)
    assert result["is_valid"] is False
    assert any("fewer than 2 samples" in e for e in result["errors"])
    assert result["recommended_cv_splits"] is None


def test_validate_classification_target_high_cardinality_identifier():
    # e.g. a customer ID or name column mistakenly picked as target
    y = pd.Series([f"user_{i}" for i in range(200)])
    result = validate_classification_target(y)
    assert result["is_valid"] is False
    assert any("identifier" in e for e in result["errors"])


def test_validate_classification_target_small_min_class_reduces_cv_folds():
    y = pd.Series((["a"] * 20) + (["b"] * 3))
    result = validate_classification_target(y)
    assert result["is_valid"] is True
    assert result["recommended_cv_splits"] == 3
    assert any("folds" in w for w in result["warnings"])


def test_validate_regression_target_zero_variance():
    y = pd.Series([5.0] * 50)
    result = validate_regression_target(y)
    assert result["is_valid"] is False


def test_validate_regression_target_healthy():
    y = pd.Series(np.random.randn(100))
    result = validate_regression_target(y)
    assert result["is_valid"] is True
