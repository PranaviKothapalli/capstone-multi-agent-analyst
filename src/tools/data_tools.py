"""
Deterministic, unit-testable data tools. Agents call these functions rather
than touching files directly (Section 4.3 of the handbook) so behaviour is
safe, sandboxed and independently testable.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np


class ValidationError(Exception):
    pass


def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValidationError(f"Unsupported file format '{ext}'. Please upload a .csv or .parquet file.")
    return df


def validate_dataset(
    df: pd.DataFrame,
    min_rows: int = 5000,
    max_rows: int = 150_000,
    min_cols: int = 8,
    max_cols: int = 45,
) -> dict:
    """Returns {"errors": [...], "warnings": [...]}. Errors block progress;
    warnings are informational (Section 8.2 thresholds are guidelines, not
    hard requirements, for real-world student datasets)."""
    errors, warnings = [], []

    if df is None or df.empty:
        errors.append("The uploaded dataset is empty.")
        return {"errors": errors, "warnings": warnings}

    n_rows, n_cols = df.shape
    if n_rows < min_rows:
        warnings.append(f"Row count ({n_rows:,}) is below the recommended minimum of {min_rows:,}.")
    if n_rows > max_rows:
        errors.append(f"Row count ({n_rows:,}) exceeds the maximum supported {max_rows:,} rows.")
    if not (min_cols <= n_cols <= max_cols):
        warnings.append(f"Column count ({n_cols}) is outside the recommended range of {min_cols}-{max_cols}.")
    if df.columns.duplicated().any():
        errors.append("Dataset contains duplicate column names.")
    if n_cols == 0:
        errors.append("Dataset has no columns.")

    return {"errors": errors, "warnings": warnings}


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Median for numeric columns, mode for categorical — computed per-column
    on the full cleaning-stage dataframe (train/val separation happens later,
    inside the ML pipeline, per Section 10.2)."""
    df = df.copy()
    report = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            report[col] = {"strategy": "median", "value": float(fill_value), "count": n_missing}
        else:
            mode_vals = df[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            report[col] = {"strategy": "mode", "value": str(fill_value), "count": n_missing}
    return df, report


def correct_invalid_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces object columns that are 'numeric-looking' (>=90% parseable)
    back into numeric dtype to fix type-drift such as '1,000' -> NaN-safe ints."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        converted = pd.to_numeric(cleaned, errors="coerce")
        non_null_ratio = converted.notna().mean() if len(df) else 0
        if non_null_ratio >= 0.9:
            df[col] = converted
    # Replace infinities produced by any upstream transform with NaN, then re-impute.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full cleaning routine used by the Data Cleaning Agent."""
    original_shape = list(df.shape)
    df, dup_count = remove_duplicates(df)
    df = correct_invalid_data_types(df)
    df, impute_report = impute_missing_values(df)
    delta = {
        "original_shape": original_shape,
        "duplicates_removed": dup_count,
        "imputation": impute_report,
        "final_shape": list(df.shape),
    }
    return df, delta


def compute_eda_report(df: pd.DataFrame) -> dict:
    """Descriptive statistics, correlation matrix, skew, outlier bounds."""
    numeric_df = df.select_dtypes(include=np.number)
    summary = df.describe(include="all").fillna("").to_dict()
    correlation = numeric_df.corr(numeric_only=True).round(3).to_dict() if numeric_df.shape[1] > 1 else {}
    skewness = numeric_df.skew(numeric_only=True).round(3).to_dict() if not numeric_df.empty else {}

    outliers = {}
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((numeric_df[col] < low) | (numeric_df[col] > high)).sum())
        outliers[col] = {"lower_bound": float(low), "upper_bound": float(high), "outlier_count": n_out}

    missing_pct = (df.isna().mean() * 100).round(2).to_dict()

    return {
        "summary": summary,
        "correlation": correlation,
        "skew": skewness,
        "outliers": outliers,
        "missing_pct": missing_pct,
        "numeric_columns": list(numeric_df.columns),
        "categorical_columns": [c for c in df.columns if c not in numeric_df.columns],
    }


def infer_task_type(y: pd.Series) -> str:
    """Numeric target -> regression (unless it's really a small set of coded
    classes). Object / category / boolean target -> classification, always."""
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique(dropna=True)
        if n_unique <= max(20, int(0.02 * len(y))) and n_unique <= 20:
            return "classification"
        return "regression"
    return "classification"


def validate_classification_target(
    y: pd.Series,
    min_samples_per_class: int = 2,
    max_cv_splits: int = 5,
    high_cardinality_abs: int = 50,
    high_cardinality_ratio: float = 0.5,
) -> dict:
    """Validates a target column before it is ever handed to a classifier.

    Returns a dict with:
      is_valid, errors (list[str]), warnings (list[str]),
      n_classes, n_samples, min_class_count, class_counts (dict, largest first),
      recommended_cv_splits (int or None if invalid).

    This is the single source of truth used by both the ML Studio UI (to show
    friendly guidance and disable the Train button) and the Machine Learning
    Agent (as a defensive re-check so training can never surface a raw
    scikit-learn stack trace to the user).
    """
    errors, warnings = [], []
    y_valid = y.dropna()
    n_samples = int(len(y_valid))

    if n_samples == 0:
        errors.append("The target column has no non-missing values to train on.")
        return {
            "is_valid": False, "errors": errors, "warnings": warnings,
            "n_classes": 0, "n_samples": 0, "min_class_count": 0,
            "class_counts": {}, "recommended_cv_splits": None,
        }

    class_counts_series = y_valid.value_counts()
    n_classes = int(class_counts_series.shape[0])
    min_class_count = int(class_counts_series.min())
    class_counts = {str(k): int(v) for k, v in class_counts_series.items()}
    cardinality_ratio = n_classes / n_samples if n_samples else 0.0

    if n_classes < 2:
        errors.append(
            f"The target column only has {n_classes} unique value after removing missing rows. "
            f"Classification needs at least 2 distinct classes — please choose a different target column."
        )

    # High-cardinality identifier detection (IDs, names, free text) — check this
    # before the rare-class check since it's the more useful explanation for the user.
    is_high_cardinality = n_classes > high_cardinality_abs and cardinality_ratio > high_cardinality_ratio
    if is_high_cardinality:
        errors.append(
            f"This column has {n_classes:,} unique values across {n_samples:,} rows "
            f"({cardinality_ratio:.0%} unique) — that pattern usually means it's an identifier, name, "
            f"or free-text field rather than a category label, so it isn't suitable for classification. "
            f"Please choose a column with a small number of repeating categories instead."
        )
    else:
        rare_classes = class_counts_series[class_counts_series < min_samples_per_class]
        if not rare_classes.empty:
            rare_list = ", ".join(f"'{idx}' ({cnt} sample{'s' if cnt != 1 else ''})" for idx, cnt in rare_classes.items())
            errors.append(
                f"The following classes have fewer than {min_samples_per_class} samples, which is not "
                f"enough for a train/test split and cross-validation: {rare_list}. Consider merging rare "
                f"classes, collecting more data for them, or choosing a different target column."
            )
        elif n_classes > 20:
            warnings.append(
                f"The target has {n_classes} distinct classes. Training will still run, but that many "
                f"classes can slow training down and make the results harder to interpret."
            )

    recommended_cv_splits = None
    if not errors:
        recommended_cv_splits = max(2, min(max_cv_splits, min_class_count))
        if min_class_count < max_cv_splits:
            warnings.append(
                f"The smallest class has only {min_class_count} sample(s), so cross-validation will "
                f"automatically use {recommended_cv_splits} folds instead of the default {max_cv_splits} "
                f"to stay statistically valid."
            )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_classes": n_classes,
        "n_samples": n_samples,
        "min_class_count": min_class_count,
        "class_counts": class_counts,
        "recommended_cv_splits": recommended_cv_splits,
    }


def validate_regression_target(y: pd.Series, min_samples: int = 10) -> dict:
    """Lightweight sanity checks for a regression target (kept minimal so
    existing, already-working regression behavior is fully preserved)."""
    errors, warnings = [], []
    y_valid = y.dropna()
    n_samples = int(len(y_valid))

    if n_samples < min_samples:
        errors.append(f"Only {n_samples} non-missing values are available for this target — at least {min_samples} are needed to train and validate a regression model.")
    elif y_valid.nunique(dropna=True) <= 1:
        errors.append("The target column has the same value in every row (zero variance), so a regression model has nothing to learn. Please choose a different target column.")

    return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings, "n_samples": n_samples}


def validate_target_distribution(y: pd.Series, task_type: str, min_rows: int = 10) -> dict:
    """Validates that a target column is actually trainable *before* handing it
    to scikit-learn, so unsuitable targets (e.g. classes with only 1 sample,
    which break StratifiedKFold / stratified train_test_split) produce a
    clear, friendly message instead of a raw exception.

    Returns {"errors": [...], "warnings": [...], "min_class_count": int|None}.
    """
    errors, warnings = [], []
    y = y.dropna()
    n_rows = len(y)
    min_class_count = None

    if n_rows < min_rows:
        errors.append(
            f"After removing missing values, only {n_rows} rows remain for the target column — "
            f"at least {min_rows} are needed to train and validate a model. Please choose a different "
            "target column, or upload a larger / more complete dataset."
        )
        return {"errors": errors, "warnings": warnings, "min_class_count": min_class_count}

    if task_type == "classification":
        counts = y.value_counts()
        n_classes = len(counts)

        if n_classes < 2:
            errors.append(
                "The selected target column has only one unique category after removing missing values, "
                "so a classification model cannot be trained. Please choose a different target column."
            )
            return {"errors": errors, "warnings": warnings, "min_class_count": min_class_count}

        min_class_count = int(counts.min())
        rare_singletons = counts[counts < 2]
        if not rare_singletons.empty:
            examples = ", ".join(str(v) for v in rare_singletons.index[:5])
            errors.append(
                "The target column has categories with only 1 sample "
                f"({examples}), which is too few to both train and validate a model. Please choose a target "
                "column where every category has at least 2 samples, or group/remove the rare categories "
                "and try again."
            )
        elif min_class_count < 5:
            warnings.append(
                f"Some target categories have very few samples (as few as {min_class_count}). The system "
                "will automatically use fewer cross-validation folds, but performance metrics may be less "
                "statistically reliable — consider collecting more data for those categories if possible."
            )
    else:
        if y.nunique() < 2:
            errors.append(
                "The target column has no variation (every value is identical), so a regression model "
                "cannot be trained. Please choose a different target column."
            )

    return {"errors": errors, "warnings": warnings, "min_class_count": min_class_count}
