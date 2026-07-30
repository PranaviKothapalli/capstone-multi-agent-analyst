import pandas as pd
from src.database import log_event
from src.tools.data_tools import infer_task_type, validate_target_distribution
from src.tools.ml_tools import train_and_evaluate, serialize_model


class MachineLearningAgent:
    """Detects task type, validates the target column is actually trainable,
    trains candidate models under leakage-safe cross-validation, and
    serializes the winning pipeline (Section 6.5).

    NOTE: trains on the *cleaned* dataframe (not the encoded feature preview)
    because the leakage-safe scikit-learn Pipeline built in ml_tools performs
    its own encoding/scaling internally, fit per training fold only.
    """

    name = "MachineLearningAgent"

    def run(self, clean_path: str, target_col: str, model_out_path: str) -> tuple[bool, dict]:
        log_event(self.name, "train_models", "started", {"target": target_col})
        try:
            df = pd.read_csv(clean_path)

            if target_col not in df.columns:
                msg = f"Target column '{target_col}' was not found in the dataset."
                log_event(self.name, "train_models", "error", msg)
                return False, {"error": msg}

            target_series = df[target_col].dropna()
            task_type = infer_task_type(target_series)

            # --- Validate BEFORE training so unsuitable targets (e.g. classes
            # with only 1 sample) never reach scikit-learn as a raw crash. ---
            validation = validate_target_distribution(target_series, task_type)
            if validation["errors"]:
                friendly_msg = " ".join(validation["errors"])
                log_event(self.name, "train_models", "error", {
                    "reason": "target_validation_failed",
                    "task_type": task_type,
                    "issues": validation["errors"],
                })
                return False, {"error": friendly_msg, "task_type": task_type}

            best_model, metrics, importances, label_encoder = train_and_evaluate(df, target_col, task_type)

            if validation["warnings"]:
                metrics["data_warnings"] = validation["warnings"]

            serialize_model(best_model, model_out_path)
            log_event(self.name, "train_models", "success", {"model": metrics.get("model_name"), "task_type": task_type})
            return True, {
                "model": best_model,
                "model_path": model_out_path,
                "metrics": metrics,
                "task_type": task_type,
                "feature_importances": importances,
                "label_encoder": label_encoder,
            }
        except Exception as e:
            # Final safety net: never let a raw exception surface to the UI.
            friendly_msg = (
                "Model training could not be completed with the selected target column. This can happen "
                "when the target's categories or values aren't suitable for modeling (e.g. too imbalanced, "
                "too few samples, or no variation). Please choose a different target column and try again. "
                f"(Technical detail: {e})"
            )
            log_event(self.name, "train_models", "error", str(e))
            return False, {"error": friendly_msg}
