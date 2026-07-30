import os
import pandas as pd
from src.database import log_event
from src.tools.ml_tools import build_feature_preview


class FeatureEngineeringAgent:
    """Encodes categoricals, scales numerics, creates interaction terms for
    highly-correlated pairs (Section 6.4)."""

    name = "FeatureEngineeringAgent"

    def run(self, clean_path: str, eda_report: dict, out_dir: str) -> tuple[bool, dict]:
        log_event(self.name, "engineer_features", "started", {"clean_path": clean_path})
        try:
            df = pd.read_csv(clean_path)
            feat_df, pipeline_map = build_feature_preview(df, eda_report.get("correlation", {}))
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(clean_path))[0].replace("_cleaned", "")
            feat_path = os.path.join(out_dir, f"{base}_features.csv")
            feat_df.to_csv(feat_path, index=False)
            log_event(self.name, "engineer_features", "success", pipeline_map)
            return True, {"feat_path": feat_path, "feat_df": feat_df, "pipeline_map": pipeline_map}
        except Exception as e:
            log_event(self.name, "engineer_features", "error", str(e))
            return False, {"error": str(e)}
