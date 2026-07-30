import os
import pandas as pd
from src.database import log_event
from src.tools.data_tools import clean_dataset


class DataCleaningAgent:
    """Inspects layouts, removes duplicates, imputes missing values, and
    fixes type drift (Section 6.2)."""

    name = "DataCleaningAgent"

    def run(self, raw_path: str, out_dir: str) -> tuple[bool, dict]:
        log_event(self.name, "clean_dataset", "started", {"raw_path": raw_path})
        try:
            df = pd.read_csv(raw_path) if raw_path.endswith(".csv") else pd.read_parquet(raw_path)
            cleaned_df, delta = clean_dataset(df)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(raw_path))[0]
            clean_path = os.path.join(out_dir, f"{base}_cleaned.csv")
            cleaned_df.to_csv(clean_path, index=False)
            log_event(self.name, "clean_dataset", "success", delta)
            return True, {"cleaned_path": clean_path, "cleaned_df": cleaned_df, "delta": delta}
        except Exception as e:
            log_event(self.name, "clean_dataset", "error", str(e))
            return False, {"error": str(e)}
