import pandas as pd
from src.database import log_event
from src.tools.data_tools import compute_eda_report


class EDAAgent:
    """Sweeps cleaned features, computes skew, correlation, outlier bounds
    (Section 6.3)."""

    name = "EDAAgent"

    def run(self, clean_path: str) -> tuple[bool, dict]:
        log_event(self.name, "run_eda", "started", {"clean_path": clean_path})
        try:
            df = pd.read_csv(clean_path)
            report = compute_eda_report(df)
            log_event(self.name, "run_eda", "success", {"n_numeric": len(report["numeric_columns"])})
            return True, report
        except Exception as e:
            log_event(self.name, "run_eda", "error", str(e))
            return False, {"error": str(e)}
