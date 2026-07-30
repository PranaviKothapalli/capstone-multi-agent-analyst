import pandas as pd
from src.database import log_event
from src.tools.viz_tools import generate_all_visuals


class VisualizationAgent:
    """Generates distribution plots, correlation heatmaps, feature
    importance and model comparison charts (Section 6.6)."""

    name = "VisualizationAgent"

    def run(self, data_path: str, target_col: str | None, metrics: dict, importances: dict, out_dir: str) -> tuple[bool, list]:
        log_event(self.name, "generate_visuals", "started", {})
        try:
            df = pd.read_csv(data_path)
            paths = generate_all_visuals(df, target_col, metrics, importances, out_dir)
            log_event(self.name, "generate_visuals", "success", {"n_charts": len(paths)})
            return True, paths
        except Exception as e:
            log_event(self.name, "generate_visuals", "error", str(e))
            return False, []
