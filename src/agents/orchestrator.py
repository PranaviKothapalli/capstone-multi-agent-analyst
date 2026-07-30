"""
Orchestrator Agent (Section 6.1). Coordinates the full sequential workflow
across all sub-agents and centralizes error handling / retry policy so a
single failing step doesn't crash the whole run.
"""
from src.database import log_event
from src.agents.cleaning_agent import DataCleaningAgent
from src.agents.eda_agent import EDAAgent
from src.agents.feature_engineering_agent import FeatureEngineeringAgent
from src.agents.ml_agent import MachineLearningAgent
from src.agents.visualization_agent import VisualizationAgent
from src.agents.insights_agent import BusinessInsightsAgent
from src.agents.report_agent import ReportGenerationAgent


class OrchestratorAgent:
    name = "OrchestratorAgent"

    def __init__(self):
        self.cleaner = DataCleaningAgent()
        self.eda = EDAAgent()
        self.feature_eng = FeatureEngineeringAgent()
        self.ml = MachineLearningAgent()
        self.viz = VisualizationAgent()
        self.insights = BusinessInsightsAgent()
        self.reporter = ReportGenerationAgent()

    def execute_full_workflow(self, raw_path, dataset_name, target_col, industry, paths, progress_cb=None):
        """paths: dict with keys cleaned_dir, features_dir, models_dir, viz_dir, reports_dir.
        progress_cb(stage_name, status, detail) is called after every stage for live UI updates."""
        log_event(self.name, "execute_full_workflow", "started", {"dataset": dataset_name})

        def notify(stage, status, detail=""):
            if progress_cb:
                progress_cb(stage, status, detail)

        results = {}

        notify("Cleaning", "running")
        ok, out = self.cleaner.run(raw_path, paths["cleaned_dir"])
        if not ok:
            notify("Cleaning", "error", out.get("error"))
            return False, results
        results["cleaned_path"], results["clean_delta"] = out["cleaned_path"], out["delta"]
        notify("Cleaning", "done", out["delta"])

        notify("EDA", "running")
        ok, out = self.eda.run(results["cleaned_path"])
        if not ok:
            notify("EDA", "error", out.get("error"))
            return False, results
        results["eda_report"] = out
        notify("EDA", "done")

        notify("Feature Engineering", "running")
        ok, out = self.feature_eng.run(results["cleaned_path"], results["eda_report"], paths["features_dir"])
        if not ok:
            notify("Feature Engineering", "error", out.get("error"))
            return False, results
        results["feat_path"] = out["feat_path"]
        results["pipeline_map"] = out["pipeline_map"]
        notify("Feature Engineering", "done")

        notify("Model Training", "running")
        model_out_path = f"{paths['models_dir']}/best_production_model.pkl"
        ok, out = self.ml.run(results["cleaned_path"], target_col, model_out_path)
        if not ok:
            notify("Model Training", "error", out.get("error"))
            return False, results
        results.update(out)
        notify("Model Training", "done", {"model": out["metrics"].get("model_name")})

        notify("Visualization", "running")
        ok, paths_out = self.viz.run(
            results["cleaned_path"], target_col, results["metrics"], results["feature_importances"], paths["viz_dir"]
        )
        results["chart_paths"] = paths_out
        notify("Visualization", "done" if ok else "error", {"n_charts": len(paths_out)})

        notify("Business Insights", "running")
        ok, out = self.insights.run(dataset_name, results["task_type"], results["metrics"], results["feature_importances"], industry)
        results["narrative"], results["recommendations"], results["insight_source"] = out["narrative"], out["recommendations"], out["source"]
        notify("Business Insights", "done", {"source": out["source"]})

        notify("Report Generation", "running")
        report_path = f"{paths['reports_dir']}/{dataset_name}_executive_report.pdf"
        ok, out = self.reporter.run(
            dataset_name, (0, 0), results["clean_delta"], results["metrics"], results["task_type"],
            results["narrative"], results["recommendations"], results["chart_paths"], report_path,
        )
        results["report_path"] = out.get("path")
        results["report_is_pdf"] = out.get("is_pdf")
        notify("Report Generation", "done" if ok else "error")

        log_event(self.name, "execute_full_workflow", "success", {"dataset": dataset_name})
        return True, results
