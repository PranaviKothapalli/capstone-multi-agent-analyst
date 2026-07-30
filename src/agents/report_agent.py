import uuid
from src.database import log_event
from src.tools.report_tools import build_html_report, compile_pdf


class ReportGenerationAgent:
    """Assembles insights, charts and metrics into a structured executive
    PDF (falls back to HTML if PDF system libraries are unavailable)."""

    name = "ReportGenerationAgent"

    def run(self, dataset_name, df_shape, clean_delta, metrics, task_type, narrative, recommendations, chart_paths, out_pdf_path):
        log_event(self.name, "compile_report", "started", {})
        try:
            run_id = uuid.uuid4().hex[:8].upper()
            html_content = build_html_report(
                run_id, dataset_name, df_shape, clean_delta, metrics, task_type,
                narrative, recommendations, chart_paths,
            )
            final_path, is_pdf = compile_pdf(html_content, out_pdf_path)
            log_event(self.name, "compile_report", "success", {"format": "pdf" if is_pdf else "html", "path": final_path})
            return True, {"path": final_path, "is_pdf": is_pdf, "run_id": run_id}
        except Exception as e:
            log_event(self.name, "compile_report", "error", str(e))
            return False, {"error": str(e)}
