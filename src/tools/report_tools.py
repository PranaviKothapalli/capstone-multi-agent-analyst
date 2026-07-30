"""
Automated Report Generation tools (Section 12). Builds the 5-section
executive HTML report and compiles it to PDF via WeasyPrint. WeasyPrint
needs native system libraries (cairo/pango); if they are not present on the
host machine, we fall back to delivering the polished HTML report instead of
crashing the whole pipeline (see README for installing the system deps).
"""
from __future__ import annotations
import os
import base64
import datetime
import html as html_lib

TEMPLATE = """
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #1E2130; margin: 40px; }}
  h1 {{ color: #4A4EDB; border-bottom: 3px solid #4A4EDB; padding-bottom: 8px; }}
  h2 {{ color: #2C2F55; margin-top: 34px; border-left: 5px solid #8B6CF1; padding-left: 10px; }}
  .meta {{ color: #666; font-size: 0.85rem; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; font-size: 0.85rem; text-align: left; }}
  th {{ background: #F0F0FA; }}
  .card {{ background: #F7F8FC; border-radius: 10px; padding: 16px 20px; margin: 10px 0; }}
  .chart {{ max-width: 100%; margin: 14px 0; border: 1px solid #eee; border-radius: 8px; }}
  .badge {{ display:inline-block; background:#EAEAFF; color:#4A4EDB; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600;}}
  ul {{ line-height: 1.6; }}
</style>
</head>
<body>
<h1>📊 Executive Analytics Report</h1>
<div class="meta">Project Identifier: <b>{run_id}</b> &nbsp;|&nbsp; Generated: {timestamp} &nbsp;|&nbsp; Dataset: <b>{dataset_name}</b></div>

<h2>1. Executive Summary</h2>
<div class="card">{executive_summary}</div>

<h2>2. Data Diagnostics Profile</h2>
<div class="card">
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Extracted Records</td><td>{n_rows}</td></tr>
<tr><td>Total Features</td><td>{n_cols}</td></tr>
<tr><td>Duplicate Rows Removed</td><td>{duplicates_removed}</td></tr>
<tr><td>Columns With Imputed Values</td><td>{n_imputed_cols}</td></tr>
</table>
</div>

<h2>3. Predictive Performance Enforcement</h2>
<div class="card">
<span class="badge">Best Model: {best_model}</span> &nbsp; <span class="badge">Task: {task_type}</span>
{metrics_table}
</div>
{chart_blocks}

<h2>4. Automated Business Insights</h2>
<div class="card">{business_insights}</div>

<h2>5. Strategic Recommendations</h2>
<div class="card">{recommendations}</div>

</body>
</html>
"""


def _metrics_table_html(metrics: dict, task_type: str) -> str:
    rows = []
    if task_type == "classification":
        keys = ["f1_macro", "precision_macro", "recall_macro", "roc_auc"]
    else:
        keys = ["rmse", "mae", "r2", "adjusted_r2"]
    for k in keys:
        v = metrics.get(k)
        if v is not None:
            rows.append(f"<tr><td>{k.replace('_', ' ').upper()}</td><td>{round(v, 4)}</td></tr>")
    leaderboard = metrics.get("cv_leaderboard", {})
    lb_rows = "".join(
        f"<tr><td>{name}</td><td>{round(info['cv_score'], 4)}</td></tr>" for name, info in leaderboard.items()
    )
    return f"""
    <table><tr><th>Metric</th><th>Value</th></tr>{''.join(rows)}</table>
    <b>Model Leaderboard (CV score)</b>
    <table><tr><th>Model</th><th>CV Score</th></tr>{lb_rows}</table>
    """


def build_html_report(
    run_id: str, dataset_name: str, df_shape: tuple, clean_delta: dict,
    metrics: dict, task_type: str, narrative: str, recommendations: str,
    chart_paths: list[str],
) -> str:
    n_imputed = len(clean_delta.get("imputation", {})) if clean_delta else 0
    duplicates_removed = clean_delta.get("duplicates_removed", 0) if clean_delta else 0

    chart_blocks = ""
    for path in chart_paths or []:
        try:
            with open(path, "rb") as img:
                encoded = base64.b64encode(img.read()).decode("utf-8")

            chart_blocks += f"""
            <img class="chart"
                src="data:image/png;base64,{encoded}">
            <br>
            """
        except Exception:
            continue

    exec_summary = narrative.split("\n\n")[0] if narrative else (
        f"This report summarizes the automated analysis of '{dataset_name}', covering data quality, "
        f"model performance and actionable business recommendations."
    )

    html_out = TEMPLATE.format(
        run_id=run_id,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        dataset_name=html_lib.escape(str(dataset_name)),
        executive_summary=exec_summary.replace("\n", "<br>"),
        n_rows=df_shape[0],
        n_cols=df_shape[1],
        duplicates_removed=duplicates_removed,
        n_imputed_cols=n_imputed,
        best_model=metrics.get("model_name", "N/A"),
        task_type=task_type,
        metrics_table=_metrics_table_html(metrics, task_type),
        chart_blocks=chart_blocks,
        business_insights=(narrative or "No narrative generated.").replace("\n", "<br>"),
        recommendations=(recommendations or "No recommendations generated.").replace("\n", "<br>"),
    )
    return html_out


def compile_pdf(html_content: str, out_path: str) -> tuple[str, bool]:
    """Returns (file_path, is_pdf). Falls back to writing an .html file next
    to the requested pdf path if WeasyPrint's native dependencies are missing."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        from weasyprint import HTML

        HTML(
            string=html_content,
            base_url="."
        ).write_pdf(out_path)

        return out_path, True

    except Exception as e:

        print(f"PDF generation failed: {e}")

        html_path = out_path.replace(".pdf", ".html")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path, False
