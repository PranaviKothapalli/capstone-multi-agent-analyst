"""
Business Insights Agent (Section 6.7). Uses Groq's free-tier, OpenAI-compatible
chat completion API to turn quantitative metrics + feature importances into a
narrative executive summary and strategic recommendations. If no GROQ_API_KEY
is configured (or the call fails for any reason — network, rate limit, etc.)
the agent falls back to a deterministic, template-based narrative so the rest
of the pipeline (and the final report) never breaks.
"""
from src.config import settings
from src.database import log_event

INDUSTRY_KPI_HINTS = {
    "Corporate Finance": "credit risk exposure, default rates, and portfolio F1-score",
    "Healthcare Systems": "patient risk recall, early anomaly detection, and clinical safety margins",
    "Retail Commerce": "customer churn, lifetime value, and feature-driven purchase behavior",
    "Manufacturing": "asset lifetime regression accuracy and predictive maintenance windows",
    "Marketing Operations": "channel ROI allocation and campaign-driven conversion uplift",
    "General / Cross-Industry": "overall data quality, predictive accuracy, and operational efficiency",
}


class BusinessInsightsAgent:
    name = "BusinessInsightsAgent"

    def __init__(self):
        self._client = None
        if settings.llm_enabled:
            try:
                from groq import Groq
                self._client = Groq(api_key=settings.groq_api_key)
            except Exception:
                self._client = None

    def _compile_prompt(self, dataset_name, task_type, metrics, importances, industry) -> str:
        kpi_hint = INDUSTRY_KPI_HINTS.get(industry, INDUSTRY_KPI_HINTS["General / Cross-Industry"])
        top_features = ", ".join(list(importances.keys())[:5]) if importances else "not available"
        return f"""
You are a senior data science consultant writing an executive report section for
the '{industry}' sector. Dataset: {dataset_name}. Task type: {task_type}.
Model metrics: {metrics}. Top contributing features: {top_features}.
Industry KPI focus: {kpi_hint}.

Write two clearly separated parts:
PART 1 - EXECUTIVE NARRATIVE: 2-3 short paragraphs explaining what the model found
and why it matters commercially. Plain, confident, executive tone. No jargon.
PART 2 - STRATEGIC RECOMMENDATIONS: 4-6 bullet points, each a concrete, actionable
business recommendation tied directly to a specific feature or metric above.
"""

    def run(self, dataset_name, task_type, metrics, importances, industry="General / Cross-Industry"):
        log_event(self.name, "generate_insights", "started", {"llm_enabled": settings.llm_enabled})
        prompt = self._compile_prompt(dataset_name, task_type, metrics, importances, industry)

        if self._client is not None:
            try:
                completion = self._client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=900,
                )
                text = completion.choices[0].message.content
                narrative, recommendations = self._split_sections(text)
                log_event(self.name, "generate_insights", "success", {"source": "groq"})
                return True, {"narrative": narrative, "recommendations": recommendations, "source": "groq"}
            except Exception as e:
                log_event(self.name, "generate_insights", "error", f"Groq call failed, using fallback: {e}")

        # ---- Graceful fallback: deterministic template-based narrative ----
        narrative, recommendations = self._fallback_narrative(dataset_name, task_type, metrics, importances, industry)
        log_event(self.name, "generate_insights", "success", {"source": "fallback_template"})
        return True, {"narrative": narrative, "recommendations": recommendations, "source": "fallback_template"}

    @staticmethod
    def _split_sections(text: str):
        upper = text.upper()
        if "PART 2" in upper:
            idx = upper.index("PART 2")
            narrative = text[:idx].replace("PART 1", "").replace("- EXECUTIVE NARRATIVE:", "").strip()
            recommendations = text[idx:].split(":", 1)[-1].strip()
        else:
            narrative, recommendations = text, "See narrative above for recommended actions."
        return narrative, recommendations

    @staticmethod
    def _fallback_narrative(dataset_name, task_type, metrics, importances, industry):
        model_name = metrics.get("model_name", "the selected model")
        kpi_hint = INDUSTRY_KPI_HINTS.get(industry, INDUSTRY_KPI_HINTS["General / Cross-Industry"])
        if task_type == "classification":
            perf_line = (
                f"achieved a macro F1-score of {metrics.get('f1_macro', 0):.3f} "
                f"with precision {metrics.get('precision_macro', 0):.3f} and recall {metrics.get('recall_macro', 0):.3f}"
            )
        else:
            perf_line = (
                f"achieved an R\u00b2 of {metrics.get('r2', 0):.3f} with an RMSE of {metrics.get('rmse', 0):.3f} "
                f"and MAE of {metrics.get('mae', 0):.3f}"
            )
        top_features = list(importances.keys())[:5] if importances else []
        narrative = (
            f"An automated analysis of '{dataset_name}' was completed for the {industry} sector, focused on "
            f"{kpi_hint}. The best-performing model, {model_name}, {perf_line} on held-out validation data.\n\n"
            f"The most influential drivers of the outcome were "
            f"{', '.join(top_features) if top_features else 'a combination of engineered features'}. "
            f"These signals indicate where operational attention and data collection effort will have the "
            f"largest measurable impact on the target outcome."
        )
        bullets = []
        for feat in top_features[:5]:
            bullets.append(f"- Monitor and act on **{feat}**: it is among the strongest predictors identified by {model_name}.")
        if not bullets:
            bullets.append("- Continue collecting clean, labeled data to further improve model confidence.")
        bullets.append(f"- Re-run this pipeline periodically to track drift in {kpi_hint}.")
        recommendations = "\n".join(bullets)
        return narrative, recommendations
