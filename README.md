# 🧠 Multi-Agent AI Data Analyst

A production-grade, multi-agent platform that takes any tabular dataset (CSV/Parquet)
and autonomously cleans it, explores it, engineers features, trains and evaluates
machine learning models, generates visualizations, writes business insights, and
compiles an executive PDF report — with a polished, guided Streamlit UI.

Built to satisfy the *Capstone Project Handbook: Multi-Agent AI Data Analyst*
(Techible × IIT Jammu) end to end.

## Features

- Multi-agent AI workflow
- Automatic data cleaning
- Exploratory Data Analysis
- Feature engineering
- Automatic task detection (Regression / Classification)
- Leakage-safe machine learning pipelines
- Multiple model comparison
- Interactive visualizations
- AI-generated business insights (Groq + offline fallback)
- Executive PDF report generation
- SQLite audit logs
- Docker support
- Streamlit deployment

## ✨ What's inside

| Layer | Tech |
|---|---|
| UI | Streamlit, custom CSS, multi-stage guided workflow (not sidebar-only) |
| Agents | Orchestrator, Cleaner, EDA, FeatureEng, ML, Visualization, Insights, ReportGen |
| ML | scikit-learn Pipelines + ColumnTransformer (leakage-safe CV), GridSearchCV |
| Insights LLM | Groq (free tier, OpenAI-compatible) with automatic offline fallback |
| Reporting | HTML → PDF via WeasyPrint (falls back to HTML if system libs are missing) |
| Audit trail | SQLite, queried live from the System Log Explorer page |
| Tests | pytest — unit tests for tools + a full pipeline integration test |

## 1. Prerequisites

- Python 3.11 (recommended) or 3.10+
- pip
- (Optional, for PDF export) native PDF rendering libraries — see step 5

## 2. Get the code onto your machine

1. Download/extract the project zip you were given.
2. Open the extracted `capstone-multi-agent-analyst/` folder in **VS Code**
   (`File → Open Folder…`) or `cd` into it from a terminal.

## 3. Create a virtual environment & install dependencies

```bash
cd capstone-multi-agent-analyst
python -m venv .venv

# Activate it:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 4. Configure your environment file

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and fill in:

```env
GROQ_API_KEY=your_free_groq_key_here     # optional but recommended
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a **free** Groq API key at https://console.groq.com/keys — no credit card
needed for the free tier. **The app works fully without a key too** — the
Business Insights Agent automatically falls back to a deterministic,
template-based narrative generator if `GROQ_API_KEY` is empty or the API
call fails for any reason (network, rate limit, etc.). Never commit your
real `.env` file — it's already excluded via `.gitignore`.

## 5. (Optional) Enable true PDF report export

The Report Generation Agent uses WeasyPrint, which needs native system
libraries for PDF rendering. If they're missing, the app **still works** —
it just delivers the report as a polished `.html` file instead of `.pdf`
(open it in any browser and use "Print → Save as PDF" if needed).

To get native PDF output:

- **macOS:** `brew install pango gdk-pixbuf libffi`
- **Ubuntu/Debian:** `sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0`
- **Windows:** install the [GTK3 runtime](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases) or run the app inside Docker (see below), which bundles everything for you.

## 6. Run the app

```bash
streamlit run app/Home.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## 7. Run the test suite

```bash
pytest -v
```

This runs unit tests for the data-cleaning and ML tools plus a full
end-to-end pipeline integration test using the Orchestrator agent.

## 8. Using the app

1. **📤 Data Ingestion** — upload a `.csv`/`.parquet` file (5,000–150,000 rows,
   8–45 columns recommended; the app warns but won't hard-block outside that range).
2. **🧹 Cleaning & 🔍 EDA** — one click runs both agents; review duplicates removed,
   imputation log, correlation heatmap, skew, and missingness.
3. **🤖 ML Studio** — pick your target column, build the engineered feature
   preview, then train & compare models (leakage-safe cross-validation).
4. **📊 Visualizations** — auto-generated distribution plots, correlation
   heatmap, feature importances, and model comparison charts.
5. **💡 Business Insights** — Groq-powered (or offline) executive narrative
   and strategic recommendations, tailored to the industry you selected at upload.
6. **📄 Reports Hub** — compile and download the final executive report.
7. **🗂️ System Log Explorer** — full audit trail of every agent action.

A workflow tracker at the top of every page always shows what's done, what's
active, and what's next — you're never stuck guessing.

## 9. Deploying

### Docker (recommended — includes PDF rendering libs out of the box)

```bash
docker build -t ai-data-analyst .
docker run -p 8501:8501 --env-file .env ai-data-analyst
```

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app → point to `app/Home.py`.
3. In the app's **Secrets** panel, add `GROQ_API_KEY = "..."` (and any other
   `.env` values) — never commit secrets to the repo.

## 10. Repository structure

```
capstone-multi-agent-analyst/
├── app/
│   ├── Home.py
│   └── pages/                # 1..7, one Streamlit page per workflow stage
├── src/
│   ├── config.py             # env-driven settings, no hardcoded secrets
│   ├── database.py           # SQLite audit trail
│   ├── state.py               # session-state + shared UI components
│   ├── agents/                # 8 specialized agents + orchestrator
│   └── tools/                 # deterministic, independently-tested functions
├── tests/                     # pytest unit + integration tests
├── workspace/                  # runtime data (git-ignored except structure)
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| "No `GROQ_API_KEY` found" warning | Expected if you skipped step 4 — insights still work via the offline template. |
| Report downloads as `.html` instead of `.pdf` | Install WeasyPrint's system libs (step 5) or use Docker. |
| `ModuleNotFoundError` on run | Make sure your virtual environment is activated and `pip install -r requirements.txt` completed without errors. |
| Upload rejected as "too many rows" | Adjust `MAX_ROWS` in `.env`, or trim the dataset. |
| Training feels slow on large files | Reduce dataset size for local testing, or increase machine resources — GridSearchCV is CPU-bound. |
