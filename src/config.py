"""
Central configuration module.
All secrets/config are read from environment variables (.env). Nothing is
ever hardcoded here. See .env.example for the full list of supported keys.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load variables from a local .env file if present (no-op in prod containers
# where env vars are injected directly).
load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- LLM provider (Groq free tier) ---
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    # --- Workspace / storage ---
    workspace_dir: str = field(default_factory=lambda: os.getenv("SYSTEM_WORKSPACE_DIR", "./workspace/"))
    sqlite_db_path: str = field(default_factory=lambda: os.getenv("SQLITE_DB_PATH", "./workspace/metadata/audit_telemetry.db"))
    logging_level: str = field(default_factory=lambda: os.getenv("LOGGING_LEVEL", "INFO"))

    # --- Dataset guardrails (Section 8.2 of handbook) ---
    min_rows: int = field(default_factory=lambda: _int_env("MIN_ROWS", 5000))
    max_rows: int = field(default_factory=lambda: _int_env("MAX_ROWS", 150000))
    min_cols: int = 8
    max_cols: int = 45
    max_file_mb: int = field(default_factory=lambda: _int_env("MAX_FILE_MB", 200))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()


def ensure_workspace() -> None:
    """Create the standard workspace sub-directories if they don't exist."""
    for sub in ("uploads", "cleaned", "features", "models", "visualizations", "reports", "metadata"):
        os.makedirs(os.path.join(settings.workspace_dir, sub), exist_ok=True)


def workspace_path(*parts: str) -> str:
    ensure_workspace()
    return os.path.join(settings.workspace_dir, *parts)
