"""
Zero-configuration SQLite audit trail. Every agent action, tool call and
status transition is written here so the System Log Explorer page (P8) can
render a live, queryable history — required by Section 14.2 of the handbook.
"""
import sqlite3
import os
import json
import datetime
import threading

from src.config import settings, ensure_workspace

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    ensure_workspace()
    os.makedirs(os.path.dirname(settings.sqlite_db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_db_path, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            agent TEXT NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL,
            detail TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS run_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value TEXT
        )"""
    )
    conn.commit()
    return conn


def log_event(agent: str, action: str, status: str, detail=None) -> None:
    """status should be one of: 'started', 'success', 'error', 'info'."""
    if detail is not None and not isinstance(detail, str):
        try:
            detail = json.dumps(detail, default=str)
        except TypeError:
            detail = str(detail)
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO audit_log (timestamp, agent, action, status, detail) VALUES (?,?,?,?,?)",
            (datetime.datetime.now().isoformat(timespec="seconds"), agent, action, status, detail or ""),
        )
        conn.commit()
        conn.close()


def log_metric(name: str, value) -> None:
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO run_metrics (timestamp, metric_name, metric_value) VALUES (?,?,?)",
            (datetime.datetime.now().isoformat(timespec="seconds"), name, value),
        )
        conn.commit()
        conn.close()


def fetch_logs(limit: int = 300):
    with _lock:
        conn = _connect()
        cur = conn.execute(
            "SELECT timestamp, agent, action, status, detail FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
    return [
        {"timestamp": r[0], "agent": r[1], "action": r[2], "status": r[3], "detail": r[4]}
        for r in rows
    ]


def clear_logs() -> None:
    with _lock:
        conn = _connect()
        conn.execute("DELETE FROM audit_log")
        conn.execute("DELETE FROM run_metrics")
        conn.commit()
        conn.close()
