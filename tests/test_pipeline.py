import os
import sys
import shutil
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import OrchestratorAgent

TEST_WORKSPACE = os.path.join(os.path.dirname(__file__), "_tmp_workspace")


@pytest.fixture(scope="module")
def workspace_paths():
    paths = {
        "cleaned_dir": os.path.join(TEST_WORKSPACE, "cleaned"),
        "features_dir": os.path.join(TEST_WORKSPACE, "features"),
        "models_dir": os.path.join(TEST_WORKSPACE, "models"),
        "viz_dir": os.path.join(TEST_WORKSPACE, "visualizations"),
        "reports_dir": os.path.join(TEST_WORKSPACE, "reports"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    yield paths
    shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)


def test_full_pipeline_end_to_end(workspace_paths):
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "segment": rng.choice(["A", "B", "C"], n),
    })
    df.loc[rng.choice(n, 15, replace=False), "income"] = np.nan
    df["churn"] = ((df["age"] < 30) | (df["segment"] == "A")).astype(int)

    raw_path = os.path.join(TEST_WORKSPACE, "sample.csv")
    os.makedirs(TEST_WORKSPACE, exist_ok=True)
    df.to_csv(raw_path, index=False)

    events = []
    orchestrator = OrchestratorAgent()
    ok, results = orchestrator.execute_full_workflow(
        raw_path, "sample", "churn", "General / Cross-Industry", workspace_paths,
        progress_cb=lambda stage, status, detail="": events.append((stage, status)),
    )

    assert ok is True
    assert os.path.exists(results["cleaned_path"])
    assert results["metrics"]["model_name"] is not None
    assert os.path.exists(results["model_path"])
    assert results["report_path"] is not None
    assert any(status == "error" for _, status in events) is False
