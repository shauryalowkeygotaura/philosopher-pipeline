"""Run-metrics writer: emits a small JSON the Command Center dashboard reads.

Shared schema with the other pipelines (see client-acquisition-pipeline's
modules/run_metrics.py). The dashboard fetches runs/latest.json over
raw.githubusercontent and the GitHub Actions workflow commits it each run.

    status: ok       posted/generated at least one reel
            degraded  ran clean but produced nothing new
            error     unhandled exception bubbled out
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RUNS_DIR = Path(__file__).resolve().parent / "runs"
PIPELINE_NAME = "philosopher"


def write(
    mode: str,
    status: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    budgets: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "pipeline": PIPELINE_NAME,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "status": status,
        "summary": summary,
        "metrics": metrics or {},
        "budgets": budgets or {},
    }
    try:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        latest = _RUNS_DIR / "latest.json"
        latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (_RUNS_DIR / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        return latest
    except Exception as e:  # best effort, never break a real run
        print(f"[run_metrics] failed to write metrics: {e}")
        return _RUNS_DIR / "latest.json"
