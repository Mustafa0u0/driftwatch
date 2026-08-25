"""HTTP surface.

Deliberately thin: it validates, calls the engine, and returns. Every decision
worth testing lives in detectors.py, which knows nothing about HTTP.
"""

from __future__ import annotations

from fastapi import FastAPI

from .engine import analyse
from .models import AnalyseRequest, AnalyseResponse

app = FastAPI(
    title="driftwatch",
    version="0.1.0",
    summary="Finds spikes, stuck sensors and baseline shifts in time-series readings.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalyseResponse)
def analyse_series(request: AnalyseRequest) -> AnalyseResponse:
    """Analyse one sensor's readings.

    Stateless on purpose. The caller owns the history, which keeps this
    horizontally scalable and means it can run against a backfill as easily as
    against the last hour.
    """
    return analyse(request)
