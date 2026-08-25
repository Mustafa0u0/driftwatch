"""The HTTP surface: validation, shapes and status codes."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from driftwatch.api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def series(values: list[float], *, sensor: str = "coop-2-temp") -> dict:
    base = datetime(2026, 1, 1)
    return {
        "sensor": sensor,
        "unit": "C",
        "readings": [
            {"at": (base + timedelta(minutes=5 * i)).isoformat(), "value": v}
            for i, v in enumerate(values)
        ],
    }


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_a_clean_series_comes_back_healthy(client: TestClient) -> None:
    values = list(np.random.default_rng(3).normal(20, 0.4, 200))

    body = client.post("/analyse", json={"series": series(values)}).json()

    assert body["healthy"] is True
    assert body["findings"] == []
    assert body["readings"] == 200


def test_a_spike_is_reported_with_its_timestamp(client: TestClient) -> None:
    values = list(np.random.default_rng(3).normal(20, 0.4, 200))
    values[120] = 40.0

    body = client.post("/analyse", json={"series": series(values)}).json()

    assert body["healthy"] is False
    spike = next(f for f in body["findings"] if f["kind"] == "spike")
    # Index 120 at five-minute intervals from midnight is 10:00.
    assert spike["at"].startswith("2026-01-01T10:00")


def test_findings_are_ordered_most_severe_first(client: TestClient) -> None:
    values = list(np.random.default_rng(3).normal(20, 0.4, 300))
    values[100] = 30.0
    values[200] = 60.0

    body = client.post("/analyse", json={"series": series(values)}).json()
    severities = [f["severity"] for f in body["findings"]]

    assert severities == sorted(severities, reverse=True)


def test_out_of_order_readings_are_rejected_not_silently_sorted(
    client: TestClient,
) -> None:
    payload = series([1.0, 2.0, 3.0])
    payload["readings"][1]["at"] = "2025-01-01T00:00:00"

    response = client.post("/analyse", json={"series": payload})

    assert response.status_code == 422
    assert "chronological" in response.text


def test_a_series_that_is_too_short_is_rejected(client: TestClient) -> None:
    payload = series([1.0])
    assert client.post("/analyse", json={"series": payload}).status_code == 422


def test_settings_are_validated(client: TestClient) -> None:
    payload = {"series": series([1.0, 2.0]), "settings": {"threshold": 0}}
    assert client.post("/analyse", json=payload).status_code == 422


def test_a_lower_threshold_reports_more(client: TestClient) -> None:
    values = list(np.random.default_rng(5).normal(20, 0.4, 300))
    values[150] = 20 + 3.5 * 0.4

    strict = client.post(
        "/analyse", json={"series": series(values), "settings": {"threshold": 5}}
    ).json()
    loose = client.post(
        "/analyse", json={"series": series(values), "settings": {"threshold": 1.5}}
    ).json()

    assert len(loose["findings"]) >= len(strict["findings"])


def test_a_stuck_sensor_is_reported_as_a_flatline(client: TestClient) -> None:
    values = list(np.random.default_rng(3).normal(20, 0.4, 60)) + [20.0] * 40

    body = client.post("/analyse", json={"series": series(values)}).json()

    assert any(f["kind"] == "flatline" for f in body["findings"])
