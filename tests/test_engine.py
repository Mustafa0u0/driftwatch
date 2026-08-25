"""How findings from different detectors are combined into one answer."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from driftwatch.engine import analyse
from driftwatch.models import AnalyseRequest, Reading, Series, Settings


def request(values: list[float], **settings: object) -> AnalyseRequest:
    base = datetime(2026, 1, 1)
    return AnalyseRequest(
        series=Series(
            sensor="s",
            readings=[
                Reading(at=base + timedelta(minutes=5 * i), value=v) for i, v in enumerate(values)
            ],
        ),
        settings=Settings(**settings) if settings else Settings(),
    )


def test_one_physical_fault_produces_one_finding() -> None:
    """The regression this exists for.

    A level shift leaves every reading at the new baseline looking extreme
    against a trailing window still holding the old one. Unsuppressed, a single
    stuck vent reported one shift and eleven spikes.
    """
    rng = np.random.default_rng(7)
    values = list(rng.normal(21, 0.4, 90)) + list(rng.normal(16, 0.4, 90))

    result = analyse(request(values))

    assert [f.kind for f in result.findings] == ["level_shift"]


def test_a_real_spike_after_a_shift_is_still_reported() -> None:
    """Suppression must not swallow an unrelated fault that follows."""
    rng = np.random.default_rng(7)
    values = list(rng.normal(21, 0.4, 90)) + list(rng.normal(16, 0.4, 150))
    values[220] = 60.0  # well clear of the shift's shadow

    result = analyse(request(values))
    kinds = [f.kind for f in result.findings]

    assert "level_shift" in kinds
    assert "spike" in kinds


def test_a_clean_series_is_healthy() -> None:
    values = list(np.random.default_rng(2).normal(20, 0.4, 200))
    result = analyse(request(values))

    assert result.healthy
    assert result.findings == []


def test_findings_carry_the_timestamp_not_the_index() -> None:
    values = list(np.random.default_rng(2).normal(20, 0.4, 200))
    values[120] = 45.0

    result = analyse(request(values))
    spike = next(f for f in result.findings if f.kind == "spike")

    assert spike.at == datetime(2026, 1, 1) + timedelta(minutes=5 * 120)


def test_the_most_severe_finding_comes_first() -> None:
    values = list(np.random.default_rng(2).normal(20, 0.4, 400))
    values[150] = 30.0
    values[300] = 90.0

    result = analyse(request(values))

    assert result.findings[0].value == 90.0
