"""Detector behaviour, tested against series where the answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.detectors import flatlines, level_shifts, spikes


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_clean_noise_is_left_alone(rng: np.random.Generator) -> None:
    assert spikes(rng.normal(20, 0.4, 300)) == []


def test_nothing_is_reported_before_the_detector_has_history() -> None:
    """The warm-up is deliberate, so it is worth pinning down.

    Until enough readings have been seen the scale estimate is too noisy to
    accuse anything on, so the opening stretch is never checked.
    """
    values = np.zeros(60)
    values[10] = 1000.0  # inside the warm-up

    assert spikes(values, window=24) == []


def test_a_single_spike_is_found_at_its_index(rng: np.random.Generator) -> None:
    values = rng.normal(20, 0.4, 200)
    values[120] = 20 + 8 * 0.4

    found = spikes(values)

    assert [f.index for f in found] == [120]
    assert found[0].kind == "spike"
    assert found[0].severity > 1


def test_a_series_shorter_than_the_window_yields_nothing() -> None:
    assert spikes(np.array([1.0, 2.0, 3.0]), window=24) == []


def test_severity_grows_with_the_size_of_the_spike(rng: np.random.Generator) -> None:
    def severity_of(sigma: float) -> float:
        values = rng.normal(20, 0.4, 200)
        values[120] = 20 + sigma * 0.4
        return next(f.severity for f in spikes(values) if f.index == 120)

    assert severity_of(12) > severity_of(6)


def test_a_stuck_sensor_is_caught_even_though_it_looks_perfect() -> None:
    """The case a variance-based test cannot see.

    A sensor repeating its last value has zero spread, so every outlier test
    says it is the best-behaved series on the farm.
    """
    values = np.concatenate([np.random.default_rng(1).normal(20, 0.4, 40), np.full(30, 20.0)])

    assert spikes(values) == [], "a flat run has no outliers to find"
    assert [f.kind for f in flatlines(values)] == ["flatline"]


def test_a_short_repeat_is_not_a_flatline() -> None:
    values = np.array([20.0, 20.0, 20.0, 21.0, 22.0])
    assert flatlines(values, run=12) == []


def test_a_flatline_reports_where_it_started() -> None:
    values = np.concatenate([np.arange(10, dtype=float), np.full(20, 9.0)])
    found = flatlines(values, run=12)

    # The run begins at index 9 — the last of the ramp already holds the value.
    assert found[0].index == 9


def test_a_step_change_is_a_shift_not_a_spike(rng: np.random.Generator) -> None:
    values = np.concatenate([rng.normal(20, 0.4, 60), rng.normal(26, 0.4, 60)])

    found = level_shifts(values)

    assert len(found) == 1
    assert found[0].kind == "level_shift"
    # Two windows straddle the boundary before reaching it, so the strongest
    # reading lands a little early. Half a window is the honest resolution.
    assert abs(found[0].index - 60) <= 12


def test_one_step_reports_once_not_once_per_offset(rng: np.random.Generator) -> None:
    """A sliding window crosses a step many times; the caller wants one row."""
    values = np.concatenate([rng.normal(20, 0.4, 80), rng.normal(30, 0.4, 80)])

    assert len(level_shifts(values)) == 1


def test_a_spike_alone_does_not_register_as_a_shift(rng: np.random.Generator) -> None:
    values = rng.normal(20, 0.4, 200)
    values[100] = 40.0

    assert level_shifts(values) == [], "the baseline returned, so nothing shifted"


def test_false_positive_rate_stays_within_budget(rng: np.random.Generator) -> None:
    """Guards the SCALE_FLOOR trade-off documented in spikes().

    Without the floor this detector fired on ~175 of every 10,000 clean
    readings. The budget here is deliberately loose enough not to be flaky and
    tight enough to catch a regression of that size.
    """
    readings = 0
    found = 0
    for _ in range(20):
        values = rng.normal(20, 0.4, 400)
        readings += len(values)
        found += len(spikes(values))

    per_10k = found / readings * 10_000
    assert per_10k < 30, f"{per_10k:.1f} false positives per 10,000 readings"


def test_large_spikes_are_always_caught(rng: np.random.Generator) -> None:
    caught = 0
    for _ in range(30):
        values = rng.normal(20, 0.4, 200)
        values[120] = 20 + 6 * 0.4
        caught += any(f.index == 120 for f in spikes(values))

    assert caught == 30
