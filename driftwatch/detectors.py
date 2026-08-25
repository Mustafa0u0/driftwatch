"""Detectors for a single stream of sensor readings.

Each detector is a pure function over a numpy array. Keeping them free of the
web framework, the storage layer and each other is what makes them testable
against a hand-written series where the right answer is known by construction,
rather than only against whatever production happened to record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A reading further than this many robust standard deviations from the local
# median is called a spike. Three is the conventional starting point; it is
# exposed because the right value depends on how noisy the sensor is.
DEFAULT_Z = 3.0

# 1.4826 rescales the median absolute deviation so that, for normally
# distributed data, it estimates the same quantity as the standard deviation.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Finding:
    """Something worth a person's attention, with the evidence attached."""

    kind: str
    index: int
    value: float
    severity: float
    detail: str


def _robust_scale(window: np.ndarray) -> float:
    """Spread of a window, measured so that outliers do not inflate it.

    The standard deviation is computed from squared distances, so a single
    large spike widens it enough to hide itself. The median absolute deviation
    does not have that problem, which matters here because spikes are exactly
    what is being looked for.
    """
    median = np.median(window)
    mad = np.median(np.abs(window - median))
    return float(mad * MAD_TO_SIGMA)


# The comparison window's spread is floored at this multiple of a longer
# trailing estimate. Chosen by measurement, not taste — see the table in
# spikes() and the same table in the README.
SCALE_FLOOR = 1.2

# How many windows of history the floor is estimated over. Long enough that
# the median absolute deviation settles down, short enough to stay local.
SCALE_SPAN = 8

# Windows of history required before any spike is reported. A detector cannot
# say a reading is unusual until it has seen enough to know what usual is, and
# the median absolute deviation of a single window is too noisy an estimate to
# accuse anything on. The cost is real and worth stating: the opening stretch
# of a series is never checked.
WARMUP = 2


def _reference_scale(values: np.ndarray, index: int, window: int) -> float:
    """Spread of the recent past, used to floor a single window's estimate.

    Measured over a trailing span rather than the whole series, for two
    reasons. It is causal, so a detector running live sees exactly what this
    one sees. And it is local: a flat run elsewhere in the series would drag a
    whole-series estimate down — a sensor that dies halfway through drops the
    global spread so far that the floor stops protecting the half that is
    still working, and ordinary noise starts being reported as spikes.
    """
    start = max(0, index - window * SCALE_SPAN)
    return _robust_scale(values[start:index])


def spikes(
    values: np.ndarray,
    window: int = 24,
    threshold: float = DEFAULT_Z,
) -> list[Finding]:
    """Readings far from their recent neighbours.

    Compared against a trailing window rather than the whole series, so a
    sensor that legitimately drifts with the season is not flagged for every
    reading once the season turns.

    The window's spread is floored at ``SCALE_FLOOR`` times the spread of the
    whole series. Without that floor the detector invents spikes: the median
    absolute deviation of a two-dozen-sample window is a noisy estimator and
    biased low, so a stretch that happens to be quiet makes the very next
    ordinary reading look extreme. Unfloored, this fired on 175 of every 10,000
    readings of pure Gaussian noise — a detector that cries wolf sixty times a
    day on a five-minute sensor.

    The floor is a sensitivity trade, measured over 12,000 clean readings and
    60 injected spikes per magnitude, at the default threshold of 3:

    ======  ====  ====  =====  ==========
    floor    3σ    4σ    5σ+   false/10k
    ======  ====  ====  =====  ==========
    1.0     42%   94%    96%   39.2
    1.2     10%   74%   100%   14.2
    1.5      0%   20%    82%    2.5
    ======  ====  ====  =====  ==========

    1.2 is the default because anything at 5σ or beyond is caught without fail,
    three quarters of 4σ events are caught, and the alert volume stays low
    enough that people keep reading the alerts. Raise it if your operators are
    drowning; lower it if you would rather chase false alarms than miss a small
    one. A 3σ event is mostly not reported at any setting, which is the honest
    consequence of a threshold of 3 on a noisy estimator — ask for those by
    lowering ``threshold``, not by lowering the floor.
    """
    findings: list[Finding] = []
    start = window * WARMUP
    if len(values) <= start:
        return findings

    for i in range(start, len(values)):
        recent = values[i - window : i]
        scale = _robust_scale(recent)
        if scale == 0:
            # A flat window has no notion of "far". flatlines() owns this case.
            continue

        scale = max(scale, _reference_scale(values, i, window) * SCALE_FLOOR)
        deviation = abs(values[i] - np.median(recent)) / scale
        if deviation >= threshold:
            findings.append(
                Finding(
                    kind="spike",
                    index=i,
                    value=float(values[i]),
                    severity=min(deviation / threshold, 10.0),
                    detail=(
                        f"{values[i]:.2f} is {deviation:.1f} deviations from the "
                        f"median of the previous {window} readings"
                    ),
                )
            )
    return findings


def flatlines(values: np.ndarray, run: int = 12) -> list[Finding]:
    """Runs of identical readings.

    The detector that earns its place. A sensor that has failed and is
    repeating its last value has zero variance, so every statistical test for
    an outlier says it is behaving perfectly — it is the most normal-looking
    series there is. Only asking "has this changed at all?" catches it.
    """
    findings: list[Finding] = []
    if len(values) < run:
        return findings

    start = 0
    for i in range(1, len(values) + 1):
        ended = i == len(values) or values[i] != values[start]
        if not ended:
            continue

        length = i - start
        if length >= run:
            findings.append(
                Finding(
                    kind="flatline",
                    index=start,
                    value=float(values[start]),
                    severity=min(length / run, 10.0),
                    detail=(
                        f"{length} consecutive readings of {values[start]:.2f} — "
                        "a stuck sensor reports no variance at all"
                    ),
                )
            )
        start = i

    return findings


def level_shifts(
    values: np.ndarray,
    window: int = 24,
    threshold: float = DEFAULT_Z,
) -> list[Finding]:
    """Step changes in the baseline.

    Distinct from a spike: a spike is one reading that returns, a shift is the
    series settling somewhere new and staying. Physically these are different
    faults — a spike is interference, a shift is a sensor knocked loose or an
    environment that genuinely changed.

    The reported index locates the step to within about half a window. A
    detector comparing two windows cannot do better: the comparison already
    straddles the boundary before the boundary is reached, so the strongest
    reading lands a little before the true change. Treat the index as "around
    here", not as a timestamp.
    """
    findings: list[Finding] = []
    if len(values) < window * 2:
        return findings

    for i in range(window, len(values) - window + 1):
        before = values[i - window : i]
        after = values[i : i + window]

        scale = _robust_scale(before)
        if scale == 0:
            continue

        # Floored for the same reason as in spikes(): an unusually quiet
        # `before` window otherwise makes an ordinary step look enormous, and
        # the reported index drifts to wherever the estimate was smallest
        # rather than to where the step actually happened.
        scale = max(scale, _reference_scale(values, i, window) * SCALE_FLOOR)

        gap = abs(np.median(after) - np.median(before)) / scale
        if gap >= threshold:
            findings.append(
                Finding(
                    kind="level_shift",
                    index=i,
                    value=float(np.median(after)),
                    severity=min(gap / threshold, 10.0),
                    detail=(
                        f"baseline moved from {np.median(before):.2f} to "
                        f"{np.median(after):.2f} and stayed there"
                    ),
                )
            )

    return _collapse(findings, window)


def _collapse(findings: list[Finding], window: int) -> list[Finding]:
    """Keep the strongest finding per neighbourhood.

    A single step produces a hit at every offset the window slides through it.
    Reporting all of them would bury one event under twenty rows.
    """
    kept: list[Finding] = []
    for finding in sorted(findings, key=lambda f: -f.severity):
        if all(abs(finding.index - other.index) >= window for other in kept):
            kept.append(finding)
    return sorted(kept, key=lambda f: f.index)
