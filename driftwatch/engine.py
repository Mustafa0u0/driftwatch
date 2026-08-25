"""Runs the detectors over a series and turns indices back into timestamps."""

from __future__ import annotations

import numpy as np

from .detectors import Finding, flatlines, level_shifts, spikes
from .models import AnalyseRequest, AnalyseResponse, FindingOut


def analyse(request: AnalyseRequest) -> AnalyseResponse:
    series, settings = request.series, request.settings
    values = np.array([r.value for r in series.readings], dtype=float)
    times = [r.at for r in series.readings]

    shifts = level_shifts(values, window=settings.window, threshold=settings.threshold)
    findings = [
        *_without_shift_shadow(
            spikes(values, window=settings.window, threshold=settings.threshold),
            shifts,
            settings.window,
        ),
        *flatlines(values, run=settings.flatline_run),
        *shifts,
    ]

    # Most severe first: a response truncated in a UI should lose the least
    # important rows, not an arbitrary selection.
    findings.sort(key=lambda f: (-f.severity, f.index))

    return AnalyseResponse(
        sensor=series.sensor,
        readings=len(values),
        findings=[
            FindingOut(
                kind=f.kind,
                at=times[f.index],
                value=f.value,
                severity=round(f.severity, 2),
                detail=f.detail,
            )
            for f in findings
        ],
        healthy=not findings,
    )


def _without_shift_shadow(
    found: list[Finding],
    shifts: list[Finding],
    window: int,
) -> list[Finding]:
    """Drop the spikes a level shift casts behind it.

    When the baseline moves, every reading at the new level is genuinely far
    from a trailing window that still holds the old one — and stays far until
    the window refills. The detector is not wrong; it is describing the same
    physical event a window's worth of times. On a real series this turned one
    stuck vent into a level shift plus eleven spikes, which is the alert
    fatigue this whole thing exists to avoid.

    The shift is the explanation, so the shift is what gets reported and its
    shadow is discarded. Spikes outside that region are untouched: a genuine
    outlier shortly after a shift is still worth knowing about.
    """
    if not shifts:
        return found

    def shadowed(finding: Finding) -> bool:
        return any(0 <= finding.index - shift.index <= window * 2 for shift in shifts)

    return [f for f in found if not shadowed(f)]
