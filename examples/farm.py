"""Three faults a farm sensor actually produces, and what driftwatch says.

Run with:  python examples/farm.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from driftwatch import analyse
from driftwatch.models import AnalyseRequest, Reading, Series

rng = np.random.default_rng(7)
BASE = datetime(2026, 1, 1)


def report(name: str, values: list[float]) -> None:
    request = AnalyseRequest(
        series=Series(
            sensor=name,
            unit="C",
            readings=[
                Reading(at=BASE + timedelta(minutes=5 * i), value=v) for i, v in enumerate(values)
            ],
        )
    )
    result = analyse(request)
    print(f"\n{name}  ({result.readings} readings)")
    if result.healthy:
        print("  nothing to report")
    for finding in result.findings:
        print(f"  {finding.kind:12} {finding.at:%H:%M}  {finding.detail}")


# A door left open: one reading far from its neighbours, then back to normal.
spike = list(rng.normal(21, 0.4, 200))
spike[140] = 34.5
report("coop-1-temp", spike)

# The probe fell out of the water: the reading stops moving entirely. Every
# variance-based test says this is the best-behaved sensor on the farm.
stuck = list(rng.normal(45, 1.2, 80)) + [45.0] * 60
report("coop-2-water", stuck)

# A vent stuck open: the baseline moves and stays there.
shifted = list(rng.normal(21, 0.4, 90)) + list(rng.normal(16, 0.4, 90))
report("coop-3-temp", shifted)
