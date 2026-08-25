"""Renders the README figures.

The main figure calls `analyse()` — the same entry point the API uses — rather
than the raw detectors, so it shows what a caller actually receives. The second
figure shows why that distinction matters.

Run with: python docs/plot.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from driftwatch.detectors import flatlines, level_shifts, spikes
from driftwatch.engine import analyse
from driftwatch.models import AnalyseRequest, Reading, Series

INK = "#1b1b1a"
MUTED = "#8a897f"
LINE = "#3f6f5f"
HIT = "#c0392b"
BASE = datetime(2026, 1, 1)

rng = np.random.default_rng(7)

SERIES = [
    (
        "A spike, then back to normal",
        np.concatenate([rng.normal(21, 0.4, 120), [34.5], rng.normal(21, 0.4, 79)]),
    ),
    (
        "A sensor that stopped moving",
        np.concatenate([rng.normal(45, 1.2, 80), np.full(60, 45.0)]),
    ),
    (
        "A baseline that shifted and stayed",
        np.concatenate([rng.normal(21, 0.4, 90), rng.normal(16, 0.4, 90)]),
    ),
]


def findings_from_engine(values: np.ndarray) -> list[tuple[int, str]]:
    request = AnalyseRequest(
        series=Series(
            sensor="s",
            readings=[
                Reading(at=BASE + timedelta(minutes=5 * i), value=float(v))
                for i, v in enumerate(values)
            ],
        )
    )
    result = analyse(request)
    index_of = {BASE + timedelta(minutes=5 * i): i for i in range(len(values))}
    return [(index_of[f.at], f.kind) for f in result.findings]


def draw(ax, values, marks, title):
    ax.plot(values, color=LINE, linewidth=1.1)
    for index, kind in marks:
        ax.axvline(index, color=HIT, linewidth=1.3, alpha=0.9)
        ax.annotate(
            kind.replace("_", " "),
            xy=(index, ax.get_ylim()[1]),
            xytext=(6, -12),
            textcoords="offset points",
            color=HIT,
            fontsize=9,
            va="top",
        )
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9d7cf")


fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), facecolor="white")
for ax, (title, values) in zip(axes, SERIES, strict=True):
    draw(ax, values, findings_from_engine(values), title)
fig.tight_layout(pad=1.6)
fig.savefig("docs/detections.png", dpi=170, facecolor="white")
print("wrote docs/detections.png")

# One fault should produce one alert. Without suppression a level shift leaves
# every reading at the new baseline looking extreme against a trailing window
# that still holds the old one.
shift = SERIES[2][1]
raw = [(f.index, f.kind) for f in [*spikes(shift), *level_shifts(shift), *flatlines(shift)]]

fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), facecolor="white")
draw(axes[0], shift, raw, f"Detectors alone: {len(raw)} findings for one fault")
draw(axes[1], shift, findings_from_engine(shift), "Through the engine: 1")
fig.tight_layout(pad=1.6)
fig.savefig("docs/one-fault-one-alert.png", dpi=170, facecolor="white")
engine_count = len(findings_from_engine(shift))
print(f"wrote docs/one-fault-one-alert.png - raw={len(raw)}, engine={engine_count}")
