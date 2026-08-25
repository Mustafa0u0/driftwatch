# driftwatch

Finds three things going wrong in a stream of sensor readings — a spike, a
sensor that has stopped moving, and a baseline that has shifted and stayed.

Use it as a library or run it as an HTTP service. The detectors are pure
functions over a numpy array and know nothing about the web layer.

```bash
uvicorn driftwatch.api:app --reload
```

```bash
curl -X POST localhost:8000/analyse -H 'content-type: application/json' -d '{
  "series": {
    "sensor": "coop-2-water",
    "unit": "%",
    "readings": [{"at": "2026-01-01T00:00:00", "value": 45.2}, ...]
  }
}'
```

```json
{
  "sensor": "coop-2-water",
  "readings": 140,
  "healthy": false,
  "findings": [{
    "kind": "flatline",
    "at": "2026-01-01T06:40:00",
    "value": 45.0,
    "severity": 5.0,
    "detail": "60 consecutive readings of 45.00 — a stuck sensor reports no variance at all"
  }]
}
```

## The three faults

**Spike** — one reading far from its recent neighbours, then back to normal. A
door left open, interference on the line.

**Flatline** — a run of identical readings. This is the one that earns the
project. A sensor that has failed and is repeating its last value has *zero*
variance, so every statistical test for an outlier reports it as the
best-behaved sensor you own. It is the most normal-looking series there is.
Only asking "has this changed at all?" catches it.

**Level shift** — the baseline moves and stays. Physically different from a
spike: a spike is interference, a shift is a probe knocked loose or an
environment that genuinely changed.

## One fault, one alert

A level shift leaves every reading at the new baseline looking extreme against
a trailing window that still holds the old one — and it stays that way until
the window refills. The spike detector is not wrong about any of them; it is
describing one physical event a window's worth of times.

On a real series that turned one stuck vent into a level shift **plus eleven
spikes**. The engine now reports the shift and discards its shadow, because
alerts nobody reads are worse than no alerts. A genuine spike outside that
region is still reported.

## Tuning, with the numbers

The spike detector compares each reading against the median of a trailing
window, scaled by the median absolute deviation rather than the standard
deviation — squared distances let a single large spike widen the estimate
enough to hide itself.

MAD over a two-dozen-sample window is noisy and biased low, so a stretch that
happens to be quiet makes the very next ordinary reading look extreme.
Unfloored, this fired on **175 of every 10,000 readings of pure noise** — sixty
false alarms a day on a five-minute sensor. The window's spread is therefore
floored at `SCALE_FLOOR` times a longer trailing estimate.

Measured over 12,000 clean readings and 50 injected spikes per magnitude, at
the default threshold of 3:

| floor | 3σ | 4σ | 5σ and up | false positives / 10k |
| ----- | --- | --- | --------- | --------------------- |
| 1.0 | 42% | 94% | 96% | 39.2 |
| **1.2** | **10%** | **74%** | **100%** | **14.2** |
| 1.5 | 0% | 20% | 82% | 2.5 |

1.2 is the default: everything at 5σ and beyond is caught without fail, three
quarters of 4σ events are caught, and the alert volume stays low enough that
people keep reading the alerts. Raise it if your operators are drowning; lower
it if you would rather chase false alarms than miss a small one.

A 3σ event is mostly not reported at any floor. That is the honest consequence
of a threshold of 3 on a noisy estimator — ask for those by lowering
`threshold`, not by lowering the floor.

## What it will not do

**No history before it can judge.** The first two windows of a series are never
checked. A detector cannot say a reading is unusual until it has seen enough to
know what usual looks like.

**A shift is located to within about half a window.** Two windows straddle the
boundary before reaching it, so the strongest reading lands a little early.
Treat the index as "around here", not as a timestamp.

**It is stateless.** The caller owns the history. That keeps it horizontally
scalable and means it runs against a backfill as readily as against the last
hour — but it will not remember what it told you yesterday.

## Library use

```python
from driftwatch import spikes, flatlines, level_shifts

findings = spikes(values, window=24, threshold=3.0)
```

Each detector takes a numpy array and returns `Finding` objects carrying the
index, the value, a severity and a sentence explaining itself. Nothing imports
FastAPI.

## Validation

Readings must be in chronological order and the request is rejected if they are
not, rather than quietly sorted — a shuffled series usually means the caller is
merging sources incorrectly, and sorting it would hide that.

## Development

```bash
pip install -e '.[dev,serve]'
pytest          # 27 tests
ruff check .
python examples/farm.py
```

## Licence

MIT
