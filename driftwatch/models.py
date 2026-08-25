"""Request and response shapes.

Validation lives here rather than in the handlers, so a malformed series is
rejected with a specific message before any detector sees it.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise

from pydantic import BaseModel, Field, field_validator


class Reading(BaseModel):
    at: datetime
    value: float


class Series(BaseModel):
    sensor: str = Field(min_length=1, max_length=128)
    unit: str | None = Field(default=None, max_length=32)
    readings: list[Reading] = Field(min_length=2, max_length=50_000)

    @field_validator("readings")
    @classmethod
    def _ordered(cls, readings: list[Reading]) -> list[Reading]:
        """Reject out-of-order input rather than quietly sorting it.

        A series that arrives shuffled usually means the caller is merging
        sources incorrectly. Sorting it here would hide that, and every
        detector below assumes neighbours in the list are neighbours in time.
        """
        for earlier, later in pairwise(readings):
            if later.at < earlier.at:
                raise ValueError(
                    f"readings must be in chronological order; {later.at.isoformat()} "
                    f"follows {earlier.at.isoformat()}"
                )
        return readings


class Settings(BaseModel):
    window: int = Field(default=24, ge=3, le=5_000)
    threshold: float = Field(default=3.0, gt=0, le=20)
    flatline_run: int = Field(default=12, ge=2, le=5_000)


class AnalyseRequest(BaseModel):
    series: Series
    settings: Settings = Settings()


class FindingOut(BaseModel):
    kind: str
    at: datetime
    value: float
    severity: float
    detail: str


class AnalyseResponse(BaseModel):
    sensor: str
    readings: int
    findings: list[FindingOut]
    healthy: bool
