from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SensorType(IntEnum):
    """Unified label space for sensor semantic alignment."""

    ABSTAIN = -1
    TEMPERATURE = 0
    PRESSURE = 1
    VIBRATION = 2
    CURRENT = 3
    VOLTAGE = 4


@dataclass(frozen=True)
class SensorMetadata:
    """Raw sensor metadata before semantic alignment."""

    field_name: str
    mean_value: float
    unit: str
    sampling_freq: float

