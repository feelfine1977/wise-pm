"""The calibration record of a population-dependent recipe (decision 0001, section 3).

A ``quantile_scale`` recipe divides an attribute by one of its quantiles.
Taken over whatever cases are being scored, that quantile makes the value of
a case depend on the other cases. A :class:`Calibration` freezes the divisor
together with what it was fitted on: the quantile, the fingerprint of the
log (:meth:`wise.EventLog.fingerprint`), the number of cases whose value
entered the quantile, and the time window of that log. Its status is
``fitted`` (the divisor is known) or ``unfitted`` (only the quantile is).

:func:`wise.fit_calibration` fits the records of a norm on a log. In a norm
document the record is the recipe's ``calibration`` key, written by
:meth:`Calibration.to_dict` and read by :func:`parse_calibration`::

    {"status": "fitted", "quantile": 0.95, "divisor": 1250.0,
     "log_fingerprint": "9f2c…", "size": 2000,
     "window": ["2018-01-01T00:00:00", "2019-01-31T23:59:00"]}
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any

from .errors import NormError


class CalibrationStatus(str, Enum):
    """Whether a calibration record holds a fitted divisor."""

    FITTED = "fitted"
    UNFITTED = "unfitted"


#: The keys of a calibration document.
KEYS: tuple[str, ...] = ("status", "quantile", "divisor", "log_fingerprint", "size", "window")
_FITTED_ONLY = ("divisor", "log_fingerprint", "size", "window")

Window = tuple[str | None, str | None]


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))


@dataclass(frozen=True)
class Calibration:
    """A frozen calibration record.

    ``quantile`` is the quantile in ``[0, 1]`` the divisor is (or will be) the
    value of. A fitted record also holds ``divisor`` (finite, ``> 0``),
    ``log_fingerprint`` (of the log it was fitted on), ``size`` (the number of
    cases whose value entered the quantile) and ``window`` (start and end of
    that log as ISO 8601 text, ``None`` where the log has no timestamp); an
    unfitted record holds none of them. Anything else is refused with
    :class:`~wise.errors.NormError`.
    """

    quantile: float
    status: CalibrationStatus = CalibrationStatus.UNFITTED
    divisor: float | None = None
    log_fingerprint: str | None = None
    size: int | None = None
    window: Window | None = None

    def __post_init__(self) -> None:
        try:
            status = CalibrationStatus(self.status)
        except ValueError:
            raise NormError(
                f"calibration status must be one of {[s.value for s in CalibrationStatus]}, got {self.status!r}"
            ) from None
        object.__setattr__(self, "status", status)
        if not _number(self.quantile) or not 0.0 <= float(self.quantile) <= 1.0:
            raise NormError(f"calibration quantile must be a number between 0 and 1, got {self.quantile!r}")
        object.__setattr__(self, "quantile", float(self.quantile))
        if status is CalibrationStatus.UNFITTED:
            given = [k for k in _FITTED_ONLY if getattr(self, k) is not None]
            if given:
                raise NormError(f"an unfitted calibration holds no {', '.join(given)}")
            return
        missing = [k for k in _FITTED_ONLY if getattr(self, k) is None]
        if missing:
            raise NormError(f"a fitted calibration needs {', '.join(missing)}")
        if not _number(self.divisor) or float(self.divisor) <= 0:  # type: ignore[arg-type]
            raise NormError(f"calibration divisor must be a finite number > 0, got {self.divisor!r}")
        object.__setattr__(self, "divisor", float(self.divisor))  # type: ignore[arg-type]
        if not isinstance(self.log_fingerprint, str) or not self.log_fingerprint:
            raise NormError(f"calibration log_fingerprint must be a non-empty string, got {self.log_fingerprint!r}")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise NormError(f"calibration size must be an integer >= 0, got {self.size!r}")
        object.__setattr__(self, "window", _window(self.window))

    @classmethod
    def fitted(cls, quantile: float, divisor: float, *, log_fingerprint: str, size: int, window: Window) -> Calibration:
        """A fitted record."""
        return cls(quantile, CalibrationStatus.FITTED, divisor, log_fingerprint, size, window)

    @property
    def is_fitted(self) -> bool:
        return self.status is CalibrationStatus.FITTED

    def to_dict(self) -> dict[str, Any]:
        """The document of the record (every key written)."""
        return {
            "status": self.status.value,
            "quantile": self.quantile,
            "divisor": self.divisor,
            "log_fingerprint": self.log_fingerprint,
            "size": self.size,
            "window": None if self.window is None else list(self.window),
        }


def _window(value: Any) -> Window:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise NormError(f"calibration window must be a pair [start, end], got {value!r}")
    for end in value:
        if end is not None and not isinstance(end, str):
            raise NormError(f"calibration window must hold ISO 8601 text or null, got {value!r}")
    return (value[0], value[1])


def parse_calibration(document: Any) -> Calibration:
    """Read a calibration document; refuse a malformed one with :class:`NormError` naming the key."""
    if not isinstance(document, Mapping):
        raise NormError(f"calibration must be a mapping, got {document!r}")
    unknown = [k for k in document if k not in KEYS]
    if unknown:
        raise NormError(f"calibration has unknown key {unknown[0]!r}; allowed {sorted(KEYS)}")
    for key in ("status", "quantile"):
        if key not in document:
            raise NormError(f"calibration is missing {key!r}")
    window = document.get("window")
    return Calibration(
        quantile=document["quantile"],
        status=document["status"],
        divisor=document.get("divisor"),
        log_fingerprint=document.get("log_fingerprint"),
        size=document.get("size"),
        window=None if window is None else _window(window),
    )


__all__ = ["KEYS", "Calibration", "CalibrationStatus", "parse_calibration"]
