"""Backgrounds of the chromatographic fit, by name (as the peak functions: :mod:`peaxart.shapes`).

The fit (:func:`peaxart.rtfit.fit_chromato`) adds a background under the peaks of a window; its function is
chosen by name (:class:`BackgroundName`, or its string; :data:`BACKGROUNDS`, :func:`background`;
``ExtractionConfig.rt_background``, ``ChromatogramFitConfig.background``):

- ``"constant"``: one level (legacy);
- ``"linear"``: a straight line, its values at the start and the end of the window as parameters (both >= 0);
- ``"exponential"``: a level plus a decay from the start of the window, ``c + a exp(-(t - t0) / tau)``
  (the tail of an earlier large peak).

The parameters refer to the window ``span = (t0, t1)`` of the fit. A new function is a new subclass of
:class:`Background` (its value, Jacobian, start values from the data of the window and bounds), its name in
:class:`BackgroundName` and the class in :data:`BACKGROUNDS`.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np


class BackgroundName(StrEnum):
    """The backgrounds available (:data:`BACKGROUNDS`); equal to their string, which may be given instead."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


#: fraction of the window at each end whose level starts the end values (linear, exponential)
END_FRACTION = 0.25


def _ends(x, y, span, level: Callable) -> tuple[float, float]:
    """The level (``level(y)``) of the data in the first and the last :data:`END_FRACTION` of the window."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    t0, t1 = span
    length = END_FRACTION * (t1 - t0)
    first, last = x <= t0 + length, x >= t1 - length
    return (max(0.0, float(level(y[first]))) if first.any() else max(0.0, float(level(y))),
            max(0.0, float(level(y[last]))) if last.any() else max(0.0, float(level(y))))


class Background:
    """A background of ``n_params`` parameters (:attr:`params`) over a window ``span = (t0, t1)``."""

    name: BackgroundName
    params: tuple[str, ...] = ()

    @property
    def n_params(self) -> int:
        return len(self.params)

    def value(self, x, p, span) -> np.ndarray:
        raise NotImplementedError

    def jac(self, x, p, span) -> np.ndarray:
        """Derivatives of the value by the parameters, (len(x), n_params)."""
        raise NotImplementedError

    def slope(self, x, p, span) -> np.ndarray:
        """Derivative of the value by x."""
        raise NotImplementedError

    def guess(self, x, y, span, level: Callable) -> np.ndarray:
        """Start parameters from the data of the window, ``level(y)`` the level of a part of the data (e.g. a
        low quantile: the background under the peaks)."""
        raise NotImplementedError

    def limit(self, start: np.ndarray, factor: float | None) -> np.ndarray | None:
        """Upper bounds of the parameters kept over the refits: ``factor`` x the start values of the levels
        (None: no such bound, :meth:`bounds` sets them from the current start values)."""
        raise NotImplementedError

    def bounds(self, start: np.ndarray, limit: np.ndarray | None, span) -> tuple[list, list, list]:
        """(start, lower, upper) of the parameters of a fit."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Constant(Background):
    """A constant level (the legacy background)."""

    name = BackgroundName.CONSTANT
    params = ("level",)

    def value(self, x, p, span=None):
        return np.full(np.shape(x), float(p[0]))

    def jac(self, x, p, span=None):
        return np.ones((len(x), 1))

    def slope(self, x, p, span=None):
        return np.zeros(np.shape(x))

    def guess(self, x, y, span, level):
        return np.array([max(0.0, float(level(np.asarray(y, dtype=float))))])

    def limit(self, start, factor):
        return None if factor is None else np.array([factor * max(float(start[0]), 1.0)])

    def bounds(self, start, limit, span=None):
        # legacy: up to 10 x the start value (which rises with the refits), at least 3
        g = max(0.0, float(start[0]))
        hi = max(3.0, g * 10) if limit is None else max(3.0, float(limit[0]))
        p0 = min(max(1.0, g), hi)
        return [p0], [min(max(0.0, g / 3), p0)], [hi]


class Linear(Background):
    """A straight line: its values at the start and the end of the window (both >= 0)."""

    name = BackgroundName.LINEAR
    params = ("start", "end")

    @staticmethod
    def _u(x, span):
        t0, t1 = span
        return (np.asarray(x, dtype=float) - t0) / (t1 - t0)

    def value(self, x, p, span):
        u = self._u(x, span)
        return p[0] + (p[1] - p[0]) * u

    def jac(self, x, p, span):
        u = self._u(x, span)
        return np.column_stack([1.0 - u, u])

    def slope(self, x, p, span):
        return np.full(np.shape(x), (p[1] - p[0]) / (span[1] - span[0]))

    def guess(self, x, y, span, level):
        return np.array(_ends(x, y, span, level))

    def limit(self, start, factor):
        return None if factor is None else factor * np.maximum(np.asarray(start, dtype=float), 1.0)

    def bounds(self, start, limit, span=None):
        start = np.maximum(np.asarray(start, dtype=float), 0.0)
        hi = np.maximum(3.0, start * 10 if limit is None else limit)
        p0 = np.minimum(np.maximum(1.0, start), hi)
        return list(p0), [0.0, 0.0], list(hi)


class Exponential(Background):
    """A level plus a decay from the start of the window: ``level + amplitude exp(-(t - t0) / tau)`` (the tail
    of an earlier large peak); level and amplitude >= 0, tau between :attr:`tau_range` x the window length."""

    name = BackgroundName.EXPONENTIAL
    params = ("level", "amplitude", "tau")
    #: bounds of tau, times the length of the window, and its start value
    tau_range: tuple[float, float] = (0.05, 3.0)
    tau_start: float = 0.3

    def _e(self, x, p, span):
        return np.exp(-(np.asarray(x, dtype=float) - span[0]) / p[2])

    def value(self, x, p, span):
        return p[0] + p[1] * self._e(x, p, span)

    def jac(self, x, p, span):
        e = self._e(x, p, span)
        dx = np.asarray(x, dtype=float) - span[0]
        return np.column_stack([np.ones(len(e)), e, p[1] * e * dx / p[2] ** 2])

    def slope(self, x, p, span):
        return -p[1] / p[2] * self._e(x, p, span)

    def guess(self, x, y, span, level):
        first, last = _ends(x, y, span, level)
        return np.array([last, max(0.0, first - last), self.tau_start * (span[1] - span[0])])

    def limit(self, start, factor):
        if factor is None:
            return None
        # the level and the value at the window start (level + amplitude) below factor x their start values
        top = factor * max(float(start[0] + start[1]), 1.0)
        return np.array([factor * max(float(start[0]), 1.0), top, np.inf])

    def bounds(self, start, limit, span):
        length = span[1] - span[0]
        tau_lo, tau_hi = self.tau_range[0] * length, self.tau_range[1] * length
        level, amplitude = max(0.0, float(start[0])), max(0.0, float(start[1]))
        hi_level = max(3.0, level * 10 if limit is None else float(limit[0]))
        hi_amplitude = max(3.0, (level + amplitude) * 10 if limit is None else float(limit[1]))
        p0 = [min(max(1.0, level), hi_level), min(max(1.0, amplitude), hi_amplitude),
              float(np.clip(start[2], tau_lo, tau_hi))]
        return p0, [0.0, 0.0, tau_lo], [hi_level, hi_amplitude, tau_hi]


BACKGROUNDS: dict[BackgroundName, type[Background]] = {b.name: b for b in (Constant, Linear, Exponential)}
"""The backgrounds of the chromatographic fit, by name."""


def background(name: BackgroundName | str | Background = BackgroundName.CONSTANT) -> Background:
    """The background of this name (:data:`BACKGROUNDS`), or the given one."""
    if isinstance(name, Background):
        return name
    try:
        return BACKGROUNDS[name]()
    except KeyError:
        raise ValueError(f"Unknown background {name!r}, one of {[str(b) for b in BackgroundName]}") from None
