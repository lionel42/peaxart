"""Reproducibility of fits under rounding (another CPU, BLAS or numpy build).

Different architectures round the last bit of floating point operations
differently. Most fitted values do not care, but ill-determined ones (noise
peaks, peaks cut by the domain edge, a tailing near 0, the split of two
co-eluting peaks) can move by tens of percent, or the number of peaks can change.

:func:`rounding_spread` simulates this on one machine: the float64 inputs of a
fit are nudged by a few ulp (relative 2^-52) and the fit is repeated::

    from peaxart.testing import rounding_spread

    spread = rounding_spread(fit_spectrum, x, y, mu_guess, apex_guess, sigma, alpha, 1.0, config)
    spread[["mu", "mu_spread", "area", "area_rel_spread"]]
    spread.attrs["n_peaks"]  # number of peaks of each nudged fit
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

ULP = 2.0**-52
"""Relative step of a nudge: one ulp of float64 at 1."""
NUDGES = (-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32)
"""Default nudges (ulp): enough to reproduce the results seen on GitHub's runners."""


def nudge(value, k: int):
    """``value`` with its float64 values scaled by ``1 + k ulp``.

    Python floats, float64 scalars and arrays are scaled, the output of a
    callable (e.g. sigma(x)) too; anything else (ints, float32 data, configs)
    is returned unchanged.
    """
    factor = 1.0 + k * ULP
    if isinstance(value, bool | int | np.integer):
        return value
    if isinstance(value, float | np.float64):
        return type(value)(value * factor)
    if isinstance(value, np.ndarray):
        return value * factor if value.dtype == np.float64 else value
    if callable(value) and not isinstance(value, type):

        def nudged(*a, **kw):
            return nudge(value(*a, **kw), k)

        return nudged
    return value


def peak_table(result) -> pd.DataFrame:
    """Table of the fitted peaks: a DataFrame as is, a dataclass (:class:`~peaxart.mzfit.ScanPeaks`,
    :class:`~peaxart.rtfit.ChromatoPeaks`) with one row per peak (scalar fields repeated),
    None (no peak) as an empty table."""
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame):
        return result.reset_index(drop=True)
    if dataclasses.is_dataclass(result):
        fields = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
        n = len(result)
        return pd.DataFrame({k: v if np.ndim(v) else np.repeat(v, n) for k, v in fields.items() if np.ndim(v) <= 1})
    raise TypeError(f"cannot make a peak table of {type(result).__name__}")


def match_peaks(ref: pd.DataFrame, other: pd.DataFrame, key: Sequence[str]) -> np.ndarray:
    """Index in ``other`` of the peak nearest to each peak of ``ref`` (-1 if ``other`` is empty).

    The distance sums the differences of the ``key`` columns, each in units of
    its range in ``ref`` (or 1 if constant).
    """
    if len(other) == 0:
        return np.full(len(ref), -1)
    dist = np.zeros((len(ref), len(other)))
    for col in key:
        r = ref[col].to_numpy(dtype=float)
        scale = np.ptp(r) if len(r) > 1 and np.ptp(r) > 0 else 1.0
        dist += np.abs(r[:, None] - other[col].to_numpy(dtype=float)[None, :]) / scale
    return np.argmin(dist, axis=1)


def rounding_spread(
    fit: Callable,
    *args,
    key: str | Sequence[str] | None = None,
    nudges: Sequence[int] = NUDGES,
    **kwargs,
) -> pd.DataFrame:
    """How much the peaks of ``fit(*args, **kwargs)`` move when its inputs are rounded differently.

    The fit is run once as is (the reference) and once per nudge, with every float64
    input scaled by ``1 + k ulp`` (see :func:`nudge`). Each reference peak is matched
    to the nearest peak of each nudged fit (by the ``key`` columns).

    :param fit: returns a DataFrame of peaks, a peak dataclass, or None (no peak).
    :param key: column(s) identifying a peak, default ``"rt"`` or else ``"mu"``.
        Several columns (e.g. ``("rt", "mz")``) when peaks share an rt.
    :param nudges: the nudges in ulp.
    :return: the reference peaks, and for each numeric column ``c``: ``c_spread``, the
        largest absolute change over the nudges, and ``c_rel_spread`` = ``c_spread / |c|``.
        NaN if a nudged fit has no peak. ``attrs["n_peaks"]``: number of peaks of the
        reference then of each nudged fit; ``attrs["nudges"]``: the nudges.
    """
    ref = peak_table(fit(*args, **kwargs))
    runs = [
        peak_table(fit(*(nudge(a, k) for a in args), **{n: nudge(v, k) for n, v in kwargs.items()})) for k in nudges
    ]
    if key is None:
        key = "rt" if "rt" in ref.columns else "mu"
    key = [key] if isinstance(key, str) else list(key)

    out = ref.copy()
    numeric = [
        c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c]) and not pd.api.types.is_bool_dtype(ref[c])
    ]
    spread = {c: np.zeros(len(ref)) for c in numeric}
    for run in runs:
        idx = match_peaks(ref, run, key) if len(ref) else np.zeros(0, int)
        for c in numeric:
            if (idx < 0).any() or c not in run.columns:
                d = np.full(len(ref), np.nan)
            else:
                d = np.abs(run[c].to_numpy(dtype=float)[idx] - ref[c].to_numpy(dtype=float))
            spread[c] = np.maximum(spread[c], d)  # NaN (no peak) is kept
    for c in numeric:
        out[f"{c}_spread"] = spread[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{c}_rel_spread"] = spread[c] / np.abs(ref[c].to_numpy(dtype=float))
    out.attrs["n_peaks"] = [len(ref)] + [len(r) for r in runs]
    out.attrs["nudges"] = list(nudges)
    return out


__all__ = ["NUDGES", "ULP", "match_peaks", "nudge", "peak_table", "rounding_spread"]
