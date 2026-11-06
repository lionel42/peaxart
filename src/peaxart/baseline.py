"""Baselines of chromatograms (any: unit mass, high resolution m/z, tic, ...).

:func:`remove_baseline`: smoothing, baseline removal and clipping of one or several
chromatograms (time along the first axis), with the parameters of a :class:`BaselineConfig`;
the method of the legacy alpinac unit mass chromatograms (``extract_chomatograms``,
:meth:`BaselineConfig.legacy`).

Steps: the acquisition changes the background abruptly (e.g. the filament emission
switched between 0.5 and 0.05 mA: all masses x10; resonant ejection of N2 / O2 / CO2
switched on and off). A smooth baseline cannot follow a step: it lags for tens of s and
leaves false signal. With ``cuts`` (scan indices of the steps), the chromatogram is
cut there and each piece smoothed and fitted separately. :func:`step_index` and
:func:`step_jump` locate and test a step near a known time (e.g. from the instrument log).
Fitted per piece, arPLS needs a stiffer ``lam`` than over a whole run (1e7 at 6 scans/s,
default): its weights use the noise of the series, which is homogeneous within a piece.

:func:`arpls`: asymmetrically reweighted penalized least squares (Baek et al.
2015), as ``pybaselines.Baseline().arpls`` (version 1.1, used by the legacy
alpinac) without the dependency: same defaults, weights, convergence test and
returned baseline.

    corrected = peaxart.remove_baseline(chromatograms)          # (n_scans,) or (n_scans, n)
    corrected = peaxart.remove_baseline(chromatograms, cuts=[1203, 6610])   # pieces fitted separately
    baseline = peaxart.baseline(chromatograms)                  # what is removed (after smoothing)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.linalg import solve_banded, solveh_banded
from scipy.special import expit

_MIN_FLOAT = np.finfo(float).eps


def _penalty_bands(n: int, diff_order: int, lam: float) -> np.ndarray:
    """``lam D^T D`` (D: (n - diff_order, n) difference matrix of order ``diff_order``) in the upper
    banded form of :func:`scipy.linalg.solveh_banded`: (diff_order + 1, n), main diagonal in the last row."""
    coefficients = np.diff(np.eye(diff_order + 1), diff_order, axis=0)[0]
    bands = np.zeros((diff_order + 1, n))
    rows = np.arange(n - diff_order)  # row j of D covers the columns j .. j + diff_order
    for a in range(diff_order + 1):
        for b in range(a, diff_order + 1):
            np.add.at(bands[diff_order - (b - a)], rows + b, coefficients[a] * coefficients[b])
    return lam * bands


@njit(cache=True)
def _cholesky_banded(ab: np.ndarray, b: np.ndarray, x: np.ndarray) -> bool:
    """Solve A x = b, A symmetric positive definite banded in the upper form of
    :func:`scipy.linalg.solveh_banded` (``ab[u + i - j, j] = A[i, j]``, i <= j). False if not
    positive definite. ~10x faster than scipy for the small bands of the baselines (no call overhead)."""
    u = ab.shape[0] - 1
    n = ab.shape[1]
    lower = np.zeros((u + 1, n))  # lower[k, i] = L[i, i - k]
    for i in range(n):
        for k in range(min(i, u), -1, -1):
            j = i - k
            total = ab[u - k, i]  # A[j, i]
            for p in range(max(0, i - u), j):
                total -= lower[i - p, i] * lower[j - p, j]
            if k == 0:
                if total <= 0.0 or not np.isfinite(total):
                    return False
                lower[0, i] = np.sqrt(total)
            else:
                lower[k, i] = total / lower[0, j]
    for i in range(n):  # L z = b
        total = b[i]
        for k in range(1, min(i, u) + 1):
            total -= lower[k, i] * x[i - k]
        x[i] = total / lower[0, i]
    for i in range(n - 1, -1, -1):  # L^T x = z
        total = x[i]
        for k in range(1, min(n - 1 - i, u) + 1):
            total -= lower[k, i + k] * x[i + k]
        x[i] = total / lower[0, i]
    return True


def _solve(upper_bands: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the symmetric banded system (Cholesky; LU if not positive definite)."""
    x = np.empty(len(b))
    if _cholesky_banded(upper_bands, b, x):
        return x
    try:
        return solveh_banded(upper_bands, b, check_finite=False)
    except np.linalg.LinAlgError:
        u = len(upper_bands) - 1
        full = np.vstack([upper_bands, np.zeros((u, upper_bands.shape[1]))])
        for k in range(1, u + 1):  # lower bands: the upper ones shifted
            full[u + k, : -k] = upper_bands[u - k, k:]
        return solve_banded((u, u), full, b, check_finite=False)


def _norm(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v * v)))


def _arpls_weights(y: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    residual = y - baseline
    negative = residual[residual < 0]
    std = negative.std(ddof=1) if negative.size >= 2 else _MIN_FLOAT
    if std == 0:
        std = _MIN_FLOAT
    with np.errstate(invalid="ignore"):  # no negative residual: mean nan, as pybaselines
        mean = np.mean(negative) if negative.size else np.nan
    return expit(-(2 / std) * (residual - (2 * std - mean)))


def arpls(y, lam: float = 1e5, diff_order: int = 2, max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    """Baseline of ``y`` by asymmetrically reweighted penalized least squares (arPLS).

    Minimises ``sum w (y - z)^2 + lam sum (D z)^2``; the weights ``w`` are
    recomputed from the residuals (points above the baseline get low weights)
    until they change by less than ``tol`` (relative, l2 norm), at most
    ``max_iter`` times. Same results as ``pybaselines.Baseline().arpls``.

    :param y: (n,) signal (no NaN).
    :param lam: smoothness; larger: smoother baseline.
    :param diff_order: order of the differences penalised.
    :return: (n,) baseline (NaN where the weights are NaN: no point below the baseline).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= diff_order:
        return y.copy()
    penalty = _penalty_bands(n, diff_order, lam)
    weights = np.ones(n)
    baseline = y
    for _ in range(max_iter + 1):
        system = penalty.copy()
        system[-1] += weights
        baseline = _solve(system, weights * y)
        new_weights = _arpls_weights(y, baseline)
        # relative l2 difference; not np.linalg.norm: BLAS (multithreaded) is ~100x slower on these vectors
        difference = _norm(new_weights - weights) / max(_norm(weights), _MIN_FLOAT)
        if difference < tol:
            break
        weights = new_weights
    return baseline


@dataclass
class BaselineConfig:
    """Smoothing, baseline and clipping of chromatograms."""

    #: smoothing kernel (centred); None: no smoothing
    smoothing: tuple[float, ...] | None = (0.25, 0.5, 0.25)
    #: padding of the chromatogram (of each piece) for the smoothing: "edge" (the end values) or "zeros"
    #: (legacy ``np.convolve(mode="same")``: the end points are low and pull the baseline down)
    padding: str = "edge"
    #: baseline method: "arpls", or None (no baseline removed)
    method: str | None = "arpls"
    #: arPLS smoothness (legacy 1e5, over the whole run)
    lam: float = 1e7
    #: arPLS order of the differences
    diff_order: int = 2
    #: arPLS iterations
    max_iter: int = 50
    #: arPLS convergence (relative change of the weights)
    tol: float = 1e-3
    #: clip the corrected chromatograms at 0
    clip: bool = True
    #: pieces between cuts shorter than this (scans): constant baseline, the 20th percentile
    min_piece: int = 30

    @classmethod
    def legacy(cls, **changes) -> BaselineConfig:
        """The legacy alpinac unit mass chromatograms: lam 1e5, zero padding of the smoothing (no cuts)."""
        return cls(**{"padding": "zeros", "lam": 1e5, **changes})


def smooth(y: np.ndarray, kernel, padding: str = "edge") -> np.ndarray:
    """Convolution of a 1D signal with a short centred kernel, same length; ``padding``: "edge" or "zeros"."""
    kernel = np.asarray(kernel, dtype=float)
    if padding == "zeros":
        return np.convolve(y, kernel, mode="same")
    if padding != "edge":
        raise ValueError(f"padding: 'edge' or 'zeros', not {padding!r}")
    left = (len(kernel) - 1) // 2
    padded = np.pad(np.asarray(y, dtype=float), (left, len(kernel) - 1 - left), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _cut_lists(cuts, n: int, n_columns: int | None) -> list[list[int]]:
    """Cuts of each column (one list for all columns, or one per column), sorted, inside ]0, n[."""
    if cuts is None:
        cuts = []
    per_column = n_columns is not None and len(cuts) > 0 and not np.isscalar(cuts[0])
    if per_column and len(cuts) != n_columns:
        raise ValueError(f"cuts for {len(cuts)} columns, chromatograms have {n_columns}")
    lists = cuts if per_column else [cuts] * (n_columns or 1)
    return [sorted({int(k) for k in c if 0 < k < n}) for c in lists]


def _pieces(y: np.ndarray, cuts: list[int], config: BaselineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed chromatogram and its baseline, each piece between the cuts separately."""
    smoothed, base = np.empty_like(y), np.empty_like(y)
    edges = [0, *cuts, len(y)]
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        piece = y[a:b]
        if config.smoothing is not None and b - a >= len(config.smoothing):
            piece = smooth(piece, config.smoothing, config.padding)
        smoothed[a:b] = piece
        if config.method is None:
            base[a:b] = 0.0
        elif config.method != "arpls":
            raise ValueError(f"Unknown baseline method {config.method!r}: 'arpls' or None")
        elif b - a < config.min_piece and len(edges) > 2:
            base[a:b] = np.percentile(piece, 20)
        else:
            fitted = arpls(piece, config.lam, config.diff_order, config.max_iter, config.tol)
            base[a:b] = np.nan_to_num(fitted, nan=0.0)  # NaN: no point below the baseline
    return smoothed, base


def remove_baseline(y, config: BaselineConfig | None = None, return_baseline: bool = False, cuts=None):
    """Chromatograms smoothed, baseline removed and clipped at 0 ((n,) or (n, m), time along the first axis).

    :param cuts: scan indices of steps of the background: the pieces between them are smoothed and
        fitted separately (index k: a piece starts at k). One list for all columns, or one per column.
    :param return_baseline: also return the baseline (removed from the smoothed chromatograms):
        ``corrected, baseline``.
    """
    config = config or BaselineConfig()
    y = np.asarray(y, dtype=float)
    if y.ndim not in (1, 2):
        raise ValueError(f"chromatograms: (n,) or (n, m), not {y.shape}")
    columns = y[:, None] if y.ndim == 1 else y
    cut_lists = _cut_lists(cuts, len(y), None if y.ndim == 1 else y.shape[1])
    corrected, base = np.empty_like(columns), np.empty_like(columns)
    for k in range(columns.shape[1]):
        smoothed, base[:, k] = _pieces(columns[:, k], cut_lists[k], config)
        corrected[:, k] = smoothed - base[:, k]
    if config.clip:
        np.clip(corrected, 0, None, out=corrected)
    if y.ndim == 1:
        corrected, base = corrected[:, 0], base[:, 0]
    return (corrected, base) if return_baseline else corrected


def baseline(y, config: BaselineConfig | None = None, cuts=None) -> np.ndarray:
    """Baseline of chromatograms ((n,) or (n, m), time along the first axis), after the smoothing
    (see :func:`remove_baseline`)."""
    return remove_baseline(y, config, return_baseline=True, cuts=cuts)[1]


def _log_level(y: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(y, dtype=float), 0, None) + 1)


def step_index(y: np.ndarray, start: int, stop: int) -> int:
    """Scan of the largest jump of ``log10(y + 1)`` in [start, stop[: the first scan after the jump."""
    start, stop = max(start, 0), min(stop, len(y))
    if stop - start < 2:
        return start
    return start + 1 + int(np.argmax(np.abs(np.diff(_log_level(y[start:stop])))))


def level_change(before: np.ndarray, after: np.ndarray, max_drift: float = 0.25, min_level: float = 0.0) -> float:
    """Step between two stretches of a signal: log10 of (median after) / (median before). NaN if they are
    not flat (the halves of either differ by more than ``max_drift`` x the step: e.g. the slope of a
    peak), if one has less than 4 points, or if both levels are below ``min_level`` (sparse signal)."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if len(before) < 4 or len(after) < 4:
        return np.nan
    if max(np.median(before), np.median(after)) < min_level:
        return np.nan
    lb, la = _log_level(before), _log_level(after)
    jump = float(np.median(la) - np.median(lb))
    drift = max(abs(np.median(h[len(h) // 2:]) - np.median(h[: len(h) // 2])) for h in (lb, la))
    return jump if drift <= max_drift * abs(jump) else np.nan


def step_jump(y: np.ndarray, index: int, width: int = 18, gap: int = 2, max_drift: float = 0.25,
              min_level: float = 0.0) -> float:
    """Step of the level of ``y`` at scan ``index`` (:func:`level_change` of the ``width`` scans before,
    ``gap`` scans before ``index`` left out, and the ``width`` scans after). NaN if not a step between two
    flat levels, or a window incomplete."""
    before = y[max(index - gap - width, 0): max(index - gap, 0)]
    after = y[index + 1: index + 1 + width]
    if len(before) < width or len(after) < width:
        return np.nan
    return level_change(before, after, max_drift, min_level)
