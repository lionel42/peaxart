"""Peak extraction in one rt / m/z domain (legacy ``extract_peaks``)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from peaxart.config import ExtractionConfig
from peaxart.detect import histogram_centres
from peaxart.mzfit import detect_mass_peaks, fit_spectrum
from peaxart.rtfit import fit_series, rt_outliers

logger = logging.getLogger(__name__)

PEAK_COLUMNS = [
    "rt",
    "rt_u",
    "mz",
    "mz_u",
    "mz_u_ppm",
    "area",
    "area_u",
    "height",
    "height_u",
    "rt_sigma",
    "rt_tailing",
    "mz_sigma",
    "mz_alpha",
    "lod",
    "sn",
    "x",
    "x_u",
    "linear_ratio",
    "coverage",
    "n_points",
    "baseline",
]
"""Columns of the peak table, see :func:`extract_domain`."""

REJECTED_COLUMNS = [
    "rt",
    "mz",
    "height",
    "rt_sigma",
    "rt_tailing",
    "baseline",
    "linear_ratio",
    "coverage",
    "reason",
    "line_rt",
    "line",
]
"""Columns of the table of rejected peaks, see :func:`extract_domain` (``return_rejected``)."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in PEAK_COLUMNS})


def interp_mz(xq, x, mz_row):
    """m/z at positions ``xq`` of the x axis.

    Interpolated linearly in sqrt(m/z): exact for a TOF (m/z proportional to the
    square of the flight time), and close to linear for small intervals otherwise.
    """
    return np.interp(xq, x, np.sqrt(mz_row)) ** 2


class MzAxis:
    """m/z of positions of the x axis: from an m/z grid, or from a calibration function."""

    def __init__(self, mz, x: np.ndarray, rt: np.ndarray):
        self.x = x
        self.rt = rt
        if callable(mz):
            self.func = mz
            self.grid = None
        else:
            self.func = None
            self.grid = np.asarray(mz, dtype=float)

    def at(self, i_scan: int, xq) -> np.ndarray:
        """m/z of positions ``xq`` in scan ``i_scan``."""
        if self.func is not None:
            xq = np.asarray(xq, dtype=float)
            return np.asarray(self.func(np.full(xq.shape, self.rt[i_scan]), xq), dtype=float)
        row = self.grid[i_scan] if self.grid.ndim == 2 else self.grid
        return interp_mz(xq, self.x, row)

    def at_rt(self, rt: float, xq) -> np.ndarray:
        """m/z at retention time ``rt`` (calibration function) or in the closest scan (grid)."""
        if self.func is not None:
            xq = np.asarray(xq, dtype=float)
            return np.asarray(self.func(np.full(xq.shape, rt), xq), dtype=float)
        return self.at(int(np.argmin(np.abs(self.rt - rt))), xq)


def _as_sigma(sigma) -> Callable[[np.ndarray], np.ndarray]:
    if callable(sigma):
        return sigma
    value = float(sigma)
    return lambda x: np.full(np.shape(x), value) if np.ndim(x) else value


def fit_scans(intensity, rt, x, mz, sigma, alpha, config: ExtractionConfig):
    """Per-scan detection and pseudo-Voigt fits.

    :return: dict of arrays, one value per fitted mass peak: rt, scan, x, area,
        sigma, alpha, baseline, mz.
    """
    sigma_of_x = _as_sigma(sigma)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    sigma_centre = float(sigma_of_x(float(int(x[len(x) // 2]))))
    rows = {k: [] for k in ("rt", "scan", "x", "area", "sigma", "alpha", "baseline", "mz")}
    for i_scan in range(len(rt)):
        y = intensity[i_scan]
        found = detect_mass_peaks(x, y, sigma_centre, dx, config)
        if len(found) == 0:
            continue
        peaks = fit_spectrum(x, y, found[:, 0], found[:, 1], sigma_of_x, alpha, dx, config)
        if len(peaks) == 0:
            continue
        n = len(peaks)
        rows["rt"] += [rt[i_scan]] * n
        rows["scan"] += [i_scan] * n
        rows["x"] += list(peaks.mu)
        rows["area"] += list(peaks.area)
        rows["sigma"] += list(peaks.sigma)
        rows["alpha"] += list(peaks.alpha)
        rows["baseline"] += [peaks.baseline] * n
        rows["mz"] += list(mz.at(i_scan, peaks.mu))
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def group_by_mz(scan_peaks: dict, x, mz, rt, config: ExtractionConfig) -> np.ndarray:
    """Group index of each per-scan peak (-1: not grouped), legacy histogram + nearest centre.

    The histogram bin width depends on the mean area (wider bins for small
    peaks, whose positions are more scattered).
    """
    masses = scan_peaks["mz"]
    mean_area = float(np.mean(scan_peaks["area"]))
    delta = max(0.003, 0.005 - 0.001 / 200.0 * mean_area)
    # slope threshold of the histogram detection: mz_hist_threshold_x in m/z at the centre
    x_int = float(int(x[len(x) // 2]))
    ends = mz.at_rt(float(rt.mean()), np.array([x_int, x_int + config.mz_hist_threshold_x]))
    pt = float(ends[1] - ends[0])
    centres = histogram_centres(masses, delta, config.mz_hist_width, pt, 1.0)
    group = np.full(len(masses), -1, dtype=np.int64)
    if len(centres) == 0:
        return group
    # nearest centre within 4 x the maximal difference
    right = np.clip(np.searchsorted(centres, masses), 0, len(centres) - 1)
    left = np.clip(right - 1, 0, len(centres) - 1)
    d_left, d_right = np.abs(centres[left] - masses), np.abs(centres[right] - masses)
    nearest = np.where(d_left < d_right, left, right)
    dist = np.minimum(d_left, d_right)
    nearest = np.where(dist <= 4.0 * config.mz_group_max_diff, nearest, -1)
    # Note: the legacy median-clipping loop (make_bins_np, NO_SLOPE) never removes a point.
    for i in range(len(centres)):
        members = nearest == i
        if members.sum() >= config.min_points + 1:
            group[members] = i
    return group


def extract_domain(
    intensity: np.ndarray,
    rt: np.ndarray,
    mz: np.ndarray,
    sigma: float | Callable,
    alpha: float,
    x: np.ndarray | None = None,
    config: ExtractionConfig | None = None,
    dt: float | None = None,
    return_rejected: bool = False,
    return_scans: bool = False,
) -> pd.DataFrame | tuple:
    """Extract the peaks of one domain (typically one unit mass +-0.5 and ~10 s).

    :param intensity: (n_scans, n_points) signal. Keep the native precision
        (float32 for TOF data): the noise statistics are computed in it, as in the legacy code.
    :param rt: (n_scans,) retention times (s), regularly sampled.
    :param mz: (n_points,) or (n_scans, n_points): m/z of each point, or a
        calibration function ``mz(rt, x)`` (arrays of the same shape).
    :param sigma: peak width (standard deviation) in the mass domain, in x units,
        a number or a function of x.
    :param alpha: Lorentzian fraction of the pseudo-Voigt peak shape.
    :param x: (n_points,) axis of the spectra (default: point index). The mass
        domain fits are done in x; the legacy code used the tof index.
    :param dt: rt sampling (s), default from ``rt``.
    :param return_rejected: also return the chromatographic peaks removed by the
        per-peak tests (``peak_linear_ratio``, ``peak_min_coverage``): a table with
        ``rt``, ``mz``, ``height``, ``rt_sigma``, ``rt_tailing``, ``baseline``,
        ``linear_ratio``, ``coverage``, ``reason`` and the local line that
        explains the data as well (``line_rt``, ``line``: arrays).
    :param return_scans: also return the per-scan mass peaks: dict of arrays
        ``rt``, ``scan``, ``x``, ``area``, ``sigma``, ``alpha``, ``baseline``,
        ``mz`` and ``group`` (m/z group, -1: none). Returned last.
    :return: one row per chromatographic peak, columns :data:`PEAK_COLUMNS`:

        - ``rt``, ``rt_u``: retention time (s) and its standard error
        - ``mz``, ``mz_u``, ``mz_u_ppm``: m/z (mean of the per-scan positions weighted
          by the fitted peak) and the spread of the per-scan positions
        - ``area``, ``area_u``: area (intensity x x-units x s) and its uncertainty
        - ``height``, ``height_u``: height of the chromatographic peak (intensity x x-units)
        - ``rt_sigma``, ``rt_tailing``: width (s) and tailing of the chromatographic peak
        - ``mz_sigma``, ``mz_alpha``: peak shape in the mass domain (sigma in m/z)
        - ``lod``: limit of detection (area), ``sn``: signal to noise
        - ``x``, ``x_u``: position in the x domain; ``n_points``: per-scan peaks used
        - ``linear_ratio``: local test against a straight background (small = real peak)
        - ``coverage``: fraction of the scans around the apex with a per-scan peak
        - ``baseline``: baseline of the chromatographic fit
    """
    config = config or ExtractionConfig()
    intensity = np.asarray(intensity)
    rt = np.asarray(rt, dtype=float)
    x = np.arange(intensity.shape[1], dtype=float) if x is None else np.asarray(x, dtype=float)
    mz = MzAxis(mz, x, rt)
    rejected: list[dict] = []

    scans_out = {k: np.zeros(0) for k in ("rt", "scan", "x", "area", "sigma", "alpha", "baseline", "mz")}
    scans_out["group"] = np.zeros(0, dtype=np.int64)

    def result(df):
        out = [df]
        if return_rejected:
            out.append(pd.DataFrame(rejected, columns=REJECTED_COLUMNS))
        if return_scans:
            out.append(scans_out)
        return out[0] if len(out) == 1 else tuple(out)

    if intensity.size == 0 or np.max(intensity) <= config.lod:
        return result(_empty())
    dt = float(np.median(np.diff(rt))) if dt is None else dt

    scans = fit_scans(intensity, rt, x, mz, sigma, alpha, config)
    if len(scans["x"]) < config.min_points:
        return result(_empty())
    # default (quicksort) argsorts as the legacy code: with equal rt values the
    # order of the points changes the rt peak detection (np.interp)
    order = np.argsort(scans["x"])
    scans = {k: v[order] for k, v in scans.items()}
    group = group_by_mz(scans, x, mz, rt, config)
    outlier = np.zeros(len(group), bool)
    scans_out = dict(scans, group=group, outlier=outlier)

    sigma_of_x = _as_sigma(sigma)
    rows = []
    for g in np.unique(group[group >= 0]):
        idx = np.flatnonzero(group == g)
        if len(idx) < config.min_points:
            continue
        idx = idx[np.argsort(scans["rt"][idx])]
        if config.rt_outlier_k is not None:
            bad = rt_outliers(scans["area"][idx], config.rt_outlier_k)
            outlier[idx[bad]] = True
            idx = idx[~bad]
            if len(idx) < config.min_points:
                continue
        g_rt, g_area = scans["rt"][idx], scans["area"][idx]
        g_x, g_mz = scans["x"][idx], scans["mz"][idx]
        for fitted, k in fit_series(g_rt, g_area, dt, config, g_x, g_mz, rejected=rejected):
            rows.append(_row(fitted, k, sigma_of_x, alpha, x, mz, rt))
    if not rows:
        return result(_empty())
    return result(pd.DataFrame(rows, columns=PEAK_COLUMNS))


def _row(f, k, sigma_of_x, alpha, x, mz, rt) -> dict:
    # m/z sigma: x sigma converted with the local m/z spacing at the peak (scan closest to its rt)
    xs = float(sigma_of_x(f.x[k]))
    ends = mz.at_rt(float(f.rt[k]), np.array([f.x[k] - xs, f.x[k] + xs]))
    mz_sigma = (ends[1] - ends[0]) / 2
    rel_u = f.height_u[k] / f.height[k] if f.height[k] else np.nan
    return {
        "rt": f.rt[k],
        "rt_u": f.rt_u[k],
        "mz": f.mz[k],
        "mz_u": f.mz_u[k],
        "mz_u_ppm": f.mz_u[k] / f.mz[k] * 1e6,
        "area": f.area[k],
        "area_u": f.area[k] * rel_u,
        "height": f.height[k],
        "height_u": f.height_u[k],
        "rt_sigma": f.sigma[k],
        "rt_tailing": f.tailing[k],
        "mz_sigma": mz_sigma,
        "mz_alpha": alpha,
        "lod": f.lod[k],
        "sn": f.sn[k],
        "x": f.x[k],
        "x_u": f.x_u[k],
        "linear_ratio": f.linear_ratio[k],
        "coverage": f.coverage[k],
        "n_points": f.n_points,
        "baseline": f.baseline,
    }
