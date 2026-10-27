"""Peak extraction in one rt / m/z domain (legacy ``extract_peaks``)."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from peaxart.config import ExtractionConfig
from peaxart.detect import histogram_centres, peak_detect
from peaxart.mzfit import detect_mass_peaks, fit_spectrum
from peaxart.rtfit import fit_chromato, rt_outliers
from peaxart.shapes import chromato

logger = logging.getLogger(__name__)

PEAK_COLUMNS = [
    "rt", "rt_u", "mz", "mz_u", "mz_u_ppm", "area", "area_u", "height", "height_u",
    "rt_sigma", "rt_tailing", "mz_sigma", "mz_alpha", "lod", "sn", "x", "x_u",
    "linear_ratio", "coverage", "n_points", "baseline",
]
"""Columns of the peak table, see :func:`extract_domain`."""

REJECTED_COLUMNS = [
    "rt", "mz", "height", "rt_sigma", "rt_tailing", "baseline", "linear_ratio", "coverage",
    "reason", "line_rt", "line",
]
"""Columns of the table of rejected peaks, see :func:`extract_domain` (``return_rejected``)."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in PEAK_COLUMNS})


def split_into_windows(start_idx: float, stop_idx: float, dt: float, n_max: float,
                       window: float, edge: float) -> list[tuple[float, float]]:
    """Split an rt index range into windows of ~``window`` s overlapping by ``edge`` (legacy)."""
    if stop_idx - start_idx > 1.2 * (window + 2 * edge) / dt:
        n_windows = int(round((stop_idx - start_idx) / (window / dt)))
        duration = int(round(stop_idx - start_idx) / n_windows)
        edge_idx = math.ceil(edge / dt)
        start_edge = max(0, start_idx - edge_idx)
        stop_edge = min(n_max - 1, stop_idx + edge_idx)
        out = [
            [start_edge + duration * i, start_edge + duration * (i + 1) + 2 * edge_idx]
            for i in range(n_windows)
        ]
        out[-1][1] = stop_edge
        return [tuple(w) for w in out]
    return [(start_idx, stop_idx)]


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
        rt_bins = g_rt / dt
        for w_start, w_stop in split_into_windows(rt_bins[0], rt_bins[-1], dt, np.inf,
                                                  config.rt_window, config.rt_window_edge):
            # as the legacy code: the window end is excluded
            sel = np.flatnonzero((rt_bins >= w_start) & (rt_bins < w_stop))
            if len(sel) <= config.min_points + 1:
                continue
            w_rt, w_area, w_x, w_mz = g_rt[sel], g_area[sel], g_x[sel], g_mz[sel]
            rt_sigma = config.rt_sigma_at((w_start + w_stop) / 2 * dt)
            pw = max(3, 3.0 * rt_sigma)
            pt = 40.0 / pw
            grid = np.arange(int((w_rt[0] - pw) / dt), int((w_rt[-1] + pw) / dt))
            detected = peak_detect(grid * dt, np.interp(grid, w_rt / dt, w_area), pw, pt, dt)
            # not more parameters than points: drop the smallest peaks
            while len(detected) > 0 and len(sel) < 1 + len(detected) * config.min_points:
                detected = detected[detected[:, 1] > detected[:, 1].min()]
            if len(detected) == 0:
                continue
            in_window = (detected[:, 0] >= w_start * dt) & (detected[:, 0] < w_stop * dt)
            if not in_window.any():
                continue
            base = histogram_centres(np.sort(w_area), 15, 5, 0.01, 1)
            baseline_guess = float(base.min()) if len(base) else float(w_area.min())
            window_rejected: list[dict] = []

            def run_fit(mu_guess, apex_guess, base_guess, rejected_list, position_bound=config.rt_position_bound):
                return fit_chromato(
                    w_rt, w_area, mu_guess, apex_guess, base_guess, dt, rt_sigma,
                    (w_start * dt, w_stop * dt), w_x, w_mz, config.conf_int, config.rt_peak_broadness,
                    config.ratio_to_linear_fit, position_bound, config.convention,
                    config.legacy_numerics, config.peak_linear_ratio, config.peak_min_coverage,
                    rejected=rejected_list, on_abort=config.on_abort,
                )

            fitted = run_fit(detected[:, 0], detected[:, 1], baseline_guess, window_rejected)
            # Try one more peak where the residual is largest (coeluting peaks closer
            # than the detection window); keep it only if the fit is clearly better.
            for _ in range(config.rt_extra_peaks):
                if fitted is None:
                    break
                trial = _try_extra_peak(fitted, w_rt, w_area, dt, rt_sigma, config, run_fit)
                if trial is None:
                    break
                fitted, window_rejected = trial
            rejected.extend(r for r in window_rejected if w_start * dt <= r["rt"] < w_stop * dt)
            if fitted is None:
                continue
            for k in range(len(fitted)):
                if not (w_start * dt <= fitted.rt[k] < w_stop * dt):
                    continue
                rows.append(_row(fitted, k, sigma_of_x, alpha, x, mz, rt))
    if not rows:
        return result(_empty())
    return result(pd.DataFrame(rows, columns=PEAK_COLUMNS))


def _model(f, t, convention):
    """Baseline + sum of the fitted chromatographic peaks at times ``t``."""
    out = np.full(len(t), f.baseline)
    for k in range(len(f)):
        out = out + chromato(t, f.height[k], f.rt[k], f.sigma[k], f.tailing[k], convention)
    return out


def _residual_std(f, t, y, convention) -> float:
    """Residual standard deviation corrected for the 1 + 4 n parameters."""
    dof = len(t) - (1 + 4 * len(f))
    if dof <= 0:
        return np.inf
    return float(np.sqrt(np.sum((y - _model(f, t, convention)) ** 2) / dof))


def _try_extra_peak(fitted, t, y, dt, rt_sigma, config, run_fit):
    """Refit with one more peak at the largest (smoothed) residual.

    :return: (the new fit, its rejected candidates) if it keeps more peaks, its residual std (corrected
        for the number of parameters) is below ``rt_extra_peak_ratio`` times the
        current one, and the added peak is at least ``rt_extra_peak_min_height``
        times the highest one, ``rt_extra_peak_min_separation`` rt sigmas away
        from the others and (``rt_extra_peak_shoulder``) makes a shoulder or a dip
        in the data, else None.
    """
    resid = y - _model(fitted, t, config.convention)
    # not wider than the data: np.convolve(mode="same") returns max(len(t), width) values
    width = min(len(t), max(3, int(round(rt_sigma / dt))))
    smooth = np.convolve(resid, np.ones(width) / width, mode="same")
    i = int(np.argmax(smooth))
    if smooth[i] <= 0 or len(t) <= 1 + 4 * (len(fitted) + 1):
        return None
    # the current peak nearest to the new one, and where it would start if it is moved
    # (see below): two coeluting peaks would be at about these two positions
    bound = max(3 * dt, rt_sigma) if config.rt_position_bound is None else config.rt_position_bound
    near = int(np.argmin(np.abs(fitted.rt - t[i])))
    moved = fitted.rt.copy()
    moved[near] -= np.clip((t[i] - moved[near]) / 2, -bound, bound)
    smoothing = max(2, width // 2)
    if config.rt_extra_peak_shoulder and _curvature_sign_changes(t, y, dt, moved[near], t[i], smoothing) < 2:
        return None  # no shoulder: not worth the fits (checked again on the fitted positions)
    # start height of the new peak: the (smoothed) data, the residual underestimates
    # it when the current peak covers both compounds
    data = np.convolve(y, np.ones(width) / width, mode="same")
    apex = np.append(fitted.height + fitted.baseline, max(fitted.baseline + smooth[i], data[i]))
    # two starts: the current peaks unchanged, and the nearest one moved away from
    # the new one by half their distance (one peak fitted to two coeluting ones sits
    # between them); the better fit is kept. The peaks may move by up to one rt
    # sigma (the normal bound, 3 scans, keeps the current peak where it is).
    starts = [fitted.rt] if np.array_equal(moved, fitted.rt) else [fitted.rt, moved]
    trial, trial_std, trial_rejected = None, np.inf, []
    for start in starts:
        start_rejected: list[dict] = []
        f = run_fit(np.append(start, t[i]), apex, fitted.baseline, start_rejected, position_bound=bound)
        if f is not None and len(f) > len(fitted):
            std = _residual_std(f, t, y, config.convention)
            if std < trial_std:
                trial, trial_std, trial_rejected = f, std, start_rejected
    if trial is None:
        return None
    # the added peak: the one of the trial farthest from all the current ones
    added = np.argmax([np.min(np.abs(fitted.rt - r)) for r in trial.rt])
    if trial.height[added] < config.rt_extra_peak_min_height * trial.height.max():
        return None
    others = np.arange(len(trial)) != added
    distance = np.abs(trial.rt[others] - trial.rt[added])
    if np.any(distance < config.rt_extra_peak_min_separation * np.maximum(trial.sigma[others], trial.sigma[added])):
        return None
    if config.rt_extra_peak_shoulder:
        nearest = np.flatnonzero(others)[np.argmin(distance)]
        if _curvature_sign_changes(t, y, dt, trial.rt[nearest], trial.rt[added], smoothing) < 2:
            return None
    if trial_std > config.rt_extra_peak_ratio * _residual_std(fitted, t, y, config.convention):
        return None
    return trial, trial_rejected


def _curvature_sign_changes(t, y, dt, a, b, width) -> int:
    """Number of sign changes of the curvature of y between t = a and t = b.

    y is put on a regular grid of step dt (summed per scan, interpolated over the
    missing scans) and smoothed twice by a moving average of ``width`` points.
    """
    scans = np.round((np.asarray(t) - t[0]) / dt).astype(int)
    summed = np.bincount(scans, weights=y)
    present = np.bincount(scans) > 0
    grid = np.arange(len(summed))
    regular = np.interp(grid, grid[present], summed[present])
    kernel = np.ones(width) / width
    smooth = np.convolve(np.convolve(regular, kernel, mode="same"), kernel, mode="same")
    curvature = np.sign(np.diff(smooth, 2))  # at grid points 1 .. n-2
    lo, hi = sorted([a, b])
    i0 = max(0, int(np.ceil((lo - t[0]) / dt)) - 1)
    i1 = int(np.floor((hi - t[0]) / dt)) - 1
    signs = curvature[i0 : i1 + 1]
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1]))


def _row(f, k, sigma_of_x, alpha, x, mz, rt) -> dict:
    # m/z sigma: x sigma converted with the local m/z spacing at the peak (scan closest to its rt)
    xs = float(sigma_of_x(f.x[k]))
    ends = mz.at_rt(float(f.rt[k]), np.array([f.x[k] - xs, f.x[k] + xs]))
    mz_sigma = (ends[1] - ends[0]) / 2
    rel_u = f.height_u[k] / f.height[k] if f.height[k] else np.nan
    return {
        "rt": f.rt[k], "rt_u": f.rt_u[k], "mz": f.mz[k], "mz_u": f.mz_u[k],
        "mz_u_ppm": f.mz_u[k] / f.mz[k] * 1e6, "area": f.area[k], "area_u": f.area[k] * rel_u,
        "height": f.height[k], "height_u": f.height_u[k], "rt_sigma": f.sigma[k],
        "rt_tailing": f.tailing[k], "mz_sigma": mz_sigma, "mz_alpha": alpha, "lod": f.lod[k],
        "sn": f.sn[k], "x": f.x[k], "x_u": f.x_u[k], "linear_ratio": f.linear_ratio[k], "coverage": f.coverage[k],
        "n_points": f.n_points, "baseline": f.baseline,
    }
