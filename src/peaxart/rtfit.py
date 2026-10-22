"""Retention time domain: chromatographic fit of the areas of one m/z group.

Legacy ``fit_chromato``: the per-scan areas of the peaks of one m/z group
are fitted as a function of rt with a sum of chromatographic peaks (Pap and
Papai) plus a constant baseline. Peaks whose height is not significant
(Student t test against the residuals and the baseline uncertainty) are
removed and the fit is redone. The fit is rejected if it is not better than
a straight line (``ratio_to_linear_fit``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from peaxart.leastsq import fit
from peaxart.shapes import chromato, chromato_area


@dataclass
class ChromatoPeaks:
    """Fitted chromatographic peaks (arrays of the same length)."""

    rt: np.ndarray
    rt_u: np.ndarray
    """Standard error of the fitted rt (s)."""
    height: np.ndarray
    height_u: np.ndarray
    area: np.ndarray
    sigma: np.ndarray
    tailing: np.ndarray
    lod: np.ndarray
    """Limit of detection (area) of each peak."""
    sn: np.ndarray
    """Signal to noise: height / height at the LOD."""
    x: np.ndarray
    x_u: np.ndarray
    """Position in the x domain (mean over scans weighted by the fitted profile) and its spread."""
    mz: np.ndarray
    mz_u: np.ndarray
    """m/z (weighted mean of the per-scan positions) and its spread."""
    linear_ratio: np.ndarray
    """Local test of each peak against a straight background (see :func:`local_linear_ratio`):
    residual std with the peak / residual std of a line. Small = the peak is real."""
    coverage: np.ndarray
    """Fraction of the scans within +-1 sigma of the apex that have a point (see :func:`apex_coverage`)."""
    n_points: int
    """Number of per-scan peaks used by the fit."""
    baseline: float

    def __len__(self) -> int:
        return len(self.rt)


def _linear_fit_std(x, y):
    """Residual standard deviation of a least squares line (legacy ``fit_linear_np``)."""
    n = len(y)
    x_sum = np.sum(x)
    den = np.sum(x**2) - x_sum**2 / n
    if den != 0:
        y_sum = np.sum(y)
        x_y_sum = np.sum(x * y)
        a = (x_y_sum - x_sum * y_sum / n) / den
        b = ((np.sum(x**2) * y_sum - x_sum * x_y_sum) / n) / den
    else:
        a, b = 0, np.sum(y) / n
    return np.sqrt(np.sum((x * a + b - y) ** 2) / float(n - 1))


def rt_outliers(y, k: float, dominance: float = 3.0, half_width: int = 2) -> np.ndarray:
    """Mask of the single outlier of a series of per-scan areas (sorted by rt), if any.

    The deviation of each point from the median of its ``half_width`` neighbours on
    each side (itself excluded) is scored in units of the robust spread of all the
    deviations (1.4826 x their median absolute value). The point with the largest
    score is an outlier if its score is above ``k`` and ``dominance`` times the
    second largest: one scan sticking out of a smooth profile. A noisy series has
    several large scores and keeps all its points.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.zeros(n, bool)
    if n < 2 * half_width + 3:
        return out
    padded = np.pad(y, half_width, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_width + 1)
    deviation = np.abs(y - np.median(np.delete(windows, half_width, axis=1), axis=1))
    spread = 1.4826 * np.median(deviation)
    if spread <= 0:
        return out
    score = deviation / spread
    first, second = np.argsort(score)[::-1][:2]
    if score[first] > k and score[first] > dominance * score[second]:
        out[first] = True
    return out


def local_linear_test(x, y, y_peaks, baseline, k, support_fraction=0.01):
    """Does peak ``k`` explain its region better than a straight background?

    On the points where the peak is above ``support_fraction`` of its height,
    the data minus the other fitted peaks are compared with (a) the baseline
    plus this peak and (b) a least squares line. The residual standard
    deviations are corrected for the number of parameters (5 for the
    baseline + peak, 2 for the line).

    :return: (ratio std(a) / std(b), support mask, the line on the support plus
        the other peaks, i.e. comparable with ``y``). The ratio is about 1 or
        more when the "peak" is a background elevation or noise.
    """
    peak = y_peaks[k]
    support = peak > support_fraction * peak.max() if peak.max() > 0 else np.zeros(len(x), bool)
    n = int(support.sum())
    others = y_peaks.sum(axis=0) - peak
    if n <= 5:
        return np.inf, support, np.full(n, np.nan)  # too few points to support a peak
    data = (y - others)[support]
    xs = x[support]
    ssr_peak = np.sum((data - baseline - peak[support]) ** 2)
    a, b = np.polyfit(xs - xs.mean(), data, 1)
    line = a * (xs - xs.mean()) + b
    ssr_line = np.sum((data - line) ** 2)
    if ssr_line == 0:
        return np.inf, support, line + others[support]
    ratio = float(np.sqrt(ssr_peak / (n - 5)) / np.sqrt(ssr_line / (n - 2)))
    return ratio, support, line + others[support]


def local_linear_ratio(x, y, y_peaks, baseline, k, support_fraction=0.01) -> float:
    """The ratio of :func:`local_linear_test`."""
    return local_linear_test(x, y, y_peaks, baseline, k, support_fraction)[0]


def _finish_lines(rejected, start, x, y, kept_peaks):
    """Display lines of the rejected candidates: a straight line fitted to the
    data minus the finally kept peaks, over the candidate's points, plus the kept peaks."""
    if rejected is None:
        return
    kept = np.zeros_like(y) if kept_peaks is None or len(kept_peaks) == 0 else kept_peaks.sum(axis=0)
    for r in rejected[start:]:
        support = r.pop("_support")
        if support.sum() < 2:
            r["line"] = np.full(int(support.sum()), np.nan)
            continue
        xs, data = x[support], (y - kept)[support]
        a, b = np.polyfit(xs - xs.mean(), data, 1)
        r["line"] = a * (xs - xs.mean()) + b + kept[support]


def apex_coverage(x, mu, sigma, dt, rt_min, rt_max) -> float:
    """Fraction of the scans within +-sigma of ``mu`` (inside the window) that have a point."""
    lo, hi = max(mu - sigma, rt_min), min(mu + sigma, rt_max)
    expected = int(np.floor(hi / dt) - np.ceil(lo / dt) + 1) if hi >= lo else 0
    if expected <= 0:
        return 1.0
    present = int(np.sum((x >= lo - dt / 2) & (x <= hi + dt / 2)))
    return min(1.0, present / expected)


def fit_chromato(
    rt: np.ndarray,
    area: np.ndarray,
    mu_guess: np.ndarray,
    apex_guess: np.ndarray,
    baseline_guess: float,
    dt: float,
    sigma_guess_value: float,
    window: tuple[float, float],
    x_series: np.ndarray,
    mz_series: np.ndarray,
    conf_int: float = 0.90,
    broadness: float = 2.0,
    ratio_to_linear_fit: float = 0.9,
    position_bound: float | None = None,
    convention: str = "legacy",
    legacy_numerics: bool = True,
    peak_linear_ratio: float | None = None,
    peak_min_coverage: float | None = None,
    rejected: list | None = None,
    max_iterations: int | None = 2000,
    on_abort: str = "best",
) -> ChromatoPeaks | None:
    """Fit chromatographic peaks to the per-scan areas ``area(rt)``.

    :param dt: rt sampling (s).
    :param window: (start, stop) of the rt window (s), used for the number of
        points required on each side of a peak.
    :param x_series, mz_series: position (x domain) and m/z of each point.
    :param peak_linear_ratio: if given, a peak whose :func:`local_linear_ratio`
        is above it is removed (and the fit redone), like the insignificant ones.
        None: legacy behaviour (only the global test against a line).
    :param peak_min_coverage: if given, a peak is removed if less than this
        fraction of the scans within +-1 sigma of its apex have a point (the
        apex of a real peak is measured; a "peak" fitted across a gap of the
        data is not). None: legacy behaviour.
    :param rejected: if a list is given, the peaks removed by the two tests
        above are appended to it (dicts with their parameters, the reason and
        the local line that explains the data as well, for plots).
    :param max_iterations, on_abort: iteration limit of each fit and the parameters
        kept when it is reached (see :func:`peaxart.leastsq.fit`).
    :return: the peaks, or None if the fit is rejected.
    """
    x = np.asarray(rt, dtype=float)
    y = np.asarray(area, dtype=float)
    mu_guess = np.asarray(mu_guess, dtype=float)
    len_x = len(x)
    window_start = len(rejected) if rejected is not None else 0
    n_peaks = len(mu_guess)
    std_linear = _linear_fit_std(x, y)
    baseline_guess = max(0, baseline_guess)
    sigma_guess = np.zeros(n_peaks) + sigma_guess_value
    sigma_guess_min = sigma_guess * 0.2
    sigma_guess_max = sigma_guess * 2
    h_guess = np.asarray(apex_guess, dtype=float) - baseline_guess
    alpha_guess = np.full(n_peaks, 0.1)
    bound = 3 * dt if position_bound is None else position_bound
    rt_min, rt_max = window
    t_value_lod = None

    while True:
        # parameters: baseline, then (mu, sigma, height, tailing) per peak
        p0 = [max(1.0, baseline_guess)]
        lo = [max(0.0, baseline_guess / 3)]
        hi = [max(3.0, baseline_guess * 10)]
        for i in range(n_peaks):
            p0 += [mu_guess[i], sigma_guess[i], max(2.0, h_guess[i]), alpha_guess[i]]
            lo += [mu_guess[i] - bound, max(0.5, sigma_guess_min[i]), max(1.0, h_guess[i] / 10.0), 0.001]
            hi += [mu_guess[i] + bound, max(0.8, sigma_guess_max[i]), max(3.0, h_guess[i] * 10), 1.6]

        def residual(p, n_peaks=n_peaks):
            # baseline + sum of the peaks, in the legacy order
            peaks = [chromato(x, p[3 + 4 * i], p[1 + 4 * i], p[2 + 4 * i], p[4 + 4 * i], convention)
                     for i in range(n_peaks)]
            return p[0] + np.sum(peaks, axis=0) - y

        res = fit(residual, p0, lo, hi, exact=legacy_numerics, max_iterations=max_iterations, on_abort=on_abort)
        params = res.params
        baseline = params[0]
        mu = params[1::4][:n_peaks]
        sigma = params[2::4][:n_peaks]
        height = params[3::4][:n_peaks]
        tailing = params[4::4][:n_peaks]
        mu_stderr = res.stderr[1::4][:n_peaks]
        y_peaks = np.array([chromato(x, height[i], mu[i], sigma[i], tailing[i], convention) for i in range(n_peaks)])
        y_no_baseline = y_peaks.sum(axis=0)
        resid = y - (y_no_baseline + baseline)
        std_fit = np.sqrt(np.sum(resid**2) / float(len(y) - 1))
        w = 1.0 / np.clip(y_no_baseline, 1, None) ** 2
        baseline_u = np.sqrt(np.sum(w * resid**2) / np.sum(w) * len_x / (len_x - 1))
        t_baseline = student_t.ppf(conf_int, len(y) - 1) * baseline_u

        height_u = np.zeros(n_peaks)
        t_height = np.zeros(n_peaks)
        for i in range(n_peaks):
            extracted = resid + y_peaks[i]
            s, m = sigma[i], mu[i]
            in_peak = np.abs(x - m) <= broadness * s
            n_left = int(np.sum(in_peak & (x <= m)))
            n_right = int(np.sum(in_peak & (x >= m)))
            n_in = int(np.sum(in_peak))
            min_left = round(max(1, min(broadness * s, m - rt_min) / dt * 0.4)) if m - rt_min > 0 else 0
            min_right = round(max(1, min(broadness * s, rt_max - m) / dt * 0.4)) if rt_max - m > 0 else 0
            summed = np.sum(y_peaks[i][in_peak])
            if summed > 0 and n_left >= min_left and n_right >= min_right:
                height_u[i] = math.sqrt(
                    np.sum((y_peaks[i][in_peak] - extracted[in_peak]) ** 2 * y_peaks[i][in_peak]) / summed
                )
            else:
                height_u[i] = height[i]
            t_height[i] = student_t.ppf(conf_int, max(n_in - 1, 1)) * height_u[i]

        if std_fit / std_linear > ratio_to_linear_fit:
            _finish_lines(rejected, window_start, x, y, None)
            return None
        significant = (height - t_height - t_baseline >= 0) & (height - 2 * t_baseline >= 0)
        tests = [local_linear_test(x, y, y_peaks, baseline, i) for i in range(n_peaks)]
        linear_ratio = np.array([t[0] for t in tests])
        coverage = np.array([apex_coverage(x, mu[i], sigma[i], dt, rt_min, rt_max) for i in range(n_peaks)])
        fails_line = (linear_ratio > peak_linear_ratio) if peak_linear_ratio is not None else np.zeros(n_peaks, bool)
        fails_coverage = (coverage < peak_min_coverage) if peak_min_coverage is not None else np.zeros(n_peaks, bool)
        if rejected is not None:
            for i in np.flatnonzero(fails_line | fails_coverage):
                _, support, _ = tests[i]
                w = y_peaks[i]
                reasons = (["linear background fits as well"] if fails_line[i] else []) + (
                    ["apex not covered by the data"] if fails_coverage[i] else [])
                rejected.append({
                    "rt": float(mu[i]), "mz": float(np.sum(w * mz_series) / np.sum(w)) if np.sum(w) > 0 else np.nan,
                    "height": float(height[i]), "rt_sigma": float(sigma[i]), "rt_tailing": float(tailing[i]),
                    "baseline": float(baseline), "linear_ratio": float(linear_ratio[i]),
                    "coverage": float(coverage[i]), "reason": " and ".join(reasons),
                    "line_rt": x[support].copy(), "_support": support,
                })
        significant &= ~fails_line & ~fails_coverage
        if significant.all():
            break
        if not significant.any():
            _finish_lines(rejected, window_start, x, y, None)
            return None
        h_guess = height[significant]
        mu_guess = mu[significant]
        sigma_guess = sigma[significant]
        alpha_guess = tailing[significant]
        sigma_guess_min = sigma_guess_min[significant]
        sigma_guess_max = sigma_guess_max[significant]
        baseline_guess = res.raw[0]
        n_peaks = int(significant.sum())

    _finish_lines(rejected, window_start, x, y, y_peaks)
    min_pts_for_lod = np.min(sigma_guess * 3 / dt)
    t_value_lod = student_t.ppf(conf_int, min_pts_for_lod - 1)
    lod_h = np.maximum(1.0, 2.0 * t_value_lod * baseline_u) * np.ones(n_peaks)
    area_fit = np.array([chromato_area(height[i], sigma[i], tailing[i]) for i in range(n_peaks)])
    lod = np.array([chromato_area(lod_h[i], sigma[i], tailing[i]) for i in range(n_peaks)])
    sums = y_peaks.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xc = (y_peaks @ x_series) / sums
        mc = (y_peaks @ mz_series) / sums
        norm = sums * (len_x - 1) / len_x
        xc_u = np.sqrt(np.sum((x_series[None, :] - xc[:, None]) ** 2 * y_peaks, axis=1) / norm)
        mc_u = np.sqrt(np.sum((mz_series[None, :] - mc[:, None]) ** 2 * y_peaks, axis=1) / norm)
    return ChromatoPeaks(
        rt=mu,
        rt_u=mu_stderr,
        height=height,
        height_u=height_u,
        area=area_fit,
        sigma=sigma,
        tailing=tailing,
        lod=lod,
        sn=height / lod_h,
        x=xc,
        x_u=xc_u,
        mz=mc,
        mz_u=mc_u,
        linear_ratio=linear_ratio,
        coverage=coverage,
        n_points=len_x,
        baseline=float(baseline),
    )
