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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from peaxart.config import ExtractionConfig
from peaxart.detect import histogram_centres, peak_detect
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


def legacy_detect(t, y, pw, dt):
    """Peaks of the per-scan areas (legacy): slope method with the threshold ``40 / pw``."""
    return peak_detect(t, y, pw, 40.0 / pw, dt)


def legacy_baseline_guess(y) -> float:
    """Start value of the baseline (legacy): lowest centre of a histogram of the values."""
    base = histogram_centres(np.sort(y), 15, 5, 0.01, 1)
    return float(base.min()) if len(base) else float(y.min())


def fit_series(
    t: np.ndarray,
    y: np.ndarray,
    dt: float,
    config: ExtractionConfig,
    x_series: np.ndarray | None = None,
    mz_series: np.ndarray | None = None,
    detect: Callable | None = None,
    baseline_guess: Callable | None = None,
    rejected: list | None = None,
) -> list[tuple[ChromatoPeaks, int]]:
    """Chromatographic fit of a series ``y(t)`` (sorted by t), window by window.

    The series is split into windows of ``config.rt_window`` s; in each, the
    peaks are detected on a regular grid, fitted with :func:`fit_chromato`, and
    up to ``config.rt_extra_peaks`` more peaks are tried (coelution).

    :param x_series, mz_series: position and m/z of each point (zeros if None).
    :param detect: ``detect(t_grid, y_grid, pw, dt) -> (n, 2)`` array of (rt, apex)
        start values, with ``pw`` the expected peak width (s). Default :func:`legacy_detect`.
    :param baseline_guess: ``baseline_guess(y_window) -> float``. Default :func:`legacy_baseline_guess`.
    :param rejected: if a list, the rejected candidates are appended (see :func:`fit_chromato`).
    :return: (fit of a window, index of the peak) for each peak, each peak in one window only.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y)
    x_series = np.zeros(len(t)) if x_series is None else np.asarray(x_series, dtype=float)
    mz_series = np.zeros(len(t)) if mz_series is None else np.asarray(mz_series, dtype=float)
    detect = detect or legacy_detect
    baseline_guess = baseline_guess or legacy_baseline_guess
    out = []
    if len(t) == 0:
        return out
    rt_bins = t / dt
    for w_start, w_stop in split_into_windows(rt_bins[0], rt_bins[-1], dt, np.inf,
                                              config.rt_window, config.rt_window_edge):
        # as the legacy code: the window end is excluded
        sel = np.flatnonzero((rt_bins >= w_start) & (rt_bins < w_stop))
        if len(sel) <= config.min_points + 1:
            continue
        w_rt, w_area, w_x, w_mz = t[sel], y[sel], x_series[sel], mz_series[sel]
        rt_sigma = config.rt_sigma_at((w_start + w_stop) / 2 * dt)
        pw = max(3, 3.0 * rt_sigma)
        grid = np.arange(int((w_rt[0] - pw) / dt), int((w_rt[-1] + pw) / dt))
        detected = detect(grid * dt, np.interp(grid, w_rt / dt, w_area), pw, dt)
        # not more parameters than points: drop the smallest peaks
        while len(detected) > 0 and len(sel) < 1 + len(detected) * config.min_points:
            detected = detected[detected[:, 1] > detected[:, 1].min()]
        if len(detected) == 0:
            continue
        in_window = (detected[:, 0] >= w_start * dt) & (detected[:, 0] < w_stop * dt)
        if not in_window.any():
            continue
        base_guess = baseline_guess(w_area)
        window_rejected: list[dict] = []

        def run_fit(mu_guess, apex_guess, base_guess, rejected_list, position_bound=config.rt_position_bound,
                    w_rt=w_rt, w_area=w_area, w_x=w_x, w_mz=w_mz, rt_sigma=rt_sigma, w_start=w_start, w_stop=w_stop):
            return fit_chromato(
                w_rt, w_area, mu_guess, apex_guess, base_guess, dt, rt_sigma,
                (w_start * dt, w_stop * dt), w_x, w_mz, config.conf_int, config.rt_peak_broadness,
                config.ratio_to_linear_fit, position_bound, config.convention,
                config.legacy_numerics, config.peak_linear_ratio, config.peak_min_coverage,
                rejected=rejected_list, on_abort=config.on_abort,
            )

        fitted = run_fit(detected[:, 0], detected[:, 1], base_guess, window_rejected)
        # Try one more peak where the residual is largest (coeluting peaks closer
        # than the detection window); keep it only if the fit is clearly better.
        for _ in range(config.rt_extra_peaks):
            if fitted is None:
                break
            trial = _try_extra_peak(fitted, w_rt, w_area, dt, rt_sigma, config, run_fit)
            if trial is None:
                break
            fitted, window_rejected = trial
        if rejected is not None:
            rejected.extend(r for r in window_rejected if w_start * dt <= r["rt"] < w_stop * dt)
        if fitted is None:
            continue
        out.extend((fitted, k) for k in range(len(fitted)) if w_start * dt <= fitted.rt[k] < w_stop * dt)
    return out


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

    :return: (the new fit, its rejected candidates) if it keeps at least as many peaks (the
        refit may drop a weak one elsewhere), its residual std (corrected for the number of
        parameters) is below ``rt_extra_peak_ratio`` times the
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
        # >=: the refit may also drop a weak peak elsewhere in the window
        if f is not None and len(f) >= max(2, len(fitted)):
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
