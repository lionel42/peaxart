"""Integration of the peaks of a chromatogram: one signal versus rt (e.g. a unit mass chromatogram).

Several methods are available (:data:`METHODS`). They share the interface of
:class:`Method` but not their outputs: every peak table has the columns
:data:`COMMON_COLUMNS`, each method adds its own (see their ``columns``).

- :class:`Trapezoid`: area above a straight baseline between two bounds
  (half widths from the apex, the prominence bases and the valleys between peaks);
- :class:`Fit`: chromatographic peaks (Pap and Papai) fitted with a constant
  baseline, the rt stage of :func:`peaxart.extract_domain` (coeluting peaks are
  separated, with uncertainties, LOD and S/N);
- :class:`FixedShapes`: peaks of known positions and shapes (e.g. fitted on a
  clean ion of each compound), only their heights and a linear baseline are
  fitted (linear least squares, heights >= 0): the deconvolution of coeluting
  compounds on an ion they share.

Usage::

    peaks = peaxart.integrate(t, y, method="trapezoid")
    peaks = peaxart.Fit().integrate(t, y, rt=[1823.2, 1826.1])
    shapes = peaxart.Fit().integrate(t, y_clean_ion)
    peaks = peaxart.FixedShapes.from_peaks(shapes).integrate(t, y_shared_ion)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.signal import peak_prominences

from peaxart.config import ExtractionConfig
from peaxart.detect import _window, peak_detect
from peaxart.rtfit import fit_series
from peaxart.shapes import chromato, chromato_area

COMMON_COLUMNS = ["rt", "height", "area"]
"""Columns of the peak table of every method: retention time (s), height above the
baseline (signal units) and area (signal units x s)."""


def _dt(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))


def noise_std(y) -> float:
    """Standard deviation of the noise of ``y``, from its second differences (robust:
    1.4826 x their median absolute value, / sqrt(6)); the peaks barely contribute."""
    d2 = np.diff(np.asarray(y, dtype=float), 2)
    return float(1.4826 * np.median(np.abs(d2)) / np.sqrt(6)) if len(d2) else 0.0


def slope_threshold(noise: float, pw: float, dt: float, k: float) -> float:
    """``k`` standard deviations of the slope fitted over the window of :func:`peak_detect` on pure noise."""
    n = _window(pw, dt)
    return k * noise / (dt * np.sqrt(n * (n * n - 1) / 12.0))


def detect_peaks(t, y, rt_sigma: float, k: float = 5.0, noise: float | None = None) -> np.ndarray:
    """Peaks of a chromatogram: slope method (:func:`peaxart.detect.peak_detect`) with the
    window of the extraction (``max(3, 3 rt_sigma)``) and a threshold of ``k`` times the
    noise of the slope.

    :return: (n, 2) array of (rt, apex value).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = _dt(t)
    pw = max(3, 3.0 * rt_sigma)
    noise = noise_std(y) if noise is None else noise
    return peak_detect(t, y, pw, slope_threshold(noise, pw, dt, k), dt)


def _walk_down(y, start, step, climb: int, listed, max_steps: int, depth: float = 0.5) -> np.ndarray:
    """End of each peak in direction ``step`` (+-1): the first minimum of ``y`` that is deep
    (below ``depth`` times the height of the peak above ``min(y)``) or followed by one of the
    ``listed`` peaks (within ``climb`` points); shallower minima before an unlisted maximum
    (a peak that was not detected) are passed over, up to ``max_steps`` points from the start.

    The walk starts at the highest point within ``climb`` points of ``start`` (the apex of
    a smoothed signal can be a few points away from the one of the raw signal).
    """
    listed = np.asarray(listed)
    floor = float(np.min(y))
    out = []
    for i in start:
        lo, hi = max(0, i - climb), min(len(y), i + climb + 1)
        i = lo + int(np.argmax(y[lo:hi]))
        limit = i + step * max_steps
        level = floor + depth * (y[i] - floor)
        while True:
            while 0 <= i + step < len(y) and y[i + step] <= y[i]:
                i += step
            if not 0 <= i + step < len(y) or y[i] < level:
                break
            j = i  # the next maximum
            while 0 <= j + step < len(y) and y[j + step] >= y[j]:
                j += step
            if np.any(np.abs(listed - j) <= climb) or (j - limit) * step > 0:
                break
            i = j
        out.append(i)
    return np.array(out, dtype=int)


def _crossing(t, y, apex: int, end: int, level: float) -> float:
    """Time where ``y`` first falls below ``level`` from ``apex`` towards ``end`` (interpolated), else ``t[end]``."""
    step = 1 if end >= apex else -1
    for i in range(apex, end, step):
        if y[i + step] < level <= y[i]:
            return float(np.interp(level, [y[i + step], y[i]], [t[i + step], t[i]]))
    return float(t[end])


class Method:
    """A method of integration of the peaks of a chromatogram."""

    name: str = ""
    columns: list[str] = COMMON_COLUMNS
    """Columns of the peak table returned by :meth:`integrate`."""

    def integrate(self, t, y, rt=None) -> pd.DataFrame:
        """Peaks of the chromatogram ``y(t)`` (t sorted, regularly sampled).

        :param rt: expected retention times of the peaks (s), e.g. from another
            detection. None: the peaks are detected.
        :return: one row per peak, columns :attr:`columns`, sorted by rt.
        """
        raise NotImplementedError

    def bands(self, t, y, peaks: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """(lower, upper) curves of each peak at times ``t``, NaN outside the peak:
        the area between them is the area of the peak (for plots)."""
        raise NotImplementedError

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in self.columns})


class Trapezoid(Method):
    """Area above a straight baseline between the bounds of each peak (trapezoidal rule).

    The apex is the local maximum nearest to the expected rt (or to a detected
    peak) within ``rt_sigma / 2``. On each side, the peak ends at the first minimum
    of the signal smoothed over ``rt_sigma / 2`` that is deep (below half the peak)
    or before another of the peaks (a valley or the start of the baseline; the
    maxima of peaks not in the list are passed over, up to 6 rt sigma), and at most
    ``n_hwhm`` half widths from the apex (half maximum between the apex and the lower end).
    The baseline joins the signal at the two bounds.

    With ``split_at_valley``, peaks whose bounds overlap form a cluster: one
    baseline joins the signal at the outer bounds of the cluster, and the peaks
    are separated by vertical lines at the lowest point between their apexes
    (drop lines). Otherwise each peak has its own baseline between its own
    bounds (from a valley, the signal under the valley is lost).

    A shoulder (a coeluting peak without its own maximum) cannot be integrated: its area is NaN.
    """

    name = "trapezoid"
    columns = COMMON_COLUMNS + ["start", "stop", "baseline_start", "baseline_stop", "fwhm", "prominence"]

    def __init__(
        self,
        n_hwhm: float = 3.0,
        split_at_valley: bool = True,
        rt_sigma: float | None = None,
        k: float = 5.0,
        config: ExtractionConfig | None = None,
    ):
        """
        :param rt_sigma: expected rt sigma (s), default ``config.rt_sigma_at`` the centre of the chromatogram.
        :param k: threshold of the peak detection (noise standard deviations of the slope).
        """
        self.n_hwhm = n_hwhm
        self.split_at_valley = split_at_valley
        self.rt_sigma = rt_sigma
        self.k = k
        self.config = config or ExtractionConfig()

    def integrate(self, t, y, rt=None) -> pd.DataFrame:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(t) < 3:
            return self._empty()
        dt = _dt(t)
        rt_sigma = self.rt_sigma or self.config.rt_sigma_at((t[0] + t[-1]) / 2)
        guesses = detect_peaks(t, y, rt_sigma, self.k)[:, 0] if rt is None else np.atleast_1d(rt).astype(float)
        is_max = np.r_[False, (y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]), False]
        half = max(1, int(round(rt_sigma / 2 / dt)))
        apexes = []
        for g in guesses:
            i = int(np.clip(np.searchsorted(t, g), 0, len(t) - 1))
            maxima = np.flatnonzero(is_max[max(0, i - half) : i + half + 1]) + max(0, i - half)
            apexes.append(maxima[np.argmin(np.abs(t[maxima] - g))] if len(maxima) else i)
        apex = np.unique(apexes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a shoulder is not a local maximum: prominence 0
            prom = peak_prominences(y, apex)[0]
        if rt is None:  # detected at the edge of the series, without a maximum
            keep = prom > 0
            apex, prom = apex[keep], prom[keep]
        if len(apex) == 0:
            return self._empty()
        ok = prom > 0
        # each side ends at the first minimum of the smoothed signal (valley or baseline);
        # the half maximum is between the apex and the lower of the two ends
        width = max(3, int(round(rt_sigma / 2 / dt)) | 1)
        max_steps = int(round(6 * rt_sigma / dt))
        smooth = np.convolve(np.pad(y, width // 2, mode="edge"), np.ones(width) / width, mode="valid")
        left_end = np.minimum(_walk_down(smooth, apex, -1, width // 2, apex[ok], max_steps), apex)
        right_end = np.maximum(_walk_down(smooth, apex, 1, width // 2, apex[ok], max_steps), apex)
        half_max = (y[apex] + np.minimum(y[left_end], y[right_end])) / 2
        left_hm = np.array([_crossing(t, y, a, e, h) for a, e, h in zip(apex, left_end, half_max, strict=True)])
        right_hm = np.array([_crossing(t, y, a, e, h) for a, e, h in zip(apex, right_end, half_max, strict=True)])
        rt_apex = t[apex]
        start = np.maximum(rt_apex - self.n_hwhm * (rt_apex - left_hm), t[left_end])
        stop = np.minimum(rt_apex + self.n_hwhm * (right_hm - rt_apex), t[right_end])
        # baseline of each peak: the line through the signal at (base_start, base_stop)
        base_start, base_stop = start.copy(), stop.copy()
        if self.split_at_valley:
            peaks = np.flatnonzero(ok)
            clusters = [[peaks[0]]] if len(peaks) else []
            for i in peaks[1:]:
                if start[i] <= max(stop[j] for j in clusters[-1]):
                    clusters[-1].append(i)
                else:
                    clusters.append([i])
            for c in clusters:
                c_start, c_stop = min(start[c]), max(stop[c])
                base_start[c], base_stop[c] = c_start, c_stop
                for i, j in zip(c[:-1], c[1:], strict=True):
                    valley = t[apex[i] + int(np.argmin(y[apex[i] : apex[j] + 1]))]
                    stop[i], start[j] = valley, valley
                start[c[0]], stop[c[-1]] = c_start, c_stop
        rows = []
        for i, a in enumerate(apex):
            yb = np.interp([base_start[i], base_stop[i]], t, y)

            def baseline(x, i=i, yb=yb):
                return np.interp(x, [base_start[i], base_stop[i]], yb) if base_stop[i] > base_start[i] else yb[0]

            mask = (t >= start[i]) & (t <= stop[i])
            if not ok[i] or mask.sum() < 2:
                area = np.nan
            else:
                # the bounds are between samples: add them, interpolated
                ts = np.r_[start[i], t[mask], stop[i]]
                area = float(np.trapezoid(np.interp(ts, t, y) - baseline(ts), ts))
            # apex: vertex of the parabola through the 3 highest points
            r = t[a]
            if is_max[a]:
                den = y[a - 1] - 2 * y[a] + y[a + 1]
                if den < 0:
                    r = t[a] + 0.5 * (y[a - 1] - y[a + 1]) / den * dt
            rows.append(
                {
                    "rt": r,
                    "height": y[a] - baseline(r),
                    "area": area,
                    "start": start[i],
                    "stop": stop[i],
                    "baseline_start": baseline(start[i]),
                    "baseline_stop": baseline(stop[i]),
                    "fwhm": right_hm[i] - left_hm[i],
                    "prominence": prom[i],
                }
            )
        return pd.DataFrame(rows, columns=self.columns)

    def bands(self, t, y, peaks):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        out = []
        for p in peaks.itertuples():
            mask = (t >= p.start) & (t <= p.stop)
            lower = np.full(len(t), np.nan)
            upper = np.full(len(t), np.nan)
            lower[mask] = np.interp(t[mask], [p.start, p.stop], [p.baseline_start, p.baseline_stop])
            upper[mask] = y[mask]
            out.append((lower, upper))
        return out


class Fit(Method):
    """Chromatographic peaks (Pap and Papai) and a constant baseline fitted to the chromatogram.

    The rt stage of :func:`peaxart.extract_domain` (:func:`peaxart.rtfit.fit_series`):
    windows of ``config.rt_window``, per-peak tests, and up to ``config.rt_extra_peaks``
    more peaks where the data show a shoulder (coelution). The peaks are detected
    with :func:`detect_peaks`, or start at the given rts.

    The default configuration uses the standard Gaussian convention (true areas for
    weakly tailing peaks; the legacy one is for the per-scan areas of the extraction)
    and the fast numerics.
    """

    name = "fit"
    columns = COMMON_COLUMNS + [
        "rt_u",
        "height_u",
        "area_u",
        "rt_sigma",
        "rt_tailing",
        "lod",
        "sn",
        "linear_ratio",
        "baseline",
    ]

    def __init__(self, config: ExtractionConfig | None = None, k: float = 5.0):
        """:param k: threshold of the peak detection (noise standard deviations of the slope)."""
        self.config = config or ExtractionConfig(convention="standard", legacy_numerics=False)
        self.k = k

    def integrate(self, t, y, rt=None) -> pd.DataFrame:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(t) < 3:
            return self._empty()
        dt = _dt(t)
        if rt is None:
            noise = noise_std(y)

            def detect(tg, yg, pw, dt):
                return peak_detect(tg, yg, pw, slope_threshold(noise, pw, dt, self.k), dt)
        else:
            guesses = np.sort(np.atleast_1d(rt).astype(float))

            def detect(tg, yg, pw, dt):
                g = guesses[(guesses >= tg[0]) & (guesses <= tg[-1])]
                return np.column_stack([g, np.interp(g, tg, yg)])

        fitted = fit_series(t, y, dt, self.config, detect=detect, baseline_guess=lambda w: float(np.quantile(w, 0.05)))
        rows = []
        for f, k in fitted:
            rel_u = f.height_u[k] / f.height[k] if f.height[k] else np.nan
            rows.append(
                {
                    "rt": f.rt[k],
                    "height": f.height[k],
                    "area": f.area[k],
                    "rt_u": f.rt_u[k],
                    "height_u": f.height_u[k],
                    "area_u": f.area[k] * rel_u,
                    "rt_sigma": f.sigma[k],
                    "rt_tailing": f.tailing[k],
                    "lod": f.lod[k],
                    "sn": f.sn[k],
                    "linear_ratio": f.linear_ratio[k],
                    "baseline": f.baseline,
                }
            )
        if not rows:
            return self._empty()
        return pd.DataFrame(rows, columns=self.columns).sort_values("rt", ignore_index=True)

    def bands(self, t, y, peaks):
        t = np.asarray(t, dtype=float)
        out = []
        for p in peaks.itertuples():
            peak = chromato(t, p.height, p.rt, p.rt_sigma, p.rt_tailing, self.config.convention)
            visible = peak > 1e-3 * p.height
            out.append((np.where(visible, p.baseline, np.nan), np.where(visible, p.baseline + peak, np.nan)))
        return out


class FixedShapes(Method):
    """Peaks of given rts and shapes; only their heights (>= 0) and a baseline are fitted.

    Linear least squares on the points where any of the peaks is above ``support``
    times its height. The shapes are typically fitted (:class:`Fit`) on an ion of
    each compound free of the others, and this method splits an ion they share
    between them. The uncertainties are from the covariance of the linear fit.
    """

    name = "fixed_shapes"
    columns = COMMON_COLUMNS + [
        "height_u",
        "area_u",
        "rt_sigma",
        "rt_tailing",
        "baseline",
        "baseline_slope",
        "residual_std",
    ]

    def __init__(
        self, rt, rt_sigma, rt_tailing, baseline: str = "linear", support: float = 0.01, convention: str = "standard"
    ):
        """
        :param rt, rt_sigma, rt_tailing: position (s), width (s) and tailing of each peak
            (see :func:`peaxart.shapes.chromato`).
        :param baseline: ``"linear"``, ``"constant"`` or ``"none"``.
        """
        self.rt = np.atleast_1d(rt).astype(float)
        self.rt_sigma = np.broadcast_to(np.atleast_1d(rt_sigma).astype(float), self.rt.shape)
        self.rt_tailing = np.broadcast_to(np.atleast_1d(rt_tailing).astype(float), self.rt.shape)
        if baseline not in ("linear", "constant", "none"):
            raise ValueError(f"baseline: {baseline!r}")
        self.baseline = baseline
        self.support = support
        self.convention = convention

    @classmethod
    def from_peaks(cls, peaks: pd.DataFrame, **kwargs) -> FixedShapes:
        """The shapes of a peak table with ``rt``, ``rt_sigma`` and ``rt_tailing`` (e.g. of :class:`Fit`)."""
        return cls(peaks["rt"].to_numpy(), peaks["rt_sigma"].to_numpy(), peaks["rt_tailing"].to_numpy(), **kwargs)

    def profiles(self, t) -> np.ndarray:
        """(n_peaks, len(t)) peaks of height 1."""
        t = np.asarray(t, dtype=float)
        return np.array(
            [
                chromato(t, 1.0, r, s, a, self.convention)
                for r, s, a in zip(self.rt, self.rt_sigma, self.rt_tailing, strict=True)
            ]
        ).reshape(len(self.rt), len(t))

    def integrate(self, t, y, rt=None) -> pd.DataFrame:
        if rt is not None:
            raise ValueError("FixedShapes: the rts are given to the constructor")
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(self.rt)
        support = (self.profiles(t) > self.support).any(axis=0) if n else np.zeros(len(t), bool)
        ts, ys = t[support], y[support]
        t_ref = float(ts.mean()) if len(ts) else 0.0
        columns = list(self.profiles(ts))
        if self.baseline != "none":
            columns.append(np.ones(len(ts)))
        if self.baseline == "linear":
            columns.append(ts - t_ref)
        n_params = len(columns)
        if n == 0 or len(ts) <= n_params:
            return self._empty()
        a = np.column_stack(columns)
        lower = np.r_[np.zeros(n), np.full(n_params - n, -np.inf)]
        res = lsq_linear(a, ys, bounds=(lower, np.inf))
        p = res.x
        resid = ys - a @ p
        residual_std = float(np.sqrt(resid @ resid / (len(ts) - n_params)))
        cov = residual_std**2 * np.linalg.pinv(a.T @ a)
        p_u = np.sqrt(np.clip(np.diag(cov), 0, None))
        b0 = p[n] if self.baseline != "none" else 0.0
        b1 = p[n + 1] if self.baseline == "linear" else 0.0
        unit_area = np.array([chromato_area(1.0, s, a) for s, a in zip(self.rt_sigma, self.rt_tailing, strict=True)])
        out = pd.DataFrame(
            {
                "rt": self.rt,
                "height": p[:n],
                "area": p[:n] * unit_area,
                "height_u": p_u[:n],
                "area_u": p_u[:n] * unit_area,
                "rt_sigma": self.rt_sigma,
                "rt_tailing": self.rt_tailing,
                "baseline": b0 + b1 * (self.rt - t_ref),
                "baseline_slope": b1,
                "residual_std": residual_std,
            },
            columns=self.columns,
        )
        return out.sort_values("rt", ignore_index=True)

    def bands(self, t, y, peaks):
        t = np.asarray(t, dtype=float)
        out = []
        for p in peaks.itertuples():
            peak = chromato(t, p.height, p.rt, p.rt_sigma, p.rt_tailing, self.convention)
            visible = peak > 1e-3 * max(p.height, 1e-300)
            base = p.baseline + p.baseline_slope * (t - p.rt)
            out.append((np.where(visible, base, np.nan), np.where(visible, base + peak, np.nan)))
        return out


METHODS: dict[str, type[Method]] = {m.name: m for m in (Trapezoid, Fit, FixedShapes)}
"""The methods by name."""


def integrate(t, y, method: str | Method = "fit", rt=None, **params) -> pd.DataFrame:
    """Integrate the peaks of the chromatogram ``y(t)``.

    :param method: a :class:`Method`, or the name of one (:data:`METHODS`) made with ``params``.
    :param rt: expected retention times (s), None to detect the peaks (see :meth:`Method.integrate`).
    """
    if isinstance(method, str):
        method = METHODS[method](**params)
    elif params:
        raise TypeError("params are only used with a method name")
    return method.integrate(t, y, rt)
