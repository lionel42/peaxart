"""Peak detection by the slope method (legacy ``peak_detect``) and histogram centres.

The slope of a linear fit over a sliding window of ``n`` points is computed
along the signal. A peak is where the slope changes from positive to
negative; it is kept if the largest slope before it, or the steepest slope
after it, is at least the threshold ``PT``. Its position is where the slope
crosses zero (linear interpolation).
"""

from __future__ import annotations

import numpy as np


def _window(pw: float, n_per_bin: float) -> int:
    """Number of points of the slope window (odd, at least 5)."""
    n = int(pw / n_per_bin / 2)
    if n % 2 == 0:
        n += 1
    return max(n, 5)


def peak_detect(x: np.ndarray, y: np.ndarray, pw: float, pt: float, n_per_bin: float) -> np.ndarray:
    """Detect peaks in ``y(x)``.

    :param x: positions (equally spaced).
    :param y: signal.
    :param pw: expected peak width (unit of x); with ``n_per_bin`` it gives
        the number of points of the slope window.
    :param pt: slope threshold (unit of y per unit of x).
    :param n_per_bin: spacing of x.
    :return: (n_peaks, 2) array of (position, value at the apex).
    """
    x = np.asarray(x, dtype=float)
    y_in = np.asarray(y)
    y = y_in.astype(float)
    n = _window(pw, n_per_bin)
    m = len(y) - n
    if m < 5:
        return np.zeros((0, 2))

    # Sums over the windows [i, i+n). Not with cumulative sums: x is often
    # large (tof index ~1e4) and the cancellation in sxy - sx*sy/n would lose precision.
    def sliding(v):
        return np.lib.stride_tricks.sliding_window_view(v, n)[:m].sum(axis=1)

    sx = sliding(x)
    # The legacy code summed float32 intensities in float32 (h5 data): keep the input precision
    sy = sliding(y_in) if y_in.dtype == np.float32 else sliding(y)
    sxy = sliding(x * y)
    # The legacy code used the denominator of the first window for all
    # windows (exact for equally spaced x).
    den = np.sum(x[:n] ** 2) - np.sum(x[:n]) ** 2 / n
    slope = (sxy - sx * sy / n) / den if den != 0 else np.zeros(m)

    table = np.zeros((m, 4))
    table[:, 0] = x[n // 2 : n // 2 + m]
    table[:, 1] = y[n // 2 : n // 2 + m]
    table[:, 2] = slope
    s0, s1 = slope[:-1], slope[1:]
    table[:-1, 3] = np.where((s0 > 0) & (s1 <= 0), 1.0, np.where((s0 < 0) & (s1 >= 0), -1.0, 0.0))

    peak_idx = np.flatnonzero(table[:, 3] > 0)
    valley_idx = np.flatnonzero(table[:, 3] < 0)
    idx_p = 0
    while len(peak_idx) > 0 and idx_p < len(peak_idx):
        idx_peak = int(peak_idx[idx_p])
        if idx_p == 0:
            idx_start = 0
            idx_stop = m - 1 if len(peak_idx) == 1 else int(peak_idx[idx_p + 1] + 1)
        else:
            idx_start = min(int(peak_idx[idx_p - 1] + 1), m - 2)
            idx_stop = m - 1 if idx_p == len(peak_idx) - 1 else int(peak_idx[idx_p + 1] + 1)
        slope_max = table[idx_start : idx_peak + 1, 2].max()
        slope_min = table[idx_peak:idx_stop, 2].min()
        if max(slope_max, abs(slope_min)) >= pt:
            if idx_peak + 1 < m:
                table[idx_peak, 0] = table[idx_peak, 0] - table[idx_peak, 2] * (
                    table[idx_peak + 1, 0] - table[idx_peak, 0]
                ) / (table[idx_peak + 1, 2] - table[idx_peak, 2])
            table[idx_peak, 1] = table[max(0, idx_peak - 1) : min(idx_peak + 1, m - 1), 1].max()
            idx_p += 1
        else:
            table[idx_peak, 3] = 0
            peak_idx = np.flatnonzero(table[:, 3] > 0)
    # A peak at the very beginning (signal decreasing from the start)
    if len(valley_idx) > 0 and len(peak_idx) > 0 and valley_idx[0] < peak_idx[0]:
        if abs(table[: int(valley_idx[0]) + 1, 2].min()) > pt:
            table[0, 3] = 1
    return table[table[:, 3] > 0, :2].copy()


def histogram_centres(values: np.ndarray, delta_bin: float, pw: float, pt: float, n_per_bin: float) -> np.ndarray:
    """Centres of the groups of ``values`` (legacy ``find_bins_np``).

    A histogram of bin width ``delta_bin`` is made over the range of the values
    (+- ``pw``), and its peaks are detected with :func:`peak_detect`.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0)
    edges = np.arange(values.min() - pw, values.max() + pw, delta_bin)
    if len(edges) < 2:
        return np.zeros(0)
    hist, edges = np.histogram(values, bins=edges)
    centres = edges[:-1] + delta_bin / 2
    found = peak_detect(centres, hist.astype(float), pw, pt, n_per_bin)
    return found[:, 0] if len(found) else np.zeros(0)
