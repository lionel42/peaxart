"""Grouping of co-eluting peaks into compounds (legacy ``comp_bin``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from peaxart.config import ExtractionConfig
from peaxart.detect import histogram_centres
from peaxart.rtfit import split_into_windows


def group_compounds(
    peaks: pd.DataFrame,
    dt: float,
    config: ExtractionConfig | None = None,
    min_peaks: int = 2,
    peak_threshold: float = 0.7,
) -> pd.Series:
    """Compound index of each peak (-1: not part of a compound).

    In each rt window, a histogram of the peak rts (bin width rt_sigma / 6,
    with rt_sigma twice the expected chromatographic sigma) gives the elution
    times of the compounds; each peak is assigned to the nearest one within
    4 x 3 rt_sigma. Compounds need ``min_peaks`` peaks.

    Unlike the legacy code, compound indices of different rt windows never collide.
    """
    config = config or ExtractionConfig()
    out = pd.Series(-1, index=peaks.index, dtype=np.int64)
    if len(peaks) == 0:
        return out
    rt = peaks["rt"].to_numpy()
    rt_idx = np.round(rt / dt).astype(int)
    next_id = 0
    for w_start, w_stop in split_into_windows(
        rt_idx.min(), rt_idx.max(), dt, np.inf, config.rt_window, config.rt_window_edge
    ):
        in_window = (rt_idx >= w_start) & (rt_idx <= w_stop)
        if in_window.sum() < min_peaks:
            continue
        rt_sigma = 2.0 * config.rt_sigma_at((w_start + w_stop) / 2 * dt)
        pw = max(3, 3.0 * rt_sigma)
        centres = histogram_centres(rt[in_window], dt * rt_sigma / 6.0, rt_sigma, peak_threshold, dt)
        if len(centres) == 0:
            continue
        idx = np.flatnonzero(in_window)
        values = rt[idx]
        right = np.clip(np.searchsorted(centres, values), 0, len(centres) - 1)
        left = np.clip(right - 1, 0, len(centres) - 1)
        d_left, d_right = np.abs(centres[left] - values), np.abs(centres[right] - values)
        nearest = np.where(d_left < d_right, left, right)
        nearest = np.where(np.minimum(d_left, d_right) <= 4.0 * dt * pw, nearest, -1)
        used = [c for c in range(len(centres)) if (nearest == c).sum() >= min_peaks]
        for c in used:
            out.iloc[idx[nearest == c]] = next_id + c
        if used:
            next_id += max(used) + 1
    return out
