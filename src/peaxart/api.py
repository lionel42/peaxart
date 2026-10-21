"""Extraction over a full rt / m/z array: one domain per unit mass."""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from peaxart.config import ExtractionConfig
from peaxart.domain import PEAK_COLUMNS, MzAxis, extract_domain

logger = logging.getLogger(__name__)


def _x_of_mz(mz_axis: MzAxis, i_scan: int, x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """x positions of given m/z values in a scan (inverse of the m/z axis)."""
    mz_row = mz_axis.at(i_scan, x)
    order = np.argsort(mz_row)
    return np.interp(np.sqrt(targets), np.sqrt(mz_row[order]), x[order])


def domain_slices(
    rt: np.ndarray,
    x: np.ndarray,
    mz,
    masses: Sequence[float],
    mz_range: float = 0.5,
    rt_range: tuple[float, float] | None = None,
) -> list[tuple[float, slice, slice]]:
    """(mass, scan slice, point slice) of each domain.

    The points of a domain are those between floor(x(mass - mz_range)) and
    ceil(x(mass + mz_range)), with x(m) from the calibration at the centre of
    the rt range (as the legacy code).
    """
    rt = np.asarray(rt, dtype=float)
    lo, hi = (rt[0], rt[-1]) if rt_range is None else rt_range
    scans = np.flatnonzero((rt >= lo) & (rt <= hi))
    if len(scans) == 0:
        return []
    s_scan = slice(int(scans[0]), int(scans[-1]) + 1)
    mz_axis = MzAxis(mz, x, rt)
    i_mid = int(np.argmin(np.abs(rt - (lo + hi) / 2)))
    out = []
    for m in masses:
        x_lo, x_hi = _x_of_mz(mz_axis, i_mid, x, np.array([m - mz_range, m + mz_range]))
        i_lo = int(np.searchsorted(x, math.floor(x_lo), side="left"))
        i_hi = int(np.searchsorted(x, math.ceil(x_hi), side="right"))
        if i_hi - i_lo < 2:
            continue
        out.append((float(m), s_scan, slice(i_lo, i_hi)))
    return out


def _run_domain(args):
    mass, intensity, rt, mz, sigma, alpha, x, config, dt, mz_range = args
    df = extract_domain(intensity, rt, mz, sigma, alpha, x=x, config=config, dt=dt)
    if len(df):
        # each peak belongs to the domain of its nearest unit mass (no duplicates at the edges)
        df = df[np.abs(df["mz"] - mass) <= mz_range].copy()
    df.insert(0, "mass", mass)
    return df


_TASKS: list = []


def _run_task(i: int) -> pd.DataFrame:
    return _run_domain(_TASKS[i])


def extract(
    intensity: np.ndarray,
    rt: np.ndarray,
    mz: np.ndarray | Callable,
    sigma: float | Callable,
    alpha: float,
    x: np.ndarray | None = None,
    masses: Sequence[float] | None = None,
    mz_range: float = 0.5,
    rt_range: tuple[float, float] | None = None,
    config: ExtractionConfig | None = None,
    dt: float | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Extract the peaks of a full data array, one domain per unit mass.

    :param intensity: (n_scans, n_points) signal (keep float32 for TOF data).
    :param rt: (n_scans,) retention times, s.
    :param mz: m/z of the points: (n_points,), (n_scans, n_points) or a
        function ``mz(rt, x)``.
    :param sigma: mass peak width (standard deviation) in x units, number or function of x.
    :param alpha: Lorentzian fraction of the mass peaks.
    :param x: (n_points,) axis of the spectra (default point index).
    :param masses: unit masses to extract (default: every integer m/z in the data).
    :param mz_range: half width of the domains, m/z.
    :param rt_range: (start, stop) s, default all.
    :param n_jobs: domains processed in parallel.
    :return: the peak table (see :func:`peaxart.domain.extract_domain`) with a
        ``mass`` column (unit mass of the domain), sorted by rt.
    """
    config = config or ExtractionConfig()
    intensity = np.asarray(intensity)
    rt = np.asarray(rt, dtype=float)
    x = np.arange(intensity.shape[1], dtype=float) if x is None else np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(rt))) if dt is None else dt
    if masses is None:
        mz_axis = MzAxis(mz, x, rt)
        row = mz_axis.at(len(rt) // 2, x)
        masses = range(int(math.ceil(np.nanmin(row))), int(math.floor(np.nanmax(row))) + 1)
    slices = domain_slices(rt, x, mz, masses, mz_range, rt_range)

    def sub_mz(s_scan, s_pt):
        if callable(mz):
            return mz
        mz_arr = np.asarray(mz, dtype=float)
        return mz_arr[s_pt] if mz_arr.ndim == 1 else mz_arr[s_scan, s_pt]

    tasks = [
        (m, intensity[s_scan, s_pt], rt[s_scan], sub_mz(s_scan, s_pt), sigma, alpha, x[s_pt], config, dt, mz_range)
        for m, s_scan, s_pt in slices
    ]
    if n_jobs > 1 and len(tasks) > 1 and "fork" in mp.get_all_start_methods():
        # fork: the workers inherit the tasks (functions given for mz or sigma,
        # e.g. lambdas, cannot be pickled), only indices are sent.
        global _TASKS
        _TASKS = tasks
        try:
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("fork")) as ex:
                frames = list(ex.map(_run_task, range(len(tasks)),
                                     chunksize=max(1, len(tasks) // (4 * n_jobs))))
        finally:
            _TASKS = []
    else:
        frames = [_run_domain(t) for t in tasks]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["mass"] + PEAK_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["rt", "mz"], ignore_index=True)
