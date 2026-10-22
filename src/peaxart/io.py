"""Output: peak file for the alpinac identification."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

ALPINAC_COLUMNS = [
    "RT", "mass", "mass_u_ppm", "mass_cal_u_ppm", "area", "area_u", "peak_width",
    "peak_alpha", "compound_bin", "LOD", "SN", "Ionisation", "Adduct", "Spectrum_id",
]


def mass_cal_u_ppm(
    mz: np.ndarray,
    cal_masses: np.ndarray | None = None,
    cal_u_ppm: np.ndarray | float = 0.0,
    min_u_mz: float = 0.001,
    min_u_below: float = 65.0,
) -> np.ndarray:
    """Mass calibration uncertainty of each peak (legacy rule).

    Interpolated in the uncertainties of the calibrant masses; below
    ``min(min_u_below, first calibrant)`` at least ``min_u_mz`` (m/z).
    """
    mz = np.asarray(mz, dtype=float)
    if cal_masses is None:
        u = np.broadcast_to(np.asarray(cal_u_ppm, dtype=float), mz.shape).astype(float)
        first = min_u_below
    else:
        u = np.interp(mz, cal_masses, cal_u_ppm)
        first = min(min_u_below, float(np.min(cal_masses)))
    return np.where(mz < first, np.maximum(min_u_mz / mz * 1e6, u), u)


def to_alpinac(
    peaks: pd.DataFrame,
    path: str | PathLike | None = None,
    mass_cal_u: np.ndarray | float | Callable = 0.0,
    mass_u_ppm_min: float = 0.0,
    compound: pd.Series | np.ndarray | None = None,
    ionisation: str = "EI",
    adduct: str = "",
    spectrum: str = "0",
    metadata: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Peak table in the format read by the alpinac identification.

    :param peaks: output of :func:`peaxart.extract` / :func:`peaxart.extract_domain`.
    :param path: if given, the table is written there (tab separated, with a
        ``#`` header of metadata).
    :param mass_cal_u: mass calibration uncertainty, ppm: a number, an array
        (one per peak) or a function of m/z (see :func:`mass_cal_u_ppm`).
    :param compound: compound index of each peak (default: column ``compound``, else 0).
    :return: the table written (columns :data:`ALPINAC_COLUMNS`).

    ``peak_width`` is the m/z sigma of the mass peaks and ``peak_alpha`` their
    Lorentzian fraction (the legacy extraction wrote the rt tailing there).
    """
    mz = peaks["mz"].to_numpy(dtype=float)
    if callable(mass_cal_u):
        cal = np.asarray(mass_cal_u(mz), dtype=float)
    else:
        cal = np.broadcast_to(np.asarray(mass_cal_u, dtype=float), mz.shape)
    if compound is None:
        compound = peaks["compound"] if "compound" in peaks else np.zeros(len(peaks), dtype=int)
    out = pd.DataFrame(
        {
            "RT": peaks["rt"].to_numpy(),
            "mass": mz,
            "mass_u_ppm": np.maximum(peaks["mz_u_ppm"].to_numpy(), mass_u_ppm_min),
            "mass_cal_u_ppm": cal,
            "area": peaks["area"].to_numpy(),
            "area_u": peaks["area_u"].to_numpy(),
            "peak_width": peaks["mz_sigma"].to_numpy(),
            "peak_alpha": peaks["mz_alpha"].to_numpy(),
            "compound_bin": np.asarray(compound, dtype=int),
            "LOD": peaks["lod"].to_numpy(),
            "SN": peaks["sn"].to_numpy(),
            "Ionisation": ionisation,
            "Adduct": adduct if adduct else "None",
            "Spectrum_id": spectrum,
        },
        columns=ALPINAC_COLUMNS,
    )
    if path is not None:
        path = Path(path)
        formats = {"RT": "{:.2f}", "mass": "{:.7f}", "mass_u_ppm": "{:.2f}", "mass_cal_u_ppm": "{:.2f}",
                   "area": "{:.2f}", "area_u": "{:.2f}", "peak_width": "{:.5f}", "peak_alpha": "{:.3f}",
                   "LOD": "{:.2f}", "SN": "{:.2f}"}
        with path.open("w") as f:
            f.write("# peaxart peak extraction\n")
            f.write(f"# date: {datetime.now().isoformat(timespec='seconds')}\n")
            for k, v in (metadata or {}).items():
                f.write(f"# {k}: {v}\n")
            f.write("\t".join(ALPINAC_COLUMNS) + "\n")
            for row in out.itertuples(index=False):
                f.write("\t".join(formats[c].format(v) if c in formats else str(v)
                                  for c, v in zip(ALPINAC_COLUMNS, row)) + "\n")
    return out


def load_domain(path: str | PathLike) -> dict:
    """Load an example domain (``.npz`` of ``tests/data/domains`` or ``data/examples``).

    :return: dict with ``name``, ``intensity`` (float32), ``rt``, ``x`` (tof
        index), ``mz``, ``sigma`` (function of x, tof index units), ``alpha``,
        ``dt`` and ``meta`` (description, empty for the recorded test domains),
        the arguments of :func:`peaxart.extract_domain`.
    """
    import json

    path = Path(path)
    z = np.load(path)
    intercept, slope = float(z["sigma_tofidx_b"]), float(z["sigma_tofidx_slope"])
    return {
        "name": path.stem,
        "intensity": z["intensity"].astype(np.float32),
        "rt": z["rt"],
        "x": z["tofidx"],
        "mz": z["mz"],
        "sigma": lambda x: intercept + slope * x,
        "alpha": float(z["alpha"]),
        "dt": float(z["s_per_bin"]),
        "meta": json.loads(str(z["meta"])) if "meta" in z else {},
    }
