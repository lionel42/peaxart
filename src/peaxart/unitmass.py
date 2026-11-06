"""Unit mass chromatograms: the spectra summed around each unit mass, smoothed, baseline removed.

Method of the legacy alpinac extraction pipeline (``extract_chomatograms``):

1. for each unit mass m, sum the points with ``m - d_m < m/z < m + d_m`` of each scan
   (:func:`unit_mass_sums`);
2. smooth, remove the arPLS baseline, clip at 0: :func:`peaxart.baseline.remove_baseline`
   (not specific to unit masses), with ``UnitMassConfig.baseline``.

``UnitMassConfig.legacy()`` reproduces the legacy method (zero padding of the
smoothing, see :class:`peaxart.baseline.BaselineConfig`).

The sums are computed scan by scan, so a run can be read in blocks
(:func:`unit_mass_sums` on each block, then :func:`peaxart.remove_baseline` on
the whole run); :func:`unit_mass_chromatograms` does both for data in memory.

The m/z axis can be the same for all scans (``mz`` of shape (n_points,): the
legacy method, the calibration at the end of the run) or follow the drift of
the calibration (``mz`` of shape (n_scans, n_points)); or give the tof index
bounds directly (:func:`unit_mass_sums` ``bounds``).

    sums = peaxart.unit_mass_sums(intensity, peaxart.mass_bounds(mz, masses))
    chromatograms = peaxart.remove_baseline(sums)            # (n_scans, n_masses)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from peaxart.baseline import BaselineConfig, remove_baseline


@dataclass
class UnitMassConfig:
    """Parameters of the unit mass chromatograms."""

    #: half width of the m/z window of each unit mass
    d_m: float = 0.5
    #: smoothing and baseline of the sums; None: the raw sums
    baseline: BaselineConfig | None = field(default_factory=BaselineConfig)

    @classmethod
    def legacy(cls, **baseline_changes) -> UnitMassConfig:
        """The legacy alpinac pipeline (``extract_chomatograms``): :meth:`BaselineConfig.legacy`."""
        return cls(baseline=BaselineConfig.legacy(**baseline_changes))


def mass_bounds(mz: np.ndarray, masses, d_m: float = 0.5) -> np.ndarray:
    """Point index bounds [start, stop[ of the window ``m - d_m < m/z < m + d_m`` of each mass.

    :param mz: m/z of the points, increasing: (n_points,) or (n_scans, n_points).
    :return: (n_masses, 2) or (n_scans, n_masses, 2) int.
    """
    mz = np.asarray(mz, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if mz.ndim == 1:
        start = np.searchsorted(mz, masses - d_m, side="right")  # first point > m - d_m
        stop = np.searchsorted(mz, masses + d_m, side="left")  # first point >= m + d_m
        return np.stack([start, stop], axis=-1)
    return np.stack([mass_bounds(row, masses, d_m) for row in mz])


def unit_mass_sums(intensity: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Sum of the points of each window, scan by scan.

    :param intensity: (n_scans, n_points).
    :param bounds: (n_masses, 2) point index bounds [start, stop[ (same for all scans), or
        (n_scans, n_masses, 2) (one set per scan, e.g. following the drift of the calibration);
        see :func:`mass_bounds`. Clipped to the points.
    :return: (n_scans, n_masses) float64.
    """
    intensity = np.asarray(intensity)
    n_scans, n_points = intensity.shape
    bounds = np.clip(np.asarray(bounds), 0, n_points)
    if bounds.ndim == 2:
        out = np.empty((n_scans, len(bounds)))
        for k, (start, stop) in enumerate(bounds):
            out[:, k] = intensity[:, start:stop].sum(axis=1, dtype=float)
        return out
    if bounds.shape[0] != n_scans:
        raise ValueError(f"bounds for {bounds.shape[0]} scans, intensity has {n_scans}")
    # cumulative sums along the points: each window is a difference of two columns
    cumulative = np.zeros((n_scans, n_points + 1))
    np.cumsum(intensity, axis=1, dtype=float, out=cumulative[:, 1:])
    rows = np.arange(n_scans)[:, None]
    return cumulative[rows, bounds[..., 1]] - cumulative[rows, bounds[..., 0]]


def unit_mass_chromatograms(intensity: np.ndarray, mz: np.ndarray, masses,
                            config: UnitMassConfig | None = None) -> np.ndarray:
    """Unit mass chromatograms of data in memory (see the module).

    :param intensity: (n_scans, n_points).
    :param mz: m/z of the points: (n_points,) or (n_scans, n_points).
    :param masses: unit masses.
    :return: (n_scans, n_masses) corrected chromatograms.
    """
    config = config or UnitMassConfig()
    sums = unit_mass_sums(intensity, mass_bounds(mz, masses, config.d_m))
    return sums if config.baseline is None else remove_baseline(sums, config.baseline)
