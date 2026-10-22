"""Parameters of the extraction (defaults of the legacy alpinac)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ExtractionConfig:
    """Parameters of the peak extraction.

    Intensities are in the units of the input (legacy: V). Per-scan areas are
    in intensity x x-units (legacy: V x tof index), rt areas additionally x s.
    """

    # --- shape convention -------------------------------------------------
    convention: str = "legacy"
    """``"legacy"``: Gaussian with ``exp(-(x-mu)^2/(2 sigma)^2)`` as the legacy code
    (sigma models fitted with the legacy code assume it); ``"standard"``: true Gaussian."""

    legacy_numerics: bool = True
    """Reproduce the legacy fits bit for bit (lmfit-like finite differences,
    scalar evaluations). False: analytic Jacobian and vectorised evaluations,
    several times faster; results differ by rounding only, except for
    ill-conditioned fits of weak peaks."""

    max_iterations: int | None = 100
    """Maximal number of Levenberg-Marquardt iterations of a fit (the fit then
    keeps the best parameters found). Fits of spectra with many weak peaks
    can otherwise take thousands of iterations. The legacy code (lmfit) used 2000."""

    # --- mass domain (per scan) ------------------------------------------
    mz_slope_threshold: float = 1.25
    """Slope threshold of the peak detection in a spectrum (intensity per x unit), legacy PTtof."""
    mz_peak_over_median: float = 1.3
    """A detected mass peak is kept if its apex is above this times the noise median."""
    mz_median_proportion: float = 1.0 / 3.0
    """The noise median is the median of this proportion of the smallest values of the spectrum."""
    mz_noise_k: float | None = 3.0
    """A detected mass peak must also have its apex at least this many noise standard
    deviations above the noise median (spectra of strong peaks otherwise have
    dozens of noise "peaks", which make the fits slow). None: legacy rule only."""
    mz_height_threshold: float = 2.6
    """After the fit, a mass peak is kept if its fitted height is above this times the noise median."""
    mz_position_bound: float = 2.0
    """The fitted position stays within +- this many x-spacings of the detected one."""

    # --- grouping by m/z -------------------------------------------------
    mz_group_max_diff: float = 0.02
    """Maximal distance (m/z) of a per-scan peak to the median of its group."""
    mz_hist_width: float = 10.0
    """Peak width (x units, legacy PWh_tofidx) used to detect the histogram centres."""
    mz_hist_threshold_x: float = 5.0 / 3.0
    """Slope threshold of the histogram centre detection, in x units (legacy PTh_tofidx),
    converted to m/z at the domain centre."""

    # --- retention time domain -------------------------------------------
    rt_sigma: float | None = None
    """Expected rt sigma (s). If None, the legacy model
    ``rt_sigma_slope * exp(rt_sigma_exp_k * rt)`` (2.5 s below 1000 s) is used."""
    rt_sigma_slope: float = 0.05452
    rt_sigma_exp_k: float = 0.00193
    rt_sigma_scale: float = 1.0
    rt_window: float = 120.0
    """Maximal duration of a fitted rt window (s); longer groups are split."""
    rt_window_edge: float = 2.5
    """Overlap added on each side of split windows (s)."""
    rt_position_bound: float | None = None
    """The fitted rt stays within +- this (s) of the detected one (default 3 scans)."""
    rt_peak_broadness: float = 2.0
    """Points within +- this many sigma are used for the peak uncertainty (legacy rt_peak_U)."""
    min_points: int = 4
    """Minimal number of per-scan peaks for a chromatographic fit (legacy fit_chromato_nb_params)."""
    conf_int: float = 0.90
    """Confidence level of the Student t tests."""
    ratio_to_linear_fit: float = 0.9
    """The chromatographic fit is rejected if its residual std is above this times the one of a linear fit."""

    peak_linear_ratio: float | None = 0.9
    """Local test of each chromatographic peak against a straight background:
    a peak is removed if the residual std with the peak, over the points it
    covers, is not below this times the residual std of a line (both corrected
    for their number of parameters). None: only the legacy global test.
    Not in the legacy code (it fitted peaks to background elevations)."""
    peak_min_coverage: float | None = 0.5
    """A peak is removed if fewer than this fraction of the scans within +-1
    sigma of its apex have a per-scan mass peak (a "peak" interpolated across a
    gap of the data). None: no test. Not in the legacy code."""

    lod: float = 0.0
    """Domains whose maximal intensity is at or below this are skipped."""

    def rt_sigma_at(self, rt: float) -> float:
        if self.rt_sigma is not None:
            sigma = self.rt_sigma
        elif rt <= 1000:
            sigma = 2.5
        else:
            sigma = self.rt_sigma_slope * np.exp(self.rt_sigma_exp_k * rt)
        return float(sigma * self.rt_sigma_scale)

    @classmethod
    def legacy(cls, **changes) -> ExtractionConfig:
        """The behaviour of the legacy alpinac extraction (no per-peak tests)."""
        return cls(**{"peak_linear_ratio": None, "peak_min_coverage": None, "max_iterations": 2000,
                      "mz_noise_k": None, **changes})

    def replace(self, **changes) -> ExtractionConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)
