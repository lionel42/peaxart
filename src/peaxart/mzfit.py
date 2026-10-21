"""Mass domain: detection and pseudo-Voigt fit of the peaks of one spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peaxart.config import ExtractionConfig
from peaxart.detect import peak_detect
from peaxart.leastsq import fit
from peaxart.shapes import C_PV, LEGACY, SQRT_2PI, pseudo_voigt, pseudo_voigt_height


@dataclass
class ScanPeaks:
    """Peaks fitted in one spectrum (arrays of the same length)."""

    mu: np.ndarray
    """Positions (x units)."""
    area: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    baseline: float

    def __len__(self) -> int:
        return len(self.mu)


EMPTY = ScanPeaks(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0.0)


def noise_median(y: np.ndarray, proportion: float) -> float:
    """Median of the smallest values of y (legacy noise level)."""
    return float(np.median(np.sort(y)[: int(len(y) * proportion)]))


def detect_mass_peaks(x, y, sigma_centre: float, dx: float, config: ExtractionConfig) -> np.ndarray:
    """Detected peaks (position, apex) of one spectrum, above the noise."""
    found = peak_detect(x, y, sigma_centre, config.mz_slope_threshold, dx)
    if len(found) > 1:
        median = noise_median(np.asarray(y), config.mz_median_proportion)
        found = found[found[:, 1] >= config.mz_peak_over_median * median]
    return found


def fit_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    mu_guess: np.ndarray,
    apex_guess: np.ndarray,
    sigma_of_x,
    alpha: float,
    dx: float,
    config: ExtractionConfig,
) -> ScanPeaks:
    """Fit pseudo-Voigt peaks (fixed shape) and a constant baseline to one spectrum.

    Legacy ``fit_pseudoVoigt`` in extraction mode: sigma and alpha are fixed,
    position (+- ``mz_position_bound`` x-spacings), area (0.1 to 10 x the
    guess) and baseline are fitted. Peaks whose fitted height is below
    ``mz_height_threshold`` times the noise median are removed and the fit is redone.
    """
    x = np.asarray(x, dtype=float)
    y_in = np.asarray(y)
    y = y_in.astype(float)
    # The noise median and the baseline start value and bounds are computed
    # in the precision of the input (float32 for TOF data) and with the same
    # scalar types as the legacy code: it changes the starting point of the
    # fit, and so the optimum of weak peaks.
    median = np.median(np.sort(y_in)[: int(len(y_in) / 3)])
    baseline_guess = max(0.0, median)
    mu_guess = np.asarray(mu_guess, dtype=float)
    sigma = np.asarray(sigma_of_x(mu_guess), dtype=float) * np.ones_like(mu_guess)
    alphas = np.full_like(mu_guess, alpha)
    area_guess = (np.asarray(apex_guess, dtype=float) - baseline_guess) / (
        (1.0 - alpha) / (sigma * SQRT_2PI) + alpha / (np.pi * sigma * C_PV)
    )
    convention = config.convention
    y_min = min(y_in)
    baseline_lo = max(0.0, y_min - 3 * y_min)

    while True:
        baseline_hi = max(3, baseline_guess * 3, median * 3)
        # parameters: baseline, then (mu, area) for each peak
        p0, lo, hi = [baseline_guess], [baseline_lo], [baseline_hi]
        for m, a in zip(mu_guess, area_guess):
            p0 += [m, max(1.0, a)]
            lo += [m - dx * config.mz_position_bound, max(1, a * 0.1)]
            hi += [m + dx * config.mz_position_bound, max(2, a * 10.0)]

        def residual(p, sigma=sigma, alphas=alphas):
            mu = p[1::2]
            area = p[2::2]
            # one 1-D evaluation per peak, summed, as the legacy code (the
            # vectorised exp of numpy can differ in the last bit with the layout)
            peaks = [pseudo_voigt(x, area[j], mu[j], sigma[j], alphas[j], convention) for j in range(len(mu))]
            return p[0] + np.sum(peaks, axis=0) - y

        if config.legacy_numerics:
            res = fit(residual, p0, lo, hi)
        else:
            fast_residual, fast_jac = _fast_model(x, y, sigma, alphas, convention)
            res = fit(fast_residual, p0, lo, hi, jac=fast_jac, exact=False)
        baseline = res.params[0]
        mu = res.params[1::2]
        area = res.params[2::2]
        height = baseline + pseudo_voigt_height(area, sigma, alphas)
        keep = height >= config.mz_height_threshold * median
        if keep.all():
            return ScanPeaks(mu, area, sigma, alphas, float(baseline))
        if not keep.any():
            return ScanPeaks(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), float(baseline))
        mu_guess, area_guess, sigma, alphas = mu[keep], area[keep], sigma[keep], alphas[keep]
        baseline_guess = res.raw[0]  # keeps a float32 bound value, as lmfit


def _fast_model(x, y, sigma, alphas, convention):
    """Vectorised residual and analytic Jacobian of the fixed-shape model.

    Parameters: baseline, then (mu, area) for each peak.
    """
    s = sigma[:, None]
    a = alphas[:, None]
    gamma = s * C_PV
    gauss_k = 2.0 * s**2 if convention == LEGACY else s**2  # exponent: -(x-mu)^2 / (2 gauss_k)

    def shapes(p):
        d = x[None, :] - p[1::2][:, None]
        g = np.exp(-(d**2) / (2 * gauss_k)) / (s * SQRT_2PI)
        den = d**2 + gamma**2
        lo = gamma / np.pi / den
        unit = (1 - a) * g + a * lo
        dunit_dmu = (1 - a) * g * d / gauss_k + a * gamma / np.pi * 2 * d / den**2
        return unit, dunit_dmu

    def residual(p):
        unit, _ = shapes(p)
        return p[0] + (p[2::2][:, None] * unit).sum(axis=0) - y

    def jac(p):
        unit, dmu = shapes(p)
        J = np.empty((len(x), len(p)))
        J[:, 0] = 1.0
        J[:, 1::2] = (p[2::2][:, None] * dmu).T
        J[:, 2::2] = unit.T
        return J

    return residual, jac
