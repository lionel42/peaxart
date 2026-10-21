"""Peak functions: pseudo-Voigt (mass domain) and Pap-Papai (retention time)."""

from __future__ import annotations

import numpy as np
from scipy.special import gamma

C_PV = np.sqrt(2 * np.log(2))
SQRT_2PI = np.sqrt(2 * np.pi)

LEGACY = "legacy"
STANDARD = "standard"


def gaussian(x, area, mu, sigma, convention: str = STANDARD):
    """Gaussian of given area.

    ``legacy``: the legacy alpinac function ``exp(-(x-mu)^2 / (2 sigma)^2)``
    (with the normalisation of a Gaussian of standard deviation sigma).
    """
    if convention == LEGACY:
        return area / (sigma * SQRT_2PI) * np.exp(-((x - mu) ** 2) / (2 * sigma) ** 2)
    return area / (sigma * SQRT_2PI) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def pseudo_voigt(x, area, mu, sigma, alpha, convention: str = STANDARD):
    """Pseudo-Voigt: ``(1 - alpha) * Gaussian + alpha * Lorentzian``.

    The Lorentzian half width at half maximum is ``sigma * sqrt(2 ln 2)``,
    the half width at half maximum of the Gaussian of standard deviation sigma.
    """
    sigma_l = sigma * C_PV
    # Same evaluation order as the legacy code (bitwise identical results)
    return (1 - alpha) * gaussian(x, area, mu, sigma, convention) + alpha * area / np.pi * sigma_l / (
        (x - mu) ** 2 + sigma_l**2
    )


def pseudo_voigt_height(area, sigma, alpha):
    """Value at the centre (same for both conventions)."""
    return (1.0 - alpha) * area / (sigma * SQRT_2PI) + alpha * area / (np.pi * sigma * C_PV)


def multi_pseudo_voigt(x, baseline, area, mu, sigma, alpha, convention: str = STANDARD):
    """Sum of pseudo-Voigt peaks (arrays of parameters) plus a constant baseline."""
    x = np.asarray(x, dtype=float)[None, :]
    a = np.atleast_1d(area)[:, None]
    m = np.atleast_1d(mu)[:, None]
    s = np.atleast_1d(sigma)[:, None]
    al = np.atleast_1d(alpha)[:, None]
    return baseline + pseudo_voigt(x, a, m, s, al, convention).sum(axis=0)


def chromato(x, height, mu, sigma, tailing, convention: str = STANDARD):
    """Chromatographic peak of Pap and Papai (J. Chromatogr. A 2001).

    ``H exp((4/a^2 - 1) (ln(1 + v) - v))``, ``v = 2 a (x - mu) / (sigma (4 - a^2))``,
    with the tailing ``a`` in ]0, 2[ (Gaussian of standard deviation sigma when a -> 0).
    """
    x = np.asarray(x, dtype=float)
    if tailing <= 0.01:
        if convention == LEGACY:
            # legacy: Gaussian with the (2 sigma)^2 exponent, computed through
            # its area as the legacy code (same rounding)
            area = height * sigma * np.sqrt(2 * np.pi)
            return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma) ** 2)
        return height * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    a2 = tailing * tailing
    v = 2.0 * tailing * (x - mu) / (sigma * (4.0 - a2))
    out = np.zeros_like(x)
    ok = v > -0.99
    out[ok] = height * np.exp((4.0 / a2 - 1.0) * (np.log(1.0 + v[ok]) - v[ok]))
    return out


def chromato_area(height, sigma, tailing):
    """Area of :func:`chromato` (Gaussian approximation below a tailing of 0.2)."""
    if tailing > 0.2:
        w = 4.0 / tailing**2
        return gamma(w) * height * tailing * sigma / (2.0 * np.exp((w - 1) * (np.log(w - 1) - 1)))
    return height * sigma * SQRT_2PI
