"""Peak functions: pseudo-Voigt (mass domain), and Pap-Papai or exponentially modified Gaussian (retention time)."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from scipy.special import erfc, erfcx, gamma

C_PV = np.sqrt(2 * np.log(2))
SQRT_2PI = np.sqrt(2 * np.pi)

LEGACY = "legacy"
STANDARD = "standard"
SQRT_PI_2 = np.sqrt(np.pi / 2)


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


def emg(x, height, mu, sigma, ratio):
    """Exponentially modified Gaussian: a Gaussian (``mu``, ``sigma``) convolved with an exponential decay of
    time constant ``tau = ratio * sigma``, the tail independent of the body.

    ``height`` is the height of the Gaussian (the apex is lower and later when the tail grows): the area is
    ``height * sigma * sqrt(2 pi)`` whatever the tail. Gaussian below a ratio of 0.01. Computed with ``erfcx``
    (no overflow far in the tail).
    """
    x = np.asarray(x, dtype=float)
    d = x - mu
    if ratio <= 0.01:
        return height * np.exp(-0.5 * (d / sigma) ** 2)
    return height * SQRT_PI_2 / ratio * _emg_core(d, sigma, ratio)[0]


def _emg_core(d, sigma, ratio):
    """``G E`` = exp(-d^2 / 2 sigma^2) erfcx(u) (stable), ``G`` and ``u`` = (1 / ratio - d / sigma) / sqrt(2)."""
    u = (1.0 / ratio - d / sigma) / np.sqrt(2.0)
    g = np.exp(-0.5 * (d / sigma) ** 2)
    ge = np.empty_like(d)
    pos = u >= 0
    ge[pos] = g[pos] * erfcx(u[pos])
    # far in the tail (u < 0): exp(1 / (2 ratio^2) - d / (ratio sigma)) erfc(u), no overflow
    neg = ~pos
    ge[neg] = np.exp(0.5 / ratio**2 - d[neg] / (ratio * sigma)) * erfc(u[neg])
    return ge, g, u


def emg_jac(x, height, mu, sigma, ratio):
    """:func:`emg` and its derivatives: (value, d/dmu, d/dsigma, d/dheight, d/dratio)."""
    x = np.asarray(x, dtype=float)
    d = x - mu
    if ratio <= 0.01:
        f = height * np.exp(-0.5 * (d / sigma) ** 2)
        return f, f * d / sigma**2, f * d * d / sigma**3, f / height if height else np.exp(-0.5 * (d / sigma) ** 2), \
            np.zeros_like(x)
    ge, g, u = _emg_core(d, sigma, ratio)
    k = height * SQRT_PI_2 / ratio
    f = k * ge
    dge_du = 2.0 * u * ge - 2.0 * g / np.sqrt(np.pi)  # d(G E)/du at fixed G
    d_mu = k * (ge * d / sigma**2 + dge_du / (sigma * np.sqrt(2.0)))
    d_sigma = k * (ge * d * d / sigma**3 + dge_du * d / (sigma**2 * np.sqrt(2.0)))
    d_height = SQRT_PI_2 / ratio * ge
    d_ratio = -f / ratio - k * dge_du / (ratio**2 * np.sqrt(2.0))
    return f, d_mu, d_sigma, d_height, d_ratio


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


def chromato_jac(x, height, mu, sigma, tailing, convention: str = STANDARD):
    """:func:`chromato` and its derivatives: (value, d/dmu, d/dsigma, d/dheight, d/dtailing).

    The Gaussian of a tailing below 0.01 does not depend on the tailing (derivative 0).
    """
    x = np.asarray(x, dtype=float)
    f = chromato(x, height, mu, sigma, tailing, convention)
    d = x - mu
    if tailing <= 0.01:
        k2 = 2.0 * sigma**2 if convention == LEGACY else sigma**2  # f = H exp(-d^2 / (2 k2))
        shape = np.exp(-0.5 * d * d / k2)
        return f, f * d / k2, f * d * d / (k2 * sigma), shape, np.zeros_like(x)
    a = tailing
    a2 = a * a
    c = 4.0 / a2 - 1.0
    k = 2.0 * a / (sigma * (4.0 - a2))
    v = k * d
    ok = v > -0.99
    g = np.zeros_like(x)  # dE/dv / -1, with E = c (ln(1 + v) - v) the exponent
    g[ok] = c * v[ok] / (1.0 + v[ok])
    log_term = np.zeros_like(x)
    log_term[ok] = np.log1p(v[ok]) - v[ok]
    d_mu = f * g * k
    d_sigma = f * g * v / sigma
    d_height = np.where(ok, np.exp(c * log_term), 0.0)
    dv_da = v * (4.0 + a2) / (a * (4.0 - a2))
    d_tailing = f * (-8.0 / a**3 * log_term - g * dv_da)
    return f, d_mu, d_sigma, d_height, d_tailing


def chromato_area(height, sigma, tailing):
    """Area of :func:`chromato` (Gaussian approximation below a tailing of 0.2)."""
    if tailing > 0.2:
        w = 4.0 / tailing**2
        return gamma(w) * height * tailing * sigma / (2.0 * np.exp((w - 1) * (np.log(w - 1) - 1)))
    return height * sigma * SQRT_2PI


# --- chromatographic peak shapes, by name ---------------------------------------------------------


class ShapeName(StrEnum):
    """The chromatographic peak shapes available (:data:`SHAPES`); equal to their string, which may be given
    instead."""

    PAP_PAPAI = "pap_papai"
    EMG = "emg"


class PeakShape:
    """A chromatographic peak function of four parameters: ``height`` (amplitude), ``location`` (s),
    ``sigma`` (width, s) and ``tail`` (asymmetry; its meaning depends on the shape).

    The fits (:func:`peaxart.rtfit.fit_chromato`, :class:`peaxart.Fit`, :class:`peaxart.FixedShapes`) use
    a shape through this interface, chosen by name (:class:`ShapeName`, :data:`SHAPES`, :func:`peak_shape`;
    ``ExtractionConfig.rt_peak_shape``): a new function is a new subclass, its name in :class:`ShapeName` and
    the class in :data:`SHAPES`. The apex (rt and height of the
    maximum) is what the fits report as ``rt`` / ``height``; ``location`` / ``height`` are the parameters
    (``shape_rt`` / ``shape_height``), equal to the apex for a shape whose maximum is at its location.
    """

    name: ShapeName
    #: meaning of the tail parameter
    tail: str = ""
    #: start value and bounds of the tail parameter in the fits
    tail_start: float = 0.1
    tail_bounds: tuple[float, float] = (0.001, 1.6)

    def value(self, x, height, location, sigma, tail) -> np.ndarray:
        raise NotImplementedError

    def jac(self, x, height, location, sigma, tail):
        """The value and its derivatives: (value, d/dlocation, d/dsigma, d/dheight, d/dtail)."""
        raise NotImplementedError

    def area(self, height, sigma, tail) -> float:
        raise NotImplementedError

    def apex(self, height, location, sigma, tail) -> tuple[float, float]:
        """rt and height of the maximum."""
        return float(location), float(height)

    def location_offset(self, sigma, tail) -> float:
        """apex - location (s): how far before its apex the location of a peak is. The fits start a peak
        detected at its apex this much earlier, and let it move this much further back."""
        return 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PapPapai(PeakShape):
    """Pap and Papai (:func:`chromato`): the tail parameter ``a`` in ]0, 2[ sets both the asymmetry near the
    apex and the decay of the tail; maximum at the location. ``convention``: of its Gaussian limit."""

    name = ShapeName.PAP_PAPAI
    tail = "tailing a in ]0, 2[ (Gaussian when a -> 0)"
    tail_bounds = (0.001, 1.6)

    def __init__(self, convention: str = STANDARD):
        self.convention = convention

    def value(self, x, height, location, sigma, tail):
        return chromato(x, height, location, sigma, tail, self.convention)

    def jac(self, x, height, location, sigma, tail):
        return chromato_jac(x, height, location, sigma, tail, self.convention)

    def area(self, height, sigma, tail):
        return chromato_area(height, sigma, tail)

    def __repr__(self) -> str:
        return f"PapPapai({self.convention!r})"


class ExpModGaussian(PeakShape):
    """Exponentially modified Gaussian (:func:`emg`): a Gaussian body convolved with an exponential tail of
    time constant ``tail * sigma``, the tail independent of the body. Its maximum is later and lower than the
    location and height of the Gaussian (:meth:`apex`); its area does not depend on the tail."""

    name = ShapeName.EMG
    tail = "tau / sigma (Gaussian below 0.01)"
    tail_bounds = (0.001, 3.0)

    def __init__(self, convention: str = STANDARD):
        self.convention = convention  # the Gaussian is the standard one (no legacy EMG)

    def value(self, x, height, location, sigma, tail):
        return emg(x, height, location, sigma, tail)

    def jac(self, x, height, location, sigma, tail):
        return emg_jac(x, height, location, sigma, tail)

    def area(self, height, sigma, tail):
        return float(height * sigma * SQRT_2PI)

    def apex(self, height, location, sigma, tail):
        if tail <= 0.01:
            return float(location), float(height)
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(lambda x: -float(emg(np.array([x]), height, location, sigma, tail)[0]),
                              bounds=(location, location + (tail + 3.0) * sigma), method="bounded",
                              options={"xatol": 1e-6 * sigma})
        return float(res.x), float(-res.fun)

    def location_offset(self, sigma, tail) -> float:
        return self.apex(1.0, 0.0, sigma, tail)[0]


SHAPES: dict[ShapeName, type[PeakShape]] = {s.name: s for s in (PapPapai, ExpModGaussian)}
"""The chromatographic peak shapes, by name."""


def peak_shape(name: ShapeName | str | PeakShape = ShapeName.PAP_PAPAI, convention: str = STANDARD) -> PeakShape:
    """The shape of this name (:data:`SHAPES`), or the given one."""
    if isinstance(name, PeakShape):
        return name
    try:
        return SHAPES[name](convention)
    except KeyError:
        raise ValueError(f"Unknown peak shape {name!r}, one of {[str(s) for s in ShapeName]}") from None

