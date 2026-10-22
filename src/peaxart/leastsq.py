"""Bounded least squares, as lmfit's ``leastsq`` method (used by the legacy code).

MINPACK Levenberg-Marquardt (:func:`scipy.optimize.leastsq`) on internal
parameters; bounds are handled with the Minuit-style transformations of lmfit
(``x = min + (sin(u) + 1) (max - min) / 2`` for two bounds, square-root
transforms for one bound). Same tolerances as lmfit 1.3. The standard errors
are computed like lmfit: covariance of the internal parameters times the
reduced chi-square, scaled by the gradients of the transformations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import leastsq

_TINY = 1e-15


@dataclass
class FitResult:
    params: np.ndarray
    stderr: np.ndarray
    """Standard errors (nan when not available, 0 for fixed parameters)."""
    residual: np.ndarray
    success: bool
    nfev: int
    """Number of evaluations of the residual."""
    aborted: bool
    """The fit was stopped by ``max_iterations``."""
    raw: list
    """The parameters with lmfit's types: a value clipped to a bound is that
    bound (e.g. a float32 scalar). Needed to restart fits as the legacy code."""


def _to_internal(x, lo, hi):
    """Internal starting values.

    The scalars are used with their own types (as lmfit does): a float32
    starting value is transformed in float32 precision (NumPy 2 promotion
    rules), which matters to reproduce the legacy fits of weak peaks.
    """
    u = np.empty(len(x))
    for i, (v, a, b) in enumerate(zip(x, lo, hi)):
        v = min(max(v, a), b)  # lmfit clips the starting values to the bounds
        if np.isinf(a) and np.isinf(b):
            u[i] = v
        elif np.isinf(b):
            u[i] = np.sqrt((v - a + 1.0) ** 2 - 1)
        elif np.isinf(a):
            u[i] = np.sqrt((b - v + 1.0) ** 2 - 1)
        else:
            u[i] = np.arcsin(2 * (v - a) / (b - a) - 1)
        if abs(u[i]) < _TINY:
            u[i] = 0.0
    return u


def _from_internal(u, lo, hi):
    """External values. Element by element with numpy scalars, as lmfit: the
    vectorised (SIMD) np.sin can differ in the last bit from the scalar one,
    and long fits of weak peaks amplify such differences."""
    x = np.empty(len(u))
    for i in range(len(u)):
        ui, a, b = np.float64(u[i]), lo[i], hi[i]
        if np.isfinite(a) and np.isfinite(b):
            x[i] = a + (np.sin(ui) + 1) * (b - a) / 2.0
        elif np.isfinite(a):
            x[i] = a - 1.0 + np.sqrt(ui * ui + 1)
        elif np.isfinite(b):
            x[i] = b + 1 - np.sqrt(ui * ui + 1)
        else:
            x[i] = ui
    return x


def _gradient_scale(u, lo, hi):
    s = np.ones_like(u)
    both = np.isfinite(lo) & np.isfinite(hi)
    only_lo = np.isfinite(lo) & ~np.isfinite(hi)
    only_hi = ~np.isfinite(lo) & np.isfinite(hi)
    s[both] = np.cos(u[both]) * (hi[both] - lo[both]) / 2.0
    s[only_lo] = u[only_lo] / np.sqrt(u[only_lo] ** 2 + 1)
    s[only_hi] = -u[only_hi] / np.sqrt(u[only_hi] ** 2 + 1)
    return s


def _from_internal_vec(u, lo, hi):
    """Vectorised :func:`_from_internal` (faster, may differ in the last bit)."""
    x = np.empty_like(u)
    both = np.isfinite(lo) & np.isfinite(hi)
    only_lo = np.isfinite(lo) & ~np.isfinite(hi)
    only_hi = ~np.isfinite(lo) & np.isfinite(hi)
    free = ~(both | only_lo | only_hi)
    x[both] = lo[both] + (np.sin(u[both]) + 1) * (hi[both] - lo[both]) / 2.0
    x[only_lo] = lo[only_lo] - 1.0 + np.sqrt(u[only_lo] ** 2 + 1)
    x[only_hi] = hi[only_hi] + 1 - np.sqrt(u[only_hi] ** 2 + 1)
    x[free] = u[free]
    return x


def fit(residual, x0, lo=None, hi=None, vary=None, jac=None, exact: bool = True,
        max_iterations: int | None = 2000) -> FitResult:
    """Minimise ``sum(residual(params)**2)`` with bounds.

    :param residual: function params (full vector) -> residual array.
    :param x0: starting values.
    :param lo, hi: bounds (``-inf`` / ``inf`` for none).
    :param vary: boolean mask of the parameters to fit (default all).
    :param jac: optional function params -> (n_residuals, n_params) Jacobian
        (external parameters). Without it, finite differences as lmfit.
    :param exact: reproduce lmfit bit for bit (scalar transformations);
        False is faster and differs only by rounding.
    :param max_iterations: the fit stops after ``max_iterations * (n + 1)``
        evaluations of the residual (n fitted parameters; with finite
        differences one iteration costs n + 1 evaluations). 2000 is the lmfit
        default and then, as lmfit, the last evaluated parameters are returned;
        otherwise the best parameters found so far. None: no limit.
    """
    x0_raw = list(x0)
    n = len(x0_raw)
    lo_raw = [-np.inf] * n if lo is None else list(lo)
    hi_raw = [np.inf] * n if hi is None else list(hi)
    # lmfit swaps inverted bounds
    for i in range(n):
        if lo_raw[i] > hi_raw[i]:
            lo_raw[i], hi_raw[i] = hi_raw[i], lo_raw[i]
    x0 = np.asarray(x0_raw, dtype=float)
    lo = np.asarray(lo_raw, dtype=float)
    hi = np.asarray(hi_raw, dtype=float)
    vary = np.ones(n, dtype=bool) if vary is None else np.asarray(vary, dtype=bool)
    fixed = np.clip(x0, lo, hi)
    iv = np.flatnonzero(vary)
    lo_v, hi_v = lo[iv], hi[iv]

    lo_iv = [lo_raw[i] for i in iv]
    hi_iv = [hi_raw[i] for i in iv]
    # Bounds given as float32 scalars are used as such (lmfit: max - min is
    # then computed in float32): needed to reproduce the legacy fits exactly.
    raw = [i for i, (a, b) in enumerate(zip(lo_iv, hi_iv)) if isinstance(a, np.float32) or isinstance(b, np.float32)]

    to_external = _from_internal if exact else _from_internal_vec

    def full(u):
        p = fixed.copy()
        p[iv] = to_external(u, lo_v, hi_v)
        for k in raw:
            a, b = lo_iv[k], hi_iv[k]
            p[iv[k]] = a + (np.sin(u[k]) + 1) * (b - a) / 2.0
        # lmfit clips the values to the bounds (rounding of the transformation)
        return np.minimum(np.maximum(p, lo), hi)

    lmfit_like = max_iterations == 2000
    max_nfev = (max_iterations if max_iterations is not None else 10**9) * (len(iv) + 1)
    state = {"nfev": 0, "last": None, "best": None, "best_ssr": np.inf}

    class _Abort(Exception):
        pass

    def fun(u):
        # lmfit aborts the fit after max_nfev evaluations and returns the last values
        state["nfev"] += 1
        state["last"] = np.array(u, copy=True)
        if state["nfev"] > max_nfev:
            raise _Abort
        r = np.asarray(residual(full(u)), dtype=float)
        if not lmfit_like:
            ssr = float(np.dot(r, r))
            if ssr < state["best_ssr"]:
                state["best_ssr"], state["best"] = ssr, state["last"]
        return r

    dfun = None
    if jac is not None:
        def dfun(u):
            return np.asarray(jac(full(u)), dtype=float)[:, iv] * _gradient_scale(u, lo_v, hi_v)[None, :]

    u0 = _to_internal([x0_raw[i] for i in iv], [lo_raw[i] for i in iv], [hi_raw[i] for i in iv])
    try:
        u, cov, info, _, ier = leastsq(
            fun, u0, Dfun=dfun, full_output=True, col_deriv=False, ftol=1.5e-8, xtol=1.5e-8,
            gtol=0.0, maxfev=2 * max_nfev, epsfcn=1e-10, factor=100, diag=None,
        )
    except _Abort:
        u = state["last"] if lmfit_like or state["best"] is None else state["best"]
        cov, ier = None, -1
    params = full(u)
    res = np.asarray(residual(params), dtype=float)
    raw = list(fixed)
    for k, i in enumerate(iv):
        unclipped = params[i]
        if unclipped >= hi_raw[i]:
            raw[i] = hi_raw[i]
        elif unclipped <= lo_raw[i]:
            raw[i] = lo_raw[i]
        else:
            raw[i] = np.float64(unclipped)
    stderr = np.zeros(n)
    stderr[iv] = np.nan
    ndata = len(res)
    if cov is not None and ndata > len(iv):
        redchi = float(np.sum(res**2)) / (ndata - len(iv))
        scale = _gradient_scale(u, lo_v, hi_v)
        covar = cov * redchi * np.outer(scale, scale)
        stderr[iv] = np.sqrt(np.abs(np.diag(covar)))
    return FitResult(params=params, stderr=stderr, residual=res, success=ier in (1, 2, 3, 4), raw=raw,
                     nfev=state["nfev"], aborted=ier == -1)
