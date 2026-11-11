import numpy as np
import pandas as pd
import pytest

import peaxart as px
from peaxart.config import ExtractionConfig
from peaxart.detect import histogram_centres, peak_detect
from peaxart.leastsq import fit
from peaxart.mzfit import fit_peak_shapes, fit_spectrum
from peaxart.rtfit import rt_outliers
from peaxart.shapes import chromato, chromato_area, chromato_jac, pseudo_voigt, pseudo_voigt_height


def test_peak_detect_finds_peaks():
    x = np.arange(200.0)
    y = 100 * np.exp(-0.5 * ((x - 60) / 4) ** 2) + 50 * np.exp(-0.5 * ((x - 140) / 4) ** 2)
    found = peak_detect(x, y, 8.0, 1.0, 1.0)
    np.testing.assert_allclose(found[:, 0], [60, 140], atol=0.5)
    assert peak_detect(x, np.zeros(200), 8.0, 1.0, 1.0).shape == (0, 2)


def test_histogram_centres():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(10.0, 0.002, 50), rng.normal(10.1, 0.002, 50)])
    centres = histogram_centres(values, 0.003, 10, 0.001, 1)
    assert np.min(np.abs(centres - 10.0)) < 0.003 and np.min(np.abs(centres - 10.1)) < 0.003


@pytest.mark.parametrize("convention", ["standard", "legacy"])
def test_pseudo_voigt_height(convention):
    assert pseudo_voigt(0.0, 3.0, 0.0, 0.5, 0.3, convention) == pytest.approx(pseudo_voigt_height(3.0, 0.5, 0.3))


def test_standard_gaussian_area():
    x = np.linspace(-20, 20, 400001)
    assert np.trapezoid(pseudo_voigt(x, 2.0, 0.0, 0.5, 0.0, "standard"), x) == pytest.approx(2.0, rel=1e-6)


@pytest.mark.parametrize("tailing", [0.005, 0.1, 0.5, 1.2])
def test_chromato_area(tailing):
    x = np.linspace(-50, 150, 400001)
    y = chromato(x, 3.0, 0.0, 2.0, tailing, "standard")
    assert y.max() == pytest.approx(3.0, rel=1e-3)
    if tailing > 0.2 or tailing <= 0.01:
        assert np.trapezoid(y, x) == pytest.approx(chromato_area(3.0, 2.0, tailing), rel=1e-3)


@pytest.mark.parametrize("convention", ["standard", "legacy"])
@pytest.mark.parametrize("tailing", [0.005, 0.3, 1.2])
def test_chromato_jac(convention, tailing):
    x = np.linspace(0, 40, 400)
    p = np.array([1e4, 15.0, 1.4, tailing])  # height, mu, sigma, tailing
    value, *jac = chromato_jac(x, *p, convention=convention)
    np.testing.assert_array_equal(value, chromato(x, *p, convention=convention))
    for derivative, i in zip(jac, [1, 2, 0, 3], strict=True):  # d/dmu, d/dsigma, d/dheight, d/dtailing
        h = 1e-6 * max(abs(p[i]), 1.0)
        up, down = p.copy(), p.copy()
        up[i] += h
        down[i] -= h
        if i == 3 and tailing <= 0.01:
            continue  # the Gaussian branch ignores the tailing
        numeric = (chromato(x, *up, convention=convention) - chromato(x, *down, convention=convention)) / (2 * h)
        np.testing.assert_allclose(derivative, numeric, atol=1e-6 * np.abs(numeric).max())


def test_max_iterations_with_jacobian():
    # with the Jacobian, an iteration costs one evaluation of the residual
    x = np.linspace(0, 10, 50)
    y = 3.0 * np.exp(-((x - 4.0) ** 2) / 2.0)

    def residual(p):
        return p[0] * np.exp(-((x - p[1]) ** 2) / 2.0) - y

    def jac(p):
        g = np.exp(-((x - p[1]) ** 2) / 2.0)
        return np.column_stack([g, p[0] * g * (x - p[1])])

    r = fit(residual, [1.0, 6.0], jac=jac, max_iterations=3)
    assert r.aborted and r.nfev == 4
    assert fit(residual, [1.0, 6.0], jac=jac).params == pytest.approx([3.0, 4.0])


def test_bounded_fit():
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 1.0
    r = fit(lambda p: p[0] * x + p[1] - y, [1.0, 0.0], [0.0, -5.0], [1.5, 5.0])
    assert r.params[0] == pytest.approx(1.5)  # at the bound
    r = fit(lambda p: p[0] * x + p[1] - y, [1.0, 0.0])
    np.testing.assert_allclose(r.params, [2.0, 1.0], rtol=1e-8)
    # inverted bounds are swapped, as lmfit
    r = fit(lambda p: p[0] * x + p[1] - y, [1.0, 0.0], [5.0, -5.0], [0.0, 5.0])
    assert r.params[0] == pytest.approx(2.0, rel=1e-6)


def test_fit_on_abort():
    # a fit stopped by max_iterations: "best" returns the lowest sum of squares
    # evaluated, "last" (lmfit) the last parameters tried, here much worse
    x = np.linspace(0, 10, 50)
    y = 3.0 * np.exp(-((x - 4.0) ** 2) / 2.0)
    ssr = []

    def residual(p):
        r = p[0] * np.exp(-((x - p[1]) ** 2) / 2.0) - y
        ssr.append(float(r @ r))
        return r

    best = fit(residual, [1.0, 6.0], max_iterations=6, on_abort="best")
    assert np.sum(best.residual**2) == pytest.approx(min(ssr[:-1]))  # the last call is the final evaluation
    last = fit(residual, [1.0, 6.0], max_iterations=6, on_abort="last")
    assert best.aborted and last.aborted
    assert np.sum(best.residual**2) < 1e-3 * np.sum(last.residual**2)
    with pytest.raises(ValueError):
        fit(residual, [1.0, 6.0], on_abort="first")


def test_fit_spectrum_recovers_peaks():
    x = np.arange(1000.0, 1080.0)
    sigma, alpha = 0.9, 0.5
    y = (
        2.0
        + pseudo_voigt(x, 500.0, 1030.3, sigma, alpha, "legacy")
        + pseudo_voigt(x, 200.0, 1050.7, sigma, alpha, "legacy")
    ).astype(np.float32)
    r = fit_spectrum(
        x, y, np.array([1030.0, 1051.0]), np.array([y[30], y[51]]), lambda m: sigma, alpha, 1.0, ExtractionConfig()
    )
    np.testing.assert_allclose(r.mu, [1030.3, 1050.7], atol=1e-3)
    np.testing.assert_allclose(r.area, [500.0, 200.0], rtol=1e-3)


def synthetic(n_scans=60, mz0=69.0, rt0=1605.0):
    """A synthetic domain: one compound at rt0 with a peak at mz0 + 0.0, in a TOF-like axis."""
    rt = 1600.0 + np.arange(n_scans) / 6.0
    x = np.arange(19760.0, 19840.0)  # around m/z 69 with this calibration
    mz_axis = ((x + 3700.0) / 2829.0) ** 2  # TOF: sqrt(m) linear in x
    x_peak = np.interp(np.sqrt(mz0), np.sqrt(mz_axis), x)
    sigma, alpha = 0.9, 0.5
    profile = chromato(rt, 1.0, rt0, 1.5, 0.3, "legacy")
    rng = np.random.default_rng(0)
    intensity = np.array(
        [1.0 + pseudo_voigt(x, 4000.0 * h, x_peak, sigma, alpha, "legacy") for h in profile]
    ) + rng.normal(0, 0.3, (n_scans, len(x)))
    return intensity.astype(np.float32), rt, x, mz_axis, sigma, alpha


def test_extract_domain_synthetic():
    intensity, rt, x, mz, sigma, alpha = synthetic()
    df = px.extract_domain(intensity, rt, mz, sigma, alpha, x=x)
    assert len(df) == 1
    p = df.iloc[0]
    assert p["mz"] == pytest.approx(69.0, abs=2e-4)
    assert p["rt"] == pytest.approx(1605.0, abs=0.1)
    assert list(df.columns) == px.PEAK_COLUMNS
    # m/z given as a calibration function
    df2 = px.extract_domain(intensity, rt, lambda r, xx: ((xx + 3700.0) / 2829.0) ** 2, sigma, alpha, x=x)
    assert df2["mz"].iloc[0] == pytest.approx(p["mz"], rel=1e-9)


def test_extra_peak_resolves_coelution():
    # two compounds 2.9 s apart (like 2- and 3-methylhexane): one peak for the rt
    # peak detection, the extra-peak trial splits it
    _, rt, x, mz, sigma, alpha = synthetic(n_scans=90)
    x_peak = np.interp(np.sqrt(69.0), np.sqrt(mz), x)
    profile = chromato(rt, 1.0, 1605.0, 1.4, 0.3, "legacy") + chromato(rt, 0.8, 1607.9, 1.4, 0.3, "legacy")
    rng = np.random.default_rng(0)
    intensity = np.array([1.0 + pseudo_voigt(x, 4000.0 * h, x_peak, sigma, alpha, "legacy") for h in profile])
    intensity = (intensity + rng.normal(0, 0.3, intensity.shape)).astype(np.float32)
    single = px.extract_domain(intensity, rt, mz, sigma, alpha, x=x, config=ExtractionConfig(rt_extra_peaks=0))
    assert len(single) == 1
    df = px.extract_domain(intensity, rt, mz, sigma, alpha, x=x)
    assert sorted(df["rt"]) == pytest.approx([1605.0, 1607.9], abs=0.05)
    assert sorted(df["rt_sigma"]) == pytest.approx([1.4, 1.4], abs=0.05)


def test_outlier_scan_does_not_hide_peak():
    # a weak peak whose area is 9x too large in one scan (as the ringing peaks of a strong
    # mass peak at its apex scan): the rt fit put a narrow peak on that scan, rejected
    _, rt, x, mz, sigma, alpha = synthetic()
    x_peak = np.interp(np.sqrt(69.0), np.sqrt(mz), x)
    profile = chromato(rt, 1.0, 1605.0, 1.5, 0.3, "legacy")
    profile[np.argmax(profile) - 1] *= 9
    rng = np.random.default_rng(0)
    intensity = np.array([1.0 + pseudo_voigt(x, 100.0 * h, x_peak, sigma, alpha, "legacy") for h in profile])
    intensity = (intensity + rng.normal(0, 0.3, intensity.shape)).astype(np.float32)
    assert (
        len(px.extract_domain(intensity, rt, mz, sigma, alpha, x=x, config=ExtractionConfig(rt_outlier_k=None)))
        == 0
    )
    df, _, scans = px.extract_domain(intensity, rt, mz, sigma, alpha, x=x, return_rejected=True, return_scans=True)
    assert len(df) == 1 and df["rt"].iloc[0] == pytest.approx(1605.0, abs=0.05)
    assert df["rt_sigma"].iloc[0] == pytest.approx(1.5, abs=0.15)
    assert scans["outlier"].sum() == 1


def test_rt_outliers():
    t = np.arange(60) / 6.0
    smooth = 1000 * np.exp(-((t - 5) ** 2) / 4) + np.random.default_rng(1).normal(0, 5, len(t))
    assert not rt_outliers(smooth, 20).any()
    spiked = smooth.copy()
    spiked[31] *= 5
    assert np.flatnonzero(rt_outliers(spiked, 20)).tolist() == [31]
    # noise with several spikes: no single outlier, all points kept
    noisy = np.abs(np.random.default_rng(2).normal(0, 1, 60)) ** 3
    assert not rt_outliers(noisy, 20).any()


def test_extract_full_array_and_export(tmp_path):
    # two unit masses side by side in one array
    i1, rt, x1, mz1, sigma, alpha = synthetic(mz0=69.0)
    x = np.arange(19700.0, 20100.0)  # m/z 68.7 to 71.6
    mz = ((x + 3700.0) / 2829.0) ** 2
    intensity = np.zeros((len(rt), len(x)), dtype=np.float32) + 1.0
    rng = np.random.default_rng(0)
    intensity += rng.normal(0, 0.3, intensity.shape).astype(np.float32)
    for m0, a in [(69.0, 4000.0), (70.0, 1500.0)]:
        xp = np.interp(np.sqrt(m0), np.sqrt(mz), x)
        prof = chromato(rt, 1.0, 1605.0, 1.5, 0.3, "legacy")
        intensity += np.array([pseudo_voigt(x, a * h, xp, sigma, alpha, "legacy") for h in prof], dtype=np.float32)
    peaks = px.extract(intensity, rt, mz, sigma, alpha, x=x, masses=[69, 70, 71])
    assert sorted(peaks["mass"]) == [69, 70]
    peaks["compound"] = px.group_compounds(peaks, dt=rt[1] - rt[0])
    assert peaks["compound"].nunique() == 1 and (peaks["compound"] >= 0).all()
    table = px.to_alpinac(peaks, tmp_path / "x.frag.txt", mass_cal_u=5.0, metadata={"sample": "test"})
    assert list(table.columns) == px.io.ALPINAC_COLUMNS
    text = (tmp_path / "x.frag.txt").read_text()
    assert "# sample: test" in text and "RT\tmass" in text
    # readable by pandas like the alpinac reader does
    back = pd.read_csv(tmp_path / "x.frag.txt", sep=r"\s+", comment="#")
    assert len(back) == 2 and set(back["compound_bin"]) == {0}


def test_mass_cal_u_rule():
    u = px.mass_cal_u_ppm(np.array([30.0, 100.0]), np.array([50.0, 200.0]), np.array([5.0, 3.0]))
    assert u[0] == pytest.approx(max(0.001 / 30.0 * 1e6, 5.0))
    assert u[1] == pytest.approx(5.0 - 2.0 * 50 / 150)


def test_fast_numerics_same_peak():
    intensity, rt, x, mz, sigma, alpha = synthetic()
    exact = px.extract_domain(intensity, rt, mz, sigma, alpha, x=x)
    fast = px.extract_domain(intensity, rt, mz, sigma, alpha, x=x, config=ExtractionConfig(legacy_numerics=False))
    assert len(fast) == len(exact) == 1
    assert fast["mz"].iloc[0] == pytest.approx(exact["mz"].iloc[0], abs=1e-5)
    assert fast["area"].iloc[0] == pytest.approx(exact["area"].iloc[0], rel=1e-3)


def test_fit_peak_shapes():
    """Free sigma and alpha: the shape of two overlapping peaks is recovered (legacy convention)."""
    x = np.arange(0.0, 60.0)
    truth = [(25.3, 800.0, 1.2, 0.4), (31.0, 300.0, 1.4, 0.7)]
    y = 5.0 + sum(pseudo_voigt(x, a, m, s, al, "legacy") for m, a, s, al in truth)
    found = np.array([[25.0, y[25]], [31.0, y[31]]])
    peaks = fit_peak_shapes(x, y, found, sigma_guess=np.array([1.0, 1.0]), alpha_guess=0.5)
    np.testing.assert_allclose(peaks.mu, [25.3, 31.0], atol=1e-4)
    np.testing.assert_allclose(peaks.sigma, [1.2, 1.4], rtol=1e-3)
    np.testing.assert_allclose(peaks.alpha, [0.4, 0.7], atol=1e-3)
    np.testing.assert_allclose(peaks.area, [800.0, 300.0], rtol=1e-3)
    assert peaks.baseline == pytest.approx(5.0, abs=0.05)


def test_emg():
    """Exponentially modified Gaussian: area independent of the tail, Gaussian limit, tail, analytic Jacobian."""
    from scipy.integrate import trapezoid

    from peaxart.shapes import emg, peak_shape

    shape = peak_shape("emg")

    x = np.linspace(-50, 500, 550001)  # the tail of ratio 10 (tau 15 s): 30 tau
    for ratio in (0.005, 0.3, 1.0, 3.0, 10.0):
        y = emg(x, 2.0, 10.0, 1.5, ratio)
        assert trapezoid(y, x) == pytest.approx(shape.area(2.0, 1.5, ratio), rel=1e-6)
        assert np.all(np.isfinite(y)) and y.min() >= 0
    np.testing.assert_allclose(emg(x, 2.0, 10.0, 1.5, 0.005), 2.0 * np.exp(-0.5 * ((x - 10) / 1.5) ** 2), atol=1e-4)
    # the tail: exp(-d / tau) far after the apex
    y = emg(np.array([40.0, 41.0]), 1.0, 10.0, 1.5, 2.0)
    assert y[1] / y[0] == pytest.approx(np.exp(-1 / 3.0), rel=1e-6)
    # the apex is later and lower than the Gaussian's
    y = emg(x, 2.0, 10.0, 1.5, 1.0)
    assert x[np.argmax(y)] > 10.0 and y.max() < 2.0
    np.testing.assert_allclose(shape.value(x, 2.0, 10.0, 1.5, 1.0), y)
    rt, height = shape.apex(2.0, 10.0, 1.5, 1.0)  # the maximum
    assert rt == pytest.approx(x[np.argmax(y)], abs=1e-3) and height == pytest.approx(y.max(), rel=1e-6)
    # Jacobian against finite differences
    xs = np.linspace(0, 40, 400)
    p = np.array([2.0, 10.0, 1.5, 1.2])  # height, mu, sigma, ratio
    f, *derivatives = shape.jac(xs, *p[[0, 1]], p[2], p[3])
    order = [1, 2, 0, 3]  # d/dmu, d/dsigma, d/dheight, d/dratio
    for k, analytic in zip(order, derivatives, strict=True):
        h = 1e-6 * max(1.0, abs(p[k]))
        up, down = p.copy(), p.copy()
        up[k] += h
        down[k] -= h
        numeric = (emg(xs, *up) - emg(xs, *down)) / (2 * h)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_peak_shapes_by_name():
    """The chromatographic peak functions, chosen by name: the fits report the apex, and the parameters
    of the function (shape_rt, shape_height) to draw it."""
    from peaxart.config import ExtractionConfig
    from peaxart.shapes import SHAPES, ExpModGaussian, PapPapai, PeakShape, peak_shape

    assert set(SHAPES) >= {"pap_papai", "emg"}
    assert isinstance(peak_shape("emg"), ExpModGaussian) and isinstance(peak_shape(), PapPapai)
    assert peak_shape("pap_papai", "legacy").convention == "legacy"
    with pytest.raises(ValueError, match="Unknown peak shape"):
        peak_shape("lorentzian")
    for shape in SHAPES.values():
        assert issubclass(shape, PeakShape) and shape.name and shape.tail
    # Pap-Papai: the apex is the location; EMG: later and lower
    assert peak_shape().apex(2.0, 10.0, 1.5, 0.5) == (10.0, 2.0)
    rt, height = peak_shape("emg").apex(2.0, 10.0, 1.5, 1.0)
    assert rt > 10.0 and height < 2.0

    t = np.arange(1700.0, 1800.0, 1 / 6)
    y = 5.0 + peak_shape("emg").value(t, 1000.0, 1740.0, 1.4, 1.2)
    fitted = px.Fit(ExtractionConfig(convention="standard", legacy_numerics=False, rt_peak_shape="emg")).integrate(
        t, y, rt=[1741.0])
    [peak] = fitted.itertuples()
    apex_rt, apex_height = peak_shape("emg").apex(1000.0, 1740.0, 1.4, 1.2)
    assert peak.rt == pytest.approx(apex_rt, abs=0.05) and peak.height == pytest.approx(apex_height, rel=0.01)
    assert peak.shape_rt == pytest.approx(1740.0, abs=0.05) and peak.shape_height == pytest.approx(1000.0, rel=0.01)
    assert peak.rt_tailing == pytest.approx(1.2, rel=0.05)
    assert peak.area == pytest.approx(1000.0 * 1.4 * np.sqrt(2 * np.pi), rel=0.01)
    # the same shapes in FixedShapes: heights refitted at the apex
    fixed = px.FixedShapes.from_peaks(fitted, shape="emg", baseline="constant").integrate(t, y)
    assert fixed["rt"].iloc[0] == pytest.approx(peak.rt)
    assert fixed["height"].iloc[0] == pytest.approx(peak.height, rel=0.01)


@pytest.mark.parametrize("name", ["constant", "linear", "exponential"])
def test_backgrounds(name):
    """The backgrounds of the chromatographic fit, by name: analytic Jacobian and slope, start values in their
    bounds, and a fit of peaks on that background recovers it and the peaks."""
    from peaxart.backgrounds import BACKGROUNDS, background
    from peaxart.rtfit import fit_chromato
    from peaxart.shapes import emg, peak_shape

    bg = background(name)
    assert isinstance(bg, BACKGROUNDS[name]) and bg.name == name
    span = (100.0, 160.0)
    x = np.linspace(*span, 601)
    true = {"constant": [500.0], "linear": [3000.0, 800.0], "exponential": [400.0, 5000.0, 15.0]}[name]
    p = np.array(true)
    # Jacobian and slope against finite differences
    for k in range(bg.n_params):
        h = 1e-6 * max(1.0, abs(p[k]))
        up, down = p.copy(), p.copy()
        up[k] += h
        down[k] -= h
        numeric = (bg.value(x, up, span) - bg.value(x, down, span)) / (2 * h)
        np.testing.assert_allclose(bg.jac(x, p, span)[:, k], numeric, rtol=1e-5, atol=1e-6)
    numeric = (bg.value(x + 1e-4, p, span) - bg.value(x - 1e-4, p, span)) / 2e-4
    np.testing.assert_allclose(bg.slope(x, p, span), numeric, rtol=1e-5, atol=1e-6)
    # peaks on it: fitted with this background
    y = bg.value(x, p, span) + emg(x, 2e4, 120.0, 1.0, 1.0) + emg(x, 6e3, 140.0, 1.0, 1.0)
    start = bg.guess(x, y, span, lambda v: float(np.quantile(v, 0.05)))
    p0, lo, hi = bg.bounds(start, bg.limit(start, 3.0), span)
    assert np.all(np.array(lo) <= p0) and np.all(np.array(p0) <= hi)
    f = fit_chromato(x, y, [121.0, 141.0], [np.interp(121.0, x, y), np.interp(141.0, x, y)], start, 0.1, 1.0, span,
                     np.zeros(len(x)), np.zeros(len(x)), convention="standard", legacy_numerics=False,
                     shape=peak_shape("emg"), background=bg)
    assert len(f) == 2
    np.testing.assert_allclose(f.background_at(x), bg.value(x, p, span), rtol=1e-3, atol=2.0)
    np.testing.assert_allclose(f.height, [2e4, 6e3], rtol=1e-3)
    np.testing.assert_allclose(f.rt, [120.0, 140.0], atol=1e-3)


def test_background_constant_is_legacy():
    """The constant background keeps the legacy start values and bounds of the baseline."""
    from peaxart.backgrounds import background

    bg = background()
    for g in (0.0, 0.5, 2.0, 1e4):
        hi = max(3.0, g * 10)
        assert bg.bounds(np.array([g]), None) == ([min(max(1.0, g), hi)], [min(max(0.0, g / 3), min(max(1.0, g), hi))],
                                                  [hi])
    assert bg.limit(np.array([200.0]), 1.5)[0] == 300.0 and bg.limit(np.array([200.0]), None) is None
    with pytest.raises(ValueError, match="Unknown background"):
        background("quadratic")


def test_names():
    """The enums of the backgrounds and peak shapes list exactly the registries, and their strings work alike."""
    from peaxart.backgrounds import BACKGROUNDS, BackgroundName, background
    from peaxart.shapes import SHAPES, ShapeName, peak_shape

    assert set(BackgroundName) == set(BACKGROUNDS) and set(ShapeName) == set(SHAPES)
    assert all(BACKGROUNDS[b].name is b for b in BackgroundName)
    assert all(SHAPES[s].name is s for s in ShapeName)
    assert type(background("linear")) is type(background(BackgroundName.LINEAR))
    assert type(peak_shape("emg")) is type(peak_shape(ShapeName.EMG))
    assert px.ChromatogramFitConfig().background == ("constant", "linear", "exponential")


def test_fit_background_columns():
    """Fit reports the background at each apex (baseline) and its slope; peak_background redraws it."""
    x = np.arange(0.0, 120.0, 0.2)
    base = 2000.0 - 10.0 * x
    y = base + chromato(x, 5e4, 40.0, 1.0, 0.01, "standard") + chromato(x, 3e4, 80.0, 1.0, 0.01, "standard")
    method = px.Fit(ExtractionConfig(convention="standard", legacy_numerics=False, rt_background="linear"))
    peaks = method.integrate(x, y, rt=[40.0, 80.0])
    assert len(peaks) == 2
    np.testing.assert_allclose(peaks["baseline"], [1600.0, 1200.0], rtol=1e-3)
    np.testing.assert_allclose(peaks["baseline_slope"], -10.0, rtol=1e-3)
    np.testing.assert_allclose(px.peak_background(x, peaks), np.maximum(base, 0), rtol=1e-3, atol=1.0)
    constant = px.Fit(ExtractionConfig(convention="standard", legacy_numerics=False)).integrate(x, y, rt=[40.0, 80.0])
    assert (constant["baseline_slope"] == 0).all()


def test_background_selection():
    """Several backgrounds: the first, another one kept only where clearly better."""
    x = np.arange(0.0, 60.0, 0.2)
    peaks = chromato(x, 5e3, 20.0, 1.0, 0.01, "standard") + chromato(x, 3e3, 40.0, 1.0, 0.01, "standard")
    config = ExtractionConfig(convention="standard", legacy_numerics=False, rt_background=("constant", "linear"))
    flat = px.Fit(config).integrate(x, 500.0 + peaks, rt=[20.0, 40.0])
    assert len(flat) == 2 and (flat["baseline_slope"] == 0).all()  # constant: as good, kept
    sloped = px.Fit(config).integrate(x, 2e4 - 300.0 * x + peaks, rt=[20.0, 40.0])
    assert len(sloped) == 2
    np.testing.assert_allclose(sloped["baseline_slope"], -300.0, rtol=1e-2)
    np.testing.assert_allclose(sloped["height"], [5e3, 3e3], rtol=1e-2)
