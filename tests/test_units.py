import numpy as np
import pandas as pd
import pytest

import peaxart
from peaxart.config import ExtractionConfig
from peaxart.detect import histogram_centres, peak_detect
from peaxart.leastsq import fit
from peaxart.mzfit import fit_spectrum
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
    for derivative, i in zip(jac, [1, 2, 0, 3]):  # d/dmu, d/dsigma, d/dheight, d/dtailing
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
    y = (2.0 + pseudo_voigt(x, 500.0, 1030.3, sigma, alpha, "legacy")
         + pseudo_voigt(x, 200.0, 1050.7, sigma, alpha, "legacy")).astype(np.float32)
    r = fit_spectrum(x, y, np.array([1030.0, 1051.0]), np.array([y[30], y[51]]), lambda m: sigma, alpha, 1.0,
                     ExtractionConfig())
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
    intensity = np.array([
        1.0 + pseudo_voigt(x, 4000.0 * h, x_peak, sigma, alpha, "legacy") for h in profile
    ]) + rng.normal(0, 0.3, (n_scans, len(x)))
    return intensity.astype(np.float32), rt, x, mz_axis, sigma, alpha


def test_extract_domain_synthetic():
    intensity, rt, x, mz, sigma, alpha = synthetic()
    df = peaxart.extract_domain(intensity, rt, mz, sigma, alpha, x=x)
    assert len(df) == 1
    p = df.iloc[0]
    assert p["mz"] == pytest.approx(69.0, abs=2e-4)
    assert p["rt"] == pytest.approx(1605.0, abs=0.1)
    assert list(df.columns) == peaxart.PEAK_COLUMNS
    # m/z given as a calibration function
    df2 = peaxart.extract_domain(intensity, rt, lambda r, xx: ((xx + 3700.0) / 2829.0) ** 2, sigma, alpha, x=x)
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
    single = peaxart.extract_domain(intensity, rt, mz, sigma, alpha, x=x, config=ExtractionConfig(rt_extra_peaks=0))
    assert len(single) == 1
    df = peaxart.extract_domain(intensity, rt, mz, sigma, alpha, x=x)
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
    assert len(peaxart.extract_domain(intensity, rt, mz, sigma, alpha, x=x, config=ExtractionConfig(rt_outlier_k=None))) == 0
    df, _, scans = peaxart.extract_domain(intensity, rt, mz, sigma, alpha, x=x, return_rejected=True, return_scans=True)
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
    peaks = peaxart.extract(intensity, rt, mz, sigma, alpha, x=x, masses=[69, 70, 71])
    assert sorted(peaks["mass"]) == [69, 70]
    peaks["compound"] = peaxart.group_compounds(peaks, dt=rt[1] - rt[0])
    assert peaks["compound"].nunique() == 1 and (peaks["compound"] >= 0).all()
    table = peaxart.to_alpinac(peaks, tmp_path / "x.frag.txt", mass_cal_u=5.0, metadata={"sample": "test"})
    assert list(table.columns) == peaxart.io.ALPINAC_COLUMNS
    text = (tmp_path / "x.frag.txt").read_text()
    assert "# sample: test" in text and "RT\tmass" in text
    # readable by pandas like the alpinac reader does
    back = pd.read_csv(tmp_path / "x.frag.txt", sep=r"\s+", comment="#")
    assert len(back) == 2 and set(back["compound_bin"]) == {0}


def test_mass_cal_u_rule():
    u = peaxart.mass_cal_u_ppm(np.array([30.0, 100.0]), np.array([50.0, 200.0]), np.array([5.0, 3.0]))
    assert u[0] == pytest.approx(max(0.001 / 30.0 * 1e6, 5.0))
    assert u[1] == pytest.approx(5.0 - 2.0 * 50 / 150)


def test_fast_numerics_same_peak():
    intensity, rt, x, mz, sigma, alpha = synthetic()
    exact = peaxart.extract_domain(intensity, rt, mz, sigma, alpha, x=x)
    fast = peaxart.extract_domain(intensity, rt, mz, sigma, alpha, x=x,
                                  config=ExtractionConfig(legacy_numerics=False))
    assert len(fast) == len(exact) == 1
    assert fast["mz"].iloc[0] == pytest.approx(exact["mz"].iloc[0], abs=1e-5)
    assert fast["area"].iloc[0] == pytest.approx(exact["area"].iloc[0], rel=1e-3)
