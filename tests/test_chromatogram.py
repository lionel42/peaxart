from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import peaxart
from peaxart.chromatogram import COMMON_COLUMNS, noise_std
from peaxart.shapes import chromato, chromato_area

EXAMPLE = Path(__file__).parents[1] / "data" / "examples" / "methylhexanes_chromatograms.npz"
DT = 1 / 6


def chromatogram(peaks, noise=50.0, baseline=500.0, seed=0):
    """Synthetic chromatogram, ``peaks``: (height, rt, sigma, tailing) of each peak."""
    t = 1800.0 + np.arange(0, 50, DT)
    y = baseline + sum(chromato(t, h, r, s, a) for h, r, s, a in peaks)
    return t, y + np.random.default_rng(seed).normal(0, noise, len(t))


ONE = [(1e5, 1820.0, 1.4, 0.3)]
TWO = [(1e5, 1823.2, 1.4, 0.3), (8e4, 1826.1, 1.4, 0.3)]  # 2- and 3-methylhexane: a shoulder
RESOLVED = [(1e5, 1823.2, 1.0, 0.3), (8e4, 1826.1, 1.0, 0.3)]  # with a valley


def area(h, r, s, a):
    return chromato_area(h, s, a)


def test_noise_std():
    t, y = chromatogram(ONE, noise=50.0)
    # the curvature of the peak adds a little
    assert noise_std(y) == pytest.approx(50.0, rel=0.3)


@pytest.mark.parametrize("method", ["trapezoid", "fit"])
def test_single_peak(method):
    t, y = chromatogram(ONE)
    peaks = peaxart.integrate(t, y, method=method)
    assert len(peaks) == 1
    assert list(peaks.columns) == peaxart.METHODS[method].columns
    assert peaks["rt"].iloc[0] == pytest.approx(1820.0, abs=0.1)
    assert peaks["area"].iloc[0] == pytest.approx(area(*ONE[0]), rel=0.02)


def test_methods_have_their_own_columns():
    t, y = chromatogram(ONE)
    tables = {
        "trapezoid": peaxart.Trapezoid().integrate(t, y),
        "fit": peaxart.Fit().integrate(t, y),
        "fixed_shapes": peaxart.FixedShapes([1820.0], 1.4, 0.3).integrate(t, y),
    }
    for name, df in tables.items():
        assert list(df.columns[:3]) == COMMON_COLUMNS
        assert list(df.columns) == peaxart.METHODS[name].columns
    assert "start" in tables["trapezoid"] and "start" not in tables["fit"]
    assert "lod" in tables["fit"] and "lod" not in tables["fixed_shapes"]


def test_fit_separates_coelution():
    t, y = chromatogram(TWO)
    peaks = peaxart.Fit().integrate(t, y)
    assert peaks["rt"].to_list() == pytest.approx([1823.2, 1826.1], abs=0.1)
    assert peaks["area"].to_list() == pytest.approx([area(*p) for p in TWO], rel=0.05)


def test_trapezoid_drop_line():
    t, y = chromatogram(RESOLVED)
    split = peaxart.Trapezoid().integrate(t, y, rt=[1823.2, 1826.1])
    assert len(split) == 2
    assert split["stop"].iloc[0] == split["start"].iloc[1]  # one drop line
    assert split["area"].sum() == pytest.approx(sum(area(*p) for p in RESOLVED), rel=0.03)
    # each peak with its own baseline, from the valley: the signal under the valley is lost
    alone = peaxart.Trapezoid(split_at_valley=False).integrate(t, y, rt=[1823.2, 1826.1])
    assert alone["area"].sum() < 0.8 * split["area"].sum()


def test_trapezoid_shoulder_is_nan():
    t, y = chromatogram(TWO, noise=0.0)
    peaks = peaxart.Trapezoid().integrate(t, y, rt=[1823.2, 1826.1])
    assert np.isnan(peaks["area"].iloc[1])


def test_fixed_shapes_recovers_heights():
    t, y = chromatogram(TWO, noise=20.0)
    method = peaxart.FixedShapes([p[1] for p in TWO], 1.4, 0.3)
    peaks = method.integrate(t, y)
    assert peaks["height"].to_list() == pytest.approx([p[0] for p in TWO], rel=0.01)
    assert (peaks["area_u"] < 0.01 * peaks["area"]).all()
    assert peaks["baseline"].to_list() == pytest.approx([500.0, 500.0], abs=10)
    # a compound absent from the ion: height 0, not negative
    t, y = chromatogram(TWO[1:], noise=20.0)
    assert method.integrate(t, y)["height"].iloc[0] == pytest.approx(0.0, abs=500.0)
    with pytest.raises(ValueError):
        method.integrate(t, y, rt=[1823.0])


def test_bands_enclose_the_area():
    t, y = chromatogram(RESOLVED, noise=0.0)
    for method in (peaxart.Trapezoid(), peaxart.Fit()):
        peaks = method.integrate(t, y, rt=[1823.2, 1826.1])
        for (lower, upper), a in zip(method.bands(t, y, peaks), peaks["area"]):
            inside = np.isfinite(lower)
            assert np.trapezoid(upper[inside] - lower[inside], t[inside]) == pytest.approx(a, rel=0.03)


@pytest.fixture(scope="module")
def methylhexanes():
    z = np.load(EXAMPLE)
    return z["rt"], dict(zip(z["masses"].tolist(), z["intensity"].T))


def window(df):
    return df[df["rt"].between(1820, 1830)]


def test_methylhexanes_fit(methylhexanes):
    # unit mass chromatograms of 251104.1724.air.1: one peak for the detection on m/z 43,
    # split by the extra-peak search
    t, y = methylhexanes
    peaks = window(peaxart.Fit().integrate(t, y[43]))
    assert peaks["rt"].to_list() == pytest.approx([1823.3, 1826.2], abs=0.2)


def test_methylhexanes_fixed_shapes(methylhexanes):
    # shapes from m/z 85 (2-methylhexane only) and m/z 70 (3-methylhexane only)
    t, y = methylhexanes
    shapes = pd.concat([window(peaxart.Fit().integrate(t, y[m])) for m in (85, 70)])
    assert len(shapes) == 2
    method = peaxart.FixedShapes.from_peaks(shapes)
    on_85, on_70, on_43 = (method.integrate(t, y[m]) for m in (85, 70, 43))
    assert on_85["area"].iloc[1] < 0.01 * on_85["area"].iloc[0]
    assert on_70["area"].iloc[0] < 0.01 * on_70["area"].iloc[1]
    assert (on_43["area"] > 10 * on_43["area_u"]).all()
