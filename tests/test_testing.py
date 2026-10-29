"""Unit tests of peaxart.testing (spread of fits under rounding)."""

import numpy as np
import pandas as pd
import pytest

from peaxart.config import ExtractionConfig
from peaxart.mzfit import ScanPeaks, fit_spectrum
from peaxart.shapes import pseudo_voigt
from peaxart.testing import ULP, match_peaks, nudge, peak_table, rounding_spread


def test_nudge_scales_float64_only():
    assert nudge(2.0, 3) == 2.0 * (1 + 3 * ULP) != 2.0
    assert nudge(np.float64(2.0), -1) == np.float64(2.0) * (1 - ULP)
    np.testing.assert_array_equal(nudge(np.array([1.0, 4.0]), 2), np.array([1.0, 4.0]) * (1 + 2 * ULP))
    y32 = np.array([1.0, 2.0], dtype=np.float32)
    assert nudge(y32, 5) is y32
    assert nudge(3, 5) == 3 and isinstance(nudge(3, 5), int)
    assert nudge(True, 5) is True
    config = ExtractionConfig.legacy()
    assert nudge(config, 5) is config


def test_nudge_callable_output():
    sigma = nudge(lambda x: 0.5 + 1e-4 * x, 4)
    assert sigma(1000.0) == pytest.approx(0.6, rel=1e-12)
    assert sigma(1000.0) == (0.5 + 1e-4 * 1000.0) * (1 + 4 * ULP)
    assert nudge(float, 4) is float  # classes are not nudged


def test_peak_table():
    peaks = ScanPeaks(np.array([1.0, 2.0]), np.array([10.0, 20.0]), np.array([0.5, 0.5]), np.array([0.3, 0.3]), 4.0)
    t = peak_table(peaks)
    assert list(t.columns) == ["mu", "area", "sigma", "alpha", "baseline"]
    assert t["baseline"].tolist() == [4.0, 4.0]
    assert len(peak_table(None)) == 0
    df = pd.DataFrame({"rt": [3.0, 1.0]}, index=[7, 9])
    assert peak_table(df).index.tolist() == [0, 1]
    with pytest.raises(TypeError):
        peak_table([1.0, 2.0])


def test_match_peaks():
    ref = pd.DataFrame({"rt": [10.0, 10.0, 20.0], "mz": [43.0, 44.0, 43.0]})
    other = pd.DataFrame({"rt": [20.1, 10.05, 9.95], "mz": [43.0, 44.0, 43.0]})
    np.testing.assert_array_equal(match_peaks(ref, other, ["rt", "mz"]), [2, 1, 0])
    # rt alone cannot tell the two peaks at rt 10 apart
    assert match_peaks(ref, other, ["rt"])[0] == match_peaks(ref, other, ["rt"])[1]
    np.testing.assert_array_equal(match_peaks(ref, other.iloc[:0], ["rt"]), [-1, -1, -1])


def test_well_determined_fit_does_not_move():
    """A straight line fitted by least squares: spreads of the order of the rounding."""
    x = np.linspace(0, 10, 50)
    y = 3.0 + 0.5 * x + np.sin(7 * x) * 0.01

    def line(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return pd.DataFrame({"mu": [intercept], "slope": [slope]})

    s = rounding_spread(line, x, y)
    assert s["mu"].iloc[0] == pytest.approx(3.0, abs=0.01)
    assert s["mu_rel_spread"].iloc[0] < 1e-13
    assert s["slope_rel_spread"].iloc[0] < 1e-13
    assert s.attrs["n_peaks"] == [1] * 13
    assert s.attrs["nudges"] == [-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32]


def test_chaotic_fit_moves():
    """The logistic map amplifies the last bit: the spread is of the order of the value."""

    def chaotic(r):
        v = 0.3
        for _ in range(200):
            v = r * v * (1 - v)
        return pd.DataFrame({"mu": [0.0], "value": [v]})

    s = rounding_spread(chaotic, 3.99)
    assert s["mu_spread"].iloc[0] == 0
    assert s["value_spread"].iloc[0] > 0.05


def test_inputs_nudged_by_position_keyword_and_callable():
    seen = []

    def fit(a, b, *, c, sigma):
        seen.append((a, b, c, sigma(2.0)))
        return pd.DataFrame({"mu": [1.0]})

    rounding_spread(fit, 1.0, np.float32(1.0), c=1.0, sigma=lambda x: x, nudges=[1, -1])
    (a0, b0, c0, s0), (a1, b1, c1, s1), (a2, _, _, _) = seen
    assert (a0, c0, s0) == (1.0, 1.0, 2.0)
    assert a1 == c1 == 1.0 + ULP and s1 == 2.0 * (1.0 + ULP) and a2 == 1.0 - ULP
    assert b1 == b0  # float32 untouched


def test_changing_and_missing_peaks():
    """A peak that appears / disappears with the rounding: counts recorded, NaN if no peak."""

    def threshold(v):
        n = 2 if v > 1.0 else (0 if v < 1.0 else 1)
        return pd.DataFrame({"mu": [5.0, 9.0][:n], "area": [100.0, 3.0][:n]})

    s = rounding_spread(threshold, 1.0, nudges=[-1, 1])
    assert s.attrs["n_peaks"] == [1, 0, 2]
    assert np.isnan(s["area_spread"].iloc[0])
    s = rounding_spread(threshold, 1.0, nudges=[1, 2])
    assert s["area_spread"].iloc[0] == 0  # the extra peak is not matched to the reference


def test_empty_reference():
    s = rounding_spread(lambda v: None, 1.0, key="rt")
    assert len(s) == 0
    assert s.attrs["n_peaks"] == [0] * 13


def test_pseudo_voigt_fit():
    """A strong peak next to a weak one (both fitted): the strong peak is stable."""
    config = ExtractionConfig.legacy()
    x = np.arange(0.0, 60.0)
    rng = np.random.default_rng(1)
    y = (
        pseudo_voigt(x, 1e5, 25.3, 1.1, 0.5, config.convention)
        + pseudo_voigt(x, 40.0, 31.0, 1.1, 0.5, config.convention)
        + rng.normal(3.0, 1.0, x.size)
    ).astype(np.float32)
    mu_guess, apex_guess = np.array([25.0, 31.0]), y[[25, 31]].astype(float)
    s = rounding_spread(fit_spectrum, x, y, mu_guess, apex_guess, lambda x: 1.1, 0.5, 1.0, config)
    big = s["area"].idxmax()
    assert s.loc[big, "mu"] == pytest.approx(25.3, abs=0.05)
    assert s.loc[big, "mu_spread"] < 1e-3
    assert s.loc[big, "area_rel_spread"] < 1e-3
