"""Legacy deconvolution: find_peaks on each chromatogram, grouping into compounds."""

from __future__ import annotations

import numpy as np
import pytest

import peaxart as px
from peaxart.deconvolution import PEAK_COLUMNS_LEGACY


def gauss(t, rt, sigma, height):
    return height * np.exp(-0.5 * ((t - rt) / sigma) ** 2)


@pytest.fixture
def run():
    """3 masses: a compound at 100 s (masses 0 and 1), a weaker one at 103 s (mass 2, overlapping in
    time but another shape), a compound at 200 s (mass 1 only)."""
    rt = np.arange(0, 300, 0.25)
    y = np.zeros((len(rt), 3))
    y[:, 0] = gauss(rt, 100, 2.0, 1000)
    y[:, 1] = gauss(rt, 100.05, 2.0, 400) + gauss(rt, 200, 2.0, 600)
    y[:, 2] = gauss(rt, 103, 4.0, 300)
    return y, rt


def test_find_chromatogram_peaks(run):
    y, rt = run
    peaks = px.find_chromatogram_peaks(y, rt, [40, 41, 42])
    assert tuple(peaks.columns) == PEAK_COLUMNS_LEGACY
    assert peaks["mass"].tolist() == [40, 41, 41, 42]  # by mass, then rt
    np.testing.assert_allclose(peaks["peak_rt"], [100, 100, 200, 103], atol=0.25)
    np.testing.assert_allclose(peaks["peak_heights"], [1000, 400, 600, 300], rtol=1e-3)
    assert (peaks["start_ip"] < peaks["peak_rt"]).all() and (peaks["stopp_ip"] > peaks["peak_rt"]).all()
    # "inflexion points": the crossings at half prominence (find_peaks widths): FWHM = 2.355 sigma
    np.testing.assert_allclose(peaks["stopp_ip"] - peaks["start_ip"], 2.355 * np.array([2, 2, 2, 4]), atol=0.1)
    np.testing.assert_allclose(peaks["tailing_ratio"], 0, atol=0.05)
    # below the height / prominence / width limits: no peak
    assert px.find_chromatogram_peaks(y * 0.01, rt, [40, 41, 42]).empty
    with pytest.raises(ValueError, match="expected"):
        px.find_chromatogram_peaks(y, rt, [40, 41])


def test_group_peaks_legacy(run):
    y, rt = run
    compounds, peaks = px.legacy_deconvolution(y, rt, [40, 41, 42])
    # largest prominence first: compound 0 = 100 s (masses 40, 41); then 200 s; then the wide peak of 42
    assert peaks["compound_id"].tolist() == [0, 0, 1, 2]
    assert peaks["is_max_of_cmp"].tolist() == [True, False, True, True]
    assert compounds.index.tolist() == [0, 1, 2]
    assert compounds[("mai", "")].tolist() == [40, 41, 42]
    assert compounds[("peak_rt", "count")].tolist() == [2, 1, 1]
    np.testing.assert_allclose(compounds[("prominences", "sum")], [1400, 600, 300], rtol=1e-3)
    assert np.isnan(compounds.loc[1, ("peak_rt", "std")])  # one peak
    # looser thresholds (x the 4.7 s duration of the largest peak) group the wide peak with it
    _, loose = px.group_peaks_legacy(px.find_chromatogram_peaks(y, rt, [40, 41, 42]),
                                         rt_threshold=1.0, rt_threshold_start_ip=1.5, rt_threshold_stopp_ip=1.5)
    assert loose["compound_id"].tolist() == [0, 0, 1, 0]


def test_overlapping_groups():
    """Connected extents, swept by their start: a broad peak joins the narrow ones before it."""
    groups = px.overlapping_groups(rt=[100.0, 105.0, 110.0, 300.0], start=[99.7, 104.7, 100.0, 299.7],
                                   stop=[100.3, 105.3, 110.3, 300.3])
    assert [g.tolist() for g, _, _ in groups] == [[0, 1, 2], [3]]
    assert groups[0][1] == pytest.approx(110.0 - 6 * 10.0)  # front: 6 half widths (clipped to 10 s)
    assert groups[1][2] == pytest.approx(300.0 + 8 * 0.3)  # tail: 8 half widths
    # given in another order: the indices refer to the inputs
    [(members, _, _)] = px.overlapping_groups(rt=[105.0, 100.0], start=[104.7, 99.7], stop=[105.3, 105.0])
    assert members.tolist() == [1, 0]


def test_fit_chromatogram():
    """Two coeluting peaks detected as one: the fit splits them; both given to the detected peak."""
    from peaxart.shapes import chromato

    t = np.arange(1700.0, 1800.0, 1 / 6)
    y = 5.0 + chromato(t, 1000.0, 1740.0, 1.4, 0.3) + chromato(t, 600.0, 1745.0, 1.4, 0.3)
    y += 1000.0 * np.exp(-0.5 * ((t - 1775) / 1.4) ** 2)  # a separate group
    fitted = px.fit_chromatogram(t, y, rt=[1741.0, 1775.0], start=[1738.0, 1773.5], stop=[1747.0, 1776.5])
    assert list(fitted.columns) == ["peak", *px.deconvolution.FIT_COLUMNS]
    assert fitted["peak"].tolist() == [0, 0, 1]
    np.testing.assert_allclose(fitted["rt"], [1740.0, 1745.0, 1775.0], atol=0.1)
    np.testing.assert_allclose(fitted["baseline"], 5.0, atol=1.0)
    assert (fitted["signal_prominence"] > 0).all() and fitted.attrs["n_warnings"] >= 0
    empty = px.fit_chromatogram(t, y, rt=[], start=[], stop=[])
    assert empty.empty and list(empty.columns) == ["peak", *px.deconvolution.FIT_COLUMNS]
