"""Known problems of the chromatogram fit (:func:`peaxart.fit_chromatogram`), on real chromatograms.

``data/chromatograms/<case>.npz``: a unit mass chromatogram of 251104.1724.air.1 (legacy
``unit_mass_chromatos.parquet``), +- 60 s around the group of interest, and the peaks the legacy
detection found in it (``rt``, ``start``, ``stop``, ``prominence``); provenance in ``source``.

Each case has a test of what holds today, and a strict ``xfail`` test of what a fix should give:
when the fix comes, the ``xfail`` test passes, which fails the run, and is then turned into a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import peaxart as px

DATA = Path(__file__).parent / "data" / "chromatograms"


def load(name: str) -> dict:
    with np.load(DATA / f"{name}.npz") as d:
        case = {k: d[k] for k in d.files}
    case["source"] = json.loads(str(case["source"]))
    return case


def fit(case: dict, y=None, **config):
    return px.fit_chromatogram(case["t"], case["y"] if y is None else y, case["rt"], case["start"], case["stop"],
                               px.ChromatogramFitConfig(**config))


# --- unstable fit: m/z 39, 1641-1664 s --------------------------------------------------------------
#
# Two detected peaks (1646.2, 1657.4 s), fitted as three: the peak at 1657 s has a coelution at 1662 s.
# The tailing of the last two trades with the position / height of the third: the fit has two nearby
# solutions (tailing ~0 / ~0 and ~0.37 / ~0.92) and stops at the iteration limit (100) between them, so
# the last bits of the data (or of a sum: memory alignment, after other fits in the same process) decide
# the result: up to 20 % on a height or area, ~8 % on their sum. Stopping at the limit gives no warning.
# More iterations (1000) mostly reach one solution, not always (a 1e-13 perturbation lands elsewhere).

UNSTABLE = "unstable_fit_mz039"
UNSTABLE_RT = [1646.24, 1657.37, 1661.83]


def _unstable_group(fitted, case):
    return fitted[np.isin(fitted["peak"], np.flatnonzero((case["rt"] > 1640) & (case["rt"] < 1660)))]


def _perturbed(case, eps):
    return case["y"] * (1 + eps * np.sin(np.arange(len(case["y"]))))


def test_unstable_fit_what_holds():
    """Three peaks at the same rts and about the same total area, whatever the last bits."""
    case = load(UNSTABLE)
    assert case["source"]["mass"] == 39
    areas = []
    for eps in (0.0, 1e-15, 1e-13):
        group = _unstable_group(fit(case, _perturbed(case, eps)), case)
        assert len(group) == 3
        np.testing.assert_allclose(group["rt"], UNSTABLE_RT, atol=0.1)
        areas.append(group["area"].sum())
    assert max(areas) / min(areas) < 1.1


@pytest.mark.xfail(strict=True, reason="fit between two solutions at the iteration limit: a last-bit change of "
                                       "the data changes the heights / areas by up to 20 %")
def test_unstable_fit_reproducible():
    case = load(UNSTABLE)
    reference = _unstable_group(fit(case), case)
    for eps in (1e-15, 1e-13):
        group = _unstable_group(fit(case, _perturbed(case, eps)), case)
        np.testing.assert_allclose(group[["height", "area", "rt_sigma"]], reference[["height", "area", "rt_sigma"]],
                                   rtol=1e-3)


# --- truncated window: m/z 40, 1641-1664 s ----------------------------------------------------------
#
# Two clear peaks (1646.2 s, 1657.4 s; prominence ~2400 and ~7400) whose half maximum crossings are
# narrow (1.7-2 s): their group is fitted in 1641.1-1664.2 s, which ends on the rising flank of the next
# peak of this mass (window end at 43 % of the window maximum). A straight line then explains the window
# as well as the peaks ("linear background fits as well"): the fit returns nothing for them, while the
# groups before and after are fitted. On this run, 132 groups (214 detected peaks) get no fitted peak:
# 78 of them (109 peaks) have a window starting or ending above 30 % of its maximum, as here; the other 54
# (105 peaks, some of the largest of the run: m/z 20, 55 at 1600-1720 s) are lost for another reason.

TRUNCATED = "truncated_window_mz040"
TRUNCATED_RT = [1646.25, 1657.41]


def _truncated_peaks(case):
    return [int(np.argmin(np.abs(case["rt"] - rt))) for rt in TRUNCATED_RT]


def test_truncated_window_what_holds():
    """The window of the group ends on the flank of the next peak; its peaks are not fitted, the others are."""
    case = load(TRUNCATED)
    assert case["source"]["mass"] == 40
    peaks = _truncated_peaks(case)
    [(members, lo, hi)] = [g for g in px.overlapping_groups(case["rt"], case["start"], case["stop"])
                           if set(g[0]) & set(peaks)]
    assert sorted(members.tolist()) == peaks
    window = (case["t"] >= lo) & (case["t"] <= hi)
    y = case["y"][window]
    assert y[-1] > 0.3 * y.max()  # cut on the flank of the next peak
    assert case["prominence"][peaks].min() > 1000  # clear peaks
    fitted = fit(case)
    assert not np.isin(peaks, fitted["peak"]).any()
    assert np.isin([peaks[0] - 1, peaks[-1] + 1], fitted["peak"]).all()  # the neighbouring groups are fitted


@pytest.mark.xfail(strict=True, reason="a group fitted in a window cut on the flank of the next peak is rejected "
                                       "(a line fits as well): its peaks are lost")
def test_truncated_window_peaks_fitted():
    case = load(TRUNCATED)
    fitted = fit(case)
    for peak, rt in zip(_truncated_peaks(case), TRUNCATED_RT, strict=True):
        mine = fitted[fitted["peak"] == peak]
        assert len(mine) >= 1
        assert np.abs(mine["rt"] - rt).min() < 0.3
