"""The chromatogram fit (:func:`peaxart.fit_chromatogram`) on real chromatograms.

``data/chromatograms/<case>.npz``: a unit mass chromatogram (legacy ``unit_mass_chromatos.parquet``
of the run in ``source``), +- 60 s around a group of peaks (``source["group_window"]``), and the peaks
the legacy detection found in it (``rt``, ``start``, ``stop``, ``prominence``).

- ``guard_*``: groups fitted well today, kept so (regressions of the fixes below);
- the known problems: a test of what holds today, and a strict ``xfail`` test of what a fix should
  give: when the fix comes, the ``xfail`` test passes, which fails the run, and is then made a test.

On 251104.1724.air.1, without the added seeds, the fit returns nothing for 132 of 2744 groups (214 of
5913 detected peaks). The causes, by case: no prior for a narrow peak (the legacy detection needs 9 scans
of width: a peak of ~1 s FWHM is missed) and its window cut on a flank (fixed: ``add_seeds``); small peaks
on a sloping background (the baseline of the fit is constant); peaks wider than the sigma bounds (0.2-2 x
the expected sigma of the rt model, at least 0.5 s), e.g. the broad air hump.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import peaxart as px
from peaxart.shapes import peak_shape

DATA = Path(__file__).parent / "data" / "chromatograms"


def load(name: str) -> dict:
    with np.load(DATA / f"{name}.npz") as d:
        case = {k: d[k] for k in d.files}
    case["source"] = json.loads(str(case["source"]))
    return case


def fit(case: dict, y=None, **config):
    return px.fit_chromatogram(case["t"], case["y"] if y is None else y, case["rt"], case["start"], case["stop"],
                               px.ChromatogramFitConfig(**config))


OLD = {"add_seeds": False, "sigma_bounds": (0.2, 2.0), "valley_split": None, "baseline_max_factor": None,
       "peak_shape": "pap_papai"}
"""The fit before the fixes (no added seeds, the sigma bounds of the HR extraction, groups not split, the
baseline bound rising with the refits, Pap-Papai peaks): the cases fail with it."""


def in_window(case: dict, fitted):
    """The fitted peaks of the group window of the case."""
    lo, hi = case["source"]["group_window"]
    return fitted[(fitted["rt"] >= lo) & (fitted["rt"] <= hi)]


def window_r2(case: dict, fitted) -> float:
    """R2 of the fitted signal (baseline + the fitted peaks in the window, with the peak function of the fit:
    ``attrs["peak_shape"]``) over the group window."""
    lo, hi = case["source"]["group_window"]
    w = (case["t"] >= lo) & (case["t"] <= hi)
    t, y = case["t"][w], case["y"][w]
    mine = in_window(case, fitted)
    if mine.empty:
        return float("nan")
    shape = peak_shape(fitted.attrs.get("peak_shape", "pap_papai"))
    model = mine["baseline"].iloc[0] + sum(shape.value(t, p.shape_height, p.shape_rt, p.rt_sigma, p.rt_tailing)
                                           for p in mine.itertuples())
    return float(1 - np.sum((y - model) ** 2) / np.sum((y - y.mean()) ** 2))


def assert_peaks_at(fitted, expected_rt, tolerance: float):
    """A fitted peak within ``tolerance`` s of each expected rt."""
    rt = fitted["rt"].to_numpy()
    missing = [r for r in expected_rt if not len(rt) or np.abs(rt - r).min() > tolerance]
    assert not missing, f"no fitted peak near {missing} (fitted: {np.round(rt, 2).tolist()})"


# --- guards: fitted well today --------------------------------------------------------------------


GUARDS = [
    ("guard_isolated_mz043", [1519.46]),           # 251104.1724.air.1, 5.8e6
    ("guard_isolated_mz069_tank", [1542.46]),      # 240404.1135.tank.5, 4.4e5
    ("guard_three_peaks_mz042", [1460.98, 1464.65, 1476.31]),  # two small peaks before a large one
]


@pytest.mark.parametrize("config", [{}, OLD], ids=["now", "old"])
@pytest.mark.parametrize("name, expected_rt", GUARDS)
def test_guard(name, expected_rt, config):
    case = load(name)
    fitted = in_window(case, fit(case, **config))
    assert len(fitted) == len(expected_rt)
    assert_peaks_at(fitted, expected_rt, 0.3)
    assert window_r2(case, fitted) > 0.995


# --- missed narrow peak: m/z 55, 1641-1727 s --------------------------------------------------------
#
# A dense series of peaks; the largest (2.2e5 at 1664.9 s) is not detected: its FWHM (1.3 s, 8 scans) is
# below the 9 scans of the legacy detection. Without a prior for it the fit of the group fails as a whole;
# fixed by the added seeds (``ChromatogramFitConfig.add_seeds``, :func:`peaxart.added_seeds`).

MISSED = "missed_narrow_peak_mz055"
MISSED_RT = [1646.25, 1657.41, 1664.91, 1671.91, 1678.07, 1687.07, 1692.73, 1711.39]
TAIL_PEAK_RT = 1703.06  # on the tail of the peak at 1692.7 s: see test_tail_peak_of_split_part


def test_missed_narrow_peak_without_seeds():
    """The case: the largest peak is not detected, and without it the group is not fitted."""
    case = load(MISSED)
    assert np.abs(case["rt"] - 1664.91).min() > 3  # not detected
    assert in_window(case, fit(case, **OLD)).empty


def test_missed_narrow_peak_fitted():
    case = load(MISSED)
    seeds, _, _ = px.added_seeds(case["t"], case["y"], case["rt"], case["start"], case["stop"])
    assert np.abs(seeds - 1664.91).min() < 0.1
    fitted = in_window(case, fit(case))
    assert_peaks_at(fitted, MISSED_RT, 0.5)
    # R2 of the window: 0.982 with the EMG, 0.956 with Pap-Papai (it loses the peak at 1703 s: see below)
    assert window_r2(case, fitted) > 0.97
    assert fitted["added"].sum() == 1  # the peak at 1664.9 s


def test_tail_peak_of_split_part():
    """A weak peak on the tail of a larger one (1703.1 s, on the tail of 1692.7 s): with Pap-Papai, lost when its
    part of the group (``valley_split``) is fitted alone; found with the EMG, whose tail follows the larger one."""
    case = load(MISSED)
    pap = in_window(case, fit(case, peak_shape="pap_papai"))
    assert np.abs(pap["rt"] - TAIL_PEAK_RT).min() > 0.5
    assert_peaks_at(in_window(case, fit(case)), [TAIL_PEAK_RT], 0.5)


# --- small peaks on a sloping background: m/z 28, 1351-1400 s -------------------------------------
#
# Bumps of ~1e4 on the tail of N2 (1e5 -> 2e4 over the window): with a constant baseline a straight line
# explains the window as well as the peaks.

SLOPED = "sloped_background_mz028"
SLOPED_RT = [1364.85, 1372.68, 1384.67, 1392.34]


def test_sloped_background_what_holds():
    case = load(SLOPED)
    lo, hi = case["source"]["group_window"]
    y = case["y"][(case["t"] >= lo) & (case["t"] <= hi)]
    assert y[:20].mean() > 3 * y[-20:].mean()  # the background falls across the window
    assert in_window(case, fit(case)).empty


@pytest.mark.xfail(strict=True, reason="constant baseline: the bumps on a sloping background are rejected")
def test_sloped_background_fitted():
    case = load(SLOPED)
    assert_peaks_at(in_window(case, fit(case)), SLOPED_RT, 0.5)


# --- broad hump: m/z 40, 1225-1288 s ---------------------------------------------------------------
#
# A hump ~45 s wide (6e4, the air region) with three maxima: wider than the sigma bounds of the fit
# (at most 2 x the expected sigma, ~2.5 s here), so a line explained it as well. Fixed by wider bounds
# (``ChromatogramFitConfig.sigma_bounds``: 0.2-5 x the expected sigma).

BROAD = "broad_hump_mz040"


def test_broad_hump_old():
    case = load(BROAD)
    assert in_window(case, fit(case, **OLD)).empty


def test_broad_hump_fitted():
    """With the EMG: 3 peaks, R2 0.976 (Pap-Papai: 5 peaks, R2 0.999: the one case where it is better)."""
    case = load(BROAD)
    fitted = in_window(case, fit(case))
    assert window_r2(case, fitted) > 0.97
    assert_peaks_at(fitted, [1235.06, 1257.72], 1.5)  # the two main maxima
    expected = px.config.ExtractionConfig().rt_sigma_at(1250.0)  # ~0.6 s
    assert fitted["rt_sigma"].max() > 2 * expected  # broader than the old upper bound


# --- unstable fit: m/z 39, 1641-1664 s --------------------------------------------------------------
#
# Two detected peaks (1646.2, 1657.4 s), fitted as three: the peak at 1657 s has a coelution at 1662 s,
# both narrow and undetected, as the large peak at 1664.9 s just after the group window. Without the
# 1664.9 s peak, the rise of its flank at the window end is explained by the tailing of the last two peaks,
# which trades with the position / height of the third: two nearby solutions (tailing ~0 / ~0 and ~0.37 /
# ~0.92), and the fit stops at the iteration limit (100) between them, so the last bits of the data (or of a
# sum: memory alignment, after other fits in the same process) decided the result: up to 20 % on a height
# or area, ~8 % on their sum. Stopping at the limit gives no warning. Fixed by the added seeds: the peaks at
# 1661.9 and 1664.9 s are priors of the fit.

UNSTABLE = "unstable_fit_mz039"
UNSTABLE_RT = [1646.24, 1657.37, 1661.83]


def _unstable_group(fitted, case):
    return fitted[np.isin(fitted["peak"], np.flatnonzero((case["rt"] > 1640) & (case["rt"] < 1660)))]


def _perturbed(case, eps):
    return case["y"] * (1 + eps * np.sin(np.arange(len(case["y"]))))


def test_unstable_fit_without_seeds():
    """The case: without the seeds, three peaks at the same rts and about the same total area, whatever the
    last bits, but their heights / areas depend on them (more than 1 %)."""
    case = load(UNSTABLE)
    assert case["source"]["mass"] == 39
    groups = [_unstable_group(fit(case, _perturbed(case, eps), **OLD), case) for eps in (0.0, 1e-15, 1e-13)]
    for group in groups:
        assert len(group) == 3
        np.testing.assert_allclose(group["rt"], UNSTABLE_RT, atol=0.1)
    areas = [g["area"].sum() for g in groups]
    assert max(areas) / min(areas) < 1.1
    spread = max(np.abs(g["area"].to_numpy() / groups[0]["area"].to_numpy() - 1).max() for g in groups)
    assert spread > 0.01


def test_unstable_fit_reproducible():
    case = load(UNSTABLE)
    reference = _unstable_group(fit(case), case)
    for eps in (1e-15, 1e-13):
        group = _unstable_group(fit(case, _perturbed(case, eps)), case)
        # the fit before the seeds moved them by up to 20 %; the EMG fit by 0.15 % (a sigma, at 1e-13)
        np.testing.assert_allclose(group[["height", "area", "rt_sigma"]], reference[["height", "area", "rt_sigma"]],
                                   rtol=5e-3)
    assert len(reference) == 3
    np.testing.assert_allclose(reference["rt"], UNSTABLE_RT, atol=0.1)


# --- truncated window: m/z 40, 1641-1664 s ----------------------------------------------------------
#
# Two clear peaks (1646.2 s, 1657.4 s; prominence ~4600 and ~6900) whose half maximum crossings are
# narrow (1.7-2 s): their group is fitted in 1641.1-1664.2 s, which ends on the rising flank of the next
# peak of this mass (window end at 43 % of the window maximum): the narrow, undetected peak at 1664.9 s
# (and one at 1661.9 s inside the window). A straight line then explains the window as well as the peaks
# ("linear background fits as well"): the fit returned nothing for them, while the groups before and after
# are fitted. On this run, 132 groups (214 detected peaks) got no fitted peak: 78 of them (109 peaks) have a
# window starting or ending above 30 % of its maximum, as here. Fixed by the added seeds: the two narrow
# peaks join the group, whose window then covers them.

TRUNCATED = "truncated_window_mz040"
TRUNCATED_RT = [1646.25, 1657.41]


def _truncated_peaks(case):
    return [int(np.argmin(np.abs(case["rt"] - rt))) for rt in TRUNCATED_RT]


def test_truncated_window_without_seeds():
    """The case, without the seeds: the window of the group ends on the flank of the next peak; its peaks
    are not fitted, the others are."""
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
    fitted = fit(case, **OLD)
    assert not np.isin(peaks, fitted["peak"]).any()
    assert np.isin([peaks[0] - 1, peaks[-1] + 1], fitted["peak"]).all()  # the neighbouring groups are fitted


def test_truncated_window_peaks_fitted():
    case = load(TRUNCATED)
    fitted = fit(case)
    for peak, rt in zip(_truncated_peaks(case), TRUNCATED_RT, strict=True):
        mine = fitted[fitted["peak"] == peak]
        assert len(mine) >= 1
        assert np.abs(mine["rt"] - rt).min() < 0.3
    assert_peaks_at(fitted, [1661.91, 1664.91], 0.3)  # the narrow peaks too
    assert window_r2(case, fitted) > 0.95


# --- large group: m/z 71, 1792-1862 s ---------------------------------------------------------------
#
# Five peaks fitted together (1797.0, 1802.4, 1816.4, 1826.0, 1848.2 s; the one at 1826 s is 6e5, the one at
# 1802 s 2e4). The constant baseline of the group rises to ~14000 to absorb the tails of the large peaks
# (the data go down to ~500 between them), and the small peaks then fail the test against a line. Fitted
# in smaller windows they are found: the signal returns to the background at 1811.5 and 1843.3 s.

LARGE = "large_group_mz071"
LARGE_RT = [1797.03, 1802.36, 1816.35, 1826.02, 1848.18]


def test_large_group_old():
    case = load(LARGE)
    fitted = in_window(case, fit(case, **OLD))
    assert np.abs(fitted["rt"].to_numpy()[:, None] - np.array([1802.36, 1816.35])[None, :]).min(axis=0).min() > 2


def test_large_group_fitted():
    """Fixed by splitting the group at its valleys (``valley_split``) and bounding the baseline
    (``baseline_max_factor``)."""
    case = load(LARGE)
    fitted = in_window(case, fit(case))
    assert_peaks_at(fitted, LARGE_RT, 0.5)
    assert window_r2(case, fitted) > 0.99
