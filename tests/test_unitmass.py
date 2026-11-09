"""Unit mass chromatograms and the baseline (arPLS) of chromatograms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solveh_banded

import peaxart as px
from peaxart.baseline import _cholesky_banded, _penalty_bands


def signal(n: int = 3000, seed: int = 0) -> np.ndarray:
    """Peaks on a drifting background with noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    background = 50 + 20 * np.sin(x / n * 6) + 0.01 * x
    peaks = sum(h * np.exp(-((x - c) ** 2) / (2 * w**2)) for c, h, w in [(500, 400, 6), (1200, 80, 10),
                                                                          (1230, 150, 8), (2500, 1000, 5)])
    return background + peaks + rng.normal(0, 3, n)


@pytest.mark.parametrize("diff_order", [1, 2, 3])
def test_penalty_and_solver(diff_order):
    n = 50
    d = np.diff(np.eye(n), diff_order, axis=0)
    bands = _penalty_bands(n, diff_order, 1.0)
    dense = sum(np.diag(bands[diff_order - k][k:], k) + (np.diag(bands[diff_order - k][k:], -k) if k else 0)
                for k in range(diff_order + 1))
    np.testing.assert_allclose(dense, d.T @ d)
    rng = np.random.default_rng(1)
    bands[-1] += rng.uniform(0.1, 1.0, n)
    b = rng.normal(size=n)
    x = np.empty(n)
    assert _cholesky_banded(bands, b, x)
    np.testing.assert_allclose(x, solveh_banded(bands, b), rtol=1e-10, atol=1e-12)


def test_arpls_as_pybaselines():
    pybaselines = pytest.importorskip("pybaselines")
    for seed in range(3):
        y = signal(seed=seed)
        reference = pybaselines.Baseline().arpls(y)[0]
        np.testing.assert_allclose(px.arpls(y), reference, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(px.arpls(y, lam=1e3, diff_order=1),
                                   pybaselines.Baseline().arpls(y, lam=1e3, diff_order=1)[0], rtol=1e-8, atol=1e-8)


def test_arpls_follows_the_background():
    y = signal()
    x = np.arange(len(y))
    background = 50 + 20 * np.sin(x / len(y) * 6) + 0.01 * x
    baseline = px.arpls(y)
    away = np.ones(len(y), dtype=bool)
    for c in (500, 1215, 2500):
        away[c - 60:c + 60] = False
    assert np.abs(baseline - background)[away].max() < 8  # within ~3 noise sigma


def test_flat_signal():
    # no point below the baseline: NaN baseline (as pybaselines), counted as 0
    assert np.isnan(px.arpls(np.zeros(100))).all()
    np.testing.assert_array_equal(px.remove_baseline(np.zeros(100)), np.zeros(100))


def test_smoothing_padding():
    y = np.array([4.0, 4.0, 8.0, 4.0, 4.0])
    np.testing.assert_allclose(px.smooth(y, (0.25, 0.5, 0.25)), [4, 5, 6, 5, 4])
    np.testing.assert_allclose(px.smooth(y, (0.25, 0.5, 0.25), "zeros"),
                               np.convolve(y, [0.25, 0.5, 0.25], mode="same"))  # legacy: ends 3, not 4


def test_end_effect_of_the_legacy_smoothing():
    """Legacy zero padding: low end points pull the baseline down near both ends of the run."""
    rng = np.random.default_rng(4)
    y = 100 + 40 * np.sin(np.arange(2000) / 1500) + rng.normal(0, 0.45, 2000)
    fixed = px.remove_baseline(y)
    legacy = px.remove_baseline(y, px.BaselineConfig.legacy())
    assert fixed.max() < 3  # only noise is left
    assert legacy[:50].max() > 15 and legacy[-50:].max() > 15


def test_mass_bounds_strict():
    mz = np.array([1.4, 1.5, 1.6, 2.0, 2.4, 2.5, 2.6, 3.0])
    np.testing.assert_array_equal(px.mass_bounds(mz, [2, 3], 0.5), [[2, 5], [6, 8]])  # 1.5 and 2.5 excluded
    per_scan = px.mass_bounds(np.stack([mz, mz + 0.15]), [2], 0.5)
    np.testing.assert_array_equal(per_scan[:, 0], [[2, 5], [0, 4]])  # 1.55 > 1.5: inside


def test_unit_mass_sums():
    rng = np.random.default_rng(2)
    intensity = rng.uniform(0, 10, (20, 40)).astype(np.float32)
    bounds = np.array([[0, 5], [5, 12], [30, 45]])  # the last one clipped to the points
    sums = px.unit_mass_sums(intensity, bounds)
    np.testing.assert_allclose(sums[:, 1], intensity[:, 5:12].sum(axis=1), rtol=1e-6)
    np.testing.assert_allclose(sums[:, 2], intensity[:, 30:].sum(axis=1), rtol=1e-6)
    # the same bounds for each scan: same result
    np.testing.assert_allclose(px.unit_mass_sums(intensity, np.repeat(bounds[None], 20, axis=0)), sums)
    with pytest.raises(ValueError, match="scans"):
        px.unit_mass_sums(intensity, np.repeat(bounds[None], 3, axis=0))


def test_unit_mass_chromatograms():
    """A chromatographic peak on mass 69 over a drifting background is recovered, the background removed."""
    n_scans, n_points = 2000, 300
    mz = np.linspace(60.0, 75.0, n_points)  # 20 points per m/z
    t = np.arange(n_scans, dtype=float)
    rng = np.random.default_rng(3)
    background = (5 + 2 * np.sin(t / 1500))[:, None] * np.ones(n_points)
    peak = 50 * np.exp(-((t - 800) ** 2) / (2 * 8.0**2))[:, None] * np.exp(-((mz - 69.0) ** 2) / (2 * 0.05**2))
    intensity = (background + peak + rng.normal(0, 0.1, (n_scans, n_points))).astype(np.float32)
    chromatograms = px.unit_mass_chromatograms(intensity, mz, [68, 69, 70])
    assert chromatograms.shape == (n_scans, 3) and (chromatograms >= 0).all()
    y = chromatograms[:, 1]
    assert abs(int(np.argmax(y)) - 800) <= 1
    assert np.abs(np.delete(y, np.s_[700:900])).max() < 0.05 * y.max()  # background removed
    assert chromatograms[:, [0, 2]].max() < 0.05 * y.max()
    raw = px.unit_mass_chromatograms(intensity, mz, [69], px.UnitMassConfig(baseline=None))
    np.testing.assert_allclose(raw[:, 0], intensity[:, (mz > 68.5) & (mz < 69.5)].sum(axis=1), rtol=1e-5)


def test_remove_baseline_columns():
    """Several chromatograms at once (columns): as one by one; the baseline is what is removed."""
    y = np.stack([signal(seed=s) for s in range(3)], axis=1)
    config = px.BaselineConfig(clip=False)
    corrected = px.remove_baseline(y, config)
    for k in range(3):
        np.testing.assert_allclose(corrected[:, k], px.remove_baseline(y[:, k], config))
    smoothed = px.remove_baseline(y, px.BaselineConfig(method=None, clip=False))
    np.testing.assert_allclose(smoothed - px.baseline(y), corrected)
    clipped, baseline = px.remove_baseline(y, return_baseline=True)
    np.testing.assert_allclose(baseline, px.baseline(y))
    np.testing.assert_allclose(clipped, np.clip(corrected, 0, None))
    one, one_baseline = px.remove_baseline(y[:, 0], return_baseline=True)
    np.testing.assert_allclose(one_baseline, baseline[:, 0])
    with pytest.raises(ValueError, match="method"):
        px.remove_baseline(y, px.BaselineConfig(method="snip"))


def stepped(n: int = 3000, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Background x10 from scan 1000 (as the filament emission 0.05 -> 0.5 mA), a peak at 2000."""
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    background = np.where(x < 1000, 20.0, 200.0) * (1 + 0.1 * np.sin(x / 800))
    peak = 3000 * np.exp(-((x - 2000) ** 2) / (2 * 15.0**2))
    return background + peak + rng.normal(0, 1, n) * np.sqrt(background), background


def test_cuts_follow_steps():
    y, background = stepped()
    whole = px.remove_baseline(y)
    cut = px.remove_baseline(y, cuts=[1000])
    after = slice(1000, 1150)  # the whole-run baseline lags behind the step: false signal
    assert whole[after].sum() > 5 * cut[after].sum()
    np.testing.assert_allclose(px.baseline(y, cuts=[1000])[200:1800], background[200:1800], rtol=0.1)
    assert cut[1950:2050].sum() == pytest.approx(3000 * 15 * np.sqrt(2 * np.pi), rel=0.1)  # the peak is kept


def test_cuts_per_column_and_short_pieces():
    y, _ = stepped()
    both = np.stack([y, y], axis=1)
    out = px.remove_baseline(both, cuts=[[1000], []])
    np.testing.assert_allclose(out[:, 0], px.remove_baseline(y, cuts=[1000]))
    np.testing.assert_allclose(out[:, 1], px.remove_baseline(y))
    np.testing.assert_allclose(px.remove_baseline(both, cuts=[1000]), np.stack([out[:, 0]] * 2, axis=1))
    # a piece shorter than min_piece: constant baseline (20th percentile), no arPLS
    _, base = px.remove_baseline(y, cuts=[1000, 1010], return_baseline=True)
    assert np.ptp(base[1000:1010]) == 0
    with pytest.raises(ValueError, match="columns"):
        px.remove_baseline(both, cuts=[[1000], [], [5]])


def test_step_detection():
    y, _ = stepped()
    k = px.step_index(y, 990, 1010)
    assert k == 1000
    assert px.step_jump(y, k) == pytest.approx(1.0, abs=0.1)
    assert np.isnan(px.step_jump(y, 1985))  # the rising side of the peak is not a step
    assert np.isnan(px.step_jump(y, 500))  # flat: no step (drift > 0.25 x ~0)
    assert np.isnan(px.step_jump(y, k, min_level=1000))  # both levels too low
    assert np.isnan(px.step_jump(y, 5))  # incomplete window


def test_step_cuts():
    """A step of all masses (x10) and the ejection of one mass between two switches: cut there only."""
    rng = np.random.default_rng(0)
    rt = np.arange(1, 501) * 0.2
    sums = rng.normal(100.0, 3.0, (500, 3)).clip(0)
    sums[rt < 20] /= 10  # emission 0.05 -> 0.5 mA at 20 s
    ejected = (rt >= 50) & (rt < 70)
    sums[ejected, 2] /= 50  # column 2 ejected between 50 and 70 s
    logged = [int(np.searchsorted(rt, t + 0.4)) for t in (20.0, 50.0, 70.0)]  # logged 0.4 s later
    cuts, summary = px.step_cuts(sums, rt, steps=[logged[0]],
                                      switches=[("REX1", logged[1], 0.0, 3.0), ("REX1", logged[2], 3.0, 0.0)],
                                      masses=[68.0, 69.0, 100.0])
    step = int(np.searchsorted(rt, 20.0))
    assert summary["steps"] == [pytest.approx(20.0, abs=0.01)]
    assert cuts[0] == cuts[1] == [step]
    assert cuts[2] == [step, int(np.searchsorted(rt, 50.0)), int(np.searchsorted(rt, 70.0))]
    [switch] = summary["switches"]
    assert switch["masses"] == [100.0] and switch["channels"] == ["REX1"] and len(switch["rt"]) == 2
    corrected = px.remove_baseline(sums, cuts=cuts)
    assert np.abs(corrected[(rt > 45) & (rt < 75), 2]).max() < 30  # only noise left at the steps
    # disabled, or nothing logged: no cut
    none, _ = px.step_cuts(sums, rt, steps=[logged[0]], config=px.StepConfig(emission=False))
    assert none == [[], [], []]
    assert px.step_cuts(sums, rt)[0] == [[], [], []]
