"""End-to-end checks on the example domains of ``data/examples`` (slow: ``pytest -m slow``)."""

from pathlib import Path

import pytest

import peaxart
from peaxart.io import load_domain

EXAMPLES = Path(__file__).parents[1] / "data" / "examples"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def methylhexanes_mz043():
    d = load_domain(EXAMPLES / "methylhexanes_mz043.npz")
    return peaxart.extract_domain(d["intensity"], d["rt"], d["mz"], d["sigma"], d["alpha"], x=d["x"], dt=d["dt"])


def near(peaks, mz, tol=0.004):
    return peaks[(peaks["mz"] - mz).abs() < tol]


def test_methylhexanes_coelution_split(methylhexanes_mz043):
    # 2- and 3-methylhexane, 2.9 s apart: one peak for the rt peak detection
    main = near(methylhexanes_mz043, 43.054)
    assert sorted(main["rt"]) == pytest.approx([1823.3, 1826.2], abs=0.15)


@pytest.mark.parametrize("mz", [43.107, 43.123])
def test_methylhexanes_ringing_peaks_found(methylhexanes_mz043, mz):
    # ringing of the C3H7+ peak (+0.053 and +0.070 m/z, ~5e-4 of it): weak, and their
    # per-scan areas have one outlier at the apex scan of C3H7+ (rt_outlier_k)
    peaks = near(methylhexanes_mz043, mz)
    assert len(peaks) >= 1
    assert peaks["rt"].between(1822.5, 1827.0).all()
    assert (peaks["sn"] > 5).all()
