"""peaxart reproduces the legacy alpinac extraction (domains recorded from the legacy code)."""

import numpy as np
import pytest

from peaxart.config import ExtractionConfig
from peaxart.detect import peak_detect
from peaxart.domain import extract_domain
from peaxart.mzfit import fit_spectrum
from peaxart.rtfit import fit_chromato


def test_peak_detection(domain):
    for d in domain.records["detect"]:
        y = np.array(d["y"], dtype=np.float32 if d["mode"].startswith("tofidx") else float)
        out = peak_detect(np.array(d["x"]), y, d["PW"], d["PT"], d["n_per_bin"])
        np.testing.assert_array_equal(out, np.array(d["out"]).reshape(-1, 2))


def test_pseudo_voigt_fits(domain):
    config = ExtractionConfig.legacy()
    for d in domain.records["pv"]:
        r = fit_spectrum(
            domain.x,
            np.array(d["y"], dtype=np.float32),
            np.array(d["mu_guess"]),
            np.array(d["apex_guess"]),
            domain.sigma,
            domain.alpha,
            1.0,
            config,
        )
        np.testing.assert_allclose(r.mu, d["mu"], rtol=1e-7)
        np.testing.assert_allclose(r.area, d["A"], rtol=1e-3)
        assert r.baseline == pytest.approx(d["baseline"], rel=1e-3, abs=1e-9)


def test_chromatographic_fits(domain):
    for c in domain.records["chromato"]:
        r = fit_chromato(
            np.array(c["x"]),
            np.array(c["y"]),
            np.array(c["mu_guess"]),
            np.array(c["apex_guess"]),
            c["baseline_guess"],
            c["n_per_bin"],
            c["sigma_guess_value"],
            (c["rt_window_min"], c["rt_window_max"]),
            np.array(c["tof_series"]),
            np.array(c["mass_series"]),
            c["conf_int"],
            c["broadness"],
            c["ratio_to_linear_fit"],
        )
        out = c["out"]
        if len(out["mu"]) == 0:
            assert r is None
            continue
        # A few long multi-peak fits diverge from the last bit (~1e-15) after
        # hundreds of evaluations: small tolerances, not bitwise.
        np.testing.assert_allclose(r.rt, out["mu"], rtol=1e-6)
        np.testing.assert_allclose(r.area, out["area"], rtol=1e-2)
        np.testing.assert_allclose(r.lod, out["LOD"], rtol=1e-2)  # LOD: residual-weighted baseline u
        np.testing.assert_allclose(r.mz, out["mass"], rtol=1e-9)


def test_domain(domain):
    """End to end. The m/z come from the grid (not the analytic calibration),
    which can change the m/z grouping of noise; these domains are not affected."""
    df = extract_domain(
        domain.intensity,
        domain.rt,
        domain.mz,
        domain.sigma,
        domain.alpha,
        x=domain.x,
        dt=domain.dt,
        config=ExtractionConfig.legacy(),
    )
    old = domain.peaks[np.argsort(domain.peaks[:, 0])]
    df = df.sort_values("rt")
    assert len(df) == len(old)
    for col, i in [
        ("rt", 0),
        ("x", 1),
        ("area", 3),
        ("height_u", 4),
        ("rt_sigma", 5),
        ("rt_tailing", 6),
        ("mz", 7),
        ("lod", 9),
        ("sn", 10),
    ]:
        tol = 1e-2 if col in ("lod", "sn", "height_u") else 1e-4
        np.testing.assert_allclose(df[col].to_numpy(), old[:, i], rtol=tol, err_msg=col)


def test_per_peak_tests_remove_background_peaks():
    """HFC-236ea m/z 64: the legacy code also fitted 3 'peaks' to noise / a background
    elevation at m/z 63.962 (S/N 1-3). The per-peak tests keep only the real one."""
    from conftest import DOMAINS, Domain

    d = Domain(DOMAINS / "HFC-236ea_mz064.npz")
    legacy = extract_domain(d.intensity, d.rt, d.mz, d.sigma, d.alpha, x=d.x, dt=d.dt, config=ExtractionConfig.legacy())
    new = extract_domain(d.intensity, d.rt, d.mz, d.sigma, d.alpha, x=d.x, dt=d.dt)
    assert len(legacy) == 4
    assert len(new) == 1
    assert new["mz"].iloc[0] == pytest.approx(64.0119, abs=1e-4)
    assert new["linear_ratio"].iloc[0] < 0.1 and new["coverage"].iloc[0] == 1.0
