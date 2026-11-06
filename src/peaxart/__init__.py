"""peaxart: peak extraction in the retention time / m/z domain of GC-TOF-MS data.

The method is the one of the legacy alpinac extraction (Guillevic et al. 2021):
peaks are detected and fitted (pseudo-Voigt) in each spectrum, grouped by
m/z, and each m/z group is fitted in retention time with a chromatographic
peak function (Pap and Papai 2001).

Quick start::

    import peaxart

    # intensity: (n_scans, n_points) float32, rt: (n_scans,) s,
    # mz: m/z of each point (n_points,) or (n_scans, n_points), or a function mz(rt, x)
    peaks = peaxart.extract(intensity, rt, mz, sigma=0.9, alpha=0.5,
                            masses=range(20, 200), rt_range=(1600, 1630))
    peaks["compound"] = peaxart.group_compounds(peaks, dt=rt[1] - rt[0])
    peaxart.to_alpinac(peaks, "run.frag.txt")   # input of the alpinac identification

    # unit mass chromatograms (peaxart.unitmass): summed, smoothed, arPLS baseline removed
    chromatograms = peaxart.unit_mass_chromatograms(intensity, mz, masses=range(2, 301))
    # any chromatograms (peaxart.baseline), time along the first axis
    corrected = peaxart.remove_baseline(chromatograms, peaxart.BaselineConfig(lam=1e6))
    # baseline cut at the steps of the acquisition (peaxart.steps), from the logged changes
    cuts, summary = peaxart.step_cuts(sums, rt, steps=[i_emission], switches=[("REX1", i_on, 0, 3)])
    corrected = peaxart.remove_baseline(sums, cuts=cuts)

    # one chromatogram (e.g. a unit mass), several methods (peaxart.chromatogram)
    peaks = peaxart.integrate(t, y, method="trapezoid")
"""

from peaxart.api import domain_slices, extract
from peaxart.baseline import (
    BaselineConfig,
    arpls,
    baseline,
    level_change,
    remove_baseline,
    smooth,
    step_index,
    step_jump,
)
from peaxart.chromatogram import METHODS, Fit, FixedShapes, Method, Trapezoid, detect_peaks, integrate
from peaxart.compounds import group_compounds
from peaxart.config import ExtractionConfig
from peaxart.domain import PEAK_COLUMNS, extract_domain
from peaxart.io import mass_cal_u_ppm, to_alpinac
from peaxart.mzfit import detect_mass_peaks, fit_peak_shapes
from peaxart.steps import StepConfig, step_cuts
from peaxart.unitmass import (
    UnitMassConfig,
    mass_bounds,
    unit_mass_chromatograms,
    unit_mass_sums,
)

__version__ = "0.1.1"

__all__ = [
    "BaselineConfig",
    "ExtractionConfig",
    "Fit",
    "FixedShapes",
    "METHODS",
    "Method",
    "PEAK_COLUMNS",
    "StepConfig",
    "Trapezoid",
    "UnitMassConfig",
    "arpls",
    "baseline",
    "detect_mass_peaks",
    "detect_peaks",
    "domain_slices",
    "extract",
    "extract_domain",
    "fit_peak_shapes",
    "group_compounds",
    "integrate",
    "level_change",
    "mass_bounds",
    "mass_cal_u_ppm",
    "remove_baseline",
    "smooth",
    "step_cuts",
    "step_index",
    "step_jump",
    "to_alpinac",
    "unit_mass_chromatograms",
    "unit_mass_sums",
]
