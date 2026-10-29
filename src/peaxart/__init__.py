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

    # one chromatogram (e.g. a unit mass), several methods (peaxart.chromatogram)
    peaks = peaxart.integrate(t, y, method="trapezoid")
"""

from peaxart.api import domain_slices, extract
from peaxart.chromatogram import METHODS, Fit, FixedShapes, Method, Trapezoid, detect_peaks, integrate
from peaxart.compounds import group_compounds
from peaxart.config import ExtractionConfig
from peaxart.domain import PEAK_COLUMNS, extract_domain
from peaxart.io import mass_cal_u_ppm, to_alpinac

__version__ = "0.1.0.dev0"

__all__ = [
    "ExtractionConfig",
    "Fit",
    "FixedShapes",
    "METHODS",
    "Method",
    "PEAK_COLUMNS",
    "Trapezoid",
    "detect_peaks",
    "domain_slices",
    "extract",
    "extract_domain",
    "group_compounds",
    "integrate",
    "mass_cal_u_ppm",
    "to_alpinac",
]
