"""Deconvolution: from the (unit mass) chromatograms to compounds, each with its rt and spectrum.

``legacy`` method: the extraction of the legacy alpinac pipeline (``mode_extraction/peaks.py``),
ported verbatim, so that the compound tables are the same:

1. :func:`find_chromatogram_peaks`: ``scipy.signal.find_peaks`` on each chromatogram (width 9
   scans, height 40, prominence 120); the rts are interpolated from the scan indices;
2. :func:`group_peaks_legacy`: the peaks are grouped into compounds, largest prominence first:
   the peaks not yet assigned whose apex, start and stop inflexion points are within a fraction of
   the duration (``stopp_ip - start_ip``) of the largest one form a compound.

    peaks = px.find_chromatogram_peaks(chromatograms, rt, masses)   # (n_scans, n_masses) array
    compounds, peaks = px.group_peaks_legacy(peaks)
    compounds, peaks = px.legacy_deconvolution(chromatograms, rt, masses)   # both

The column names are those of the legacy ``pipeline.nc`` (``start_ip``, ``stopp_ip``, ...).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

#: columns of :func:`find_chromatogram_peaks`, in order
PEAK_COLUMNS_LEGACY = ("peak_heights", "prominences", "left_bases", "right_bases", "widths", "width_heights",
                       "start_ip", "stopp_ip", "peak_rt", "mass", "flatness", "tailing_ratio")


@dataclass(frozen=True)
class LegacyDeconvolutionConfig:
    """Parameters of the legacy extraction (``run_extraction_pipeline`` defaults)."""

    #: minimum width of a peak, scans (``find_peaks`` ``width``)
    width: float = 9
    #: minimum height (``find_peaks`` ``height``)
    height: float = 40
    #: minimum prominence (``find_peaks`` ``prominence``)
    prominence: float = 120
    #: a peak joins a compound if its apex is within this fraction of the compound's peak duration
    rt_threshold: float = 0.2
    #: ... and its start inflexion point
    rt_threshold_start_ip: float = 0.25
    #: ... and its stop inflexion point
    rt_threshold_stopp_ip: float = 0.25


def find_chromatogram_peaks(chromatograms: np.ndarray, rt: np.ndarray, masses: Sequence,
                            width: float = 9, height: float = 40, prominence: float = 120) -> pd.DataFrame:
    """Peaks of each chromatogram (``scipy.signal.find_peaks``), legacy ``extract_peaks_from_chromatograms``.

    :param chromatograms: (n_scans, n_masses), one column per mass (corrected chromatograms).
    :param rt: (n_scans,) rt of the scans, s: the scan indices of the peaks are interpolated in it.
    :param masses: (n_masses,) label of each column (``mass`` of its peaks).
    :return: one row per peak, ordered by mass then rt: the ``find_peaks`` properties (``peak_heights``,
        ``prominences``, ``left_bases``, ``right_bases`` (scans), ``widths`` (scans), ``width_heights``), the
        inflexion points ``start_ip`` / ``stopp_ip`` and the apex ``peak_rt`` (s), ``mass``, ``flatness``
        (height / width) and ``tailing_ratio`` (asymmetry of the apex between the inflexion points).
    """
    chromatograms = np.asarray(chromatograms)
    rt = np.asarray(rt)
    masses = list(masses)
    if chromatograms.ndim != 2 or chromatograms.shape != (len(rt), len(masses)):
        raise ValueError(f"chromatograms {chromatograms.shape}: expected ({len(rt)}, {len(masses)})")
    scans = np.arange(len(rt))
    found = [find_peaks(chromatograms[:, j], width=width, height=height, prominence=prominence)
             for j in range(len(masses))]
    keys = ("peak_heights", "prominences", "left_bases", "right_bases", "widths", "width_heights",
            "left_ips", "right_ips")
    peaks = pd.DataFrame({key: np.concatenate([props[key] for _, props in found]) for key in keys})
    peaks["start_ip"] = np.interp(peaks["left_ips"], scans, rt)
    peaks["stopp_ip"] = np.interp(peaks["right_ips"], scans, rt)
    peaks["peak_rt"] = np.interp(np.concatenate([index for index, _ in found]), scans, rt)
    peaks["mass"] = np.concatenate([np.full(len(index), mass) for (index, _), mass in zip(found, masses,
                                                                                            strict=True)])
    peaks["flatness"] = peaks["peak_heights"] / peaks["widths"]
    peaks["tailing_ratio"] = (
        (peaks["stopp_ip"] - peaks["peak_rt"]) - (peaks["peak_rt"] - peaks["start_ip"])
    ) / (peaks["stopp_ip"] - peaks["start_ip"])
    return peaks.drop(columns=["left_ips", "right_ips"])


def group_peaks_legacy(peaks: pd.DataFrame, rt_threshold: float = 0.2, rt_threshold_start_ip: float = 0.25,
                       rt_threshold_stopp_ip: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group the peaks into compounds, legacy ``group_peaks_into_compounds``.

    While peaks are not assigned: the one of largest prominence (the first one if equal) starts a
    compound with all the peaks not assigned whose ``peak_rt``, ``start_ip`` and ``stopp_ip`` differ
    from its own by less than ``rt_threshold`` (resp. ``rt_threshold_start_ip``, ``rt_threshold_stopp_ip``)
    x its duration ``stopp_ip - start_ip``. Compound ids in that order (0 = the largest peak).

    :param peaks: :func:`find_chromatogram_peaks`.
    :return: compounds (index ``compound_id``; columns ``(peak_rt, mean|std|count)``, ``(prominences, sum)``,
        ``(start_ip, mean|std)``, ``(stopp_ip, mean|std)``, ``(mai, "")``: the mass of its largest peak), and the
        peaks with ``compound_id`` and ``is_max_of_cmp``.
    """
    peaks = peaks.copy()
    rt, start, stop = (peaks[c].to_numpy() for c in ("peak_rt", "start_ip", "stopp_ip"))
    prominence = peaks["prominences"].to_numpy(dtype=float)
    assigned = np.zeros(len(peaks), dtype=bool)
    compound = -np.ones(len(peaks), dtype=int)
    is_max = np.zeros(len(peaks), dtype=bool)
    cmp_id = 0
    while not assigned.all():
        i = int(np.argmax(np.where(assigned, -np.inf, prominence)))  # first of the largest not assigned
        duration = stop[i] - start[i]
        members = (
            (np.abs(rt - rt[i]) < rt_threshold * duration)
            & (np.abs(start - start[i]) < rt_threshold_start_ip * duration)
            & (np.abs(stop - stop[i]) < rt_threshold_stopp_ip * duration)
            & ~assigned
        )
        compound[members] = cmp_id
        is_max[i] = True
        assigned |= members
        if not assigned[i]:  # zero duration: the peak alone (the legacy loop would never end)
            compound[i] = cmp_id
            assigned[i] = True
        cmp_id += 1
    peaks["compound_id"] = compound
    peaks["is_max_of_cmp"] = is_max

    compounds = peaks.groupby("compound_id").agg({
        "peak_rt": ["mean", "std", "count"],
        "prominences": ["sum"],
        "start_ip": ["mean", "std"],
        "stopp_ip": ["mean", "std"],
    })
    # most abundant ion: the mass of the largest peak
    compounds["mai"] = peaks.loc[peaks["is_max_of_cmp"], ["mass", "compound_id"]].set_index("compound_id")
    return compounds, peaks


def legacy_deconvolution(chromatograms: np.ndarray, rt: np.ndarray, masses: Sequence,
                         config: LegacyDeconvolutionConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """:func:`find_chromatogram_peaks` then :func:`group_peaks_legacy`: (compounds, peaks)."""
    config = config or LegacyDeconvolutionConfig()
    peaks = find_chromatogram_peaks(chromatograms, rt, masses, config.width, config.height, config.prominence)
    return group_peaks_legacy(peaks, config.rt_threshold, config.rt_threshold_start_ip, config.rt_threshold_stopp_ip)


# --- fit of the peaks of each chromatogram (step 2 of the deconvolution) ---------------------

#: columns of :func:`fit_chromatogram`, besides ``peak`` (from the columns of :class:`peaxart.Fit`)
FIT_COLUMNS = ("rt", "rt_u", "height", "height_u", "rt_sigma", "rt_tailing", "baseline", "area", "area_u", "sn",
               "signal_height", "signal_prominence")


@dataclass(frozen=True)
class ChromatogramFitConfig:
    """Fit of the detected peaks of a chromatogram (:func:`fit_chromatogram`)."""

    #: Gaussian convention of the peak shapes ("standard": the areas are true areas)
    convention: str = "standard"
    #: iteration limit of each fit: the few fits that do not converge (weak, unresolved peaks)
    #: would otherwise take most of the time
    max_iterations: int = 100
    #: extent of a peak for the groups of overlapping peaks, in half widths at half maximum of
    #: each side from the apex, longer on the tail side
    front_hwhm: float = 6.0
    tail_hwhm: float = 8.0
    #: half widths of the detection (apex - start, stop - apex) clipped to this range, s
    hwhm_range: tuple[float, float] = (0.3, 10.0)
    #: groups with fewer points are not fitted
    min_points: int = 10

    def method(self):
        """The :class:`peaxart.Fit` of these parameters."""
        from peaxart.chromatogram import Fit
        from peaxart.config import ExtractionConfig

        return Fit(ExtractionConfig(convention=self.convention, legacy_numerics=False,
                                    rt_max_iterations=self.max_iterations))


def overlapping_groups(rt: np.ndarray, start: np.ndarray, stop: np.ndarray,
                       config: ChromatogramFitConfig | None = None) -> list[tuple[np.ndarray, float, float]]:
    """Groups of overlapping peaks of one chromatogram, fitted together.

    The extent of a peak is ``front_hwhm`` x its front half width before its apex and ``tail_hwhm`` x its
    tail half width after it (half widths: ``rt - start``, ``stop - rt``, clipped to ``hwhm_range``; the
    smallest if not given); connected extents form a group.

    :param rt, start, stop: apex and half maximum crossings of the peaks (e.g. ``peak_rt``, ``start_ip``,
        ``stopp_ip`` of :func:`find_chromatogram_peaks`), s.
    :return: (indices of the peaks in the inputs (sorted by rt), rt range fitted) of each group.
    """
    config = config or ChromatogramFitConfig()
    rt, start, stop = (np.asarray(v, dtype=float) for v in (rt, start, stop))
    order = np.argsort(rt, kind="quicksort")  # the peaks sorted by rt (as pandas sort_values), as the groups list them
    rt, start, stop = rt[order], start[order], stop[order]
    front = np.nan_to_num(np.clip(rt - start, *config.hwhm_range), nan=config.hwhm_range[0])
    tail = np.nan_to_num(np.clip(stop - rt, *config.hwhm_range), nan=config.hwhm_range[0])
    lo, hi = rt - config.front_hwhm * front, rt + config.tail_hwhm * tail
    # connected intervals, swept by their start (a broad peak can start before the narrow ones eluting before it)
    groups, members, end = [], [], -np.inf
    for i in np.argsort(lo, kind="stable"):
        if members and lo[i] >= end:
            groups.append(members)
            members = []
        members.append(i)
        end = max(end, hi[i]) if len(members) > 1 else hi[i]
    if members:
        groups.append(members)
    return [(order[sorted(g)], float(lo[g].min()), float(hi[g].max())) for g in groups]


def fitted_signal_peaks(t: np.ndarray, fitted: pd.DataFrame,
                        convention: str = "standard") -> tuple[np.ndarray, np.ndarray]:
    """Height and prominence of the fitted peaks of one group, on the fitted signal.

    The fitted signal is the baseline plus all the fitted peaks of the group; the height is its value at
    the apex, the prominence is scipy's (``find_peaks``, as the detection). A shoulder (apex not a local
    maximum) has none by scipy's definition: it gets its prominence alone, its fitted height above the baseline.

    :param fitted: :class:`peaxart.Fit` peaks (``rt``, ``height``, ``rt_sigma``, ``rt_tailing``, ``baseline``).
    """
    import warnings

    from scipy.signal import peak_prominences

    from peaxart.shapes import chromato

    t = np.asarray(t, dtype=float)
    signal = fitted["baseline"].iloc[0] + sum(
        np.nan_to_num(chromato(t, p.height, p.rt, p.rt_sigma, p.rt_tailing, convention))
        for p in fitted.itertuples())
    rt = fitted["rt"].to_numpy(dtype=float)
    nearest = np.abs(t[None, :] - rt[:, None]).argmin(axis=1)
    # the apex falls between samples: the highest of the nearest one and its neighbours
    apex = np.array([max(i - 1, 0) + int(np.argmax(signal[max(i - 1, 0): i + 2])) for i in nearest])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PeakPropertyWarning: a prominence is 0
        prominence = peak_prominences(signal, apex)[0]
    shoulder = prominence <= 0
    prominence[shoulder] = fitted["height"].to_numpy(dtype=float)[shoulder]
    return signal[apex], prominence


def fit_chromatogram(t: np.ndarray, y: np.ndarray, rt: np.ndarray, start: np.ndarray, stop: np.ndarray,
                     config: ChromatogramFitConfig | None = None) -> pd.DataFrame:
    """Fit the detected peaks of one chromatogram, by groups of overlapping peaks (:func:`overlapping_groups`).

    Pap-Papai peaks on a constant baseline (:class:`peaxart.Fit`), starting at the detected rts; the fit
    adds peaks where the data show a coelution and drops those a straight background explains.

    :param t, y: the chromatogram (e.g. a unit mass), t in s.
    :param rt, start, stop: apex and half maximum crossings of the detected peaks, s.
    :return: one row per fitted peak: ``peak`` (index in the inputs of the nearest detected peak of its
        group) and :data:`FIT_COLUMNS` (the :class:`peaxart.Fit` columns, ``signal_height`` and
        ``signal_prominence``: :func:`fitted_signal_peaks`); the number of warnings of the fits in
        ``attrs["n_warnings"]``.
    """
    import warnings

    config = config or ChromatogramFitConfig()
    t, y, rt = np.asarray(t, dtype=float), np.asarray(y, dtype=float), np.asarray(rt, dtype=float)
    method = config.method()
    rows, n_warnings = [], 0
    for members, lo, hi in overlapping_groups(rt, start, stop, config):
        inside = (t >= lo) & (t <= hi)
        if inside.sum() < config.min_points:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = method.integrate(t[inside], y[inside], rt=rt[members])
        n_warnings += len(caught)
        if fitted.empty:
            continue
        nearest = np.abs(fitted["rt"].to_numpy()[:, None] - rt[members][None, :]).argmin(axis=1)
        fitted["peak"] = members[nearest]
        fitted["signal_height"], fitted["signal_prominence"] = fitted_signal_peaks(t[inside], fitted,
                                                                                    config.convention)
        rows.append(fitted[["peak", *FIT_COLUMNS]])
    out = (pd.concat(rows, ignore_index=True) if rows
           else pd.DataFrame({c: pd.Series(dtype=int if c == "peak" else float) for c in ("peak", *FIT_COLUMNS)}))
    out.attrs["n_warnings"] = n_warnings
    return out
