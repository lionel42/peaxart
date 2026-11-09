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
