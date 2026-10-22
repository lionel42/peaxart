"""Plots of an extracted domain (requires matplotlib).

    from peaxart.io import load_domain
    from peaxart.plotting import analyse_domain, field_figure, plot_groups

    d = load_domain("data/examples/cfc113_mz070.npz")
    peaks, rejected, scans = analyse_domain(d)
    field_figure(d, peaks, rejected, scans)
    plot_groups(d, peaks, rejected, scans)

A domain ``d`` is a dict with ``intensity``, ``rt``, ``x``, ``mz``, ``sigma``
(function of x), ``alpha``, ``dt`` and ``name`` (see :func:`peaxart.io.load_domain`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from peaxart.config import ExtractionConfig
from peaxart.domain import MzAxis, extract_domain
from peaxart.mzfit import detect_mass_peaks, fit_spectrum
from peaxart.shapes import chromato, pseudo_voigt

# One sequential hue for the signal, an accent for the extracted peaks,
# red for what is rejected, neutral ink for everything else.
FIELD_CMAP = "Blues"
ACCENT = "#d95f02"
REJECTED = "#c62828"
INK = "#333333"
MUTED = "#8a8a8a"

STYLE = {
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
}
"""Matplotlib style of the plots (``plt.rcParams.update(STYLE)``)."""


def analyse_domain(d: dict, config: ExtractionConfig | None = None):
    """Extracted peaks, rejected candidates, and the per-scan mass peaks with their m/z group.

    :return: (peaks, rejected, scans) where ``scans`` is the dict of the
        per-scan peaks (rt, x, mz, area, ...) with their m/z ``group`` (-1: none).
    """
    config = config or ExtractionConfig()
    return extract_domain(d["intensity"], d["rt"], d["mz"], d["sigma"], d["alpha"], x=d["x"], dt=d["dt"],
                          config=config, return_rejected=True, return_scans=True)


def group_of(scans: dict, mz: float) -> int:
    """The m/z group whose median m/z is the closest to ``mz``."""
    groups = [g for g in np.unique(scans["group"]) if g >= 0]
    return min(groups, key=lambda g: abs(np.median(scans["mz"][scans["group"] == g]) - mz))


def plot_field(d, peaks, rejected, scans, ax_field, ax_spec, ax_chrom):
    """Signal in the rt / m/z plane with the per-scan, the extracted and the rejected peaks,
    the summed spectrum (``ax_spec``) and the summed chromatogram (``ax_chrom``)."""
    I = np.asarray(d["intensity"], dtype=float)
    floor = max(np.percentile(I[I > 0], 5) if (I > 0).any() else 1e-3, 1e-3)
    mz = np.asarray(d["mz"])
    mz2 = mz if mz.ndim == 2 else np.broadcast_to(mz[None, :], I.shape)
    rt2 = np.broadcast_to(d["rt"][:, None], I.shape)
    mesh = ax_field.pcolormesh(mz2, rt2, np.clip(I, floor, None), shading="nearest",
                               cmap=FIELD_CMAP, norm=LogNorm(vmin=floor, vmax=I.max()), rasterized=True)
    if len(scans["x"]):
        grouped = scans["group"] >= 0
        ax_field.scatter(scans["mz"][~grouped], scans["rt"][~grouped], s=6, c=MUTED, lw=0,
                         label="per-scan peak (not grouped)")
        ax_field.scatter(scans["mz"][grouped], scans["rt"][grouped], s=9, c=INK, lw=0,
                         label="per-scan peak (in an m/z group)")
    if len(rejected):
        ax_field.scatter(rejected["mz"], rejected["rt"], s=70, marker="x", c=REJECTED, linewidths=2,
                         zorder=5, label="rejected candidate")
    # labels of peaks close in rt alternate between right and left of the marker
    placed = []
    for _, p in peaks.sort_values("rt").iterrows():
        ax_field.scatter(p["mz"], p["rt"], s=110, marker="o", facecolor="none",
                         edgecolor=ACCENT, linewidth=2, zorder=5)
        near = [side for rt0, side in placed if abs(rt0 - p["rt"]) < 1.0]
        side = -1 if near and near[-1] == 1 else 1
        placed.append((p["rt"], side))
        ax_field.annotate(f"m/z {p['mz']:.4f}\nrt {p['rt']:.2f} s, S/N {p['sn']:.1f}",
                          (p["mz"], p["rt"]), xytext=(10 * side, 6 if side > 0 else -22),
                          textcoords="offset points", ha="left" if side > 0 else "right",
                          color=INK, fontsize=8,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    if len(peaks):
        ax_field.scatter([], [], s=110, facecolor="none", edgecolor=ACCENT, linewidth=2,
                         label="extracted peak")
    ax_field.set_xlabel("m/z")
    ax_field.set_ylabel("retention time [s]")
    if ax_field.get_legend_handles_labels()[0]:
        ax_field.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.85)

    ax_spec.plot(mz2[len(d["rt"]) // 2], I.sum(axis=0), color=INK, lw=1)
    ax_spec.set_ylabel("summed\nintensity")
    ax_spec.set_title(f"{d.get('name', '')}: {len(peaks)} extracted peak(s), {len(rejected)} rejected",
                      loc="left")
    ax_chrom.plot(I.sum(axis=1), d["rt"], color=INK, lw=1)
    ax_chrom.set_xlabel("summed\nintensity")
    ax_chrom.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax_chrom.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    for p in peaks.itertuples():
        ax_spec.axvline(p.mz, color=ACCENT, lw=1, alpha=0.7)
        ax_chrom.axhline(p.rt, color=ACCENT, lw=1, alpha=0.7)
    ax_spec.tick_params(labelbottom=False)
    ax_chrom.tick_params(labelleft=False)
    return mesh


def field_figure(d, peaks, rejected, scans):
    """Figure of :func:`plot_field` with a color bar."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 3, width_ratios=[5, 1.2, 0.15], height_ratios=[1.3, 5],
                              wspace=0.05, hspace=0.05)
        ax_field = fig.add_subplot(gs[1, 0])
        ax_spec = fig.add_subplot(gs[0, 0], sharex=ax_field)
        ax_chrom = fig.add_subplot(gs[1, 1], sharey=ax_field)
        mesh = plot_field(d, peaks, rejected, scans, ax_field, ax_spec, ax_chrom)
        fig.colorbar(mesh, cax=fig.add_subplot(gs[1, 2]), label="intensity (log)")
    return fig


def plot_mass_fit(ax, d, rt, mz_mark, mz_sigma, config: ExtractionConfig | None = None):
    """Spectrum of the scan closest to ``rt`` and its pseudo-Voigt fit, around ``mz_mark``."""
    config = config or ExtractionConfig()
    i = int(np.argmin(np.abs(d["rt"] - rt)))
    x, y = d["x"], d["intensity"][i]
    axis = MzAxis(d["mz"], x, d["rt"])
    dx = float(np.median(np.diff(x)))
    found = detect_mass_peaks(x, y, float(d["sigma"](float(int(x[len(x) // 2])))), dx, config)
    xf = np.linspace(x[0], x[-1], 2000)
    mz_f, mz_pts = axis.at(i, xf), axis.at(i, x)
    ax.plot(mz_pts, y, "o", ms=3, color=INK, label=f"spectrum at rt {d['rt'][i]:.2f} s")
    if len(found):
        fit = fit_spectrum(x, y, found[:, 0], found[:, 1], d["sigma"], d["alpha"], dx, config)
        total = np.full_like(xf, fit.baseline)
        for j in range(len(fit)):
            comp = pseudo_voigt(xf, fit.area[j], fit.mu[j], fit.sigma[j], fit.alpha[j], config.convention)
            total += comp
            ax.plot(mz_f, fit.baseline + comp, color=MUTED, lw=1)
        ax.plot(mz_f, total, color=ACCENT, lw=2, label="pseudo-Voigt fit")
    ax.axvline(mz_mark, color=ACCENT, lw=1, ls=":")
    lo, hi = mz_mark - 12 * mz_sigma, mz_mark + 12 * mz_sigma
    ax.set_xlim(lo, hi)
    shown = (mz_pts >= lo) & (mz_pts <= hi)
    if shown.any():
        ax.set_ylim(min(0, y[shown].min()), y[shown].max() * 1.15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("m/z")
    ax.set_ylabel("intensity")
    ax.set_title(f"mass fit at rt {d['rt'][i]:.2f} s", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="best")


def plot_group_detail(d, peaks, rejected, scans, g, config: ExtractionConfig | None = None):
    """Chromatographic fit of the m/z group ``g`` (kept peaks, rejected candidates with the
    straight background that explains them as well) and the mass fit at the apex."""
    config = config or ExtractionConfig()
    sel = scans["group"] == g
    g_peaks = peaks[[group_of(scans, m) == g for m in peaks["mz"]]] if len(peaks) else peaks
    g_rej = rejected[[group_of(scans, m) == g for m in rejected["mz"]]] if len(rejected) else rejected
    with plt.rc_context(STYLE):
        fig, (ax_rt, ax_mz) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw=dict(wspace=0.25))
        ax_rt.scatter(scans["rt"][sel], scans["area"][sel], s=14, c=INK, lw=0, label="per-scan area")
        t = np.linspace(d["rt"][0], d["rt"][-1], 600)
        for k, p in enumerate(g_peaks.itertuples()):
            ax_rt.plot(t, p.baseline + chromato(t, p.height, p.rt, p.rt_sigma, p.rt_tailing, config.convention),
                       color=ACCENT, lw=2, label="fitted chromatographic peak" if k == 0 else None)
        first_line = first_cross = True
        for r in g_rej.itertuples():
            if "linear" in r.reason and len(r.line_rt):
                ax_rt.plot(r.line_rt, r.line, ls=":", lw=2.2, color=REJECTED,
                           label="linear background (better than a peak)" if first_line else None)
                first_line = False
            else:
                ax_rt.scatter([r.rt], [r.baseline + r.height], marker="x", s=60, c=REJECTED, linewidths=2,
                              label="rejected: apex not covered" if first_cross else None, zorder=5)
                first_cross = False
        if len(g_peaks):
            ax_rt.axhline(g_peaks["baseline"].iloc[0], color=MUTED, lw=1, ls="--", label="baseline")
        ax_rt.set_xlabel("retention time [s]")
        ax_rt.set_ylabel("area of the mass peak")
        ax_rt.set_title("chromatographic fit", loc="left")
        ax_rt.legend(fontsize=7, frameon=False, loc="best")
        ax_rt.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax_rt.ticklabel_format(axis="x", style="plain", useOffset=False)

        if len(g_peaks):
            p = g_peaks.sort_values("area").iloc[-1]
            plot_mass_fit(ax_mz, d, p["rt"], p["mz"], p["mz_sigma"], config)
        else:
            r = g_rej.iloc[0]
            plot_mass_fit(ax_mz, d, r["rt"], r["mz"], 0.006, config)
        fig.suptitle(f"{d.get('name', '')}, m/z {np.median(scans['mz'][sel]):.4f}: "
                     f"{len(g_peaks)} peak(s), {len(g_rej)} rejected",
                     x=0.06, ha="left", y=1.02, fontsize=11, fontweight="bold", color=INK)
    return fig


def plot_groups(d, peaks, rejected, scans, max_groups: int | None = 8, config: ExtractionConfig | None = None):
    """Detail figures (:func:`plot_group_detail`) of the m/z groups with an extracted peak or
    a rejected candidate, the most intense first, at most ``max_groups`` (None: all).

    :return: the figures.
    """
    candidates = list(peaks["mz"]) + list(rejected["mz"])
    if not candidates:
        return []
    groups = sorted({group_of(scans, m) for m in candidates},
                    key=lambda g: -scans["area"][scans["group"] == g].max())
    if max_groups is not None:
        groups = groups[:max_groups]
    return [plot_group_detail(d, peaks, rejected, scans, g, config) for g in groups]


__all__ = ["STYLE", "analyse_domain", "field_figure", "group_of", "plot_field",
           "plot_group_detail", "plot_groups", "plot_mass_fit"]
