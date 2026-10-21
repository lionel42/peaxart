"""Generate (and run) ``notebooks/examples.ipynb``.

    .venv/bin/python notebooks/make_examples_notebook.py [--no-run]
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# peaxart: extracted peaks on example domains

The example domains are real data recorded from the legacy alpinac extraction
(MassBank reprocessing, `tests/data/domains/*.npz`): for each, the signal of
~10 s of retention time around one unit mass (+-0.5 m/z), the m/z of each
point, and the mass peak shape model of the instrument.

For each domain the notebook shows

1. the **field**: signal in the rt / m/z plane, the peaks fitted in each
   spectrum (small dots), the extracted chromatographic peaks (orange circles)
   and the rejected candidates (red crosses), with the summed spectrum and the
   summed chromatogram on the sides;
2. for each m/z group, the **chromatographic fit** (per-scan areas vs rt) and
   the **mass-domain fit** of the spectrum at the apex.

A chromatographic candidate is rejected when a straight background explains
its data as well (red dotted line; `peak_linear_ratio`, column `linear_ratio`)
or when its apex is not covered by the data (`peak_min_coverage`, column `coverage`).
""")

code("""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import peaxart
from peaxart.config import ExtractionConfig
from peaxart.domain import MzAxis, extract_domain, fit_scans, group_by_mz
from peaxart.mzfit import detect_mass_peaks, fit_spectrum
from peaxart.shapes import chromato, pseudo_voigt

DATA = Path("../tests/data/domains")
CONFIG = ExtractionConfig()

# Colors: one sequential hue for the signal, an accent for the extracted peaks,
# red for what is rejected, neutral ink for everything else.
FIELD_CMAP = "Blues"
ACCENT = "#d95f02"
REJECTED = "#c62828"
INK = "#333333"
MUTED = "#8a8a8a"

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
})

print("example domains:", sorted(p.stem for p in DATA.glob("*.npz")))
""")

code('''
def load(name):
    """One recorded domain as plain arrays (the inputs of peaxart)."""
    z = np.load(DATA / f"{name}.npz")
    intercept, slope = float(z["sigma_tofidx_b"]), float(z["sigma_tofidx_slope"])
    return dict(
        name=name,
        intensity=z["intensity"].astype(np.float32),  # TOF data are float32
        rt=z["rt"], x=z["tofidx"], mz=z["mz"],
        sigma=lambda x: intercept + slope * x,         # mass peak sigma, tof index units
        alpha=float(z["alpha"]), dt=float(z["s_per_bin"]),
    )


def analyse(d, config=CONFIG):
    """Extracted peaks, rejected candidates, and the per-scan mass peaks with their m/z group."""
    peaks, rejected = extract_domain(d["intensity"], d["rt"], d["mz"], d["sigma"], d["alpha"],
                                     x=d["x"], dt=d["dt"], config=config, return_rejected=True)
    axis = MzAxis(d["mz"], d["x"], d["rt"])
    scans = fit_scans(d["intensity"], d["rt"], d["x"], axis, d["sigma"], d["alpha"], config)
    if len(scans["x"]):
        order = np.argsort(scans["x"])
        scans = {k: v[order] for k, v in scans.items()}
        scans["group"] = group_by_mz(scans, d["x"], axis, d["rt"], config)
    else:
        scans["group"] = np.zeros(0, dtype=int)
    return peaks, rejected, scans


def group_of(scans, mz):
    """The m/z group (index) whose median m/z is the closest to ``mz``."""
    groups = [g for g in np.unique(scans["group"]) if g >= 0]
    return min(groups, key=lambda g: abs(np.median(scans["mz"][scans["group"] == g]) - mz))
''')

code('''
def plot_field(d, peaks, rejected, scans, ax_field, ax_spec, ax_chrom):
    """Signal in the rt / m/z plane with the per-scan, the extracted and the rejected peaks."""
    I = np.asarray(d["intensity"], dtype=float)
    floor = max(np.percentile(I[I > 0], 5) if (I > 0).any() else 1e-3, 1e-3)
    rt2 = np.broadcast_to(d["rt"][:, None], I.shape)
    mesh = ax_field.pcolormesh(d["mz"], rt2, np.clip(I, floor, None), shading="nearest",
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
        ax_field.annotate(f"m/z {p['mz']:.4f}\\nrt {p['rt']:.2f} s, S/N {p['sn']:.1f}",
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

    mid = len(d["rt"]) // 2
    mz_row = d["mz"][mid] if d["mz"].ndim == 2 else d["mz"]
    ax_spec.plot(mz_row, I.sum(axis=0), color=INK, lw=1)
    ax_spec.set_ylabel("summed\\nintensity")
    ax_spec.set_title(f"{d['name']}: {len(peaks)} extracted peak(s), {len(rejected)} rejected", loc="left")
    ax_chrom.plot(I.sum(axis=1), d["rt"], color=INK, lw=1)
    ax_chrom.set_xlabel("summed\\nintensity")
    ax_chrom.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax_chrom.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    for p in peaks.itertuples():
        ax_spec.axvline(p.mz, color=ACCENT, lw=1, alpha=0.7)
        ax_chrom.axhline(p.rt, color=ACCENT, lw=1, alpha=0.7)
    ax_spec.tick_params(labelbottom=False)
    ax_chrom.tick_params(labelleft=False)
    return mesh


def field_figure(d, peaks, rejected, scans):
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[5, 1.2, 0.15], height_ratios=[1.3, 5],
                          wspace=0.05, hspace=0.05)
    ax_field = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[0, 0], sharex=ax_field)
    ax_chrom = fig.add_subplot(gs[1, 1], sharey=ax_field)
    mesh = plot_field(d, peaks, rejected, scans, ax_field, ax_spec, ax_chrom)
    fig.colorbar(mesh, cax=fig.add_subplot(gs[1, 2]), label="intensity (log)")
    return fig
''')

code('''
def plot_mass_fit(ax, d, rt, mz_mark, mz_sigma, config=CONFIG):
    """Spectrum of the scan closest to ``rt`` and its pseudo-Voigt fit, around ``mz_mark``."""
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
    ax.set_xlabel("m/z")
    ax.set_ylabel("intensity")
    ax.set_title(f"mass fit at rt {d['rt'][i]:.2f} s", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="best")


def plot_group_detail(d, peaks, rejected, scans, g, config=CONFIG):
    """Chromatographic fit of m/z group g (kept peaks, rejected candidates) and a mass fit."""
    sel = scans["group"] == g
    g_peaks = peaks[[group_of(scans, m) == g for m in peaks["mz"]]] if len(peaks) else peaks
    g_rej = rejected[[group_of(scans, m) == g for m in rejected["mz"]]] if len(rejected) else rejected
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

    mz_group = float(np.median(scans["mz"][sel]))
    if len(g_peaks):
        p = g_peaks.sort_values("area").iloc[-1]
        plot_mass_fit(ax_mz, d, p["rt"], p["mz"], p["mz_sigma"], config)
    else:
        r = g_rej.iloc[0]
        plot_mass_fit(ax_mz, d, r["rt"], r["mz"], 0.006, config)
    ax_mz.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.suptitle(f"{d['name']}, m/z {mz_group:.4f}: {len(g_peaks)} peak(s), {len(g_rej)} rejected",
                 x=0.06, ha="left", y=1.02, fontsize=11, fontweight="bold", color=INK)
    return fig


def plot_groups(d, peaks, rejected, scans):
    """Detail figures of the m/z groups with an extracted peak or a rejected candidate."""
    groups = sorted({group_of(scans, m) for m in list(peaks["mz"]) + list(rejected["mz"])},
                    key=lambda g: -scans["area"][scans["group"] == g].max())
    for g in groups:
        plot_group_detail(d, peaks, rejected, scans, g)
''')

md("""
## m/z 64 of the HFC-236ea measurement

The strong peak at m/z 64.012 (C2H2F2+ of HFC-236ea). At m/z 63.962 there is
only noise and a background elevation (the wing of the strong mass peak):
the candidates found there are rejected.
""")

code("""
d = load("HFC-236ea_mz064")
peaks, rejected, scans = analyse(d)
cols = ["rt", "mz", "area", "lod", "sn", "linear_ratio", "coverage"]
display(peaks[cols].round(3))
display(rejected[["rt", "mz", "linear_ratio", "coverage", "reason"]].round(3))
""")

code("""
field_figure(d, peaks, rejected, scans);
""")

code("""
plot_groups(d, peaks, rejected, scans)
""")

md("""
## All example domains

Change `NAME` to look at the details of another domain.
""")

code("""
for path in sorted(DATA.glob("*.npz")):
    d = load(path.stem)
    peaks, rejected, scans = analyse(d)
    field_figure(d, peaks, rejected, scans)
    plt.show()
""")

code("""
NAME = "HFC-143_mz043"
d = load(NAME)
peaks, rejected, scans = analyse(d)
display(peaks.round(4))
plot_groups(d, peaks, rejected, scans)
""")

md("""
## Your own data

The same functions work on any domain given as arrays:

```python
d = dict(name="my domain", intensity=intensity, rt=rt, x=x, mz=mz, sigma=sigma, alpha=alpha, dt=rt[1] - rt[0])
peaks, rejected, scans = analyse(d)
field_figure(d, peaks, rejected, scans)
plot_groups(d, peaks, rejected, scans)
```

`sigma` must be a function of `x` here (wrap a constant in `lambda x: value`).
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "examples.ipynb"
nbf.write(nb, out)
print("written", out)

if "--no-run" not in sys.argv:
    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbf.write(nb, out)
    print("executed", out)
