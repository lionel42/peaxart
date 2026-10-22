"""Generate ``notebooks/examples.ipynb`` (the plots are in :mod:`peaxart.plotting`).

    .venv/bin/python notebooks/make_examples_notebook.py [--run]

``--run`` also executes it (slow: every domain is analysed).
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

For each domain (~10 s of retention time around one unit mass, +-0.5 m/z):

1. the **field** (`field_figure`): signal in the rt / m/z plane, the peaks
   fitted in each spectrum (small dots), the extracted chromatographic peaks
   (orange circles) and the rejected candidates (red crosses), with the summed
   spectrum and the summed chromatogram on the sides;
2. for each m/z group (`plot_groups`), the **chromatographic fit** (per-scan
   areas vs rt) and the **mass-domain fit** of the spectrum at the apex.

A chromatographic candidate is rejected when a straight background explains
its data as well (red dotted line; `peak_linear_ratio`, column `linear_ratio`)
or when its apex is not covered by the data (`peak_min_coverage`, column `coverage`).

The analysis of a domain takes a few seconds; `ExtractionConfig(legacy_numerics=False)`
is faster (approximate, see the README).
""")

code("""
from pathlib import Path

import matplotlib.pyplot as plt

from peaxart.config import ExtractionConfig
from peaxart.io import load_domain
from peaxart.plotting import analyse_domain, field_figure, plot_groups

RECORDED = Path("../tests/data/domains")  # recorded from the legacy extraction
EXAMPLES = Path("../data/examples")       # cut from h5 files
CONFIG = ExtractionConfig()
plt.rcParams["figure.dpi"] = 110


def show(name, max_groups=8):
    \"\"\"Analyse one domain and plot its field and its m/z groups.\"\"\"
    path = RECORDED / f"{name}.npz"
    d = load_domain(path if path.exists() else EXAMPLES / f"{name}.npz")
    if d["meta"]:
        print(f"{name}: {d['meta']['compound']} ({d['meta']['file']}): {d['meta']['comment']}")
    peaks, rejected, scans = analyse_domain(d, CONFIG)
    field_figure(d, peaks, rejected, scans)
    plt.show()
    plot_groups(d, peaks, rejected, scans, max_groups=max_groups, config=CONFIG)
    plt.show()
    return peaks, rejected


print("recorded:", sorted(p.stem for p in RECORDED.glob("*.npz")))
print("examples:", sorted(p.stem for p in EXAMPLES.glob("*.npz")))
""")

md("""
## m/z 64 of the HFC-236ea measurement

The strong peak at m/z 64.012 (C2H2F2+ of HFC-236ea). At m/z 63.962 there is
only noise and a background elevation (the wing of the strong mass peak):
the candidates found there are rejected.
""")

code("""
peaks, rejected = show("HFC-236ea_mz064")
display(peaks[["rt", "mz", "area", "lod", "sn", "linear_ratio", "coverage"]].round(3))
display(rejected[["rt", "mz", "linear_ratio", "coverage", "reason"]].round(3))
""")

md("""
## Coeluting compounds separated by the exact mass: CFC-113 and 2-pentene

m/z 70: Cl2+ (69.937) of CFC-113 at 1675.0 s and C5H10+ (70.078) of
2-pentene at 1675.6 s. m/z 58: 57.984 and 58.074 at 1674.9-1675.0 s.
""")

code("""
show("cfc113_mz070");
""")

code("""
show("cfc113_mz058");
""")

md("""
## Two compounds eluting 2.9 s apart: 2- and 3-methylhexane

The rt peak detection finds one peak; the extra-peak trial (`rt_extra_peaks`) splits it at
m/z 43, where the data show a shoulder. At m/z 85 the second compound is weak and makes no
shoulder: one peak.
""")

code("""
show("methylhexanes_mz043");
""")

code("""
show("methylhexanes_mz085");
""")

md("""
## A long peak: acetone (~13 s)
""")

code("""
show("acetone_mz058");
""")

code("""
show("acetone_mz043");
""")

md("""
## A broad peak late in the run: m- and p-xylene
""")

code("""
show("mp-xylene_mz091");
""")

code("""
show("mp-xylene_mz106");
""")

md("""
## The other recorded domains
""")

code("""
for name in ["HFC-236ea_mz051", "HFC-236ea_mz131", "HFC-143_mz043", "CH3I_mz064", "CH3I_mz139", "HFC-236ea_mz200"]:
    show(name, max_groups=3)
""")

md("""
## Your own data

```python
d = dict(name="my domain", intensity=intensity, rt=rt, x=x, mz=mz,
         sigma=lambda x: ..., alpha=alpha, dt=rt[1] - rt[0])
peaks, rejected, scans = analyse_domain(d)
field_figure(d, peaks, rejected, scans)
plot_groups(d, peaks, rejected, scans)
```
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "examples.ipynb"
nbf.write(nb, out)
print("written", out)

if "--run" in sys.argv:
    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbf.write(nb, out)
    print("executed", out)
