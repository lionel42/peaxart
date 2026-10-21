# peaxart

Peak extraction in the retention time / m/z domain of GC-TOF-MS data.

This is a rewrite of the peak extraction of the legacy alpinac
(`halosearch/alpinac/mode_extraction`), independent of the HDF5 files: the
input is plain numpy arrays, the output a table of peaks. With the default
settings it reproduces the legacy extraction (bit for bit on most data).
The output can be written in the peak-file format read by the alpinac
identification.

## Method

For each domain (typically one unit mass +-0.5 m/z and ~10 s of rt):

1. **Mass domain, each spectrum**: peaks are detected with a slope method and
   fitted together with pseudo-Voigt functions of fixed shape (sigma from a
   model of the instrument, fixed Lorentzian fraction) plus a constant
   baseline: position, area and baseline are fitted. Weak peaks are removed and
   the fit redone.
2. **Grouping by m/z**: the per-scan peaks are grouped with a histogram of their
   m/z (bin width depending on the mean area) and assigned to the nearest centre.
3. **Retention time**: for each m/z group, the per-scan areas versus rt are
   fitted with chromatographic peaks (Pap and Papai 2001) and a baseline;
   insignificant peaks (Student t test) are removed, and the fit is rejected if
   it is not better than a straight line. The m/z of a peak is the mean of the
   per-scan positions weighted by the fitted chromatographic peak.

## Install

```bash
pip install -e .   # numpy, scipy, pandas, numba
```

## Use

```python
import numpy as np
import peaxart

# intensity: (n_scans, n_points), keep float32 for TOF data
# rt:        (n_scans,) seconds
# mz:        m/z of each point: (n_points,), (n_scans, n_points), or a function mz(rt, x)
# x:         axis of the spectra used for the mass-domain fits (e.g. the tof index)
# sigma:     mass peak width in x units (number or function of x), alpha: Lorentzian fraction
peaks = peaxart.extract(intensity, rt, mz, sigma=lambda x: 0.4 + 2e-5 * x, alpha=0.49,
                        x=tofidx, masses=range(20, 300), rt_range=(1612, 1621), n_jobs=8)

peaks["compound"] = peaxart.group_compounds(peaks, dt=rt[1] - rt[0])

# input of the alpinac identification
peaxart.to_alpinac(peaks, "sample.frag.txt", mass_cal_u=5.0, metadata={"sample": "tank 5"})

# one domain only
df = peaxart.extract_domain(intensity_domain, rt_domain, mz_domain, sigma, alpha, x=x_domain)
```

Parameters (all with the legacy defaults) are in `peaxart.ExtractionConfig`,
e.g. `ExtractionConfig(rt_sigma=1.5)` for another chromatography.

### Output columns

| column | meaning |
|---|---|
| `mass` | unit mass of the domain (`extract` only) |
| `rt`, `rt_u` | retention time (s) and its standard error |
| `mz`, `mz_u`, `mz_u_ppm` | m/z and the spread of the per-scan positions |
| `area`, `area_u` | area (intensity x x-units x s) and its uncertainty |
| `height`, `height_u` | height of the chromatographic peak |
| `rt_sigma`, `rt_tailing` | width (s) and tailing of the chromatographic peak |
| `mz_sigma`, `mz_alpha` | mass peak shape (sigma in m/z, Lorentzian fraction) |
| `lod`, `sn` | limit of detection (area) and signal to noise |
| `x`, `x_u` | position on the x axis |
| `n_points`, `baseline` | per-scan peaks used, baseline of the rt fit |

### Data from TOFWERK h5 files

peaxart does not read h5 files. `benchmarks/compare_full.py` (`load_case`)
shows how to get the arrays and the calibration with the legacy alpinac.

## Example notebook

`notebooks/examples.ipynb` plots the example domains (real data, `tests/data/domains`):
the signal in the rt / m/z plane with the per-scan and the extracted peaks, and for
each peak the chromatographic fit and the mass fit at the apex.

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/examples.ipynb
python notebooks/make_examples_notebook.py   # regenerate and re-run it
```

## Numerics and speed

- `ExtractionConfig(legacy_numerics=True)` (default) reproduces the legacy fits
  (lmfit's algorithm and rounding). It is 2-10 times faster than the legacy
  code per domain, and `extract(..., n_jobs=N)` runs domains in parallel.
- `legacy_numerics=False` uses an analytic Jacobian: much faster for spectra
  with many peaks, same fit quality, but the rt fits of the legacy method are
  sensitive: a ~1 % change of a per-scan area can change the area of some peaks
  by tens of percent. Use it for exploration.
- `convention="legacy"` (default) keeps the legacy Gaussian
  `exp(-(x-mu)^2 / (2 sigma)^2)`, needed with sigma models fitted by the legacy
  mass calibration; `"standard"` is the usual Gaussian.

## Differences to the legacy extraction

With `ExtractionConfig.legacy()` the peaks are the same as the legacy ones.
By default, each chromatographic peak is also tested individually (the legacy
code only compared the whole rt window with a straight line, and fitted peaks
to noise and background elevations):

- `peak_linear_ratio = 0.9`: on the points it covers, the peak must explain the
  data better than a straight background (residual std, corrected for the
  number of parameters, below 0.9 x the one of the line) - column `linear_ratio`;
- `peak_min_coverage = 0.5`: at least half of the scans around the apex must
  have a per-scan mass peak (no "peak" interpolated across a gap) - column `coverage`.

`extract_domain(..., return_rejected=True)` also returns the rejected
candidates, with the reason and the straight background that explains their
data as well (`line_rt`, `line`), as plotted in the example notebook.

The reported values also differ where the legacy output was wrong:

- `peak_width` in the alpinac file is the m/z sigma of the peaks (legacy:
  computed from the *rt* sigma, in seconds, used as a tof-index offset);
- `peak_alpha` is the Lorentzian fraction of the mass peaks (legacy: the rt tailing);
- `area_u` is the uncertainty of the area (legacy: of the height);
- a peak on the edge of two unit-mass domains is reported once;
- compound indices of different rt windows do not collide.

See `PLAN.md` for the validation (MassBank reprocessing case, 4 compounds,
1196 domains) and all details.

## Development

```bash
python -m pytest
# validation against the legacy code (needs the legacy alpinac, /shared/pyenv):
PYTHONPATH=src /shared/pyenv/bin/python benchmarks/legacy/dump_legacy_domains.py benchmarks/legacy/data
PYTHONPATH=src /shared/pyenv/bin/python benchmarks/compare_legacy.py --calibration legacy
PYTHONPATH=src /shared/pyenv/bin/python benchmarks/compare_full.py HFC-236ea -j 8
```
