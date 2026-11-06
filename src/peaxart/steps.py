"""Cuts of the baseline at the steps of the acquisition.

The background of the chromatograms changes abruptly when the acquisition switches
a setting: e.g. the filament emission (all masses x10) or the resonant ejection of
some ions in the quadrupole (N2, O2, CO2 ejected while the air peaks elute). The
baseline of :func:`peaxart.remove_baseline` is then fitted separately between the
steps (``cuts``). The changes are logged by the instrument, slightly after they
take effect: the step is located in the data around each logged change
(:func:`peaxart.step_index`).

Two kinds of changes (:func:`step_cuts`):

- ``steps``: change the level of all masses (filament emission): all masses are cut,
  at the step of their sum;
- ``switches``: a setting switched on and off (resonant ejection channels): the masses
  whose level drops when it is switched on (:func:`peaxart.level_change`) are cut at
  all the switches of the channels. Channels switched on within ``cluster`` s are
  tested together.

    cuts, summary = peaxart.step_cuts(sums, rt, steps=[i_emission], switches=[("REX1", i_on, 0, 3), ...])
    corrected = peaxart.remove_baseline(sums, cuts=cuts)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from peaxart.baseline import level_change, step_index


@dataclass(frozen=True)
class StepConfig:
    """Cuts of the baseline at the steps of the acquisition (:func:`step_cuts`)."""

    #: cut all masses at the ``steps`` (e.g. changes of the filament emission)
    emission: bool = True
    #: cut the masses ejected by the ``switches`` (e.g. resonant ejection channels)
    ejection: bool = True
    #: window where the step is searched, s around the logged change (logged ~0.5 s after it)
    search: tuple[float, float] = (-2.0, 1.0)
    #: an ejected mass: its level drops by more than this (log10) when the switch is turned on
    #: (between flat levels, :func:`peaxart.level_change`); it is then cut at all the switches
    min_jump: float = 0.3
    #: ... from a level of at least this (sparse masses: no cut)
    min_level: float = 20.0
    #: s of flat level on each side of a step (at most: not reaching another change)
    width: float = 3.0
    #: switched on within this time (s) of each other: one cluster, tested together
    cluster: float = 6.0


Switch = tuple[str, int, float, float]
"""(channel, index of the logged change, old value, new value)."""


def step_cuts(sums: np.ndarray, rt: np.ndarray, steps: Sequence[int] = (), switches: Sequence[Switch] = (),
              masses: Sequence[float] | None = None,
              config: StepConfig | None = None) -> tuple[list[list[int]], dict]:
    """Indices where the baseline of each column of ``sums`` is cut (``cuts`` of :func:`peaxart.remove_baseline`).

    :param sums: (n_scans, n_masses) raw sums, e.g. :func:`peaxart.unit_mass_sums`.
    :param rt: (n_scans,) rt of the scans, s.
    :param steps: indices (in ``sums``) of the logged changes that change the level of all masses.
    :param switches: logged changes of the switched channels: (channel, index, old, new), sorted by index;
        switched on when the value increases.
    :param masses: names of the columns in the summary (default their index).
    :return: the cuts of each column, and a summary {"steps": [rt of the cuts], "switches": [{"channels",
        "rt" (logged switches, on and off), "masses" (cut)}, one per cluster]}.
    """
    config = config or StepConfig()
    sums = np.asarray(sums, dtype=float)
    rt = np.asarray(rt, dtype=float)
    n, n_masses = sums.shape
    masses = list(range(n_masses)) if masses is None else list(masses)
    cuts: list[set[int]] = [set() for _ in range(n_masses)]
    summary: dict = {"steps": [], "switches": []}
    steps = list(steps) if config.emission else []
    switches = sorted(switches, key=lambda c: c[1]) if config.ejection else []
    if n < 2 or not (steps or switches):
        return [[] for _ in range(n_masses)], summary
    dt = float(np.median(np.diff(rt)))
    lo, hi = round(config.search[0] / dt), round(config.search[1] / dt) + 1
    width = max(4, round(config.width / dt))

    def locate(y, index):
        return step_index(y, index + lo, index + hi)

    tic = sums.sum(axis=1)
    for index in steps:
        k = locate(tic, index)
        for c in cuts:
            c.add(k)
        summary["steps"].append(float(rt[k]))
    # switched on in clusters (e.g. 191.1, 192.6, 195.1 s); a mass whose level drops across a
    # cluster (before its first switch -> after its last) is ejected: cut at all the switches of the
    # channels of the cluster (switched on and off)
    clusters: list[list[tuple[str, int]]] = []
    for name, index, old, new in switches:
        if new <= old:
            continue
        if clusters and index - clusters[-1][-1][1] <= round(config.cluster / dt):
            clusters[-1].append((name, index))
        else:
            clusters.append([(name, index)])
    other = np.array(sorted({*steps, *(index for _, index, _, _ in switches)}))
    for cluster in clusters:
        first, last = cluster[0][1], cluster[-1][1]
        channels = sorted({name for name, _ in cluster})
        previous = other[other < first + lo]
        following = other[other > last]
        b1, a0 = max(first + lo, 0), min(last + hi, n)  # before the first switch, after the last
        b0 = max(b1 - width, previous.max() + hi if len(previous) else 0, 0)
        a1 = min(a0 + width, following.min() + lo if len(following) else n)
        ejected = []
        for j in range(n_masses):
            y = sums[:, j]
            if level_change(y[b0:b1], y[a0:a1], min_level=config.min_level) < -config.min_jump:
                ejected.append(j)
                for _, index, _, _ in (c for c in switches if c[0] in channels):
                    cuts[j].add(locate(y, index))
        summary["switches"].append({
            "channels": channels,
            "rt": [float(rt[min(index, n - 1)]) for c, index, _, _ in switches if c in channels],
            "masses": [masses[j] for j in ejected]})
    return [sorted(c) for c in cuts], summary
