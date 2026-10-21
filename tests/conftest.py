import json
from pathlib import Path

import numpy as np
import pytest

DOMAINS = Path(__file__).parent / "data" / "domains"


def domain_files() -> list[Path]:
    return sorted(DOMAINS.glob("*.npz"))


class Domain:
    """A domain recorded from the legacy extraction (inputs and outputs)."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.name = path.stem
        self.intensity = z["intensity"].astype(np.float32)  # h5 data are float32
        self.rt = z["rt"]
        self.x = z["tofidx"]
        self.mz = z["mz"]
        self.slope = float(z["sigma_tofidx_slope"])
        self.intercept = float(z["sigma_tofidx_b"])
        self.alpha = float(z["alpha"])
        self.dt = float(z["s_per_bin"])
        self.peaks = z["peaks"]  # rt, tof, tof_u, area, height_u, sigma, tailing, mass, mass_u, LOD, SN, -
        self.records = json.loads(str(z["records"]))

    def sigma(self, x):
        return self.intercept + self.slope * x


@pytest.fixture(params=domain_files(), ids=lambda p: p.stem)
def domain(request) -> Domain:
    return Domain(request.param)
