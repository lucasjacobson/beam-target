"""Shared helpers for the test suite."""

import os
from pathlib import Path

import pytest

from shine_neutron_source import Region, ShineSource
from shine_neutron_source.source import DEFAULT_DATA_DIR, dt_gas

REPO_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_INPUT = REPO_DIR / "examples" / "generic.toml"


def data_path(fname):
    """Path of a file in the package's data directory."""
    return os.path.join(DEFAULT_DATA_DIR, fname)


# A small, fast case: a 50 cm, 5 torr D/T gas target that the beam passes through (it does not
# stop), with a 1 cm wall radius so that part of the diverging beam is scraped. The region
# settings are lists (one entry per region); make_source turns them into Regions.
SMALL_CASE = dict(
    name="test",
    material=dt_gas(0.5),
    voltage=280.0,
    current=10.0,
    pressure=[5.0],
    temperature=[25.0],
    density=None,
    length=[50.0],
    start_radius=0.2,
    wall_radius=1.0,
    divergence=[30.0],
    num_x_bins=[40],
    num_x_bins_out=[10],
    angle_step=1.0,
    angle_step_out=10.0,
    energy_bin_width=0.01,
)

# Region settings: one value per region (a scalar applies to every region)
REGION_SETTINGS = (
    "material",
    "length",
    "pressure",
    "temperature",
    "density",
    "wall_radius",
    "divergence",
    "num_x_bins",
    "num_x_bins_out",
)


def make_regions(settings):
    """Split per-region settings (lists, or scalars for every region) into Regions."""
    n = len(settings["length"]) if isinstance(settings["length"], list) else 1

    def value(key, r):
        v = settings.get(key)
        return v[r] if isinstance(v, (list, tuple)) else v

    return [Region(**{key: value(key, r) for key in REGION_SETTINGS}) for r in range(n)]


def make_source(**overrides):
    """Return a ShineSource for SMALL_CASE with some settings overridden. Region settings are
    lists, one entry per region. T_frac=x is short for material=dt_gas(x), a D/T gas with that
    tritium fraction."""
    if "T_frac" in overrides:
        overrides["material"] = dt_gas(overrides.pop("T_frac"))
    settings = {**SMALL_CASE, **overrides}
    regions = make_regions(settings)
    rest = {k: v for k, v in settings.items() if k not in REGION_SETTINGS}
    return ShineSource(regions=regions, **rest)


@pytest.fixture(scope="session")
def mixed_source():
    """A 50/50 D/T case, shared by tests that only read its results."""
    return make_source()
