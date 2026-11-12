"""Shared helpers for the test suite."""

import os
from pathlib import Path

import numpy as np
import pytest

from beamtarget import BeamTarget, Region
from beamtarget.source import DEFAULT_DATA_DIR, dt_gas

REPO_DIR = Path(__file__).resolve().parents[1]

# Ignore the user's own configuration file (their nuclear-data libraries), so the results do
# not depend on the machine: point it at a file that does not exist. Set before any source is
# made; tests of the configuration set it themselves (monkeypatch).
os.environ["BEAM_TARGET_CONFIG"] = str(REPO_DIR / "tests" / "no-user-config.toml")
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
    """Return a BeamTarget for SMALL_CASE with some settings overridden. Region settings are
    lists, one entry per region. T_frac=x is short for material=dt_gas(x), a D/T gas with that
    tritium fraction."""
    if "T_frac" in overrides:
        overrides["material"] = dt_gas(overrides.pop("T_frac"))
    settings = {**SMALL_CASE, **overrides}
    regions = make_regions(settings)
    rest = {k: v for k, v in settings.items() if k not in REGION_SETTINGS}
    return BeamTarget(regions=regions, **rest)


@pytest.fixture(scope="session")
def mixed_source():
    """A 50/50 D/T case, shared by tests that only read its results."""
    return make_source()


# --- Statistical checks of sampled source particles (OpenMC and MCNP sampling tests) ---------

MAX_Z = 5.0  # Largest deviation allowed in any bin, in standard deviations


def assert_counts_match(counts, probabilities, what):
    """Check sampled counts per bin against expected probabilities: none where the
    probability is zero, and within MAX_Z standard deviations everywhere else."""
    counts = np.asarray(counts, dtype=float).ravel()
    p = np.asarray(probabilities, dtype=float).ravel()
    p = p / p.sum()
    n = counts.sum()
    assert np.all(counts[p == 0] == 0), f"{what}: particles in bins with no neutrons"
    filled = (p > 0) & (p < 1)  # A bin holding everything carries no information
    if not np.any(filled):
        return
    z = (counts[filled] - n * p[filled]) / np.sqrt(n * p[filled] * (1 - p[filled]))
    assert np.max(np.abs(z)) < MAX_Z, f"{what}: max deviation {np.max(np.abs(z)):.1f} sigma"


def assert_mean_matches(values, mean, variance, what):
    """Check a sample mean against the expected mean, within MAX_Z standard errors."""
    error = np.sqrt(variance / len(values))
    z = (np.mean(values) - mean) / error
    assert abs(z) < MAX_Z, f"{what}: mean off by {z:.1f} sigma"
