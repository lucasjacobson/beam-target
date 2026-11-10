"""Shared helpers for the test suite."""

import os
from pathlib import Path

import pytest

from shine_neutron_source import ShineSource
from shine_neutron_source.source import DEFAULT_DATA_DIR

REPO_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_INPUT = REPO_DIR / "examples" / "generic.toml"


def data_path(fname):
    """Path of a file in the package's data directory."""
    return os.path.join(DEFAULT_DATA_DIR, fname)


# A small, fast case: a 50 cm, 5 torr target that the beam passes through (it does not stop),
# with a 1 cm wall radius so that part of the diverging beam is scraped
SMALL_CASE = dict(
    name="test",
    T_frac=0.5,
    voltage=280.0,
    current=10.0,
    pressure=[5.0],
    temperature=[25.0],
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


def make_source(**overrides):
    """Return a ShineSource for SMALL_CASE with some settings overridden."""
    return ShineSource(**{**SMALL_CASE, **overrides})


@pytest.fixture(scope="session")
def mixed_source():
    """A 50/50 D/T case, shared by tests that only read its results."""
    return make_source()
