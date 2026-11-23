"""The ICRU 49 stopping model (stopping.py): NIST tables, the model for other elements, ZBL."""

import os

import numpy as np
import pytest

from beamtarget import stopping
from beamtarget.nuclear import BEAMS, atomic_mass
from beamtarget.source import DEFAULT_DATA_DIR, SRIM_STOPPING_FILE, read_srim_h5


@pytest.fixture(scope="module")
def data():
    return stopping.load(DEFAULT_DATA_DIR)


@pytest.mark.parametrize("key, median, worst", [("p", 0.025, 0.30), ("a", 0.040, 0.16)])
def test_model_matches_nist(data, key, median, worst):
    """For the 25 elements NIST tabulates, the model (used for every other element) is close to
    ICRU 49 from 10 keV/u up: median worst deviation 2.1% for protons and 3.4% for alphas. The
    largest is carbon at low energy (26%): NIST's is graphite, the parameterization amorphous."""
    per_u = 4.0 if key == "a" else 1.0
    deviations = []
    for Z, (E, S) in data[key].items():
        keep = (E >= 10 * per_u) & (E <= 1e5 * per_u)
        deviations.append(np.max(np.abs(stopping.model(data, Z, key, E[keep]) / S[keep] - 1)))
    assert np.median(deviations) < median
    assert max(deviations) < worst


@pytest.mark.parametrize("key", ["p", "a"])
def test_nist_tables_used_as_printed(data, key):
    for Z, (E, S) in data[key].items():
        np.testing.assert_allclose(stopping.electronic(data, Z, key, E), S, rtol=1e-12)


@pytest.mark.parametrize("beam", list(BEAMS))
@pytest.mark.parametrize("Z", [3, 5, 40, 73])
def test_model_continuous_at_join(data, beam, Z):
    """The parameterization and the scaled Bethe formula meet at JOIN_ENERGY (protons)."""
    E_join = stopping.JOIN_ENERGY * BEAMS[beam].mass / stopping.M_PROTON
    below, above = stopping.model(data, Z, beam, E_join * np.array([1 - 1e-9, 1 + 1e-9]))
    assert above == pytest.approx(below, rel=1e-7)


@pytest.mark.parametrize("beam, reference", [("d", "p"), ("t", "p"), ("h", "a")])
@pytest.mark.parametrize("Z", [1, 4, 5, 22, 73])
def test_same_velocity_scaling(data, beam, reference, Z):
    """Deuterons and tritons stop as protons, and helions as alphas, at the same velocity."""
    E = np.geomspace(10.0, 1e5, 30)  # [keV]
    E_ref = E * BEAMS[reference].mass / BEAMS[beam].mass
    np.testing.assert_allclose(
        stopping.electronic(data, Z, beam, E),
        stopping.electronic(data, Z, reference, E_ref),
        rtol=1e-12,
    )


@pytest.mark.parametrize("beam", list(BEAMS))
def test_every_element_finite_and_positive(data, beam):
    E = stopping.energy_grid(data, beam)
    assert np.all(np.diff(E) > 0)
    for Z in range(1, 93):
        S = stopping.electronic(data, Z, beam, E)
        assert np.all(np.isfinite(S) & (S > 0)), Z
        N = stopping.nuclear(beam, Z, 2.5 * Z, E)  # A typical mass [u]
        assert np.all(np.isfinite(N) & (N > 0)), Z


@pytest.mark.parametrize(
    "atom, Z, A, phase", [("T", 1, 3, "gas"), ("Be", 4, 9, "solid"), ("Ti", 22, 48, "solid")]
)
def test_nuclear_stopping_matches_srim(atom, Z, A, phase):
    """SRIM's nuclear stopping is also the ZBL universal formula: within 1.5% from 1 keV to
    1 MeV (SRIM uses the natural atomic weight, and its own fit to it)."""
    E, _, srim = read_srim_h5(os.path.join(DEFAULT_DATA_DIR, SRIM_STOPPING_FILE), particle="d")
    keep = (E > 1) & (E < 1000)
    ratio = stopping.nuclear("d", Z, atomic_mass(Z, A), E[keep]) / srim[phase][atom][1][keep]
    assert np.all((ratio > 0.98) & (ratio < 1.0))


def test_nuclear_stopping_is_small_at_high_energy(data):
    """Above 100 keV/u, nuclear stopping of hydrogen ions is below 1% of electronic, even in U."""
    E = np.geomspace(200.0, 1e4, 20)  # Deuterons [keV]
    for Z, A in [(1, 2), (22, 48), (92, 238)]:
        ratio = stopping.nuclear("d", Z, atomic_mass(Z, A), E) / stopping.electronic(
            data, Z, "d", E
        )
        assert np.all(ratio < 0.01), Z
