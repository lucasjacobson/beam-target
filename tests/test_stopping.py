"""Stopping powers: SRIM and ICRU 49 sets, and per-atom mixing over a material's atoms."""

import numpy as np
import pytest
from conftest import data_path, make_source

from beamtarget.source import (
    SRIM_STOPPING_FILE,
    get_atom_density,
    get_molecule_density,
    read_srim_h5,
)
from beamtarget.units import KEV, TORR, ZERO_CELSIUS


def _stopping(source, region=-1):
    """(E [keV], eps [keV cm^2 per atom]) of one region."""
    E, eps = source.stopping_per_region[region]
    return E / KEV, eps / KEV


@pytest.mark.parametrize("T_frac, isotope", [(0.0, "D"), (1.0, "T")])
def test_pure_gas_stopping_is_srim(T_frac, isotope):
    """In pure D or T gas the code's stopping per atom is SRIM's gas-phase value."""
    E_srim, _, srim = read_srim_h5(data_path(SRIM_STOPPING_FILE))
    elec, nuc = srim["gas"][isotope]
    E, eps = _stopping(make_source(T_frac=T_frac))
    np.testing.assert_allclose(E, E_srim, rtol=1e-15)
    np.testing.assert_allclose(eps, elec + nuc, rtol=1e-14)


def test_mixing_is_per_atom():
    """Stopping in a D/T mix is (1 - T_frac) x deuterium + T_frac x tritium, per atom."""
    eps = {f: _stopping(make_source(T_frac=f))[1] for f in (0.0, 0.3, 1.0)}
    np.testing.assert_allclose(eps[0.3], 0.7 * eps[0.0] + 0.3 * eps[1.0], rtol=1e-12)


def test_same_pressure_same_energy_loss():
    """D and T have one electron each, so dE/dx at 100 keV agrees within 0.1%."""
    loss = [np.interp(100.0, *_stopping(make_source(T_frac=f))) for f in (0.0, 1.0)]
    assert loss[0] == pytest.approx(loss[1], rel=1e-3)


def test_gas_densities():
    # 1 atm at 0 C: Loschmidt number, x 2 atoms per molecule for D2 or T2
    assert get_molecule_density(101325.0, ZERO_CELSIUS) == pytest.approx(2.686780111e19, rel=1e-9)
    assert get_atom_density(101325.0, ZERO_CELSIUS) == pytest.approx(2 * 2.686780111e19, rel=1e-9)
    s = make_source(pressure=[5.0], temperature=[25.0])  # torr, degC
    expected = get_atom_density(5 * TORR, 25 + ZERO_CELSIUS)
    assert s.atom_density_per_region[0] == pytest.approx(expected, rel=1e-14)


def test_icru49_close_to_srim():
    E_s, eps_s = _stopping(make_source(T_frac=1.0, stopping_power="srim"))
    E_i, eps_i = _stopping(make_source(T_frac=1.0, stopping_power="icru49"))
    assert E_i[0] >= 1.99  # ICRU 49 starts at 1 keV protons = 2 keV deuterons
    E = np.linspace(2, 400, 200)
    ratio = np.interp(E, E_s, eps_s) / np.interp(E, E_i, eps_i)
    assert np.all((ratio > 0.96) & (ratio < 1.01))
    assert np.all(ratio[E > 200] < 0.98)  # SRIM is 2-3.5% lower where the beam enters


def test_icru49_slows_beam_faster():
    """ICRU 49 stopping is higher, so the beam leaves the (thin) target with less energy."""
    srim, icru = make_source(stopping_power="srim"), make_source(stopping_power="icru49")
    assert icru.beam_energy_grid[-1] < srim.beam_energy_grid[-1]
    E_srim, E_icru = srim.beam_energy_grid[-1] / KEV, icru.beam_energy_grid[-1] / KEV
    assert 0.97 < (280 - E_srim) / (280 - E_icru) < 0.99


def test_invalid_stopping_power_rejected():
    with pytest.raises(ValueError, match="stopping_power must be one of"):
        make_source(stopping_power="pstar")
