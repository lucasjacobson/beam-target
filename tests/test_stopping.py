"""Stopping powers: SRIM and ICRU 49 sets, and per-atom D/T mixing."""

import re
import shutil

import numpy as np
import pytest
from conftest import data_path, make_source

from shine_neutron_source.source import SRIM_FILES, get_atom_density


def _srim_linear_per_atom(path):
    """dE/dx per unit atom density [keV cm^2] using only the SRIM file's own numbers:
    the table times its keV/micron factor, divided by its own atom density."""
    text = open(path).read()
    n_file = float(re.search(r"=\s*(\S+)\s*atoms/cm3", text).group(1))
    kev_per_um = float(re.search(r"^\s*(\S+)\s+keV/micron", text, re.M).group(1))
    unit = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("-"):
            inside = not inside
        elif inside:
            k = line.split()
            rows.append([float(k[0]) * unit[k[1]], float(k[2]) + float(k[3])])
    a = np.array(rows)
    return a[:, 0], a[:, 1] * kev_per_um * 1e4 / n_file


@pytest.mark.parametrize("T_frac, isotope", [(0.0, "D"), (1.0, "T")])
def test_energy_loss_matches_srim(T_frac, isotope):
    """The code's dE/dx in pure D or T agrees with SRIM's own linear stopping, scaled to the
    same atom density, without using any of the code's mass conversions."""
    source = make_source(T_frac=T_frac)
    E_ref, per_atom_ref = _srim_linear_per_atom(data_path(SRIM_FILES[isotope]))
    E = np.array([2, 10, 50, 100, 200, 300, 400.0])
    code = np.interp(E, source.sp_data["E"], source.sp_data["eps"])
    ref = np.interp(E, E_ref, per_atom_ref)
    np.testing.assert_allclose(code, ref, rtol=1e-4)


def test_mixing_is_per_atom():
    """Stopping in a D/T mix is (1 - T_frac) x deuterium + T_frac x tritium, per atom."""
    eps = {f: make_source(T_frac=f).sp_data["eps"] for f in (0.0, 0.3, 1.0)}
    np.testing.assert_allclose(eps[0.3], 0.7 * eps[0.0] + 0.3 * eps[1.0], rtol=1e-12)


def test_same_pressure_same_energy_loss():
    """D and T have one electron each, so dE/dx at 100 keV agrees within 0.1 %."""
    loss = [
        np.interp(100.0, s.sp_data["E"], s.sp_data["eps"])
        for s in map(lambda f: make_source(T_frac=f), (0.0, 1.0))
    ]
    assert loss[0] == pytest.approx(loss[1], rel=1e-3)


def test_atom_density():
    # 1 atm at 0 C: Loschmidt number x 2 atoms per molecule
    assert get_atom_density(760.0, 0.0) == pytest.approx(2 * 2.686780111e19, rel=1e-9)


def test_icru49_close_to_srim():
    srim = make_source(T_frac=1.0, stopping_power="srim").sp_data
    icru = make_source(T_frac=1.0, stopping_power="icru49").sp_data
    assert icru["E"][0] >= 1.99  # ICRU 49 starts at 1 keV protons = 2 keV deuterons
    E = np.linspace(2, 400, 200)
    ratio = np.interp(E, srim["E"], srim["eps"]) / np.interp(E, icru["E"], icru["eps"])
    assert np.all((ratio > 0.96) & (ratio < 1.01))
    assert np.all(ratio[E > 200] < 0.98)  # SRIM is 2-3.5 % lower where the beam enters


def test_icru49_slows_beam_faster():
    """ICRU 49 stopping is higher, so the beam leaves the (thin) target with less energy."""
    srim, icru = make_source(stopping_power="srim"), make_source(stopping_power="icru49")
    assert icru.EB_grid[-1] < srim.EB_grid[-1]
    assert 0.97 < (280 - srim.EB_grid[-1]) / (280 - icru.EB_grid[-1]) < 0.99


def test_invalid_stopping_power_rejected():
    with pytest.raises(ValueError, match="stopping_power must be one of"):
        make_source(stopping_power="pstar")


def test_swapped_srim_files_rejected(tmp_path):
    data_dir = tmp_path / "data"
    shutil.copytree(data_path(""), data_dir)
    shutil.copyfile(data_dir / SRIM_FILES["T"], data_dir / SRIM_FILES["D"])
    with pytest.raises(ValueError, match="SRIM target is T, expected D"):
        make_source(data_dir=str(data_dir))
