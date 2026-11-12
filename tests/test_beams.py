"""Proton and triton beams: the reactions other than D(d,n)3He and T(d,n)4He."""

import re

import numpy as np
import pytest
from conftest import make_source

from beamtarget import Material
from beamtarget.nuclear import REACTIONS, beam_key, mean_atomic_mass

LI7 = Material("7Li", "solid", atoms={"7Li": 1}, density=0.534)

# A thin lithium target: 10 um of 7Li metal, which takes about 60 keV off a 2.5 MeV proton
LITHIUM_CASE = dict(
    beam="proton",
    material=LI7,
    beam_energy_entry=2500.0,
    pressure=[None],
    temperature=[None],
    length=[1e-3],
    divergence=[0.0],
    position_bins=[20],
    position_bins_out=[5],
)


def _gas(formula):
    return Material(formula, "gas", molecules={formula: 1})


def _sdef_energies(si):
    """Return the energy segments [(E_min, E_max), ...] of an SI card such as
    " 0.1 9i 0.2 1.5 4i 2.0"."""
    tokens = si.split()
    return [(float(tokens[i]), float(tokens[i + 2])) for i in range(0, len(tokens), 3)]


def test_proton_on_lithium():
    s = make_source(**LITHIUM_CASE)
    # No tritium in the target, so no T(p,n) reaction at all
    assert list(s.reactions) == ["p7Li_n0", "p7Li_n1"]
    assert s.reactions_present() == ["p7Li_n0", "p7Li_n1"]
    assert "pT" not in s.yields
    # Below about 3 MeV the 7Be* line is a small fraction of the ground-state one
    assert 0 < s.yields["p7Li_n1"] < 0.3 * s.yields["p7Li_n0"]
    # At 2.5 MeV, n0 neutrons run from about 0.78 MeV forward to 0.24 MeV backward (less
    # as the beam slows), n1 neutrons below 0.4 MeV
    E0 = s.neutron_energy["p7Li_n0"]
    assert 0.2 < np.nanmin(E0) and np.nanmax(E0) < 0.79
    assert np.nanmax(s.neutron_energy["p7Li_n1"]) < 0.4
    assert s.backstop_power > 0


def test_natural_lithium_is_mostly_7Li():
    """Natural Li gives the 7Li yield times its 7Li abundance, at the same atom density."""
    natural = Material("Li", "solid", atoms={"Li": 1}, density=0.534)
    natural = Material(
        "Li", "solid", atoms={"Li": 1}, density=0.534 * natural.mean_mass / LI7.mean_mass
    )
    enriched = make_source(**LITHIUM_CASE)
    s = make_source(**{**LITHIUM_CASE, "material": natural})
    np.testing.assert_allclose(s.atom_density_per_region, enriched.atom_density_per_region)
    ratio = s.yields["p7Li_n0"] / enriched.yields["p7Li_n0"]
    assert ratio == pytest.approx(natural.isotope_fractions[(3, 7)], rel=1e-9)
    assert mean_atomic_mass(natural.isotope_fractions) == natural.mean_mass


def test_lithium_limits():
    """Above 3.70 MeV, 7Li(p,n3He)4He opens: a warning, not an error. Above the end of the
    Legendre data (23 MeV) there is nothing to calculate with: an error."""
    s = make_source(**{**LITHIUM_CASE, "beam_energy_entry": 3700.0})
    assert len(s.warnings) == 1 and s.warnings[0].startswith("7Li(p,n3He)4He is open")
    assert s.yield_total > 0
    with pytest.raises(ValueError, match="above the end of the 7Li"):
        make_source(**{**LITHIUM_CASE, "beam_energy_entry": 23500.0})
    with pytest.raises(ValueError, match="no neutrons are produced"):
        make_source(
            **{**LITHIUM_CASE, "beam_energy_entry": 1850.0}
        )  # Below the 1.8808 MeV threshold


@pytest.mark.parametrize(
    "energy, expected",
    [
        (280.0, ["13C(d,n)14N"]),  # 12C(d,n) is closed below 329 keV
        (1000.0, ["12C(d,n)13N", "13C(d,n)14N"]),
        (3000.0, ["12C(d,n)13N", "12C(d,np)12C", "13C(d,n)14N", "13C(d,np)13C"]),
    ],
)
def test_unmodelled_channels_warned(energy, expected):
    """Deuterons on CD4: the carbon (d,n) channels are open but not modeled, so the case
    runs (with the DD neutrons) and warns about each."""
    s = make_source(
        material=_gas("CD4"), beam_energy_entry=energy, position_bins=[4], position_bins_out=[2]
    )
    assert [w.split(" is open")[0] for w in s.warnings] == expected
    assert s.yields["DD"] > 0
    assert "warnings" in s.get_summary()


def test_coulomb_barrier_suppresses_warnings():
    """2% argon at 280 keV: 40Ar(d,n) is exothermic, but the beam is at 6% of the Coulomb
    barrier, so it is negligible and not warned about. At 1 MeV it is."""
    gas = Material("DT_argon", "gas", molecules={"D2": 0.49, "T2": 0.49, "Ar": 0.02})
    assert make_source(material=gas).warnings == []
    s = make_source(
        material=gas, beam_energy_entry=1000.0, position_bins=[4], position_bins_out=[2]
    )
    assert any(w.startswith("40Ar(d,n)41K") for w in s.warnings)


def test_no_warnings_for_the_small_case(mixed_source):
    assert mixed_source.warnings == []
    assert "warnings" not in mixed_source.get_summary()


def test_overlapping_lines_share_sdef_segments():
    """The n0 and n1 lines overlap in energy, so the SDEF energy distributions add them on one
    segment: energies must increase through each SI card."""
    for angle_bin_width_out in (180.0, 10.0):
        s = make_source(**{**LITHIUM_CASE, "angle_bin_width_out": angle_bin_width_out})
        sdef = s.get_mcnp_sdef()
        cards = re.findall(r"^  si(?:4 |5\d\d)(.*)$", sdef, flags=re.M)
        assert cards
        for si in cards:
            bounds = np.ravel(_sdef_energies(si))
            assert np.all(np.diff(bounds) > 0), si


def test_proton_on_tritium():
    s = make_source(beam="proton", material=_gas("T2"), beam_energy_entry=2000.0)
    assert s.reactions_present() == ["pT"]
    # T(p,n)3He at 2 MeV: about 1.2 MeV neutrons forward
    E = s.neutron_energy["pT"]
    assert 1.0 < np.nanmax(E) < 1.25


def test_inverse_reaction_uses_flipped_data():
    """D(t,n)4He at E_t has the T(d,n)4He cross section at E_d = E_t m_d / m_t, with the CM
    angle measured from the other particle."""
    thin = dict(length=[1e-6], position_bins=[1], position_bins_out=[1], divergence=[0.0])
    E_t = 600.0
    E_d = REACTIONS["tD"].data_energy(E_t)
    tD = make_source(beam="triton", material=_gas("D2"), beam_energy_entry=E_t, **thin)
    dT = make_source(beam="deuteron", material=_gas("T2"), beam_energy_entry=E_d, **thin)
    np.testing.assert_allclose(tD.cm_xs["tD"][0], dT.cm_xs["DT"][0][::-1], rtol=1e-6)


def test_triton_on_hydrogen_is_a_forward_cone():
    """H(t,n)3He just above its 3.05 MeV threshold sends every neutron forward in a cone;
    the SDEF still has a valid energy distribution for the empty angle bins."""
    s = make_source(beam="triton", material=_gas("1H2"), beam_energy_entry=3300.0, length=[1.0])
    assert s.reactions_present() == ["tH"]
    angular = np.sum(s.neutrons["tH"], (0, 2))
    assert angular[0] > 0 and np.all(angular[-6:] == 0)  # Nothing beyond 120 degrees
    assert "no neutrons, never sampled" in s.get_mcnp_sdef()


def test_natural_hydrogen_holds_deuterium():
    """Natural hydrogen is 0.0145% deuterium, so a triton beam also makes D(t,n)4He
    neutrons in it, at every angle."""
    s = make_source(beam="triton", material=_gas("H2"), beam_energy_entry=3300.0, length=[1.0])
    assert s.reactions_present() == ["tD", "tH"]
    assert np.all(np.sum(s.neutrons["tD"], (0, 2)) > 0)


def test_proton_stopping_in_hydrogen_icru49():
    """ICRU 49 is protons in hydrogen: it applies to a proton beam directly."""
    srim = make_source(beam="proton", material=_gas("T2"), beam_energy_entry=2000.0)
    icru = make_source(
        beam="proton", material=_gas("T2"), beam_energy_entry=2000.0, stopping_power="icru49"
    )
    assert icru.beam_energy_grid[-1] == pytest.approx(srim.beam_energy_grid[-1], rel=0.01)


@pytest.mark.parametrize("beam", ["x", "p", "d", "Proton", "He3"])
def test_unknown_beam_rejected(beam):
    """Beam particles are named by nuclide or particle name only (not the one-letter keys)."""
    with pytest.raises(ValueError, match='beam must be one of "1H", "proton", "2H"'):
        make_source(beam=beam)


@pytest.mark.parametrize(
    "nuclide, name, key",
    [("1H", "proton", "p"), ("2H", "deuteron", "d"), ("3H", "triton", "t"),
     ("3He", "helion", "h"), ("4He", "alpha", "a")],
)  # fmt: skip
def test_beam_names(nuclide, name, key):
    """Each beam particle can be named by its nuclide or its particle name."""
    assert beam_key(nuclide) == beam_key(name) == key
