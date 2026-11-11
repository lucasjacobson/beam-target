"""Masses, reactions and relativistic two-body kinematics (nuclear.py)."""

import numpy as np
import pytest

from shine_neutron_source.nuclear import (
    BEAMS,
    REACTIONS,
    atom_name,
    cm_energy,
    cm_to_lab,
    coulomb_barrier,
    isotope_fractions,
    mean_atomic_mass,
    natural_abundances,
    neutron_channels,
    nuclear_mass,
    parse_atom,
)

COS = np.cos(np.radians(np.linspace(0.05, 179.95, 1800)))
BEAM_ENERGIES = [1e-6, 0.01, 0.1, 0.3, 1.0, 3.0]  # MeV
EXOTHERMIC = ["DD", "DT", "tD"]
ENDOTHERMIC = ["pT", "p7Li_n0", "p7Li_n1", "tH"]

# --- Masses and isotopes -----------------------------------------------------------------------


def test_masses_match_codata():
    """Bare-nucleus masses from AME2020 atomic masses agree with the CODATA 2022 nuclear
    mass-energies [MeV] to within 0.1 eV."""
    codata = {
        (0, 1): 939.56542194,
        (1, 1): 938.27208943,
        (1, 2): 1875.61294500,
        (1, 3): 2808.92113668,
        (2, 3): 2808.39161112,
        (2, 4): 3727.3794118,
    }
    for (Z, A), mass in codata.items():
        assert nuclear_mass(Z, A) == pytest.approx(mass, abs=1e-7), (Z, A)


@pytest.mark.parametrize(
    "name, za, canonical",
    [
        ("H", (1, None), "H"),
        ("1H", (1, 1), "1H"),
        ("2H", (1, 2), "D"),
        ("D", (1, 2), "D"),
        ("T", (1, 3), "T"),
        ("3He", (2, 3), "3He"),
        ("Li", (3, None), "Li"),
        ("7Li", (3, 7), "7Li"),
    ],
)
def test_atom_names(name, za, canonical):
    assert parse_atom(name) == za
    assert atom_name(*za) == canonical


@pytest.mark.parametrize("name", ["Xx", "li", "7", "99Li", "Li7"])
def test_bad_atom_names(name):
    with pytest.raises(ValueError):
        parse_atom(name)


def test_natural_abundances():
    """IUPAC 2009 representative isotopic compositions (data/natural_abundances.csv)."""
    li = isotope_fractions("Li")
    assert li == {(3, 6): 0.0759, (3, 7): 0.9241}
    assert isotope_fractions("H") == {(1, 1): 0.999885, (1, 2): 0.000115}
    assert isotope_fractions("7Li") == {(3, 7): 1.0}
    assert mean_atomic_mass(li) == pytest.approx(6.940, abs=1e-3)  # Standard atomic weight
    assert isotope_fractions("U")[(92, 235)] == 0.007204
    assert isotope_fractions("Ta")[(73, 180)] == 0.00012  # 180mTa, by its mass number
    with pytest.raises(ValueError, match="no natural isotopes"):
        isotope_fractions("Tc")


def test_abundance_table_complete():
    """Every element in the table sums to 1, and every isotope is in the mass table."""
    for Z, isotopes in natural_abundances().items():
        assert sum(isotopes.values()) == pytest.approx(1, abs=1e-6), Z
        for A in isotopes:
            assert nuclear_mass(Z, A) > 0


# --- Reactions ---------------------------------------------------------------------------------


def test_reactions_per_beam():
    by_beam = {b: [k for k, r in REACTIONS.items() if r.beam == b] for b in BEAMS}
    assert by_beam == {"p": ["pT", "p7Li_n0", "p7Li_n1"], "d": ["DD", "DT"], "t": ["tD", "tH"]}


@pytest.mark.parametrize(
    "rxn, Q",
    [
        ("DD", 3.268857),
        ("DT", 17.589248),
        ("pT", -0.763807),
        ("p7Li_n0", -1.644420),
        ("p7Li_n1", -1.644420 - 0.42908),
        ("tD", 17.589248),
        ("tH", -0.763807),
    ],
)
def test_q_values(rxn, Q):
    assert REACTIONS[rxn].q_value == pytest.approx(Q, abs=2e-6)


@pytest.mark.parametrize(
    "rxn, threshold, complete",
    [
        ("DD", 0.0, 4.4505),
        ("DT", 0.0, 3.7109),
        ("pT", 1.0190, 8.3543),
        ("p7Li_n0", 1.8808, 3.6961),
        ("p7Li_n1", 2.3716, 3.6961),
        ("tD", 0.0, 5.5574),
        ("tH", 3.0507, 25.0105),
    ],
)
def test_thresholds_and_limits(rxn, threshold, complete):
    """Thresholds and the energies where another neutron channel opens [MeV] (lab beam
    energy). Standard values: 7Li(p,n) at 1.8807 MeV, T(p,n) at 1.0190 MeV."""
    r = REACTIONS[rxn]
    assert r.threshold == pytest.approx(threshold, abs=1e-4)
    assert r.complete_up_to()[0] == pytest.approx(complete, abs=1e-4)


def test_inverse_data_energy():
    """D(t,n)4He at E_t uses the T(d,n)4He data at E_d = E_t m_d / m_t (same CM energy)."""
    r = REACTIONS["tD"]
    assert r.data_energy(3.0) == pytest.approx(3.0 * nuclear_mass(1, 2) / nuclear_mass(1, 3))
    assert REACTIONS["DT"].data_energy(3.0) == 3.0
    # The threshold of the inverse reaction maps onto that of the direct one
    assert REACTIONS["tH"].data_energy(REACTIONS["tH"].threshold) == pytest.approx(
        REACTIONS["pT"].threshold, rel=1e-9
    )


# --- Kinematics -----------------------------------------------------------------------------


@pytest.mark.parametrize("rxn, E_rest", [("DD", 2.4487), ("DT", 14.0284)])
def test_neutron_energy_at_rest(rxn, E_rest):
    """Neutron energy for a reaction at rest (standard values, MeV)."""
    T3, _ = cm_to_lab(1e-12, [0.0], *REACTIONS[rxn].masses())
    assert T3[0] == pytest.approx(E_rest, abs=1e-4)


def _residual_mass(T1, T3, cos_lab, m1, m2, m3):
    """Invariant mass of what is left after the neutron, from 4-momentum conservation."""
    E1 = T1 + m1
    p1 = np.sqrt(E1**2 - m1**2)
    E3 = T3 + m3
    p3 = np.sqrt(E3**2 - m3**2)
    E4 = E1 + m2 - E3
    p4_sq = p1**2 + p3**2 - 2 * p1 * p3 * cos_lab
    return np.sqrt(E4**2 - p4_sq)


@pytest.mark.parametrize("rxn", list(REACTIONS))
@pytest.mark.parametrize("above", [1e-4, 0.01, 0.1, 1.0, 3.0])
def test_four_momentum_conserved(rxn, above):
    """Every CM angle gives a lab energy and angle that conserve energy and momentum, near
    and well above the threshold (where endothermic reactions are double-valued in the lab)."""
    m1, m2, m3, m4 = REACTIONS[rxn].masses()
    T1 = REACTIONS[rxn].threshold + above
    T3, cos_lab = cm_to_lab(T1, COS, m1, m2, m3, m4)
    assert np.all(np.isfinite(T3)) and np.all(T3 > 0)
    assert np.all(np.abs(cos_lab) <= 1)
    np.testing.assert_allclose(_residual_mass(T1, T3, cos_lab, m1, m2, m3), m4, rtol=1e-12)


@pytest.mark.parametrize("rxn", EXOTHERMIC)
@pytest.mark.parametrize("T1", BEAM_ENERGIES)
def test_exothermic_angles_map_one_to_one(rxn, T1):
    """Exothermic reactions: 0 and 180 degrees map onto themselves, lab angle grows with CM
    angle, and forward neutrons are faster."""
    T3, cos_lab = cm_to_lab(T1, COS, *REACTIONS[rxn].masses())
    _, ends = cm_to_lab(T1, [1.0, -1.0], *REACTIONS[rxn].masses())
    np.testing.assert_allclose(ends, [1.0, -1.0], atol=1e-12)
    assert np.all(np.diff(cos_lab) < 0)
    assert np.all(np.diff(T3) < 0)


@pytest.mark.parametrize("rxn", ENDOTHERMIC)
def test_endothermic_near_threshold_is_forward(rxn):
    """Just above threshold every neutron goes forward in the lab, in a narrow cone, and a
    lab angle has two neutron energies (from CM angles forward and backward)."""
    T1 = REACTIONS[rxn].threshold * (1 + 1e-4)
    T3, cos_lab = cm_to_lab(T1, COS, *REACTIONS[rxn].masses())
    assert np.all(cos_lab > 0.9)
    assert T3[0] > T3[-1]  # Forward in the CM: the fast branch
    assert cos_lab[0] == pytest.approx(cos_lab[-1], abs=1e-9)  # Both along the beam


@pytest.mark.parametrize("rxn", ENDOTHERMIC)
def test_below_threshold_is_nan(rxn):
    T3, cos_lab = cm_to_lab(REACTIONS[rxn].threshold * 0.999, COS, *REACTIONS[rxn].masses())
    assert np.all(np.isnan(T3)) and np.all(np.isnan(cos_lab))


# --- Other neutron channels ----------------------------------------------------------------


@pytest.mark.parametrize(
    "beam, target, expected",
    [
        ("d", (6, 12), [(0.3288, "12C(d,n)13N"), (2.5982, "12C(d,np)12C")]),
        ("d", (6, 13), [(0.0, "13C(d,n)14N")]),
        ("p", (4, 9), [(1.8508, "9Be(p,np)8Be"), (2.0577, "9Be(p,n)9B")]),
        ("p", (1, 2), [(3.3388, "D(p,np)H")]),
        ("p", (2, 3), []),  # No neutron channel with a bound residual
    ],
)
def test_generic_neutron_channels(beam, target, expected):
    """The lowest generic channels and their thresholds [MeV]. Standard values: 12C(d,n) at
    0.328 MeV, 9Be(p,np) at 1.851 and 9Be(p,n) at 2.057 MeV, deuteron break-up by protons at 3.34 MeV."""
    channels = neutron_channels(beam, target)[: len(expected)]
    assert [label for _, label in channels] == [label for _, label in expected]
    for (threshold, _), (value, _) in zip(channels, expected):
        assert threshold == pytest.approx(value, abs=1e-4)


def test_coulomb_barrier():
    """About 0.44 MeV for d + D and 4.3 MeV for d + 40Ar (R0 = 1.3 fm)."""
    assert coulomb_barrier(1, 2, 1, 2) == pytest.approx(0.44, abs=0.01)
    assert coulomb_barrier(1, 2, 18, 40) == pytest.approx(4.26, abs=0.01)
    # Centre-of-mass energy: about E_lab * m2 / (m1 + m2)
    assert cm_energy("d", (1, 3), 0.1) == pytest.approx(0.1 * 3 / 5, rel=2e-3)
