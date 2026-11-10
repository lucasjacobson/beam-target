"""Relativistic two-body kinematics of D(d,n)3He and T(d,n)4He."""

import numpy as np
import pytest

from shine_neutron_source.source import (
    M_H2,
    M_H3,
    M_HE3,
    M_HE4,
    M_N1,
    Q_DD,
    Q_DT,
    REACTIONS,
    two_body_kinematics,
)

COS = np.cos(np.radians(np.linspace(0.05, 179.95, 1800)))
BEAM_ENERGIES = [1e-6, 0.01, 0.1, 0.3, 1.0, 10.0, 40.0]  # MeV


def test_q_values():
    assert Q_DD == pytest.approx(3.26885694, abs=1e-8)
    assert Q_DT == pytest.approx(17.58924794, abs=1e-8)
    # Q is consistent with the masses used in the kinematics
    for m1, m2, m3, m4, Q in REACTIONS.values():
        assert (m1 + m2) - (m3 + m4) == Q


@pytest.mark.parametrize("rxn, E_rest", [("DD", 2.4487), ("DT", 14.0284)])
def test_neutron_energy_at_rest(rxn, E_rest):
    """Neutron energy for a reaction at rest (standard values, MeV)."""
    m1, m2, m3, m4, _ = REACTIONS[rxn]
    T3, _, _ = two_body_kinematics(1e-12, np.array([0.0]), m1, m2, m3, m4)
    assert T3[0] == pytest.approx(E_rest, abs=1e-4)


def _independent_energy(m1, m2, m3, m4, T1, cos):
    """Neutron lab energy from 4-momentum conservation, solved as a quadratic in p3."""
    E1 = T1 + m1
    p1 = np.sqrt(E1**2 - m1**2)
    Et = E1 + m2
    A = Et**2 - p1**2 + m3**2 - m4**2
    a = 4 * (Et**2 - p1**2 * cos**2)
    b = -4 * A * p1 * cos
    c = 4 * Et**2 * m3**2 - A**2
    p3 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return np.sqrt(p3**2 + m3**2) - m3


@pytest.mark.parametrize("rxn", ["DD", "DT"])
@pytest.mark.parametrize("T1", BEAM_ENERGIES)
def test_energy_matches_independent_solution(rxn, T1):
    m1, m2, m3, m4, _ = REACTIONS[rxn]
    T3, _, _ = two_body_kinematics(T1, COS, m1, m2, m3, m4)
    np.testing.assert_allclose(T3, _independent_energy(m1, m2, m3, m4, T1, COS), atol=1e-10)


@pytest.mark.parametrize("rxn", ["DD", "DT"])
@pytest.mark.parametrize("T1", BEAM_ENERGIES)
def test_angle_conversion_and_jacobian(rxn, T1):
    m1, m2, m3, m4, _ = REACTIONS[rxn]
    # 0 and 180 degrees map onto themselves
    _, ends, _ = two_body_kinematics(T1, np.array([1.0, -1.0]), m1, m2, m3, m4)
    np.testing.assert_allclose(ends, [1.0, -1.0], atol=1e-12)
    # The Jacobian is d(cos_cm)/d(cos_lab): compare with a central difference
    cos = COS[np.abs(COS) < 0.999]
    _, _, jac = two_body_kinematics(T1, cos, m1, m2, m3, m4)
    h = 1e-6
    up = two_body_kinematics(T1, cos + h, m1, m2, m3, m4)[1]
    down = two_body_kinematics(T1, cos - h, m1, m2, m3, m4)[1]
    np.testing.assert_allclose(jac, (up - down) / (2 * h), rtol=1e-6)


@pytest.mark.parametrize("rxn", ["DD", "DT"])
@pytest.mark.parametrize("T1", [0.05, 0.3, 1.0, 40.0])
def test_jacobian_conserves_cross_section(rxn, T1):
    """A CM angular distribution integrated over lab solid angle equals its CM integral."""
    m1, m2, m3, m4, _ = REACTIONS[rxn]
    edges = np.linspace(0, np.pi, 100001)
    cos = np.cos(0.5 * (edges[1:] + edges[:-1]))
    d_omega = 2 * np.pi * (np.cos(edges[:-1]) - np.cos(edges[1:]))
    _, cos_cm, jac = two_body_kinematics(T1, cos, m1, m2, m3, m4)

    def f(mu):  # An arbitrary anisotropic CM distribution; its 4 pi integral is 4 pi (1 + 0.5/3)
        return 1 + 0.3 * mu + 0.5 * mu**2 - 0.2 * mu**3

    assert np.sum(f(cos_cm) * jac * d_omega) == pytest.approx(4 * np.pi * (1 + 0.5 / 3), rel=1e-8)


def test_forward_neutrons_are_faster():
    """With a moving beam, neutrons emitted forward are more energetic than backward ones."""
    for m1, m2, m3, m4, _ in REACTIONS.values():
        T3, _, _ = two_body_kinematics(0.3, COS, m1, m2, m3, m4)
        assert np.all(np.diff(T3) < 0)  # COS runs from 0 to 180 degrees


def test_masses_are_nuclear():
    """Kinematics use bare-nucleus masses (CODATA 2022 mass-energies, MeV)."""
    assert (M_N1, M_H2, M_HE3, M_H3, M_HE4) == (
        939.56542194,
        1875.61294500,
        2808.39161112,
        2808.92113668,
        3727.3794118,
    )
