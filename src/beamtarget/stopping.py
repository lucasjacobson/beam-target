"""Stopping powers of ICRU Report 49, for every element and beam particle (stopping_power =
"icru49"), with no SRIM data.

Electronic stopping per target atom:

- The 25 elements NIST tabulates (H, He, Be, C as graphite, N, O, Ne, Al, Si, Ar, Ti, Fe, Cu,
  Ge, Kr, Mo, Ag, Sn, Xe, Gd, W, Pt, Au, Pb, U): ICRU 49 itself, from NIST PSTAR (protons)
  and ASTAR (alpha particles), interpolated log-log. Deuterons and tritons use the proton
  values, and 3He ions the alpha values, at the same velocity (electronic stopping depends on
  the velocity and the charge, not the mass). Above the end of a NIST table (10 GeV protons,
  1 GeV alphas), the model below, scaled to meet the table there.
- Every other element: the model ICRU 49 and Geant4 use where there are no data. Up to
  JOIN_ENERGY (2 MeV for a proton, at the same velocity for the others), the ICRU 49 / Ziegler
  parameterizations (Geant4's G4BraggModel for protons, G4BraggIonModel for He ions); above
  it, Bethe's formula with Geant4's standard corrections (shell, Barkas, Bloch, and Mott), with
  the element's ICRU 37 mean excitation energy, scaled to meet the parameterization at
  JOIN_ENERGY (the scale fading as JOIN_ENERGY / energy, as in Geant4). The density effect is
  left out: it matters only above about 1 GeV per nucleon. Against the 25 NIST elements, the
  model alone is within 2.1% of ICRU 49 for protons and 3.4% for alphas (median over the
  elements of the largest difference from 10 keV/u to 100 MeV/u; up to about 10% for some
  elements, mostly heavy ones around 2-5 MeV/u, and 26% for carbon below 30 keV, where NIST's
  is graphite), as close as SRIM is to ICRU 49 (6.4% and 5.4%).

Nuclear stopping: the universal formula of Ziegler, Biersack and Littmark (ZBL, The Stopping
and Range of Ions in Solids, 1985), with each isotope's own mass.

The same values are used for gas and solid targets, and electronic stopping is the same for
every isotope of an element. The data come from DATA_FILE (see scripts/make_icru49_tables.py).
"""

import functools
import os

import h5py
import numpy as np

from .nuclear import BEAMS, U_MEV, nuclear_mass

DATA_FILE = "icru49_stopping.h5"
# Proton energy [keV] (2 MeV) at which the parameterizations hand over to Bethe's formula, and
# above which the shell correction is fully applied (8 MeV); other particles at the same velocity
JOIN_ENERGY = 2000.0
SHELL_FULL_ENERGY = 8000.0

M_PROTON = nuclear_mass(1, 1)  # [MeV]
M_ALPHA = nuclear_mass(2, 4)  # [MeV]
M_ELECTRON_EV = 510998.95  # [eV]
R_ELECTRON = 2.8179403262e-13  # Classical electron radius [cm]
FINE_STRUCTURE = 1 / 137.035999084
# 4 pi r_e^2 m_e c^2 [keV cm^2]: Bethe's stopping per electron is this z^2 / beta^2 x L
BETHE_CONSTANT = 4 * np.pi * R_ELECTRON**2 * M_ELECTRON_EV * 1e-3
# Standard atomic weights of the NIST elements (as ICRU 49 uses), to convert per gram to per atom
NIST_ATOMIC_WEIGHT = {1: 1.00794, 2: 4.002602, 4: 9.012182, 6: 12.0107, 7: 14.0067,
                      8: 15.9994, 10: 20.1797, 13: 26.981538, 14: 28.0855, 18: 39.948,
                      22: 47.867, 26: 55.845, 29: 63.546, 32: 72.64, 36: 83.798, 42: 95.94,
                      47: 107.8682, 50: 118.71, 54: 131.293, 64: 157.25, 74: 183.84,
                      78: 195.078, 79: 196.96655, 82: 207.2, 92: 238.02891}  # fmt: skip
AVOGADRO = 6.02214076e23


@functools.cache
def load(data_dir):
    """The contents of DATA_FILE in data_dir: {"p" and "a": {Z: (energy [keV], electronic
    stopping per atom [keV cm^2])} from PSTAR and ASTAR, "proton_fit", "helium_fit",
    "mean_excitation_energy" [eV] (index Z - 1), "barkas" ((W, value) pairs)}."""
    out = {}
    with h5py.File(os.path.join(data_dir, DATA_FILE), "r") as f:
        for key, group in [("p", "pstar"), ("a", "astar")]:
            tables = {}
            for name, data in f[group].items():
                Z = int(name)
                per_atom = NIST_ATOMIC_WEIGHT[Z] / AVOGADRO * 1e3  # MeV cm^2/g -> keV cm^2
                tables[Z] = (data[:, 0] * 1e3, data[:, 1] * per_atom)
            out[key] = tables
        for name in ("proton_fit", "helium_fit", "mean_excitation_energy", "barkas"):
            out[name] = f["geant4"][name][()]
    return out


def _reference(beam):
    """(table key, reference rest mass [MeV], charge) for a beam particle: protons for p, d,
    and t, alphas for 3He and 4He."""
    return ("a", M_ALPHA, 2) if BEAMS[beam].charge == 2 else ("p", M_PROTON, 1)


def _proton_fit(coefficients, Z, T):
    """ICRU 49 / Ziegler proton electronic stopping [eV/(1e15 atoms/cm^2)], T [keV/u]
    (Geant4's G4BraggModel::ElectronicStoppingPower)."""
    a = coefficients[Z - 1]
    T = np.asarray(T, dtype=float)
    T_low = 40.0 if Z == 6 else 10.0  # Below this, free-electron-gas scaling
    fac = np.where(T < T_low, np.sqrt(np.maximum(T, 0.0) / T_low), 1.0)
    T = np.maximum(T, T_low)
    slow = a[1] * T**0.45
    shigh = np.log(1.0 + a[3] / T + a[4] * T) * a[2] / T
    return np.maximum(slow * shigh * fac / (slow + shigh), 0.0)


def _helium_fit(coefficients, Z, E):
    """ICRU 49 / Ziegler He-ion electronic stopping [eV/(1e15 atoms/cm^2)], E [MeV] (the
    alpha's kinetic energy; Geant4's G4BraggIonModel::HeElectronicStoppingPower)."""
    a = coefficients[Z - 1]
    E = np.asarray(E, dtype=float)
    T = np.maximum(E, 0.001)
    slow = a[0] * (T * 1000.0) ** a[1]
    shigh = np.log(1.0 + a[3] / T + a[4] * T) * a[2] / T
    main = slow * shigh / (slow + shigh)
    # Below 1 keV: free-electron-gas scaling
    s1 = np.log(1.0 + a[3] * 1000.0 + a[4] * 0.001) * a[2] * 1000.0
    low = a[0] * s1 / (a[0] + s1) * np.sqrt(np.maximum(E, 0.0) * 1000.0)
    return np.where(E < 0.001, low, main)


def _barkas_b(Z):
    """Geant4's b parameter of the Barkas correction for an element."""
    for limit, b in [(1, 1.8), (2, 0.6), (10, 1.8), (17, 1.4), (18, 1.8), (25, 1.4), (50, 1.35)]:
        if Z <= limit:
            return b
    return 1.3


def bethe(data, Z, z, tau):
    """Bethe's electronic stopping [keV cm^2 per atom] of a bare particle of charge z in element
    Z, at tau = kinetic energy / rest mass, with Geant4's standard shell correction
    (G4EmCorrections::ShellCorrectionSTD) and its Barkas, Bloch, and Mott corrections."""
    excitation = data["mean_excitation_energy"][Z - 1]  # Mean excitation energy [eV]
    tau = np.asarray(tau, dtype=float)
    gamma = 1 + tau
    bg2 = tau * (tau + 2)
    beta2 = bg2 / gamma**2
    beta = np.sqrt(beta2)
    ba2 = beta2 / FINE_STRUCTURE**2
    # Shell correction: Bichsel's fit in powers of 1/(beta gamma)^2, applied fully above
    # SHELL_FULL_ENERGY and phased in logarithmically above JOIN_ENERGY (for a proton)
    rate = 0.001 * excitation
    coefficients = np.array([(0.422377 + 3.858019 * rate), (0.0304043 - 0.1667989 * rate),
                             (-0.00038106 + 0.00157955 * rate)]) * rate**2 / Z  # fmt: skip
    tau_full, tau_join = SHELL_FULL_ENERGY / 1000 / M_PROTON, JOIN_ENERGY / 1000 / M_PROTON
    bg2_full = tau_full * (tau_full + 2)
    full = bg2 >= bg2_full
    x = np.where(full, bg2, bg2_full)
    shell = sum(c / x ** (k + 1) for k, c in enumerate(coefficients))
    phase = np.log(np.maximum(tau, 1e-30) / tau_join) / np.log(tau_full / tau_join)
    shell = shell * np.where(full, 1.0, phase)
    # Barkas (z^3) correction
    if Z == 47:
        barkas = 0.006812 * beta**-0.9
    elif Z >= 64:
        barkas = 0.002833 * beta**-1.2
    else:
        X = ba2 / Z
        W = _barkas_b(Z) / np.sqrt(X)
        table = data["barkas"]
        value = np.interp(W, table[:, 0], table[:, 1])
        value = np.where(W > 10.0, value * 10.0 / W, value)
        barkas = value / (np.sqrt(Z * X) * X)
    barkas = 1.29 * z * barkas
    # Bloch (z^4) correction
    y2 = z**2 / ba2
    bloch = 1.0 / (1.0 + y2)
    for j in range(2, 200):
        bloch = bloch + 1.0 / (j * (j * j + y2))
    bloch = -y2 * bloch
    mott = np.pi * FINE_STRUCTURE * beta * z
    # Geant4: ln(2 mc^2 bg2 Tmax / I^2) - 2 beta^2 - 2 shell + 2 (Barkas + Bloch) + Mott, with
    # Tmax = 2 mc^2 bg2, times 2 pi r_e^2 mc^2 z^2 Z / beta^2
    L2 = (2 * np.log(2 * M_ELECTRON_EV * bg2 / excitation) - 2 * beta2 - 2 * shell
          + 2 * (barkas + bloch) + mott)  # fmt: skip
    return BETHE_CONSTANT / 2 * z**2 * Z / beta2 * L2


def model(data, Z, beam, E):
    """Electronic stopping [keV cm^2 per atom] of a beam particle at E [keV] in element Z from
    the parameterizations and Bethe's formula (see the module docstring)."""
    key, m_ref, z = _reference(beam)
    E_ref = np.asarray(E, dtype=float) * m_ref / BEAMS[beam].mass  # Same velocity
    m_ref_u = m_ref / U_MEV

    def fit(E_r):
        if key == "a":
            return _helium_fit(data["helium_fit"], Z, E_r / 1000.0) * 1e-18
        return _proton_fit(data["proton_fit"], Z, E_r / m_ref_u) * 1e-18

    E_join = JOIN_ENERGY * m_ref / M_PROTON
    out = np.array(fit(np.minimum(E_ref, E_join)), dtype=float)
    high = E_ref > E_join
    if np.any(high):
        f = fit(np.array([E_join]))[0] / bethe(data, Z, z, np.array([E_join / 1000 / m_ref]))[0]
        out[high] = bethe(data, Z, z, E_ref[high] / 1000 / m_ref) * (
            1 + (f - 1) * E_join / E_ref[high]
        )
    return out


def electronic(data, Z, beam, E):
    """Electronic stopping [keV cm^2 per atom] of a beam particle (a nuclear.BEAMS key) at E
    [keV] in element Z: ICRU 49 (NIST) where tabulated, else the model."""
    key, m_ref, _ = _reference(beam)
    E = np.asarray(E, dtype=float)
    if Z not in data[key]:
        return model(data, Z, beam, E)
    E_table, S_table = data[key][Z]
    E_ref = E * m_ref / BEAMS[beam].mass
    out = np.exp(np.interp(np.log(np.maximum(E_ref, 1e-30)), np.log(E_table), np.log(S_table)))
    above = E_ref > E_table[-1]
    if np.any(above):  # The model, scaled to meet the table at its end
        E_end = E_table[-1] * BEAMS[beam].mass / m_ref
        scale = S_table[-1] / model(data, Z, beam, np.array([E_end]))[0]
        out[above] = model(data, Z, beam, E[above]) * scale
    return out


def nuclear(beam, Z2, M2, E):
    """Nuclear stopping [keV cm^2 per atom] of a beam particle at E [keV] (lab) by a target
    atom of atomic number Z2 and mass M2 [u]: the ZBL universal formula."""
    Z1, M1 = BEAMS[beam].Z, BEAMS[beam].mass / U_MEV
    E = np.asarray(E, dtype=float)
    screening = Z1**0.23 + Z2**0.23
    eps = 32.53 * M2 * E / (Z1 * Z2 * (M1 + M2) * screening)  # Reduced energy
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = np.where(
            eps <= 30,
            np.log(1 + 1.1383 * eps) / (2 * (eps + 0.01321 * eps**0.21226 + 0.19593 * eps**0.5)),
            np.log(eps) / (2 * eps),
        )
    return 8.462 * Z1 * Z2 * M1 * sn / ((M1 + M2) * screening) * 1e-18


def energy_grid(data, beam, points_per_decade=40):
    """Beam energies [keV] for a stopping table: from the start of the NIST tables (1 keV for a
    proton or an alpha, at the same velocity for the others) to their end."""
    key, m_ref, _ = _reference(beam)
    E_table = data[key][1][0]
    scale = BEAMS[beam].mass / m_ref
    lo, hi = E_table[0] * scale, E_table[-1] * scale
    n = int(np.ceil(np.log10(hi / lo) * points_per_decade)) + 1
    return np.geomspace(lo, hi, n)
