"""Nuclides and their masses, beam particles, and the neutron-producing reactions.

Atomic masses and element symbols come from MASS_FILE (the AME2020 atomic mass table), natural
isotopic compositions from NATURAL_ABUNDANCE_FILE (NIST Atomic Weights and Isotopic
Compositions); see data/README.md. Nuclear (bare-nucleus) masses, used for the beam and the reaction
kinematics, are the atomic masses less the electrons, plus the electrons' binding energy.

Atoms are named by element symbol for the natural element ("Li", "H"), or with the mass
number first for one isotope ("7Li", "1H", "3He"); D and T are names for 2H and 3H.
"""

import functools
import os
import re
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# AME2020 atomic mass table (in data/), as distributed by the IAEA AMDC
MASS_FILE = os.path.join(DATA_DIR, "mass_1.mas20.txt")
# Natural isotopic compositions (in data/): the NIST Atomic Weights and Isotopic Compositions
# records
NATURAL_ABUNDANCE_FILE = os.path.join(DATA_DIR, "nist_isotopic_compositions.txt")

# Energy equivalents [MeV], CODATA 2022
U_MEV = 931.49410372  # Atomic mass unit
M_ELECTRON = 0.51099895069  # Electron

# Names for hydrogen isotopes; the reverse gives their canonical names
ISOTOPE_ALIASES = {"D": (1, 2), "T": (1, 3)}
ALIAS_NAMES = {za: name for name, za in ISOTOPE_ALIASES.items()}

_ATOM = re.compile(r"^(\d*)([A-Z][a-z]?)$")


# Total electron binding energies of H and He [eV] (ionization energies, NIST ASD)
_ELECTRON_BINDING_EV = {1: 13.598434599702, 2: 24.587389011 + 54.4177655282}


def electron_binding(Z):
    """Return the total binding energy of the Z electrons of a neutral atom [MeV]: exact for
    H and He, otherwise from the fit of D. Lunney, J. M. Pearson and C. Thibault, Rev. Mod.
    Phys. 75 (2003) 1021, eq. A4 (a few keV even at Z = 10, so it matters little here)."""
    if Z in _ELECTRON_BINDING_EV:
        return _ELECTRON_BINDING_EV[Z] * 1e-6
    return (14.4381 * Z**2.39 + 1.55468e-6 * Z**5.35) * 1e-6


@functools.cache
def mass_table():
    """Return ({(Z, A): atomic mass [u]}, {Z: element symbol}) from MASS_FILE.

    The file is fixed-width (Fortran format a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,
    f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6): N - Z, N, Z, A, element, origin, mass excess,
    binding energy, beta-decay energy, and the atomic mass, as an integer part [u] and the rest
    [micro-u]. A "#" in place of the decimal point marks an estimated (non-experimental)
    value; it is read as a decimal point. The table starts after the "(micro-u)" header line.
    """
    with open(MASS_FILE, encoding="ascii") as f:
        lines = f.read().splitlines()
    start = next(i for i, line in enumerate(lines) if "(micro-u)" in line) + 1
    masses, symbols = {}, {}
    for line in lines[start:]:
        if not line.strip():
            continue
        Z, A = int(line[9:14]), int(line[14:19])
        micro_u = int(line[106:109]) * 1e6 + float(line[110:123].replace("#", "."))
        masses[(Z, A)] = micro_u * 1e-6
        symbols.setdefault(Z, line[20:23].strip())
    return masses, symbols


def atomic_mass(Z, A):
    """Return the atomic mass of the neutral atom (Z, A) [u] (the neutron's for Z = 0).
    Raises ValueError if (Z, A) is not in the mass table."""
    masses, symbols = mass_table()
    if (Z, A) not in masses:
        raise ValueError(f"no isotope {A}{symbols.get(Z, f'(Z={Z})')} in the mass table")
    return masses[(Z, A)]


def nuclear_mass(Z, A, excitation=0.0):
    """Return the rest-mass energy [MeV] of the bare nucleus (Z, A), plus an excitation
    energy [MeV] for an excited state. Z = 0, A = 1 is the neutron."""
    mass = atomic_mass(Z, A) * U_MEV
    if Z > 0:
        mass += -Z * M_ELECTRON + electron_binding(Z)
    return mass + excitation


def parse_atom(name):
    """Return (Z, A) for an atom name: A is None for the natural element ("Li"), the mass
    number for one isotope ("7Li", "D"). Raises ValueError for an unknown element or
    isotope."""
    if name in ISOTOPE_ALIASES:
        return ISOTOPE_ALIASES[name]
    m = _ATOM.match(name)
    if not m:
        raise ValueError(f"cannot read the atom {name!r}")
    numbers = {symbol: Z for Z, symbol in mass_table()[1].items() if Z > 0}
    if m.group(2) not in numbers:
        raise ValueError(f"unknown element {m.group(2)!r}")
    Z = numbers[m.group(2)]
    if not m.group(1):
        return Z, None
    A = int(m.group(1))
    atomic_mass(Z, A)  # Raises for an unknown isotope
    return Z, A


def atom_name(Z, A=None):
    """Return the canonical name of an atom: "Li" (natural), "7Li", or D or T."""
    symbol = mass_table()[1][Z]
    if A is None:
        return symbol
    return ALIAS_NAMES.get((Z, A), f"{A}{symbol}")


@functools.cache
def natural_abundances():
    """Return {Z: {A: atom fraction}} from NATURAL_ABUNDANCE_FILE.

    The file holds one record per nuclide, separated by blank lines, of "key = value" lines
    ("Atomic Number = 3", "Mass Number = 7", "Isotopic Composition = 0.9241(4)", ...); the
    uncertainty in brackets is dropped, and nuclides with no composition are left out.
    """
    table = {}
    with open(NATURAL_ABUNDANCE_FILE, encoding="ascii") as f:
        records = f.read().strip().split("\n\n")
    for record in records:
        fields = {}
        for line in record.strip().splitlines():
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
        composition = fields.get("Isotopic Composition", "").split("(")[0]
        if composition:
            Z, A = int(fields["Atomic Number"]), int(fields["Mass Number"])
            table.setdefault(Z, {})[A] = float(composition)
    return table


def isotope_fractions(name):
    """Return {(Z, A): fraction of the atoms} for an atom name: the natural isotopic
    composition for an element (see natural_abundances), or 1 for one isotope. Raises
    ValueError for an element with no natural isotopes (such as Tc)."""
    Z, A = parse_atom(name)
    if A is not None:
        return {(Z, A): 1.0}
    fractions = {(Z, A): f for A, f in natural_abundances().get(Z, {}).items() if f > 0}
    if not fractions:
        raise ValueError(f"{name} has no natural isotopes; name one, such as {name}99")
    total = sum(fractions.values())
    return {za: f / total for za, f in fractions.items()}


def mean_atomic_mass(fractions):
    """Return the mean atomic mass [u] of atoms with the given {(Z, A): fraction}."""
    return sum(f * atomic_mass(Z, A) for (Z, A), f in fractions.items())


@dataclass(frozen=True)
class Beam:
    """A beam particle: its name, description, nucleus (Z, A) and charge state."""

    name: str
    description: str
    Z: int
    A: int
    charge: int = 1

    @property
    def mass(self):
        """Rest-mass energy of the bare nucleus [MeV]."""
        return nuclear_mass(self.Z, self.A)


# Beam particles by their one-letter keys (as in TENDL's file names and the reaction keys,
# such as p9Be_xn). Inputs name them by BEAM_NAMES instead.
BEAMS = {
    "p": Beam("p", "proton", 1, 1),
    "d": Beam("d", "deuteron", 1, 2),
    "t": Beam("t", "triton", 1, 3),
    "h": Beam("h", "helion", 2, 3, charge=2),
    "a": Beam("a", "alpha", 2, 4, charge=2),
}

# The names a beam particle may be given by: its nuclide or its particle name
BEAM_NAMES = {
    "1H": "p", "proton": "p",
    "2H": "d", "deuteron": "d",
    "3H": "t", "triton": "t",
    "3He": "h", "helion": "h",
    "4He": "a", "alpha": "a",
}  # fmt: skip


def beam_key(name):
    """Return the BEAMS key of a beam particle named as in BEAM_NAMES (such as "proton" or
    "1H"); raise ValueError for any other name."""
    if name not in BEAM_NAMES:
        names = ", ".join('"' + n + '"' for n in BEAM_NAMES)
        raise ValueError(f"beam must be one of {names}, got {name!r}")
    return BEAM_NAMES[name]


# Particles in final states: (Z, A, excitation energy [MeV])
_N, _P, _D, _T = (0, 1, 0.0), (1, 1, 0.0), (1, 2, 0.0), (1, 3, 0.0)
_HE3, _HE4, _LI6 = (2, 3, 0.0), (2, 4, 0.0), (3, 6, 0.0)
_BE7_STAR2 = (4, 7, 4.57)  # Second excited state of 7Be

# 7Be first excited state [MeV]
BE7_FIRST_EXCITED = 0.42908


@dataclass(frozen=True)
class Reaction:
    """A two-body neutron-producing reaction, beam + target -> neutron + residual.

    Attributes:
        key: Short name, used in output names (such as "DD").
        label: Reaction in the usual notation (such as "D(d,n)3He").
        beam: Beam particle name (see BEAMS).
        target: Target nucleus (Z, A).
        residual: Residual nucleus (Z, A), with excitation [MeV] for an excited state.
        data_file: DROSG-2000 cross-section file (in data/).
        inverse: True if the data are for the target as the beam (and the beam as the
            target): for example D(t,n)4He from the T(d,n)4He data.
        competing: Final states of the other neutron-producing channels of the same beam and
            target, as tuples of (Z, A, excitation) particles. The reaction data (with any
            others sharing them, such as the two 7Li(p,n) lines) give all the neutrons only
            below the lowest of their thresholds (see complete_up_to).
    """

    key: str
    label: str
    beam: str
    target: tuple
    residual: tuple
    data_file: str
    excitation: float = 0.0
    inverse: bool = False
    competing: tuple = ()

    def masses(self):
        """Return (m1, m2, m3, m4) [MeV]: beam, target, neutron, and residual."""
        return (
            BEAMS[self.beam].mass,
            nuclear_mass(*self.target),
            nuclear_mass(0, 1),
            nuclear_mass(*self.residual, self.excitation),
        )

    @property
    def q_value(self):
        """Q value [MeV] (positive: exothermic)."""
        m1, m2, m3, m4 = self.masses()
        return (m1 + m2) - (m3 + m4)

    def _threshold(self, final_masses):
        m1, m2 = self.masses()[:2]
        return max(0.0, ((sum(final_masses)) ** 2 - (m1 + m2) ** 2) / (2 * m2))

    @property
    def threshold(self):
        """Lab beam kinetic energy at which the reaction opens [MeV] (0 if exothermic)."""
        return self._threshold(self.masses()[2:])

    def competing_channels(self):
        """Return [(threshold [MeV], label)] for the competing channels (lab beam energy at
        which each opens), lowest first."""
        return sorted(
            (self._threshold([nuclear_mass(Z, A, e) for Z, A, e in state]),
             _channel_label(self, state))
            for state in self.competing
        )  # fmt: skip

    def complete_up_to(self):
        """Return (energy [MeV], channel): the lab beam energy below which this reaction (with
        any others sharing its competing channels) gives all the neutrons from this beam and
        target, and the label of the channel that opens there. (inf, "") if none."""
        channels = self.competing_channels()
        return channels[0] if channels else (np.inf, "")

    def data_energy(self, beam_energy):
        """Convert lab beam energies to the energies the cross-section data are tabulated at
        (the same center-of-mass energy): unchanged, or for an inverse reaction scaled by
        target mass / beam mass, which is exact relativistically too."""
        if not self.inverse:
            return beam_energy
        m1, m2 = self.masses()[:2]
        return beam_energy * m2 / m1


def _channel_label(reaction, state):
    """Return a channel such as "T(d,np)T" for a final state of a reaction's beam and target."""
    target = "H" if reaction.target == (1, 1) else atom_name(*reaction.target)
    names = []
    for Z, A, excitation in state:
        if (Z, A) == (0, 1):
            names.append("n")
        elif (Z, A) == (1, 1):
            names.append("p")
        else:
            name = atom_name(Z, A)
            names.append(name + (f"* ({excitation:g} MeV)" if excitation else ""))
    light = [n for n in names if n in ("n", "p")]
    heavy = [n for n in names if n not in ("n", "p")]
    if not heavy:
        heavy, light = [light[-1]], light[:-1]
    return f"{target}({reaction.beam},{''.join(light) + ''.join(heavy[:-1])}){heavy[-1]}"


REACTIONS = {
    r.key: r
    for r in [
        Reaction("DD", "D(d,n)3He", "d", (1, 2), (2, 3), "d2hags.koe",
                 competing=((_N, _P, _D),)),
        Reaction("DT", "T(d,n)4He", "d", (1, 3), (2, 4), "d3hags.koe",
                 competing=((_N, _P, _T), (_N, _N, _HE3))),
        Reaction("pT", "T(p,n)3He", "p", (1, 3), (2, 3), "p3hags.koe",
                 competing=((_N, _P, _D),)),
        Reaction("p7Li_n0", "7Li(p,n0)7Be", "p", (3, 7), (4, 7), "p7ligs.koe",
                 competing=((_N, _HE3, _HE4), (_N, _P, _LI6), (_N, _BE7_STAR2))),
        Reaction("p7Li_n1", "7Li(p,n1)7Be*", "p", (3, 7), (4, 7), "p7lies.koe",
                 excitation=BE7_FIRST_EXCITED,
                 competing=((_N, _HE3, _HE4), (_N, _P, _LI6), (_N, _BE7_STAR2))),
        Reaction("tD", "D(t,n)4He", "t", (1, 2), (2, 4), "d3hags.koe", inverse=True,
                 competing=((_N, _P, _T), (_N, _N, _HE3))),
        Reaction("tH", "H(t,n)3He", "t", (1, 1), (2, 3), "p3hags.koe", inverse=True,
                 competing=((_N, _P, _D),)),
    ]
}  # fmt: skip


def cm_to_lab(T1, cos_cm, m1, m2, m3, m4):
    """Relativistic kinematics of 1 + 2 -> 3 + 4 with particle 2 at rest, from the center of
    mass (CM) to the lab.

    Args:
        T1: Lab kinetic energy of the beam particle 1 [MeV] (scalar, or an array that
            broadcasts against cos_cm, such as a column of beam energies against a row of
            cosines).
        cos_cm: Cosine of the CM emission angle of particle 3, from the beam direction
            (scalar or array).
        m1, m2, m3, m4: Rest-mass energies [MeV].

    Returns (T3, cos_lab), shaped like T1 and cos_cm broadcast together: the lab kinetic
    energy of particle 3 [MeV] and the cosine of its lab angle. Every CM angle has exactly one
    lab angle, so this also covers endothermic reactions near threshold, where a lab angle can
    have two neutron energies. Below the threshold both are NaN.
    """
    cos_cm = np.asarray(cos_cm, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    E1 = T1 + m1
    p1 = np.sqrt(E1**2 - m1**2)
    E_tot = E1 + m2
    s = m1**2 + m2**2 + 2 * E1 * m2
    sqrt_s = np.sqrt(s)
    shape = np.broadcast_shapes(T1.shape, cos_cm.shape)
    below = np.broadcast_to(sqrt_s <= m3 + m4, shape)
    if np.all(below):
        nan = np.full(shape, np.nan)
        return nan, nan.copy()
    beta, gamma = p1 / E_tot, E_tot / sqrt_s
    E3_cm = (s + m3**2 - m4**2) / (2 * sqrt_s)
    p3_cm = np.sqrt(np.clip(E3_cm**2 - m3**2, 0, None))
    sin_cm = np.sqrt(np.clip(1 - cos_cm**2, 0, None))
    p_par = gamma * (p3_cm * cos_cm + beta * E3_cm)
    p_perp = p3_cm * sin_cm
    E3 = gamma * (E3_cm + beta * p3_cm * cos_cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_lab = p_par / np.hypot(p_par, p_perp)
    T3 = np.where(below, np.nan, E3 - m3)
    return T3, np.where(below, np.nan, cos_lab)


# Radius parameter of the Coulomb barrier [fm], and e^2 / (4 pi eps0) [MeV fm]
R0_COULOMB = 1.3
E2_COULOMB = 1.439964548


def coulomb_barrier(Z1, A1, Z2, A2):
    """Return the Coulomb barrier [MeV] of two nuclei touching at radius
    R0_COULOMB * (A1^(1/3) + A2^(1/3)), in the center of mass (an estimate)."""
    return E2_COULOMB * Z1 * Z2 / (R0_COULOMB * (A1 ** (1 / 3) + A2 ** (1 / 3)))


def cm_energy(beam, target, beam_energy):
    """Return the kinetic energy in the center of mass [MeV] of a beam particle (a BEAMS key)
    of lab energy beam_energy [MeV] on a target nucleus (Z, A) at rest."""
    m1, m2 = BEAMS[beam].mass, nuclear_mass(*target)
    return np.sqrt(m1**2 + m2**2 + 2 * (beam_energy + m1) * m2) - m1 - m2


def neutron_channels(beam, target):
    """Return [(threshold [MeV], label)] for the generic neutron-producing channels of a beam
    on a target nucleus (Z, A): (b,n), (b,2n) and (b,np), lowest threshold first. Channels
    whose residual nucleus is not in the mass table are left out."""
    b = BEAMS[beam]
    Z, A = target
    finals = [
        (_N, (Z + b.Z, A + b.A - 1, 0.0)),
        (_N, _N, (Z + b.Z, A + b.A - 2, 0.0)),
        (_N, _P, (Z + b.Z - 1, A + b.A - 2, 0.0)),
    ]
    m1, m2 = b.mass, nuclear_mass(*target)
    channels = []
    for state in finals:
        residual = state[-1]
        if residual[0] < 0 or residual[1] < 1 or residual[1] < residual[0]:
            continue
        try:
            masses = [nuclear_mass(*particle) for particle in state]
        except (ValueError, KeyError, IndexError):
            continue
        threshold = max(0.0, (sum(masses) ** 2 - (m1 + m2) ** 2) / (2 * m2))
        channels.append((threshold, _generic_label(beam, target, state)))
    return sorted(channels)


def _generic_label(beam, target, state):
    """Return a channel label such as "12C(d,n)13N", "48Ti(d,2n)48V" or "D(p,np)H"."""
    light = ["n" if (Z, A) == (0, 1) else "p" for Z, A, _ in state[:-1]]
    light = "2n" if light == ["n", "n"] else "".join(light)

    def name(za):
        return "H" if za == (1, 1) else atom_name(*za)

    return f"{name(target)}({beam},{light}){name(state[-1][:2])}"
