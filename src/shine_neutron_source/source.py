"""Beam-target fusion neutron source term for Monte Carlo transport codes.

A deuteron beam passes through one or more gas regions containing a deuterium/tritium
mixture. For each region the user gives a pressure, temperature, length and beam
divergence. The calculation:

1. Slows the beam down along its path using SRIM or ICRU 49 stopping powers.
2. Grows the beam radius with its divergence and removes the fraction of the beam that
   strikes the target wall (beam scraping).
3. Evaluates the D(d,n)3He and T(d,n)4He double-differential cross sections at each
   position from tabulated Legendre data, transformed from the centre-of-mass frame to
   the lab frame, and the neutron energy at each emission angle, using relativistic
   two-body kinematics.
4. Tallies neutrons on a fine internal (position, angle) mesh and condenses them onto a
   coarser output (position, angle, energy) mesh.

The result can be written as an HDF5 file, as an MCNP SDEF source card, as text files and
as plots. The caller chooses every output path.

Cases are normally described in a TOML input file and run from the command line (see
cli.py); the ShineSource class can also be used directly from Python.

Naming conventions used throughout:

- A mesh is divided into bins. ``*_grid`` arrays hold values on bin boundaries (length
  n + 1); ``*_mids`` arrays hold one value per bin, at or averaged over the bin (length n).
- An ``_out`` suffix marks the coarser output mesh; no suffix is the fine internal mesh the
  calculation runs on. Each output bin is a whole number of internal bins (a "ratio").
- ``*_per_region`` arrays have one entry per gas region. The region inputs (pressure,
  temperature, length, divergence, num_x_bins, num_x_bins_out) are also per region.
- Beam (deuteron) energies are in keV; neutron energies are in MeV.
- ``rxn`` is a reaction, "DD" for D(d,n)3He or "DT" for T(d,n)4He.
"""

import datetime
import functools
import importlib.metadata
import logging
import os
import re
import subprocess
import textwrap
import time

import h5py
import matplotlib
import numpy as np
from numpy.polynomial.legendre import legval

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # Must follow matplotlib.use
from matplotlib.colors import LogNorm

PACKAGE_NAME = "shine-neutron-source"
logger = logging.getLogger(__name__)
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_DATA_DIR = os.path.join(CODE_DIR, "data")
# DROSG-2000 cross-section file for each reaction (in data/)
DROSG_FILES = {"DD": "d2hags.koe", "DT": "d3hags.koe"}

# Stopping-power data files (in data/) and the available stopping-power sets
SRIM_STOPPING_FILE = "srim_stopping.h5"
PSTAR_FILE = "pstar_hydrogen.txt"
STOPPING_POWERS = {
    "srim": "SRIM-2012.01, per atom, mixed over each region's composition",
    "icru49": "as srim, but ICRU 49 electronic stopping for hydrogen isotopes (gas only)",
}
BEAM_PARTICLE = "d"  # Beam particle group in SRIM_STOPPING_FILE
PHASES = ("gas", "solid")
HYDROGEN_ISOTOPES = ("H", "D", "T")
# Elements whose gases are diatomic molecules: a gas Material rejects them as lone atoms
DIATOMIC_ELEMENTS = ("H", "D", "T", "N", "O", "F", "Cl", "Br", "I")

# Decimal places for the beam radii in the SDEF card [cm]
SDEF_RADIUS_DECIMALS = 5

# Version of the HDF5 output layout; increase it whenever the layout changes
HDF5_FORMAT_VERSION = 3

# Error message when a case produces no neutrons
NO_NEUTRONS = (
    "no neutrons are produced. Check that the beam voltage is high enough (the cross-section "
    "data start at 1.8 keV for DD and 10 keV for DT) and that the beam is not scraped away "
    "entirely."
)

# Highest allowed beam voltage [kV]. The DROSG Legendre tables in data/ end at 40 MeV; the
# rows above that are not angular distributions (see data/README.md).
MAX_VOLTAGE = 40000.0

# Repository root when running from a source checkout (src/shine_neutron_source/ -> root)
REPO_DIR = os.path.normcase(os.path.dirname(os.path.dirname(CODE_DIR)))

# Physical constants (exact SI values)
NA = 6.02214076e23  # Avogadro constant [1/mol]
ee = 1.602176634e-19  # Elementary charge [C]
kB = 1.380649e-23  # Boltzmann constant [J/K]

# Atomic weight of hydrogen that ICRU Report 49 used for its per-mass stopping [g/mol]
A_H_ICRU49 = 1.00794

# Nuclear (bare-nucleus) rest-mass energies [MeV], CODATA 2022. Used for the beam ion and the
# reaction kinematics, which involve bare nuclei; electron binding energies are negligible.
M_H1 = 938.27208943  # Proton (H-1)
M_N1 = 939.56542194  # Neutron (n-1)
M_H2 = 1875.61294500  # Deuteron (H-2)
M_HE3 = 2808.39161112  # Helion (He-3)
M_H3 = 2808.92113668  # Triton (H-3)
M_HE4 = 3727.3794118  # Alpha (He-4)

# Reaction Q values [MeV]
Q_DD = (M_H2 + M_H2) - (M_N1 + M_HE3)  #  3.26885694 MeV
Q_DT = (M_H2 + M_H3) - (M_N1 + M_HE4)  # 17.58924794 MeV

# Two-body kinematics for each reaction, 1 + 2 -> 3 + 4:
# (beam M1, target M2, neutron M3, residual M4, Q value)
REACTIONS = {
    "DD": (M_H2, M_H2, M_N1, M_HE3, Q_DD),  # D(d,n)3He
    "DT": (M_H2, M_H3, M_N1, M_HE4, Q_DT),  # T(d,n)4He
}

# Target isotope of each reaction: the beam deuteron reacts with this atom
REACTION_TARGETS = {"DD": "D", "DT": "T"}

# MCNP source direction -> (AXS vector, VEC vector, reverse the position distributions).
# For "-z" the extent runs along +z and the position-dependent distributions are written
# in reverse order instead of using AXS = 0 0 -1, which ADVANTG does not handle correctly.
# As a result, for "-z" POS is the downstream (backstop) end of the source; for every other
# direction POS is the point where the beam enters the first gas region. get_mcnp_sdef takes
# the beam entry point for every direction and sets POS accordingly.
SDEF_DIRECTIONS = {
    "-x": ("-1 0 0", "-1 0 0", False),
    "+x": ("1 0 0", "1 0 0", False),
    "-y": ("0 -1 0", "0 -1 0", False),
    "+y": ("0 1 0", "0 1 0", False),
    "-z": ("0 0 1", "0 0 -1", True),
    "+z": ("0 0 1", "0 0 1", False),
}


def two_body_kinematics(T1, cos_lab, m1, m2, m3, m4):
    """Relativistic kinematics of the reaction 1 + 2 -> 3 + 4 with particle 2 at rest.

    Args:
        T1: Kinetic energy of the beam particle 1 in the lab [MeV].
        cos_lab: Cosine of the lab emission angle of particle 3 (scalar or array).
        m1, m2, m3, m4: Rest-mass energies [MeV].

    Returns (T3, cos_cm, jacobian), each shaped like cos_lab:

    - T3: kinetic energy of particle 3 in the lab [MeV].
    - cos_cm: cosine of its emission angle in the centre-of-mass (CM) frame.
    - jacobian: dOmega_cm / dOmega_lab = d(cos_cm) / d(cos_lab), which converts a CM
      differential cross section to the lab frame.

    The CM frame moves with velocity beta (units of c) and Lorentz factor gamma, and particle 3
    has a fixed CM energy E3_cm. Its lab momentum p3 at each lab angle follows from
    E3_cm = gamma (E3 - beta p3 cos_lab). This assumes particle 3 moves faster in the CM frame
    than the CM frame moves in the lab, so each lab angle has one CM angle, which holds for
    neutrons from these exothermic reactions.
    """
    E1 = T1 + m1
    p1 = np.sqrt(E1**2 - m1**2)
    E_tot = E1 + m2
    sqrt_s = np.sqrt(m1**2 + m2**2 + 2 * E1 * m2)  # Total energy in the CM frame
    beta, gamma = p1 / E_tot, E_tot / sqrt_s
    E3_cm = (sqrt_s**2 + m3**2 - m4**2) / (2 * sqrt_s)
    p3_cm = np.sqrt(E3_cm**2 - m3**2)

    a = E3_cm / gamma
    bc = beta * cos_lab
    p3 = (a * bc + np.sqrt(a**2 - m3**2 * (1 - bc**2))) / (1 - bc**2)
    E3 = np.sqrt(p3**2 + m3**2)
    cos_cm = gamma * (p3 * cos_lab - beta * E3) / p3_cm
    dp3_dcos = beta * p3 / (p3 / E3 - bc)
    jacobian = gamma / p3_cm * (p3 + dp3_dcos * (cos_lab - beta * p3 / E3))
    return E3 - m3, cos_cm, jacobian


def get_molecule_density(pressure, temperature):
    """Return the molecule density [1/cm^3] of an ideal gas.

    pressure is in torr and temperature in degrees C. Works element-wise on arrays.
    """
    P_Pa = pressure * 101325 / 760  # Pressure [Pa]
    T_K = temperature + 273.15  # Temperature [K]
    return P_Pa / (kB * T_K) / 1e6


def get_atom_density(pressure, temperature):
    """Return the atom density [1/cm^3] of a diatomic ideal gas such as D2, DT or T2.

    pressure is in torr and temperature in degrees C. Works element-wise on arrays.
    """
    return get_molecule_density(pressure, temperature) * 2


_FORMULA_TOKEN = re.compile(r"(\d+[A-Z][a-z]?|[A-Z][a-z]?)(\d+(?:\.\d*)?|\.\d+)?")


def parse_formula(formula):
    """Return {atom: count} for a chemical formula such as "D2", "TiD1.5", "CD4" or "Ar".

    Atoms are element symbols, or the isotopes D, T and 3He (written with its mass number
    first, so that "He3" still means three helium atoms). Counts may be decimal and default
    to 1. Raises ValueError for anything else.
    """
    counts = {}
    pos = 0
    while pos < len(formula):
        m = _FORMULA_TOKEN.match(formula, pos)
        if not m:
            raise ValueError(f"cannot read the chemical formula {formula!r}")
        symbol = m.group(1)
        if symbol[0].isdigit():  # Isotope written as mass number + symbol: 3He -> He3
            mass_number = symbol.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            symbol = symbol[len(mass_number) :] + mass_number
        counts[symbol] = counts.get(symbol, 0.0) + float(m.group(2) or 1)
        pos = m.end()
    if not counts:
        raise ValueError("empty chemical formula")
    return counts


class Material:
    """A target material: its phase, composition and (for a solid) density.

    A solid's composition is given as atoms and a gas's as molecules, so that the molecules a
    gas pressure counts are always stated: H2O vapour, a mixture of H2 and O2, and atomic H
    and O are different gases with the same atom ratios.

    Args:
        name: Name, used in messages and outputs.
        phase: "gas" or "solid", which selects SRIM's gas- or solid-phase stopping.
        atoms: For a solid, {atom: amount}: atom ratios, such as {"Ti": 1, "D": 1.5}. Atoms
            are element symbols or the isotopes D, T and 3He.
        molecules: For a gas, {formula: amount}: mole fractions of its molecules, such as
            {"D2": 0.5, "T2": 0.5} or {"H2O": 1} (see parse_formula). A single atom is a
            monatomic molecule (such as "Ar"); hydrogen isotopes, N, O and the halogens are
            rejected on their own (write D2, not D).
        density: Mass density of a solid [g/cm^3]. A gas has none: its state (pressure and
            temperature, or density) belongs to the region it fills.
        Amounts are relative and need not sum to 1.

    Attributes: composition (the atoms or molecules given), atom_fractions ({atom: fraction
    of all atoms}) and atoms_per_unit (for a gas, the mean number of atoms per molecule, which
    converts a molecule density to an atom density; 1 for a solid).
    """

    def __init__(self, name, phase, *, atoms=None, molecules=None, density=None):
        self.name = str(name)
        self.phase = str(phase)
        self.density = None if density is None else float(density)
        where = f"material {self.name}"
        if self.phase not in PHASES:
            raise ValueError(f"{where}: phase must be one of {', '.join(PHASES)}")
        if self.phase == "solid":
            if molecules is not None or atoms is None:
                raise ValueError(f"{where}: give a solid's composition as atoms (atom ratios)")
            if not (self.density is not None and self.density > 0):
                raise ValueError(f"{where}: a solid needs a positive density (g/cm3)")
            given = atoms
        else:
            if atoms is not None or molecules is None:
                raise ValueError(
                    f"{where}: give a gas's composition as molecules (mole fractions of its "
                    "molecules, such as D2 = 0.5, T2 = 0.5)"
                )
            if self.density is not None:
                raise ValueError(
                    f"{where}: a gas's density belongs to the region it fills (with pressure "
                    "and temperature, or density), not to the material"
                )
            given = molecules
        self.composition = {str(k): float(v) for k, v in dict(given).items()}
        if not self.composition:
            raise ValueError(f"{where}: empty composition")
        if any(not (v >= 0) for v in self.composition.values()):
            raise ValueError(f"{where}: composition amounts must not be negative")
        total = sum(self.composition.values())
        if not total > 0:
            raise ValueError(f"{where}: composition amounts must not all be zero")

        counts = {}
        for formula, amount in self.composition.items():
            try:
                parsed = parse_formula(formula)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            if self.phase == "solid" and (len(parsed) != 1 or list(parsed.values()) != [1.0]):
                raise ValueError(
                    f"{where}: atoms takes single atoms (such as Ti or D), not {formula!r}"
                )
            if self.phase == "gas" and list(parsed) in ([a] for a in DIATOMIC_ELEMENTS):
                if parsed[list(parsed)[0]] == 1.0:
                    raise ValueError(
                        f"{where}: {formula} on its own is a lone atom; its gas is made of "
                        f"{formula}2 molecules, so write {formula}2"
                    )
            for atom, n in parsed.items():
                counts[atom] = counts.get(atom, 0.0) + amount * n
        atoms_total = sum(counts.values())
        self.atoms_per_unit = atoms_total / total
        self.atom_fractions = {atom: n / atoms_total for atom, n in counts.items() if n > 0}

    @property
    def kind(self):
        """ "atoms" for a solid, "molecules" for a gas: how composition is given."""
        return "atoms" if self.phase == "solid" else "molecules"

    def __repr__(self):
        density = "" if self.density is None else f", density={self.density!r}"
        return f"Material({self.name!r}, {self.phase!r}, {self.kind}={self.composition!r}{density})"


def dt_gas(T_frac, name="D/T gas"):
    """Return a gas Material of D2 and T2 molecules with T_frac of its atoms tritium (DT
    molecules would give the same atoms). A convenience for D/T gas targets."""
    if not 0 <= T_frac <= 1:
        raise ValueError(f"T_frac must be between 0 and 1, got {T_frac}")
    return Material(name, "gas", molecules={"D2": 1 - T_frac, "T2": T_frac})


@functools.lru_cache(maxsize=4)
def read_srim_h5(fname, particle=BEAM_PARTICLE):
    """Read the stopping of one beam particle in every target from SRIM_STOPPING_FILE.

    Returns (E [keV], masses, stopping): masses maps each target name (element symbols, plus
    D, T and He3) to its atomic mass [u]; stopping maps each phase ("gas", "solid") to
    {target: (electronic, nuclear)} stopping per target atom [keV cm^2]. The arrays are shared
    between calls (the result is cached), so do not modify them.
    """
    to_per_atom = 1e-18  # eV/(1e15 atoms/cm^2) -> keV cm^2 per atom
    with h5py.File(fname, "r") as f:
        E = f["energy"][()]
        names = [n.decode() for n in f["targets"][()]]
        masses = dict(zip(names, f["target_mass"][()]))
        stopping = {}
        for phase in PHASES:
            elec = f[particle][phase]["electronic"][()] * to_per_atom
            nuc = f[particle][phase]["nuclear"][()] * to_per_atom
            stopping[phase] = {name: (elec[i], nuc[i]) for i, name in enumerate(names)}
    return E, masses, stopping


def slow_down(E0, stopping_E, stopping_eps, atom_density, dx):
    """Return the beam energy [keV] at each grid point as the beam slows down through bins.

    E0 is the starting energy [keV]; stopping_E [keV] and stopping_eps [keV cm^2 per atom] the
    stopping-power table; atom_density [1/cm^3] and dx [cm] are per bin. The energy loss in
    each bin uses the stopping power at the bin's average energy, found by fixed-point
    iteration. If the beam stops or leaves the stopping-power table, the remaining energies
    are zero. The result has len(dx) + 1 entries.
    """
    grid = np.zeros(len(dx) + 1)
    grid[0] = E0
    for i in range(len(dx)):
        E_start = grid[i]
        E_avg = grid[i]
        for _ in range(5):
            eps = np.interp(E_avg, stopping_E, stopping_eps, left=np.nan, right=np.nan)
            E_end = E_start - eps * atom_density[i] * dx[i]
            E_avg = (E_start + E_end) / 2
        if np.isnan(E_end) or E_end < 0:
            break
        grid[i + 1] = E_end
    return grid


_FORTRAN_EXPONENT = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$")


def _fortran_float(field):
    """Convert one Fortran F-format field read with BLANK='NULL' to a float.

    Spaces anywhere in the field are ignored, a blank field is 0, and an exponent may be
    written without the "e" (8.619-4 = 8.619e-4).
    """
    field = field.replace(" ", "")
    if not field:
        return 0.0
    m = _FORTRAN_EXPONENT.match(field)
    if m:
        return float(f"{m.group(1)}e{m.group(2)}")
    return float(field)


def read_drosg_koe(fname):
    """Read the Legendre table of a DROSG-2000 cross-section file (*.KOE).

    Follows how DROSG-2000 itself reads these files (FORMAT(10f8.0) with BLANK='NULL' in
    LEGINT.f95): every line is ten 8-character fields, spaces inside a field are ignored and
    blank fields are zero. The first line holds EMIN, EMAX, EMINL and EMAXL [MeV]; EMAXL is the
    highest energy of the Legendre table. Each record is then two lines (20 values): incident
    deuteron energy [MeV], the centre-of-mass differential cross section at 0 degrees [mb/sr]
    and 18 Legendre coefficients, normalised to sum to 1. Reading stops at the record at
    EMAXL; the records above it are single-angle data, not Legendre coefficients, and are not
    used (see data/README.md).

    Returns (E [MeV], total [mb/sr], coeffs) as arrays; coeffs has 18 columns. Raises
    ValueError if the energies do not increase, a record's coefficients do not sum to 1 within
    0.001 (the tolerance DROSG-2000 uses) or the file ends before EMAXL.
    """
    width, per_line = 8, 10
    with open(fname, "r", encoding="ascii") as f:
        lines = [line.rstrip("\r\n") for line in f]

    def fields(line):
        line = line.ljust(width * per_line)
        return [_fortran_float(line[i * width : (i + 1) * width]) for i in range(per_line)]

    E_maxl = fields(lines[0])[3]
    records = []
    for i in range(1, len(lines), 2):
        values = fields(lines[i]) + fields(lines[i + 1] if i + 1 < len(lines) else "")
        energy, total, coeffs = values[0], values[1], values[2:]
        if records and energy <= records[-1][0]:
            raise ValueError(f"{fname}: energy {energy} MeV (line {i + 1}) does not increase")
        if abs(sum(coeffs) - 1) > 1e-3:
            raise ValueError(
                f"{fname}: Legendre coefficients at {energy} MeV (line {i + 1}) sum to "
                f"{sum(coeffs):.5f}, not 1"
            )
        records.append([energy, total] + coeffs)
        if energy >= E_maxl:
            break
    else:
        raise ValueError(f"{fname}: no record at EMAXL = {E_maxl} MeV")

    data = np.array(records)
    return data[:, 0], data[:, 1], data[:, 2:]


def read_pstar(fname):
    """Read a PSTAR table (see data/pstar_hydrogen.txt; lines starting with # are comments).

    Returns (E [MeV], S_elec, S_nuc [MeV cm^2/g]) for protons as arrays.
    """
    data = np.loadtxt(fname, comments="#", ndmin=2)
    return data[:, 0], data[:, 1], data[:, 2]


def get_package_version():
    """Return the installed package version, or "unknown" if the package is not installed."""
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_code_version():
    """Return the version of this code, for recording in output files.

    When the code runs from a git checkout of this repository (including an editable
    install), this is the git commit, with "-dirty" appended if tracked files (code or data)
    have uncommitted changes. Otherwise it is the installed package version, such as "0.1.0".
    """

    def git(*args):
        return subprocess.run(
            ["git", "-C", CODE_DIR, *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        # Only trust git if this file is in this repository, not in some other repository
        # that happens to contain the installed package (such as a project's virtual env)
        toplevel = git("rev-parse", "--show-toplevel")
        if os.path.normcase(os.path.realpath(toplevel)) == os.path.normcase(REPO_DIR):
            changes = git("status", "--porcelain", "--untracked-files=no")
            return git("rev-parse", "HEAD") + ("-dirty" if changes else "")
    except (OSError, subprocess.CalledProcessError):
        pass
    return get_package_version()


def _toml_value(value):
    """Format a string, number or sequence of numbers as a TOML value. Floats are written to 7
    significant figures and always read back as floats."""
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    text = f"{float(value):.7g}"
    return text if any(c in text for c in ".enai") else text + ".0"


def _toml_key(key):
    """Return key as a TOML key: bare if it can be, quoted otherwise."""
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else _toml_value(key)


def _make_parent_dir(path):
    """Create the directory that will contain path, if it does not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _is_multiple(a, b, rtol=1e-9):
    """Return True if a / b is an integer, allowing for floating-point round-off."""
    ratio = a / b
    return abs(ratio - round(ratio)) <= rtol * max(1.0, abs(ratio))


def _format_values(values, fmt, per_line, indent="\n       "):
    """Format values as an MCNP data list, wrapping after every per_line entries. fmt is a
    format specification or a function that returns the text for one value."""
    s = ""
    for i, x in enumerate(values):
        if i > 0 and i % per_line == 0:
            s += indent
        s += " " + (fmt(x) if callable(fmt) else format(x, fmt))
    return s


def _clean_float_text(x):
    """Return a float as Python stores it (shortest exact text), without floating-point
    artefacts such as 131.90200000000002 from adding up lengths."""
    return repr(float(f"{x:.12g}"))


def _cosine_text(x):
    """Return a cosine with 15 decimal places in a fixed width of 18 characters, so columns line
    up; values within 1e-12 of zero (such as cos(90 deg), 6.1e-17 in floating point) are 0."""
    return f"{0.0 if abs(x) < 1e-12 else float(x):18.15f}"


# Matplotlib settings for every plot: serif text, upright mathtext, 200 dpi
PLOT_STYLE = {"font.family": "serif", "mathtext.default": "regular", "savefig.dpi": 200}


def _plot_style(method):
    """Decorator: draw the plot with PLOT_STYLE, without changing global matplotlib settings."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with plt.rc_context(PLOT_STYLE):
            return method(*args, **kwargs)

    return wrapper


def _save_figure(fig, fname, close=True):
    """Lay out fig, save it to fname and (unless close is False) close it."""
    fig.tight_layout()
    fig.savefig(fname)
    if close:
        plt.close(fig)


class ShineSource:
    """Neutron source term for a deuteron beam in a target containing deuterium and/or tritium.

    The whole calculation runs when the object is created; afterwards use write_hdf5(),
    write_mcnp_sdef(), write_text_files() and make_plots() to write the results. Summary
    values are logged at DEBUG level to the "shine_neutron_source" logger (silent unless
    logging is configured).

    material, pressure, temperature, density, length, divergence, num_x_bins and
    num_x_bins_out take one value per region, in the order the beam passes through them; a
    scalar is treated as a single region. The last region is the target chamber (a gas, or a
    solid target such as a metal hydride); earlier regions are normally upstream
    differential-pumping stages.

    Each region is filled with a Material. A gas region gives its state: pressure and
    temperature (ideal gas), or density. A solid region gives none of these (None), since
    its density is part of the material. Neutrons come from the deuterium (D(d,n)3He) and
    tritium (T(d,n)4He) atoms; other atoms only slow the beam. dt_gas makes a D/T gas.

    Give either the beam current or the total neutron yield (yield_total), and the other is
    calculated: the yield is proportional to the current. Give None for the current to
    calculate it.

    Of the target chamber's gas state (its pressure, or its density if it has no
    temperature), its length and the beam energy at the backstop, give any two and the third
    is calculated (see solve_target_chamber); for a solid target chamber, give its length or
    the backstop energy. To calculate the pressure, density or length, give None as the
    target chamber's value (the last entry, or the whole argument for a single region) and
    set beam_energy_backstop. Every upstream region needs its length and, if it is a gas,
    its state.

    Args:
        name: Case name, used in output file names and titles.
        material: The Material filling each region (one Material for a single region).
        voltage: Beam voltage [kV]. The beam starts with this energy in keV. At most
            MAX_VOLTAGE (40 MeV), the upper limit of the Legendre cross-section data.
        current: Beam current [mA]; None to calculate it from yield_total.
        pressure: Gas pressure in each region [torr]; None for a solid region or a gas given
            by density, or for the target chamber to calculate it.
        temperature: Gas temperature in each region [degC]; None for a solid region or a gas
            given by density.
        length: Length of each region along the beam [cm]; None for the target chamber to
            calculate it.
        start_radius: Beam radius where it enters the first region [cm].
        wall_radius: Radius of the target chamber wall [cm]. Beam outside this radius is
            scraped.
        divergence: Beam divergence (growth in radius per unit length) in each
            region [mrad].
        num_x_bins: Number of position bins in each region, internal mesh.
        num_x_bins_out: Number of position bins in each region, output mesh.
            Must divide num_x_bins.
        angle_step: Width of the emission-angle bins of the internal mesh [deg].
        angle_step_out: Width of the emission-angle bins of the output mesh [deg]. Must
            divide 180 and be a multiple of angle_step. 180 gives an isotropic source.
        energy_bin_width: Width of the neutron energy bins of the output mesh [MeV].
        data_dir: Directory holding the data files (see data/README.md). Defaults to the
            data directory next to this file.
        stopping_power: Stopping-power set, one of STOPPING_POWERS: "srim" (default) or
            "icru49". See read_stopping_power.
        beam_energy_backstop: Beam energy wanted at the backstop, the end of the target
            chamber [keV]; give it to calculate the target chamber's pressure, density or
            length. None (default) calculates the beam energy from the others.
        yield_total: Total neutron yield wanted [n/s]; give it, with current None, to
            calculate the current. None (default) calculates the yield from the current.
        density: Mass density of a gas in each region given by density instead of pressure
            and temperature [g/cm^3]; None elsewhere (default: None everywhere).
    """

    def __init__(
        self,
        name,
        material,
        voltage,
        current,
        pressure,
        temperature,
        length,
        start_radius,
        wall_radius,
        divergence,
        num_x_bins,
        num_x_bins_out,
        angle_step,
        angle_step_out,
        energy_bin_width,
        data_dir=None,
        stopping_power="srim",
        beam_energy_backstop=None,
        yield_total=None,
        density=None,
    ):
        t0 = time.time()

        # Input parameters
        self.name = str(name)
        self.voltage = float(voltage)
        self.current = None if current is None else float(current)
        self.requested_yield_total = None if yield_total is None else float(yield_total)
        self.current_calculated = self.current is None
        self.pressure = np.atleast_1d(np.array(pressure, dtype=float))
        self.temperature = np.atleast_1d(np.array(temperature, dtype=float))
        self.length = np.atleast_1d(np.array(length, dtype=float))
        self.start_radius = float(start_radius)
        self.wall_radius = float(wall_radius)
        self.divergence = np.atleast_1d(np.array(divergence, dtype=float))
        # Bin counts are converted to int after check_input confirms they are whole numbers
        self.num_x_bins = np.atleast_1d(np.array(num_x_bins, dtype=float))
        self.num_x_bins_out = np.atleast_1d(np.array(num_x_bins_out, dtype=float))
        self.angle_step = float(angle_step)
        self.angle_step_out = float(angle_step_out)
        self.energy_bin_width = float(energy_bin_width)
        self.data_dir = DEFAULT_DATA_DIR if data_dir is None else str(data_dir)
        self.stopping_power = str(stopping_power)
        self.beam_energy_backstop = (
            None if beam_energy_backstop is None else float(beam_energy_backstop)
        )

        self.num_regions = len(self.length)
        if isinstance(material, Material):
            material = [material] * self.num_regions
        self.materials = list(material)
        if density is None:
            density = [None] * self.num_regions
        self.density = np.atleast_1d(np.array(density, dtype=float))
        self.solid = np.array([getattr(m, "phase", None) == "solid" for m in self.materials])
        # A gas region with a temperature is given by pressure; without one, by density
        self.by_pressure = ~self.solid & ~np.isnan(np.resize(self.temperature, self.num_regions))

        # Which of the target chamber's gas state (pressure or density), its length and the
        # backstop beam energy is calculated from the other two (a missing value is NaN here).
        # A solid target chamber has no gas state.
        state = "pressure" if self.by_pressure[-1] else "density"
        state_missing = not self.solid[-1] and np.isnan(getattr(self, state)[-1])
        unknown = [
            name
            for name, missing in [
                (state, state_missing),
                ("length", np.isnan(self.length[-1])),
                ("beam_energy_backstop", self.beam_energy_backstop is None),
            ]
            if missing
        ]
        self.calculated = unknown[0] if len(unknown) == 1 else None

        self.check_input()
        self.num_x_bins = self.num_x_bins.astype(int)
        self.num_x_bins_out = self.num_x_bins_out.astype(int)

        self.read_stopping_power()
        self.read_cross_sections()
        if self.calculated != "beam_energy_backstop":
            self.solve_target_chamber()

        self.setup_meshes()
        self.setup_regions()

        self.calculate_beam_energy()
        self.calculate_scraping()
        self.calculate_ddxs()
        if self.current_calculated:
            self.solve_current()
        self.calculate_beam_current()
        self.calculate_beam_power()
        self.bin_neutrons()

        # Recorded once, so every output of this case has the same creation time
        self.created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        logger.debug(f"{self.name}: calculation took {time.time() - t0:.1f} s")

    def check_input(self):
        """Raise ValueError if any input parameter is inconsistent or out of range."""

        def require(condition, message):
            if not condition:
                raise ValueError(f"{self.name}: {message}")

        for attr in [
            "pressure",
            "temperature",
            "density",
            "divergence",
            "num_x_bins",
            "num_x_bins_out",
            "materials",
        ]:
            require(
                len(getattr(self, attr)) == self.num_regions,
                f"{attr} has {len(getattr(self, attr))} entries, expected {self.num_regions}",
            )
        require(
            all(isinstance(m, Material) for m in self.materials),
            "material must be a Material for each region",
        )
        last = self.num_regions - 1
        for r, material in enumerate(self.materials):
            where = f"region {r + 1} ({material.name})"
            given = {
                key: not np.isnan(getattr(self, key)[r])
                for key in ("pressure", "temperature", "density")
            }
            if self.solid[r]:
                require(
                    not any(given.values()),
                    f"{where} is a solid, so it takes no pressure, temperature or density "
                    "(give None)",
                )
            elif given["temperature"]:
                require(
                    not given["density"],
                    f"{where}: give pressure and temperature, or density, not both",
                )
                require(
                    given["pressure"] or r == last,  # The chamber's state: see below
                    f"{where} needs its pressure (with its temperature), or its density",
                )
            else:
                require(
                    not given["pressure"],
                    f"{where}: a pressure needs a temperature (or give density instead)",
                )
                require(
                    given["density"] or r == last,
                    f"{where} needs its pressure and temperature, or its density",
                )
        require(
            not np.any(np.isnan(self.length[:-1])),
            "every region before the target chamber needs its length",
        )
        if self.solid[-1]:
            require(
                self.calculated is not None,
                "give exactly one of the target chamber's length and beam_energy_backstop (it "
                "is a solid); the other is calculated",
            )
        else:
            state = "pressure" if self.by_pressure[-1] else "density"
            require(
                self.calculated is not None,
                f"give exactly two of the target chamber's {state} and length and "
                "beam_energy_backstop; the third is calculated",
            )
        if self.beam_energy_backstop is not None:
            require(
                0 < self.beam_energy_backstop < self.voltage,
                f"beam_energy_backstop must be above 0 and below the beam energy at entry "
                f"({self.voltage:g} keV)",
            )
        require(
            0 < self.voltage <= MAX_VOLTAGE,
            f"voltage must be between 0 and {MAX_VOLTAGE:g} kV (the upper limit of the "
            "Legendre cross-section data, see data/README.md)",
        )
        require(
            (self.current is None) != (self.requested_yield_total is None),
            "give exactly one of current and yield_total; the other is calculated",
        )
        if self.current is not None:
            require(self.current > 0, "current must be positive")
        if self.requested_yield_total is not None:
            require(self.requested_yield_total > 0, "yield_total must be positive")
        for key, values in [
            ("pressures", self.pressure),
            ("densities", self.density),
            ("lengths", self.length),
        ]:
            known = values[~np.isnan(values)]
            require(np.all(known > 0), f"{key} must be positive")
        temperatures = self.temperature[~np.isnan(self.temperature)]
        require(np.all(temperatures > -273.15), "temperatures must be above 0 K")
        require(self.start_radius > 0, "start_radius must be positive")
        require(self.wall_radius > 0, "wall_radius must be positive")
        require(
            self.wall_radius >= self.start_radius,
            "wall_radius must not be smaller than start_radius (the beam radius at entry)",
        )
        require(np.all(self.divergence >= 0), "divergences must not be negative")
        for attr in ["num_x_bins", "num_x_bins_out"]:
            counts = getattr(self, attr)
            require(np.all(counts == np.round(counts)), f"{attr} must be whole numbers")
            require(np.all(counts > 0), f"{attr} must be positive")
        require(self.angle_step > 0, "angle_step must be positive")
        require(self.angle_step_out > 0, "angle_step_out must be positive")
        require(self.energy_bin_width > 0, "energy_bin_width must be positive")
        require(
            self.stopping_power in STOPPING_POWERS,
            f"stopping_power must be one of {', '.join(STOPPING_POWERS)}, "
            f"got {self.stopping_power!r}",
        )
        require(_is_multiple(180, self.angle_step_out), "angle_step_out must divide 180")
        require(
            _is_multiple(self.angle_step_out, self.angle_step),
            "angle_step_out must be a multiple of angle_step",
        )
        require(
            np.all(self.num_x_bins % self.num_x_bins_out == 0),
            "num_x_bins_out must divide num_x_bins in every region",
        )

    def solve_target_chamber(self):
        """Calculate the target chamber's pressure, density or length (whichever is None)
        that gives beam_energy_backstop at the backstop.

        The beam is slowed down through the upstream regions once, then through the target
        chamber on its internal mesh (num_x_bins), exactly as the full calculation does, for
        trial values. The backstop energy falls steadily as any of them grows, so the value is
        bracketed by doubling and then found by bisection, to a relative precision of 1e-12.
        The result replaces the NaN in self.pressure, self.density or self.length.

        Raises ValueError if the beam is already at or below beam_energy_backstop when it
        enters the target chamber, or if that energy cannot be reached because it is below
        the bottom of the stopping-power table (where the beam is treated as stopped).
        """
        target = self.beam_energy_backstop
        last = self.num_regions - 1
        atom_density = self._atom_densities(self.pressure, self.density)  # NaN if unknown
        E_in = self._slow_down_regions(self.voltage, range(last), atom_density, self.length)[-1]
        if not E_in > target:
            raise ValueError(
                f"{self.name}: the beam enters the target chamber with {E_in:g} keV, not above "
                f"beam_energy_backstop ({target:g} keV)"
            )

        def backstop_energy(value):
            trial = {
                "pressure": self.pressure.copy(),
                "density": self.density.copy(),
                "length": self.length.copy(),
            }
            trial[self.calculated][last] = value
            atoms = self._atom_densities(trial["pressure"], trial["density"])
            return self._slow_down_regions(E_in, [last], atoms, trial["length"])[-1]

        # Bracket the value (torr, g/cm3 or cm), then bisect
        lo, hi = 0.0, 1.0
        while backstop_energy(hi) > target:
            lo, hi = hi, 2 * hi
        while hi - lo > 1e-12 * hi:
            mid = (lo + hi) / 2
            if backstop_energy(mid) > target:
                lo = mid
            else:
                hi = mid
        value = (lo + hi) / 2
        if abs(backstop_energy(value) - target) > 1e-6 * target:
            raise ValueError(
                f"{self.name}: cannot reach beam_energy_backstop = {target:g} keV: below about "
                f"{self.stopping_per_region[last][0][0]:g} keV the beam is treated as stopped"
            )

        getattr(self, self.calculated)[-1] = value
        units = {"pressure": "torr", "density": "g/cm3", "length": "cm"}[self.calculated]
        logger.debug(
            f"{self.name}: target chamber {self.calculated} = {value:.6g} {units} for a beam "
            f"energy of {target:g} keV at the backstop"
        )

    def setup_regions(self):
        """Set the target properties of each region and spread them over the position mesh.

        Every region is filled with a Material (self.materials) and described by its total
        atom density [1/cm^3], the atom fractions of deuterium and tritium (the reaction
        targets) and the stopping power per atom averaged over its composition (see
        read_stopping_power), so the energy loss per unit length is the atom density times
        that stopping.

        Sets self.atom_density_per_region, self.atom_fractions_per_region (isotope -> array),
        self.target_density_per_region and self.target_density_mids (reaction -> density of
        its target atoms [1/cm^3], per region and per internal bin) and self.divergence_mids.
        """
        self.atom_density_per_region = self._atom_densities(self.pressure, self.density)
        self.atom_fractions_per_region = {
            isotope: np.array([m.atom_fractions.get(isotope, 0.0) for m in self.materials])
            for isotope in REACTION_TARGETS.values()
        }
        self.target_density_per_region = {
            rxn: self.atom_density_per_region * self.atom_fractions_per_region[isotope]
            for rxn, isotope in REACTION_TARGETS.items()
        }
        self.target_density_mids = {
            rxn: np.repeat(density, self.num_x_bins)
            for rxn, density in self.target_density_per_region.items()
        }
        self.divergence_mids = np.repeat(self.divergence, self.num_x_bins)

    def _atom_densities(self, pressure, density):
        """Return the atom density [1/cm^3] of each region for the given gas pressures [torr]
        and densities [g/cm^3]: from the ideal gas law (with self.temperature) for a gas given
        by pressure, and from the mass density for a gas given by density or a solid."""
        densities = np.zeros(self.num_regions)
        for r, material in enumerate(self.materials):
            if self.by_pressure[r]:
                molecules = get_molecule_density(pressure[r], self.temperature[r])
                densities[r] = molecules * material.atoms_per_unit
            else:
                rho = material.density if self.solid[r] else density[r]
                mean_mass = sum(
                    f * self.srim_masses[atom] for atom, f in material.atom_fractions.items()
                )
                densities[r] = rho * NA / mean_mass
        return densities

    def _slow_down_regions(self, E0, regions, atom_density, length):
        """Return the beam energy [keV] at each internal grid point through the given regions
        (indices into the region arrays), starting at E0 [keV], for the given atom density
        [1/cm^3] and length [cm] of each of them. See slow_down."""
        grid = [np.array([E0])]
        for r in regions:
            n = self.num_x_bins[r]
            E_table, eps_table = self.stopping_per_region[r]
            region_grid = slow_down(
                grid[-1][-1],
                E_table,
                eps_table,
                np.full(n, atom_density[r]),
                np.full(n, length[r] / n),
            )
            grid.append(region_grid[1:])
        return np.concatenate(grid)

    def setup_meshes(self):
        """Build the internal and output meshes in position and emission angle."""

        # Angle mesh [deg], cosines, and the solid angle of each bin [sr]
        self.num_angle_bins = int(round(180 / self.angle_step))
        self.num_angle_bins_out = int(round(180 / self.angle_step_out))
        self.angle_ratio = int(round(self.num_angle_bins / self.num_angle_bins_out))
        self.angle_grid = np.linspace(0, 180, self.num_angle_bins + 1)
        self.angle_grid_out = np.linspace(0, 180, self.num_angle_bins_out + 1)
        self.angle_mids = (self.angle_grid[:-1] + self.angle_grid[1:]) / 2
        self.angle_mids_out = (self.angle_grid_out[:-1] + self.angle_grid_out[1:]) / 2
        self.cosine_grid = np.cos(self.angle_grid * np.pi / 180)
        self.cosine_grid_out = np.cos(self.angle_grid_out * np.pi / 180)
        self.cosine_mids = np.cos(self.angle_mids * np.pi / 180)
        self.cosine_mids_out = np.cos(self.angle_mids_out * np.pi / 180)
        self.solid_angles = (self.cosine_grid[:-1] - self.cosine_grid[1:]) * 2 * np.pi
        self.solid_angles_out = (self.cosine_grid_out[:-1] - self.cosine_grid_out[1:]) * 2 * np.pi

        # Position mesh [cm]. Each region is divided into equal bins.
        self.num_x_bins_total = np.sum(self.num_x_bins)
        self.num_x_bins_total_out = np.sum(self.num_x_bins_out)
        self.x_ratio_per_region = np.array(
            np.round(self.num_x_bins / self.num_x_bins_out), dtype=int
        )
        self.dx_per_region = self.length / self.num_x_bins
        self.dx_per_region_out = self.length / self.num_x_bins_out
        self.dx = np.repeat(self.dx_per_region, self.num_x_bins)
        self.dx_out = np.repeat(self.dx_per_region_out, self.num_x_bins_out)
        self.x_grid = np.append(0, np.cumsum(self.dx))
        self.x_grid_out = np.append(0, np.cumsum(self.dx_out))
        self.x_mids = (self.x_grid[:-1] + self.x_grid[1:]) / 2
        self.x_mids_out = (self.x_grid_out[:-1] + self.x_grid_out[1:]) / 2

        # Number of internal bins in each output bin (angle_ratio for angle), and the
        # internal grid index of each output grid point
        self.x_ratio_out = np.repeat(self.x_ratio_per_region, self.num_x_bins_out)
        self.out_grid_inds = np.append(0, np.cumsum(self.x_ratio_out))

    def read_stopping_power(self):
        """Read the stopping power of deuterons in each region's material, per atom.

        Stores self.stopping_per_region, an (E, eps) pair for each region: deuteron energy
        [keV] and electronic + nuclear stopping per atom [keV cm^2], averaged over the
        material's atoms (Bragg additivity), so the energy loss per unit length is eps times
        the atom density. Also stores self.srim_masses (target -> atomic mass [u]).

        - "srim": SRIM-2012.01 stopping for deuterons in each element (SRIM_STOPPING_FILE),
          in the material's phase (gas or solid).
        - "icru49": as "srim", but the electronic stopping of hydrogen isotopes comes from ICRU
          Report 49 for protons in hydrogen gas (PSTAR_FILE), taken at the proton energy with
          the same velocity (E_p = E_d * M_H1 / M_H2); electronic stopping does not depend on
          the hydrogen isotope. Only for gas materials, since the data are for hydrogen gas.
          The table covers deuteron energies from about 2 keV upwards; below that the beam is
          treated as stopped.

        Raises ValueError if a material contains an atom with no stopping data, or if
        "icru49" is used with a solid.
        """
        path = os.path.join(self.data_dir, SRIM_STOPPING_FILE)
        E, self.srim_masses, srim = read_srim_h5(path)
        if self.stopping_power == "icru49":
            E_p, P_elec, _ = read_pstar(os.path.join(self.data_dir, PSTAR_FILE))
            E_d = E_p * 1000 * M_H2 / M_H1  # Deuteron energy at the same velocity [keV]
            P_elec = P_elec * 1000 * A_H_ICRU49 / NA  # MeV cm^2/g -> keV cm^2 per atom
            keep = (E >= E_d[0]) & (E <= E_d[-1])
            # Log-log interpolation of the ICRU 49 values onto the SRIM energy grid
            icru_elec = np.exp(np.interp(np.log(E[keep]), np.log(E_d), np.log(P_elec)))
        else:
            keep = np.ones(len(E), dtype=bool)

        cache = {}
        self.stopping_per_region = []
        for material in self.materials:
            if id(material) not in cache:
                if self.stopping_power == "icru49" and material.phase != "gas":
                    raise ValueError(
                        f"{self.name}: stopping_power icru49 is for hydrogen gas only, but "
                        f"material {material.name} is a {material.phase}"
                    )
                eps = np.zeros(int(np.sum(keep)))
                for atom, fraction in material.atom_fractions.items():
                    if atom not in srim[material.phase]:
                        raise ValueError(
                            f"{self.name}: material {material.name}: no stopping data for "
                            f"{atom!r} (elements H-U, and the isotopes D, T and 3He)"
                        )
                    elec, nuc = srim[material.phase][atom]
                    elec = elec[keep]
                    if self.stopping_power == "icru49" and atom in HYDROGEN_ISOTOPES:
                        elec = icru_elec
                    eps += fraction * (elec + nuc[keep])
                cache[id(material)] = (E[keep], eps)
            self.stopping_per_region.append(cache[id(material)])

    def read_cross_sections(self):
        """Read the DD and DT cross sections from the DROSG-2000 files (see read_drosg_koe).

        The centre-of-mass differential cross section is

            dsigma/dOmega_cm(E, mu) = total(E) * sum_l coeffs[l](E) * P_l(mu)

        Stores self.xs_data[rxn] with "E" [keV], "total" [b/sr] and "coeffs".
        """
        self.xs_data = {}
        for rxn in REACTIONS:
            E, total, coeffs = read_drosg_koe(os.path.join(self.data_dir, DROSG_FILES[rxn]))
            self.xs_data[rxn] = {
                "E": E * 1000,  # MeV -> keV
                "total": total / 1000,  # mb/sr -> b/sr
                "coeffs": coeffs,
            }

    def calculate_beam_energy(self):
        """Slow the beam down through the regions.

        Sets self.beam_energy_grid, the beam energy at each internal grid point [keV], and its
        midpoint and output-mesh versions (see slow_down).
        """
        self.beam_energy_grid = self._slow_down_regions(
            self.voltage, range(self.num_regions), self.atom_density_per_region, self.length
        )
        self.beam_energy_mids = (self.beam_energy_grid[:-1] + self.beam_energy_grid[1:]) / 2
        self.beam_energy_grid_out = np.array(self.beam_energy_grid[self.out_grid_inds])

        logger.debug(
            f"{self.name}: beam energy at the end of the target = {self.beam_energy_grid[-1]:.3f} keV"
        )

    def calculate_scraping(self, use_atan=False):
        """Grow the beam radius with divergence and work out how much is scraped.

        The beam profile is assumed flat, so the fraction scraped at radius r > wall_radius
        is 1 - (wall_radius / r)^2. This is conservative. Divergence is used as
        dr/dx = divergence / 1000 (small-angle approximation) unless use_atan is True.

        Sets the beam radius [cm] and scrape fraction on the internal and output meshes.
        Radii are capped at wall_radius after the scrape fraction is found.
        """
        if use_atan:
            divergence_mids = np.arctan(self.divergence_mids / 1000)
        else:
            divergence_mids = self.divergence_mids / 1000
        self.radius_grid = self.start_radius + np.append(0, np.cumsum(divergence_mids * self.dx))

        self.scrape_grid = 1 - (self.wall_radius / self.radius_grid) ** 2
        self.scrape_grid[self.scrape_grid < 0] = 0
        self.scrape_mids = (self.scrape_grid[:-1] + self.scrape_grid[1:]) / 2

        self.radius_grid[self.radius_grid > self.wall_radius] = self.wall_radius
        rsq_mids = (self.radius_grid[:-1] ** 2 + self.radius_grid[1:] ** 2) / 2
        self.radius_mids = np.sqrt(rsq_mids)  # RMS radius in each bin

        # Output mesh: grid values are sampled, midpoint values are averaged
        def condense(vals):
            return np.add.reduceat(vals, self.out_grid_inds[:-1]) / self.x_ratio_out

        self.radius_grid_out = np.array(self.radius_grid[self.out_grid_inds])
        self.scrape_grid_out = np.array(self.scrape_grid[self.out_grid_inds])
        self.radius_mids_out = np.sqrt(condense(rsq_mids))
        self.scrape_mids_out = condense(self.scrape_mids)

    def solve_current(self):
        """Calculate the beam current [mA] that gives the requested total neutron yield.

        The yield is proportional to the current, so the current is the requested yield
        divided by the yield per mA. The yield per mA is summed from the lab cross sections
        the same way bin_neutrons sums the yield, so the final yield matches the request to
        rounding error. Raises ValueError if the case produces no neutrons.
        """
        deuterons_per_mA = 1e-3 / ee  # [1/s]
        yield_per_mA = 0.0
        for rxn in REACTIONS:
            per_bin = np.sum(self.ddxs[rxn] * self.solid_angles[None, :], axis=1)
            yield_per_mA += np.sum(
                per_bin
                * self.target_density_mids[rxn]
                * 1e-24
                * self.dx
                * deuterons_per_mA
                * (1 - self.scrape_mids)
            )
        if yield_per_mA == 0:
            raise ValueError(f"{self.name}: {NO_NEUTRONS}")
        self.current = self.requested_yield_total / yield_per_mA
        logger.debug(
            f"{self.name}: beam current = {self.current:.6g} mA for a total neutron yield of "
            f"{self.requested_yield_total:.4e} n/s"
        )

    def calculate_beam_current(self):
        """Set the deuteron rate [1/s] and the beam current [mA] after scraping, on the
        internal and output meshes."""
        self.deuteron_rate = self.current / 1000 / ee
        self.current_grid = self.current * (1 - self.scrape_grid)
        self.current_mids = (self.current_grid[:-1] + self.current_grid[1:]) / 2
        self.current_grid_out = np.array(self.current_grid[self.out_grid_inds])
        self.current_mids_out = (
            np.add.reduceat(self.current_mids, self.out_grid_inds[:-1]) / self.x_ratio_out
        )

    def calculate_beam_power(self):
        """Work out where the beam power goes: into the target material, onto the wall or to the backstop.

        Beam power is beam energy times current (keV x mA = W). Along each bin, the beam
        still in it deposits the energy it loses in the target material (current x energy loss), and the
        beam scraped off hits the wall carrying its energy there (energy x current lost). Both
        use the bin averages of the other quantity, which makes the three parts add up to the
        entry power exactly. The power reaching the backstop is the remaining current times
        the final beam energy.

        Sets self.deposited_power_mids and self.wall_power_mids [kW] per bin on the internal and
        output meshes (output bins are sums), and the totals self.deposited_power,
        self.wall_power and self.backstop_power [kW].
        """
        beam_energy_mids = (self.beam_energy_grid[:-1] + self.beam_energy_grid[1:]) / 2
        self.deposited_power_mids = self.current_mids * -np.diff(self.beam_energy_grid) / 1000
        self.wall_power_mids = beam_energy_mids * -np.diff(self.current_grid) / 1000
        self.deposited_power_mids_out = np.add.reduceat(
            self.deposited_power_mids, self.out_grid_inds[:-1]
        )
        self.wall_power_mids_out = np.add.reduceat(self.wall_power_mids, self.out_grid_inds[:-1])
        self.deposited_power = np.sum(self.deposited_power_mids)
        self.wall_power = np.sum(self.wall_power_mids)
        self.backstop_power = self.beam_energy_grid[-1] * self.current_grid[-1] / 1000
        logger.debug(
            f"{self.name}: beam power {self.voltage * self.current / 1000:.4g} kW: "
            f"{self.deposited_power:.4g} kW deposited, {self.wall_power:.4g} kW onto the wall, "
            f"{self.backstop_power:.4g} kW to the backstop"
        )

    def calculate_ddxs(self):
        """Calculate lab-frame cross sections and neutron energies.

        For each reaction, sets self.ddxs[rxn], the double-differential cross section in the
        lab frame [b/sr], and self.neutron_energy[rxn], the neutron energy [MeV], on the internal
        (position, angle) mesh. Both use the beam energy at the bin midpoint and
        relativistic kinematics (see two_body_kinematics).
        """
        self.ddxs = {}
        self.neutron_energy = {}

        for rxn, (M1, M2, M3, M4, _Q) in REACTIONS.items():
            E_table = self.xs_data[rxn]["E"]
            coeff_table = self.xs_data[rxn]["coeffs"]

            # Double-differential cross sections [b/sr]: the CM angular distribution times
            # the CM-to-lab solid angle factor; and neutron energies [MeV]. Both come from
            # relativistic two-body kinematics at each lab angle.
            self.ddxs[rxn] = np.zeros((self.num_x_bins_total, self.num_angle_bins))
            self.neutron_energy[rxn] = np.zeros((self.num_x_bins_total, self.num_angle_bins))
            for i, E in enumerate(self.beam_energy_mids):
                neutron_energy, mu_cm, jacobian = two_body_kinematics(
                    E / 1000, self.cosine_mids, M1, M2, M3, M4
                )
                self.neutron_energy[rxn][i, :] = neutron_energy
                total = np.interp(E, E_table, self.xs_data[rxn]["total"], left=0, right=np.nan)
                coeffs = np.array(
                    [
                        np.interp(E, E_table, coeff_table[:, j], left=0, right=np.nan)
                        for j in range(coeff_table.shape[1])
                    ]
                )
                # Evaluate the Legendre series only up to its last non-zero coefficient: the
                # higher coefficients are zero at beam energies below about 400 keV
                nonzero = np.flatnonzero(coeffs)
                coeffs = coeffs[: nonzero[-1] + 1] if len(nonzero) else coeffs[:1]
                self.ddxs[rxn][i, :] = total * legval(mu_cm, coeffs) * jacobian

    def bin_neutrons(self):
        """Tally neutrons onto the output (position, angle, energy) mesh.

        The neutron rate from each internal (position, angle) cell is
        sigma [b/sr] * 1e-24 [cm^2/b] * n_target [1/cm^3] * dx [cm] * dOmega [sr]
        * deuteron rate [1/s], reduced by the scrape fraction. It is added to the output
        cell containing that position, angle and neutron energy.

        Sets, for each reaction, self.neutrons[rxn] [n/s] with shape
        (num_x_bins_total_out, num_angle_bins_out, num_energy_bins[rxn]), its energy mesh
        self.neutron_energy_grid[rxn] [MeV], the yield self.yields[rxn] [n/s] and the fraction
        of neutrons lost to scraping self.neutron_scrape_fractions[rxn]. The totals over both
        reactions are self.yield_total and self.neutron_scrape_fraction.
        """
        self.num_energy_bins = {}
        self.neutron_energy_grid = {}
        self.neutron_energy_mids = {}
        self.neutrons = {}
        self.yields = {}
        self.neutron_scrape_fractions = {}

        for rxn in REACTIONS:
            # Neutron energy mesh: spans this reaction's energies, aligned to energy_bin_width
            energy_min = (
                np.floor(np.min(self.neutron_energy[rxn]) / self.energy_bin_width)
                * self.energy_bin_width
            )
            energy_max = (
                np.ceil(np.max(self.neutron_energy[rxn]) / self.energy_bin_width)
                * self.energy_bin_width
            )
            self.num_energy_bins[rxn] = int(
                round((energy_max - energy_min) / self.energy_bin_width)
            )
            self.neutron_energy_grid[rxn] = np.linspace(
                energy_min, energy_max, self.num_energy_bins[rxn] + 1
            )
            self.neutron_energy_mids[rxn] = (
                self.neutron_energy_grid[rxn][:-1] + self.neutron_energy_grid[rxn][1:]
            ) / 2

            # Output-mesh indices (i_out, j_out, k_out) of every internal (position, angle) cell
            offsets = np.cumsum(np.concatenate(([0], self.num_x_bins_out[:-1])))
            i_out = np.concatenate(
                [
                    np.floor_divide(np.arange(num_x_bins)[:, None], x_ratio) + offset
                    for num_x_bins, x_ratio, offset in zip(
                        self.num_x_bins, self.x_ratio_per_region, offsets
                    )
                ],
                axis=0,
            )
            j_out = np.floor_divide(np.arange(self.num_angle_bins)[None, :], self.angle_ratio)
            i_out = np.repeat(i_out, self.num_angle_bins, axis=1).astype(int).ravel()
            j_out = np.repeat(j_out, self.num_x_bins_total, axis=0).astype(int).ravel()
            k_out = (
                np.searchsorted(self.neutron_energy_grid[rxn], self.neutron_energy[rxn].ravel()) - 1
            )

            # Neutron rate in each internal cell [n/s], with and without scraping
            areas = np.tile(self.solid_angles, self.num_x_bins_total)
            factor = (self.target_density_mids[rxn] * 1e-24 * self.dx * self.deuteron_rate)[:, None]
            vals_scr = (self.ddxs[rxn] * factor * (1 - self.scrape_mids[:, None])).ravel() * areas
            vals_tot = (self.ddxs[rxn] * factor).ravel() * areas

            self.neutrons[rxn] = np.zeros(
                (self.num_x_bins_total_out, self.num_angle_bins_out, self.num_energy_bins[rxn])
            )
            np.add.at(self.neutrons[rxn], (i_out, j_out, k_out), vals_scr)

            self.yields[rxn] = np.sum(self.neutrons[rxn])
            logger.debug(f"{self.name}: {rxn} neutron yield = {self.yields[rxn]:.4e} n/s")

            if np.sum(vals_tot) == 0:
                self.neutron_scrape_fractions[rxn] = 0.0
            else:
                self.neutron_scrape_fractions[rxn] = 1 - np.sum(vals_scr) / np.sum(vals_tot)

        self.yield_total = self.yields["DD"] + self.yields["DT"]
        if self.yield_total == 0:
            raise ValueError(f"{self.name}: {NO_NEUTRONS}")
        self.neutron_scrape_fraction = (
            self.yields["DD"] * self.neutron_scrape_fractions["DD"]
            + self.yields["DT"] * self.neutron_scrape_fractions["DT"]
        ) / self.yield_total
        logger.debug(
            f"{self.name}: total neutron yield = {self.yield_total:.4e} n/s, "
            f"{self.neutron_scrape_fraction * 100:.3f} % lost to beam scraping"
        )

    def unique_materials(self):
        """Return the materials used, each once, in order of first use."""
        unique = []
        for material in self.materials:
            if all(material is not m for m in unique):
                unique.append(material)
        return unique

    def get_inputs(self):
        """Return the input parameters as a list of (name, value, units).

        A calculated target chamber pressure or length is given as calculated, and its units
        say so, as does a calculated current. beam_energy_backstop and yield_total are
        included only if they were given, and density only if some gas is given by density.
        material lists each region's material by name (see unique_materials).
        """
        calculated = {
            name: ", target chamber value calculated" if self.calculated == name else ""
            for name in ("pressure", "density", "length")
        }
        unused = "; nan where unused" if not np.all(self.by_pressure) else ""
        density = []
        if np.any(~self.solid & ~self.by_pressure):
            density = [("density", self.density, "g/cm3" + calculated["density"] + unused)]
        material_names = [m.name for m in self.materials]
        requested_yield = []
        if self.requested_yield_total is not None:
            requested_yield = [("yield_total", self.requested_yield_total, "n/s, requested")]
        requested = []
        if self.beam_energy_backstop is not None:
            requested = [("beam_energy_backstop", self.beam_energy_backstop, "keV, requested")]
        return (
            [
                ("voltage", self.voltage, "kV"),
                ("current", self.current, "mA, calculated" if self.current_calculated else "mA"),
            ]
            + requested_yield
            + [
                ("material", material_names, "per region; see [materials]"),
                ("pressure", self.pressure, "torr" + calculated["pressure"] + unused),
                ("temperature", self.temperature, "degC" + unused),
            ]
            + density
            + [
                ("length", self.length, "cm" + calculated["length"]),
            ]
            + requested
            + [
                ("start_radius", self.start_radius, "cm"),
                ("wall_radius", self.wall_radius, "cm"),
                ("divergence", self.divergence, "mrad"),
                ("num_x_bins", self.num_x_bins, ""),
                ("num_x_bins_out", self.num_x_bins_out, ""),
                ("angle_step", self.angle_step, "deg"),
                ("angle_step_out", self.angle_step_out, "deg"),
                ("energy_bin_width", self.energy_bin_width, "MeV"),
                ("stopping_power", self.stopping_power, ""),
            ]
        )

    def write_hdf5(self, path, run_info=None):
        """Write the source on the output mesh to an HDF5 file at path.

        File attributes: format ("shine-neutron-source") and format_version (HDF5_FORMAT_VERSION,
        increased whenever the layout changes), name (case name), code_version (see
        get_code_version), created (UTC time) and any run_info entries (such as the input
        file). The inputs group holds every input parameter (see get_inputs) and the results
        group the key results (see get_results), one dataset each with a "units" attribute.
        The materials group has a group for each material used, whose attributes give its
        phase, density (solids), composition (composition_kind "atoms" or "molecules", with
        composition_formulas and composition_amounts) and atom fractions.

        Datasets, each with "units" and "description" attributes ("grid" arrays hold values at
        mesh boundaries, "mids" arrays one value per bin):

        - x_grid: position mesh along the beam.
        - beam_energy_grid, radius_grid, scrape_grid, current_grid: beam energy, edge radius, scraped
          fraction and current at each position grid point.
        - beam_energy_mids, radius_mids, scrape_mids, current_mids: the same per position bin, as
          used in the SDEF card and axial_profile.txt.
        - angle_grid: emission angle mesh, relative to the beam direction.
        - neutron_energy_grid_DD, neutron_energy_grid_DT: neutron energy mesh for each reaction.
        - neutrons_DD, neutrons_DT: neutron emission rate in each (position, angle, energy)
          cell, with shape (len(x_grid) - 1, len(angle_grid) - 1, len(neutron_energy_grid_<rxn>) - 1); their
          "dimensions" attribute names the axes.
        """
        _make_parent_dir(path)
        datasets = [
            ("x_grid", self.x_grid_out, "cm", "position mesh along the beam (bin boundaries)"),
            ("beam_energy_grid", self.beam_energy_grid_out, "keV", "beam energy at each position grid point"),
            ("radius_grid", self.radius_grid_out, "cm",
             "beam edge radius at each position grid point, capped at the wall radius"),
            ("scrape_grid", self.scrape_grid_out, "fraction",
             "fraction of the beam scraped off at each position grid point"),
            ("current_grid", self.current_grid_out, "mA", "beam current at each position grid point"),
            ("beam_energy_mids", (self.beam_energy_grid_out[:-1] + self.beam_energy_grid_out[1:]) / 2, "keV",
             "beam energy at the centre of each position bin"),
            ("radius_mids", self.radius_mids_out, "cm",
             "beam edge radius over each position bin (root mean square), as in the SDEF"),
            ("scrape_mids", self.scrape_mids_out, "fraction",
             "fraction of the beam scraped off, averaged over each position bin"),
            ("current_mids", self.current_mids_out, "mA",
             "beam current, averaged over each position bin"),
            ("angle_grid", self.angle_grid_out, "deg", "emission angle mesh, from the beam direction"),
            ("neutron_energy_grid_DD", self.neutron_energy_grid["DD"], "MeV", "neutron energy mesh for D(d,n)3He"),
            ("neutron_energy_grid_DT", self.neutron_energy_grid["DT"], "MeV", "neutron energy mesh for T(d,n)4He"),
            ("neutrons_DD", self.neutrons["DD"], "n/s",
             "D(d,n)3He neutron emission rate in each (position, angle, energy) cell"),
            ("neutrons_DT", self.neutrons["DT"], "n/s",
             "T(d,n)4He neutron emission rate in each (position, angle, energy) cell"),
        ]  # fmt: skip
        with h5py.File(path, "w") as hf:
            hf.attrs["format"] = "shine-neutron-source"
            hf.attrs["format_version"] = HDF5_FORMAT_VERSION
            hf.attrs["name"] = self.name
            hf.attrs["code_version"] = get_code_version()
            hf.attrs["created"] = self.created
            for key, value in (run_info or {}).items():
                hf.attrs[key] = value
            for group_name, entries in [
                ("inputs", self.get_inputs()),
                ("results", self.get_results()),
            ]:
                group = hf.create_group(group_name)
                for key, value, units in entries:
                    if isinstance(value, list) and value and isinstance(value[0], str):
                        value = np.array(value, dtype=h5py.string_dtype())
                    group.create_dataset(key, data=value)
                    group[key].attrs["units"] = units
            materials = hf.create_group("materials")
            for material in self.unique_materials():
                g = materials.create_group(material.name)
                g.attrs["phase"] = material.phase
                if material.density is not None:
                    g.attrs["density"] = material.density
                    g.attrs["density_units"] = "g/cm3"
                g.attrs["composition_kind"] = material.kind
                g.attrs["composition_formulas"] = list(material.composition)
                g.attrs["composition_amounts"] = list(material.composition.values())
                g.attrs["atoms"] = list(material.atom_fractions)
                g.attrs["atom_fractions"] = list(material.atom_fractions.values())
            for key, data, units, description in datasets:
                hf.create_dataset(key, data=data, compression="gzip")
                hf[key].attrs["units"] = units
                hf[key].attrs["description"] = description
            for rxn in ("DD", "DT"):
                hf[f"neutrons_{rxn}"].attrs["dimensions"] = ["position", "angle", "energy"]

    def get_mcnp_sdef(self, direction="-z", run_info=None, entry_position=(0.0, 0.0, 0.0)):
        """Return an MCNP SDEF source definition as a string.

        The source is a cylinder along the beam axis, starting where the beam enters the first
        gas region (entry_position) and extending along the beam direction. POS is set from
        entry_position for every direction, including -z, whose card runs along +z with POS at
        the backstop end (see SDEF_DIRECTIONS). Distributions:

        - d1: position along the axis (EXT).
        - d2: emission direction relative to VEC (omitted for an isotropic source).
        - d3: radius, dependent on position (via distributions 101, 102, ...) when the
          beam diverges.
        - d4: neutron energy, dependent on emission angle (via distributions 501, 502,
          ...) unless the source is isotropic.

        The comments at the top are the case summary (see get_summary), including the
        direction and entry position, then notes on using the source: how to normalize tallies,
        where the source lies, and that neutron energy does not depend on position.

        Args:
            direction: Beam direction, one of SDEF_DIRECTIONS ("-x", "+x", ..., "+z").
            run_info: Optional extra entries for the summary's [run] section.
            entry_position: Point where the beam enters the first gas region, (x, y, z) [cm].
        """
        if direction not in SDEF_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {', '.join(SDEF_DIRECTIONS)}, got {direction!r}"
            )
        entry = np.array(entry_position, dtype=float)
        if entry.shape != (3,) or not np.all(np.isfinite(entry)):
            raise ValueError(
                f"entry_position must be three numbers (x, y, z), got {entry_position!r}"
            )
        axs, vec, reverse_x = SDEF_DIRECTIONS[direction]
        length = self.x_grid_out[-1]
        backstop = entry + length * np.array(vec.split(), dtype=float)
        pos = backstop if reverse_x else entry  # For -z the card runs along +z from the backstop

        def point(p):
            return " ".join(f"{v + 0.0:.12g}" for v in p)  # + 0.0 turns -0.0 into 0

        divergent = not np.all(self.divergence == 0)
        isotropic = self.angle_step_out == 180
        has_DD = "DD" in self.reactions_present()
        has_DT = "DT" in self.reactions_present()
        dashed_line = "c " + "-" * 78 + "\n"

        run = {**(run_info or {}), "sdef_direction": direction, "sdef_entry_position": entry}
        s = self._sdef_header(dashed_line, run)

        # Notes on using the source
        s += "c\n"
        s += "c     Using this source:\n"
        s += "c     - Tallies are per source neutron: multiply them by yield_total\n"
        s += f"c       ({self.yield_total:.6e} n/s) for absolute rates.\n"
        s += f"c     - The beam enters at ({point(entry)}) cm, travels along {direction} and\n"
        s += (
            f"c       reaches the backstop at ({point(backstop)}) cm, {length:.6g} cm further on.\n"
        )
        if reverse_x:
            s += "c       This -z card runs along +z, so POS is the backstop end.\n"
        if isotropic:
            s += "c     - Emission is isotropic, with one energy spectrum for all positions.\n"
        else:
            s += "c     - Neutron energy depends on emission angle but not on position along the\n"
            s += "c       beam: the drop in neutron energy as the beam slows is averaged out.\n"
            s += "c       This is a limitation of MCNP: an SDEF distribution can depend on only\n"
            s += "c       one other variable, so energy depends on direction (ERG = FDIR) and\n"
            s += "c       cannot also depend on position along the beam.\n"
            s += "c     - The angular distribution is also averaged over position along the\n"
            s += "c       beam, again because of MCNP: making direction depend on position\n"
            s += "c       (DIR = FEXT) is not accepted together with ERG = FDIR.\n"
        s += "c\n"
        s += dashed_line

        # SDEF card
        s += "c\n"
        s += "  sdef  par = n\n"
        s += f"        pos = {point(pos)}\n"
        s += f"        axs = {axs}\n"
        s += f"        vec = {vec}\n"
        s += "        ext = d1\n"
        if not isotropic:
            s += "        dir = d2\n"
        s += "        rad = fext = d3\n" if divergent else "        rad = d3\n"
        s += "        erg = d4\n" if isotropic else "        erg = fdir = d4\n"
        s += "c\n"
        s += dashed_line

        # Extent distribution: region boundaries, with interpolated points inside each region
        ext = np.sum(self.neutrons["DD"], (1, 2)) + np.sum(self.neutrons["DT"], (1, 2))
        nvals = self.num_x_bins_out
        Lvals = self.length
        if reverse_x:
            ext = ext[::-1]
            nvals = nvals[::-1]
            Lvals = Lvals[::-1]
        si1 = "0"
        for n, x in zip(nvals, np.cumsum(Lvals)):  # x: position of each region's far end
            if n == 2:
                si1 += " i"
            elif n > 2:
                si1 += f" {n - 1}i"
            si1 += f" {_clean_float_text(x)}"
        si1 = textwrap.fill(si1, width=72).replace("\n", "\n        ")
        s += "c\n"
        s += "  sc1   Extent distribution\n"
        s += f"  si1   {si1}\n"
        s += "  sp1  " + _format_values(np.append(0, ext), ".5e", 6) + "\n"
        s += "c\n"
        s += dashed_line

        # Angular distribution. MCNP needs increasing cosines, so bins run from 180 to 0 deg.
        if not isotropic:
            dir_vals = np.sum(self.neutrons["DD"], (0, 2)) + np.sum(self.neutrons["DT"], (0, 2))
            s += "c\n"
            s += "  sc2   Angular distribution\n"
            # Three cosines per line keeps the lines within 80 characters
            s += "  si2  " + _format_values(self.cosine_grid_out[::-1], _cosine_text, 3) + "\n"
            s += "  sp2  " + _format_values(np.append(0, dir_vals[::-1]), "11.5e", 6) + "\n"
            s += "c\n"
            s += dashed_line

        # Radial distribution: uniform in area out to the beam edge radius of each bin
        s += "c\n"
        s += "  sc3   Radial distribution\n"
        if divergent:
            rmids = self.radius_mids_out[::-1] if reverse_x else self.radius_mids_out
            smids = self.scrape_mids_out[::-1] if reverse_x else self.scrape_mids_out
            s += "  ds3 S" + _format_values(101 + np.arange(len(rmids)), "3d", 18) + "\n"
            s += "c\n"
            for i, (r, scraped) in enumerate(zip(rmids, smids)):
                s += f"  si{101 + i:<3d} 0 {r:.{SDEF_RADIUS_DECIMALS}f}"
                if scraped > 0:
                    s += f"  $ ({scraped * 100:.2f}% scraped)"
                s += "\n"
        else:
            s += f"  si3   0 {self.start_radius:.{SDEF_RADIUS_DECIMALS}f}\n"
        s += "c\n"
        s += dashed_line

        # Energy distribution: DD and DT bins side by side, each starting with a zero bin
        s += "c\n"
        s += "  sc4   Energy distribution\n"
        if isotropic:
            spectra = [np.sum(self.neutrons[rxn], (0, 1)) for rxn in ["DD", "DT"]]
            si, sp = self._sdef_energy_bins(spectra, has_DD, has_DT, trim=False)
            s += f"  si4  {si}\n"
            s += "  sp4  " + _format_values(sp, ".5e", 6) + "\n"
            s += "c\n"
        else:
            sids = 501 + np.arange(self.num_angle_bins_out)
            s += "  ds4 S" + _format_values(sids, "3d", 18) + "\n"
            s += "c\n"
            for i, sid in enumerate(sids):
                # Distribution i covers angle bin i counted from 180 degrees, matching si2; k is
                # that bin's index in angle_grid_out, which runs from 0 degrees
                k = self.num_angle_bins_out - 1 - i
                spectra = [np.sum(self.neutrons[rxn][:, k, :], 0) for rxn in ["DD", "DT"]]
                si, sp = self._sdef_energy_bins(spectra, has_DD, has_DT, trim=True)
                s += (
                    f"  sc{sid:<3d} Energy distribution "
                    f"({self.angle_grid_out[k]:.12g} - {self.angle_grid_out[k + 1]:.12g} degrees)\n"
                )
                s += f"  si{sid:<3d}{si}\n"
                s += f"  sp{sid:<3d}" + _format_values(sp, ".5e", 6) + "\n"
                s += "c\n"
        s += dashed_line

        return s

    def _sdef_header(self, dashed_line, run_info):
        """Return the comment block at the top of the SDEF card: the case summary (see
        get_summary) with each line as an MCNP comment."""
        s = dashed_line
        s += f"c     Source definition for {self.name}, made by shine-neutron-source\n"
        s += "c     Case summary (TOML; the same as summary.toml written by write_text_files):\n"
        s += dashed_line
        s += "c\n"
        for line in self.get_summary(run_info, title=False).splitlines():
            s += f"c {line}".rstrip() + "\n"
        s += "c\n"
        s += dashed_line
        return s

    def _sdef_energy_bins(self, spectra, has_DD, has_DT, trim):
        """Return the SI entries and SP values of an energy distribution.

        spectra holds the DD and DT spectra on their own energy meshes. If trim is True,
        empty bins at either end of each spectrum are dropped.
        """
        si = ""
        sp = []
        for rxn, spectrum, present in zip(["DD", "DT"], spectra, [has_DD, has_DT]):
            if not present:
                continue
            ind_min, ind_max = 0, len(spectrum)
            if trim:
                inds = np.nonzero(spectrum)[0]
                ind_min, ind_max = inds[0], inds[-1] + 1
            E_min = round(self.neutron_energy_grid[rxn][ind_min], 12)
            E_max = round(self.neutron_energy_grid[rxn][ind_max], 12)
            si += f" {E_min} {ind_max - ind_min - 1}i {E_max}"
            sp = np.append(sp, np.append(0, spectrum[ind_min:ind_max]))
        return si, sp

    def write_mcnp_sdef(self, path, direction="-z", run_info=None, entry_position=(0.0, 0.0, 0.0)):
        """Write the MCNP SDEF card (see get_mcnp_sdef) to a text file at path.

        The file is ASCII with Unix (LF) line endings on every platform.
        """
        _make_parent_dir(path)
        with open(path, "w", encoding="ascii", newline="\n") as w:
            w.write(self.get_mcnp_sdef(direction, run_info, entry_position))

    def reactions_present(self):
        """Return the reactions the target can produce: "DD" if any region contains deuterium,
        "DT" if any contains tritium."""
        return [rxn for rxn in REACTIONS if np.any(self.target_density_per_region[rxn] > 0)]

    def neutrons_by_angle_and_energy(self):
        """Return the neutron source integrated over position, on one energy mesh for both
        reactions.

        Returns (neutron_energy_grid, neutrons): neutron_energy_grid [MeV] runs from the lowest to the highest neutron
        energy of the reactions present, in steps of energy_bin_width (the reactions' own meshes are
        aligned with it), and neutrons[rxn] [n/s] has shape (num_angle_bins_out, len(neutron_energy_grid) - 1) for
        "DD" and "DT" (zero for a reaction the gas cannot produce).
        """
        present = self.reactions_present()
        # Each reaction's mesh starts and ends on a multiple of energy_bin_width: work in those indices
        first = {
            rxn: int(round(self.neutron_energy_grid[rxn][0] / self.energy_bin_width))
            for rxn in present
        }
        i_lo = min(first.values())
        i_hi = max(
            int(round(self.neutron_energy_grid[rxn][-1] / self.energy_bin_width)) for rxn in present
        )
        neutron_energy_grid = np.arange(i_lo, i_hi + 1) * self.energy_bin_width
        neutrons = {}
        for rxn in ("DD", "DT"):
            neutrons[rxn] = np.zeros((self.num_angle_bins_out, i_hi - i_lo))
            if rxn in present:
                by_angle = np.sum(self.neutrons[rxn], 0)
                i = first[rxn] - i_lo
                neutrons[rxn][:, i : i + by_angle.shape[1]] = by_angle
        return neutron_energy_grid, neutrons

    def get_results(self):
        """Return the key results as a list of (name, value, units or description)."""
        results = [
            ("yield_total", self.yield_total, "n/s"),
            ("yield_DD", self.yields["DD"], "n/s"),
            ("yield_DT", self.yields["DT"], "n/s"),
            ("fraction_DD", self.yields["DD"] / self.yield_total, "share of the total yield"),
            ("fraction_DT", self.yields["DT"] / self.yield_total, "share of the total yield"),
        ]
        for rxn in self.reactions_present():
            spectrum = np.sum(self.neutrons[rxn], (0, 1))
            mean = np.sum(spectrum * self.neutron_energy_mids[rxn]) / np.sum(spectrum)
            results.append((f"mean_neutron_energy_{rxn}", mean, "MeV, over all angles"))
        region_ends = np.cumsum(self.num_x_bins)
        results += [
            ("atom_density", self.atom_density_per_region, "atoms/cm3, in each region"),
            (
                "beam_energy_region_end",
                self.beam_energy_grid[region_ends],
                "keV, at the end of each region",
            ),
            ("beam_energy_backstop", self.beam_energy_grid[-1], "keV"),
            ("beam_radius_backstop", self.radius_grid[-1], "cm, beam edge; capped at the wall"),
            ("beam_current_backstop", self.current_grid[-1], "mA, after scraping"),
            ("beam_power_entry", self.voltage * self.current / 1000, "kW"),
            (
                "beam_power_deposited",
                self.deposited_power,
                "kW, deposited in the target material (gas or solid)",
            ),
            ("beam_power_wall", self.wall_power, "kW, carried onto the wall by scraping"),
            ("beam_power_backstop", self.backstop_power, "kW"),
            ("neutron_scrape_fraction", self.neutron_scrape_fraction, "neutrons lost to scraping"),
        ]
        return results

    def get_summary(self, run_info=None, title=True):
        """Return the key inputs and results as TOML text (readable with tomllib).

        Sections: [run] (case name, code version, creation time and any run_info entries, such
        as the input file and SDEF direction), [inputs] (the ShineSource arguments, see
        get_inputs; lists have one value per gas region in beam order) and [results] (see
        get_results), followed by a [materials.<name>] table for each material used (phase,
        density for a solid, and its atoms or molecules). Units are given in comments. title adds a first
        comment line saying what the file is. The "=" signs are aligned across all sections, and so are the comments
        (except after values too long for the comment column).
        """
        sections = {
            "run": [
                ("name", self.name, ""),
                ("code_version", get_code_version(), ""),
                ("created", self.created, ""),  # ISO 8601 with "+00:00": UTC
            ]
            + [(name, value, "") for name, value in (run_info or {}).items()],
            "inputs": self.get_inputs(),
            "results": self.get_results(),
        }
        key_width = max(len(name) for entries in sections.values() for name, _, _ in entries)
        comment_column = key_width + 19  # Room for " = " and a 16-character value

        s = "# Neutron source summary (TOML; read it in Python with tomllib)\n\n" if title else ""
        for i, (section, entries) in enumerate(sections.items()):
            s += ("\n" if i else "") + f"[{section}]\n"
            for name, value, comment in entries:
                text = f"{name:<{key_width}} = {_toml_value(value)}"
                s += f"{text:<{comment_column}} # {comment}\n" if comment else text + "\n"
        for material in self.unique_materials():
            s += f"\n[materials.{_toml_key(material.name)}]\n"
            s += f"{'phase':<{key_width}} = {_toml_value(material.phase)}\n"
            if material.density is not None:
                text = f"{'density':<{key_width}} = {_toml_value(material.density)}"
                s += f"{text:<{comment_column}} # g/cm3\n"
            composition = ", ".join(
                f"{_toml_key(k)} = {_toml_value(v)}" for k, v in material.composition.items()
            )
            s += f"{material.kind:<{key_width}} = {{ {composition} }}\n"
        return s

    def write_text_files(self, directory, run_info=None):
        """Write a summary and integrated neutron profiles as text files in directory.

        - summary.toml: key inputs and results (see get_summary; run_info adds entries to
          its [run] section).
        - axial_profile.txt: neutrons per output position bin, integrated over angle and
          energy, with the beam energy, radius and scraped fraction, and the beam power
          deposited in the target material and carried onto the wall in each bin.
        - energy_spectrum.txt: neutrons per energy bin, integrated over position and angle.
        - angular_distribution.txt: neutrons per emission-angle bin (angle from the beam
          direction), integrated over position and energy, also per steradian.

        The profile files start with "#" comment lines describing the columns and can be read
        with numpy.loadtxt; summary.toml can be read with tomllib. Each profile sums to the
        neutron yields. All files are ASCII with Unix (LF) line endings.
        """
        os.makedirs(directory, exist_ok=True)
        summary_path = os.path.join(directory, "summary.toml")
        with open(summary_path, "w", encoding="ascii", newline="\n") as w:
            w.write(self.get_summary(run_info))

        def write_table(fname, title, columns, data):
            # savetxt puts the comment character before each line; blank lines stay a bare "#"
            header = f" {title}\n Case: {self.name}\n\n Columns:\n"
            header += "".join(f"   {i + 1}  {c}\n" for i, c in enumerate(columns))
            with open(os.path.join(directory, fname), "w", encoding="ascii", newline="\n") as w:
                np.savetxt(w, np.column_stack(data), fmt="%13.6e", header=header, comments="#")

        axial = {rxn: np.sum(self.neutrons[rxn], (1, 2)) for rxn in ("DD", "DT")}
        write_table(
            "axial_profile.txt",
            "Neutron source along the beam, integrated over emission angle and neutron energy",
            [
                "start of position bin [cm]",
                "end of position bin [cm]",
                "beam energy at the bin centre [keV]",
                "beam edge radius over the bin [cm] (root mean square; capped at the wall)",
                "fraction of the beam scraped off, averaged over the bin",
                "DD neutrons [n/s]",
                "DT neutrons [n/s]",
                "total neutrons [n/s]",
                "beam power deposited in the target material over the bin [kW]",
                "beam power carried onto the wall by scraping over the bin [kW]",
            ],
            [
                self.x_grid_out[:-1],
                self.x_grid_out[1:],
                (self.beam_energy_grid_out[:-1] + self.beam_energy_grid_out[1:]) / 2,
                self.radius_mids_out,
                self.scrape_mids_out,
                axial["DD"],
                axial["DT"],
                axial["DD"] + axial["DT"],
                self.deposited_power_mids_out,
                self.wall_power_mids_out,
            ],
        )

        neutron_energy_grid, by_angle_energy = self.neutrons_by_angle_and_energy()
        energy = {rxn: np.sum(by_angle_energy[rxn], 0) for rxn in ("DD", "DT")}
        write_table(
            "energy_spectrum.txt",
            "Neutron energy spectrum, integrated over position and emission angle",
            [
                "start of energy bin [MeV]",
                "end of energy bin [MeV]",
                "DD neutrons [n/s]",
                "DT neutrons [n/s]",
                "total neutrons [n/s]",
            ],
            [
                neutron_energy_grid[:-1],
                neutron_energy_grid[1:],
                energy["DD"],
                energy["DT"],
                energy["DD"] + energy["DT"],
            ],
        )

        angular = {rxn: np.sum(self.neutrons[rxn], (0, 2)) for rxn in ("DD", "DT")}
        total = angular["DD"] + angular["DT"]
        write_table(
            "angular_distribution.txt",
            "Neutron angular distribution (angle from the beam direction), integrated over "
            "position and neutron energy",
            [
                "start of angle bin [deg]",
                "end of angle bin [deg]",
                "solid angle of the bin [sr]",
                "DD neutrons [n/s]",
                "DT neutrons [n/s]",
                "total neutrons [n/s]",
                "total neutrons per steradian [n/s/sr]",
            ],
            [
                self.angle_grid_out[:-1],
                self.angle_grid_out[1:],
                self.solid_angles_out,
                angular["DD"],
                angular["DT"],
                total,
                total / self.solid_angles_out,
            ],
        )

    def make_plots(self, directory):
        """Write plots of the beam and neutron source to directory as PNG files.

        - beam.png: beam energy, radius (with the wall), current and the power deposited in
          the gas and on the wall, along the beam.
        - neutron_position.png, neutron_position_log.png: neutrons per unit length along the
          beam, on linear and log scales.
        - neutron_angle.png: neutrons per steradian against emission angle.
        - neutron_energy_angle_DD.png, neutron_energy_angle_DT.png (for the reactions
          present): neutron energy against emission angle, coloured by intensity.
        - neutron_energy.png, neutron_energy_log.png: the neutron energy spectrum,
          integrated over angle, on linear and log scales.

        Position plots run from where the beam enters the first gas region, with dashed lines
        at the boundaries between regions. File names do not include the case name, so use a
        separate directory per case.
        """
        os.makedirs(directory, exist_ok=True)
        self.plot_beam(directory)
        self.plot_neutron_position(directory)
        self.plot_neutron_angle(directory)
        self.plot_neutron_energy(directory)

    def _position_units(self):
        """Return (scale, unit) for position plots: micrometres if the whole source is shorter
        than 1 mm (such as a solid target), centimetres otherwise. Positions in cm are
        multiplied by scale; quantities per cm are divided by it."""
        if self.x_grid_out[-1] < 0.1:
            return 1e4, r"$\mu$m"
        return 1.0, "cm"

    def _mark_regions(self, ax):
        """Draw dashed lines at the boundaries between regions."""
        scale, _ = self._position_units()
        for x in np.cumsum(self.length)[:-1]:
            ax.axvline(x * scale, color="gray", linestyle="--", linewidth=0.8)

    def _set_position_axis(self, ax, mark_regions=True):
        """Label the position axis and (unless mark_regions is False) mark the regions."""
        scale, unit = self._position_units()
        if mark_regions:
            self._mark_regions(ax)
        ax.set_xlim(0, self.x_grid_out[-1] * scale)
        if self.num_regions > 1:
            ax.set_xlabel(f"Distance along the beam [{unit}] (dashed lines: region boundaries)")
        else:
            ax.set_xlabel(f"Distance into the target [{unit}]")

    @_plot_style
    def plot_beam(self, directory):
        """Plot the beam energy, radius, current and deposited power against position, in a
        2 x 2 grid."""
        scale, unit = self._position_units()
        x = self.x_grid_out * scale
        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(13, 9))
        (ax_energy, ax_radius), (ax_current, ax_power) = axes

        ax_energy.plot(x, self.beam_energy_grid_out)
        ax_energy.set_ylabel("Beam energy [keV]")
        ax_energy.set_ylim(bottom=0)

        ax_radius.plot(x, self.radius_grid_out, label="Beam edge")
        ax_radius.axhline(self.wall_radius, color="k", linewidth=1.5, label="Wall")
        ax_radius.set_ylabel("Radius [cm]")
        ax_radius.set_ylim(0, 1.1 * self.wall_radius)
        ax_radius.legend(loc="lower right")

        ax_current.plot(x, self.current_grid_out)
        ax_current.set_ylabel("Beam current [mA]")
        ax_current.set_ylim(0, 1.05 * self.current)

        # Power per unit length [W per cm or um] in each output bin, totals in the legend
        entry = self.voltage * self.current / 1000
        deposited_label = f"Deposited in the target ({self.deposited_power:.3g} kW)"
        wall_label = f"Onto the wall ({self.wall_power:.3g} kW)"
        per_length = 1000 / (self.dx_out * scale)
        ax_power.stairs(self.deposited_power_mids_out * per_length, x, label=deposited_label)
        ax_power.stairs(self.wall_power_mids_out * per_length, x, label=wall_label)
        ax_power.set_ylabel(f"Power deposited [W/{unit}]")
        ax_power.set_ylim(bottom=0)
        ax_power.legend(
            loc="center left",
            title=f"Beam power {entry:.3g} kW\nTo the backstop: {self.backstop_power:.3g} kW",
            alignment="left",
        )

        for ax in axes.flat:
            self._mark_regions(ax)
            ax.grid()
        for ax in axes[1]:
            self._set_position_axis(ax, mark_regions=False)
        fig.suptitle(f"Beam along the target ({self.name})")
        _save_figure(fig, os.path.join(directory, "beam.png"))

    @_plot_style
    def plot_neutron_position(self, directory):
        """Plot the neutrons per unit length against position, on linear and log scales."""
        scale, unit = self._position_units()
        x = self.x_grid_out * scale
        lines = [
            (rxn, np.sum(self.neutrons[rxn], (1, 2)) / (self.dx_out * scale))
            for rxn in self.reactions_present()
        ]
        if len(lines) > 1:
            lines.append(("Total", lines[0][1] + lines[1][1]))

        fig, ax = plt.subplots(figsize=(8, 6))
        for label, per_cm in lines:
            ax.stairs(per_cm, x, label=label)
        self._set_position_axis(ax)
        ax.set_ylabel(rf"Neutrons per unit length [n/({unit}$\cdot$s)]")
        ax.set_title(f"Neutron source along the beam ({self.name})")
        ax.set_ylim(bottom=0)
        ax.grid()
        ax.legend()
        _save_figure(fig, os.path.join(directory, "neutron_position.png"), close=False)

        ax.set_yscale("log")
        ax.autoscale(axis="y")
        _save_figure(fig, os.path.join(directory, "neutron_position_log.png"))

    @_plot_style
    def plot_neutron_angle(self, directory):
        """Plot the neutrons per steradian against emission angle (from the beam direction).

        The y axis is not forced to start at zero, so that small anisotropies show.
        """
        lines = [
            (rxn, np.sum(self.neutrons[rxn], (0, 2)) / self.solid_angles_out)
            for rxn in self.reactions_present()
        ]
        if len(lines) > 1:
            lines.append(("Total", lines[0][1] + lines[1][1]))

        fig, ax = plt.subplots(figsize=(8, 6))
        for label, per_sr in lines:
            ax.stairs(per_sr, self.angle_grid_out, label=label)
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel("Emission angle from the beam direction [degrees]")
        ax.set_ylabel(r"Neutrons per steradian [n/(s$\cdot$sr)]")
        ax.set_title(f"Neutron angular distribution ({self.name})")
        ax.grid()
        ax.legend()
        _save_figure(fig, os.path.join(directory, "neutron_angle.png"))

    @_plot_style
    def plot_neutron_energy(self, directory):
        """Plot neutron energy against emission angle for each reaction, and the spectrum.

        The maps show neutrons per unit energy and solid angle, so each column is the
        spectrum at that angle, on a log colour scale covering four decades below the peak;
        empty cells are left blank.

        The spectrum is integrated over angle and drawn on linear and log scales (the log
        one covers five decades below the peak). DD and DT are shown separately, since their
        energies do not overlap, and each drops to zero at the ends of its energy range. The
        axes follow the neutron energies, so the plots suit any beam energy.
        """
        present = self.reactions_present()
        for rxn in present:
            by_angle = np.sum(self.neutrons[rxn], 0)  # (angle, energy) [n/s]
            density = by_angle / self.solid_angles_out[:, None] / self.energy_bin_width
            filled = np.flatnonzero(np.sum(by_angle, 0))
            lo, hi = filled[0], filled[-1] + 1
            fig, ax = plt.subplots(figsize=(8, 6))
            mesh = ax.pcolormesh(
                self.angle_grid_out,
                self.neutron_energy_grid[rxn][lo : hi + 1],
                np.ma.masked_less_equal(density[:, lo:hi].T, 0),
                cmap="viridis",
                norm=LogNorm(vmin=np.max(density) * 1e-4, vmax=np.max(density)),
            )
            fig.colorbar(mesh, ax=ax, label=r"Neutrons [n/(MeV$\cdot$s$\cdot$sr)]")
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, 30))
            ax.set_xlabel("Emission angle from the beam direction [degrees]")
            ax.set_ylabel("Neutron energy [MeV]")
            ax.set_title(f"{rxn} neutron energy against emission angle ({self.name})")
            _save_figure(fig, os.path.join(directory, f"neutron_energy_angle_{rxn}.png"))

        energy_grid, on_common_grid = self.neutrons_by_angle_and_energy()
        peak = 0.0
        fig, ax = plt.subplots(figsize=(8, 6))
        for rxn in present:
            spectrum = np.sum(on_common_grid[rxn], 0) / self.energy_bin_width
            # Draw each reaction over its own energy range, dropping to zero at both ends
            filled = np.flatnonzero(spectrum)
            lo, hi = filled[0], filled[-1] + 1
            ax.stairs(spectrum[lo:hi], energy_grid[lo : hi + 1], baseline=0, label=rxn)
            peak = max(peak, np.max(spectrum))
        ax.set_xlim(0, energy_grid[-1] + 0.05 * (energy_grid[-1] - energy_grid[0]))
        ax.set_ylim(0, 1.05 * peak)
        ax.set_xlabel("Neutron energy [MeV]")
        ax.set_ylabel(r"Neutrons per unit energy [n/(MeV$\cdot$s)]")
        ax.set_title(f"Neutron energy spectrum ({self.name})")
        ax.grid()
        ax.legend()
        _save_figure(fig, os.path.join(directory, "neutron_energy.png"), close=False)

        ax.set_yscale("log")
        ax.set_ylim(peak * 10**-5.25, peak * 10**0.25)
        _save_figure(fig, os.path.join(directory, "neutron_energy_log.png"))
