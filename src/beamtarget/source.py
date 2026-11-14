"""Beam-target neutron source term for Monte Carlo transport codes.

A beam of protons, deuterons, tritons, 3He, or alphas passes through one or more regions,
each filled with a gas or solid material, and makes neutrons in reactions with the target
isotopes: the built-in two-body reactions D(d,n)3He, T(d,n)4He, T(p,n)3He and 7Li(p,n)7Be
(DROSG-2000 data, see nuclear.REACTIONS), or any reaction in an evaluated nuclear data
library (ENDF/B, FENDL, TENDL; see endf.py). For each region the user gives the material and
its state, a length, a wall radius, and the beam divergence. The calculation:

1. Slows the beam down along its path using SRIM or ICRU 49 stopping powers.
2. Grows the beam radius with its divergence and removes the fraction of the beam that
   strikes the target wall (beam scraping), and the fraction lost to nuclear reactions.
3. Evaluates the cross sections at each position: for the two-body reactions, the
   center-of-mass differential cross section from tabulated Legendre data and the lab
   energy and angle of the neutron from each center-of-mass angle (relativistic two-body
   kinematics); for evaluated data, each channel's neutron spectrum, converted to the lab.
4. Tallies neutrons onto the output (position, lab angle, energy) mesh.

The result can be written as an HDF5 file, as an MCNP SDEF source card, as text files, and
as plots. The caller chooses every output path.

Cases are normally described in a TOML input file and run from the command line (see
cli.py); the BeamTarget class can also be used directly from Python.

Naming conventions used throughout:

- A mesh is divided into bins. ``*_grid`` arrays hold values on bin boundaries (length
  n + 1); ``*_mids`` arrays hold one value per bin, at or averaged over the bin (length n).
- An ``_out`` suffix marks the coarser output mesh; no suffix is the fine internal mesh the
  calculation runs on. Each output bin is a whole number of internal bins (a "ratio").
- ``*_per_region`` arrays have one entry per gas region. The region inputs (pressure,
  temperature, length, divergence, position_bins, position_substeps) are also per region.
- Units (see units.py): energies (beam and neutron) in eV, current in A, pressure in Pa,
  temperature in K, divergence in rad, power in W, lengths in cm. The nuclear-data and
  kinematics modules (endf.py, nuclear.py) work in MeV, converted at the calls.
- ``rxn`` is a reaction key (see nuclear.REACTIONS), such as "DD" for D(d,n)3He.
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
import numba
import numpy as np

from .config import DROSG, ConfigError, load_config, resolve_nuclear_data
from .endf import EndfReaction, find_evaluation, load_evaluation
from .nuclear import (
    BEAMS,
    REACTIONS,
    atom_name,
    beam_key,
    bin_two_body,
    cm_energy,
    coulomb_barrier,
    isotope_fractions,
    legendre_cumulative,
    mean_atomic_mass,
    neutron_channels,
    nuclear_mass,
    parse_atom,
    two_body_energy_range,
)
from .parallel import in_parallel
from .units import DEFAULT_UNITS, KEV, KILOWATT, MEV, MILLIAMPERE, UNITS, from_internal

PACKAGE_NAME = "beam-target"
logger = logging.getLogger(__name__)

# Beam-energy step [relative, and at least the absolute value in MeV] of the grids on which
# the spectra of evaluated (ENDF) reactions are calculated and then interpolated: continua
# (LAW=1), and lab-frame tables (LAW=7), whose sharper features need a finer grid. Checked
# against steps half as large: within 2.2% in any energy bin for continua (ENDF/B p + Be, W
# and alpha + Be), and 9% for tables at 2% (FENDL Ta, U, C, d + Li, d + Be; bins holding
# under 0.4% of the neutrons), which is why tables use 1%.
ENDF_ENERGY_STEP = (0.04, 0.01)
ENDF_TABLE_ENERGY_STEP = (0.01, 0.0025)
# Angle cells over which evaluated continuum distributions are integrated before conversion to
# the lab: this many per output angle bin, and at least ENDF_MIN_ANGLE_CELLS (two-body lines
# are binned exactly, see nuclear.bin_two_body)
ENDF_ANGLE_CELLS_PER_BIN = 8
ENDF_MIN_ANGLE_CELLS = 180
# Warn if the explicit channels of an evaluation cover less than this fraction of its
# non-elastic cross section at the beam energy (the neutrons of the rest are missing)
ENDF_MISSING_WARNING = 0.01
# Targets from thorium (Z = 90) up can fission: warn unless their evaluation gives fission
# (MT=18) neutrons explicitly
FISSION_MIN_Z = 90
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_DATA_DIR = os.path.join(CODE_DIR, "data")
# Stopping-power data files (in data/) and the available stopping-power sets
SRIM_STOPPING_FILE = "srim_stopping.h5"
PSTAR_FILE = "pstar_hydrogen.txt"
STOPPING_POWERS = {
    "srim": "SRIM-2012.01, per atom, mixed over each region's composition",
    "icru49": "ICRU 49 electronic + SRIM nuclear stopping; hydrogen-isotope gases only",
}
PHASES = ("gas", "solid")
# Elements (by Z) whose gases are diatomic molecules: a gas Material rejects lone atoms of
# them: H, N, O, F, Cl, Br, I
DIATOMIC_ELEMENTS = (1, 7, 8, 9, 17, 35, 53)
# SRIM_STOPPING_FILE target names for single isotopes with their own tables; any other
# isotope uses its element's table (electronic stopping hardly depends on the isotope)
SRIM_ISOTOPE_TABLES = {(1, 2): "D", (1, 3): "T", (2, 3): "He3"}

# Decimal places for the beam radii in the SDEF card [cm]
SDEF_RADIUS_DECIMALS = 5
# Longest line of summary.toml before a list of strings is written one item per line
SUMMARY_LINE_LENGTH = 100
# Most energy cells in a surface plot (see plot_neutron_energy_surface)
SURFACE_MAX_CELLS = 300
# Heights at which a surface plot's color changes (see plot_neutron_energy_surface)
SURFACE_COLOR_LEVELS = 128
# Light for the shading of surface plots: azimuth and elevation [deg] in the plot box, and the
# brightness of a surface facing away from it (see _surface_shade)
SURFACE_LIGHT = (60.0, 45.0)
SURFACE_AMBIENT = 0.55

# Version of the HDF5 output layout; increase it whenever the layout changes
HDF5_FORMAT_VERSION = 10
# Internal position steps per output position bin, unless a region gives its own
POSITION_SUBSTEPS = 10
# Position-mesh check (see check_position_mesh): warn if two-body lines move more than
# MESH_LINE_SHIFT energy bins per internal step (median over their neutrons), or if the
# yield's midpoint-rule error is more than MESH_YIELD_ERROR
MESH_LINE_SHIFT = 0.5
MESH_YIELD_ERROR = 1e-3
MESH_MARGIN = 1.2  # The position_substeps suggested: this much more than the estimate

# Error message when a case produces no neutrons
NO_NEUTRONS = (
    "no neutrons are produced. Check that the target holds an isotope the beam makes neutrons "
    "with (see nuclear.REACTIONS), that the beam energy is above the reaction's threshold or the "
    "start of its cross-section data, and that the beam is not scraped away entirely."
)

# Repository root when running from a source checkout (src/beamtarget/ -> root)
REPO_DIR = os.path.normcase(os.path.dirname(os.path.dirname(CODE_DIR)))

# Physical constants (exact SI values)
NA = 6.02214076e23  # Avogadro constant [1/mol]
ee = 1.602176634e-19  # Elementary charge [C]
kB = 1.380649e-23  # Boltzmann constant [J/K]

# Atomic weight of hydrogen that ICRU Report 49 used for its per-mass stopping [g/mol]
A_H_ICRU49 = 1.00794

# A neutron channel that is open but not modeled is warned about once the beam's
# center-of-mass energy reaches this fraction of the Coulomb barrier (see check_channels)
COULOMB_FRACTION = 0.1

# Nuclear (bare-nucleus) rest-mass energies [MeV], from nuclear.nuclear_mass (AME2020)
M_H1 = nuclear_mass(1, 1)  # Proton
M_N1 = nuclear_mass(0, 1)  # Neutron

# MCNP source direction -> (AXS vector, VEC vector, reverse the position distributions).
# For "-z" the extent runs along +z and the position-dependent distributions are written
# in reverse order instead of using AXS = 0 0 -1, which ADVANTG does not handle correctly.
# As a result, for "-z" POS is the downstream (backstop) end of the source; for every other
# direction POS is the point where the beam enters the first region. get_mcnp_sdef takes
# the beam entry point for every direction and sets POS accordingly.
SDEF_DIRECTIONS = {
    "-x": ("-1 0 0", "-1 0 0", False),
    "+x": ("1 0 0", "1 0 0", False),
    "-y": ("0 -1 0", "0 -1 0", False),
    "+y": ("0 1 0", "0 1 0", False),
    "-z": ("0 0 1", "0 0 -1", True),
    "+z": ("0 0 1", "0 0 1", False),
}


def get_molecule_density(pressure, temperature):
    """Return the molecule density [1/cm^3] of an ideal gas.

    pressure is in Pa and temperature in K. Works element-wise on arrays.
    """
    return pressure / (kB * temperature) / 1e6


def get_atom_density(pressure, temperature):
    """Return the atom density [1/cm^3] of a diatomic ideal gas such as D2, DT, or T2.

    pressure is in Pa and temperature in K. Works element-wise on arrays.
    """
    return get_molecule_density(pressure, temperature) * 2


_FORMULA_TOKEN = re.compile(r"(\d*[A-Z][a-z]?)(\d+(?:\.\d*)?|\.\d+)?")


def parse_formula(formula):
    """Return {atom: count} for a chemical formula such as "D2", "TiD1.5", "CD4", "7Li" or "Ar".

    Atoms are element symbols (the natural element, such as "Li"), or one isotope with its
    mass number first (such as "7Li" or "3He", so that "He3" still means three helium atoms);
    D and T are the hydrogen isotopes. Counts may be decimal and default to 1. Atom names are
    returned in canonical form (see nuclear.atom_name). Raises ValueError for anything else.
    """
    counts = {}
    pos = 0
    while pos < len(formula):
        m = _FORMULA_TOKEN.match(formula, pos)
        if not m:
            raise ValueError(f"cannot read the chemical formula {formula!r}")
        symbol = atom_name(*parse_atom(m.group(1)))
        counts[symbol] = counts.get(symbol, 0.0) + float(m.group(2) or 1)
        pos = m.end()
    if not counts:
        raise ValueError("empty chemical formula")
    return counts


class Material:
    """A target material: its phase, composition, and (for a solid) density.

    A solid's composition is given as atoms and a gas's as molecules, so that the molecules a
    gas pressure counts are always stated: H2O vapor, a mixture of H2 and O2, and atomic H
    and O are different gases with the same atom ratios.

    Args:
        name: Name, used in messages and outputs.
        phase: "gas" or "solid", which selects SRIM's gas- or solid-phase stopping.
        atoms: For a solid, {atom: amount}: atom ratios, such as {"Ti": 1, "D": 1.5}. Atoms
            are element symbols (the natural element: "Li" is 7.59% 6Li and 92.41% 7Li) or
            single isotopes with the mass number first ("7Li"); D and T are 2H and 3H.
        molecules: For a gas, {formula: amount}: mole fractions of its molecules, such as
            {"D2": 0.5, "T2": 0.5} or {"H2O": 1} (see parse_formula). A single atom is a
            monatomic molecule (such as "Ar"); hydrogen isotopes, N, O, and the halogens are
            rejected on their own (write D2, not D).
        density: Mass density of a solid [g/cm^3]. A gas has none: its state (pressure and
            temperature, or density) belongs to the region it fills.
        Amounts are relative and need not sum to 1.

    Attributes: composition (the atoms or molecules given), atom_fractions ({atom: fraction
    of all atoms}, by the names used), isotope_fractions ({(Z, A): fraction of all atoms},
    natural elements split into their isotopes), mean_mass (mean atomic mass [u]) and
    atoms_per_unit (for a gas, the mean number of atoms per molecule, which converts a
    molecule density to an atom density; 1 for a solid).
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
            if self.phase == "gas" and len(parsed) == 1 and list(parsed.values()) == [1.0]:
                (atom,) = parsed
                if parse_atom(atom)[0] in DIATOMIC_ELEMENTS:
                    raise ValueError(
                        f"{where}: {formula} on its own is a lone atom; its gas is made of "
                        f"{formula}2 molecules, so write {formula}2"
                    )
            for atom, n in parsed.items():
                counts[atom] = counts.get(atom, 0.0) + amount * n
        atoms_total = sum(counts.values())
        self.atoms_per_unit = atoms_total / total
        self.atom_fractions = {atom: n / atoms_total for atom, n in counts.items() if n > 0}
        self.isotope_fractions = {}
        for atom, fraction in self.atom_fractions.items():
            try:
                isotopes = isotope_fractions(atom)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            for za, f in isotopes.items():
                self.isotope_fractions[za] = self.isotope_fractions.get(za, 0.0) + fraction * f
        self.mean_mass = mean_atomic_mass(self.isotope_fractions)

    def stopping_fractions(self):
        """Return {SRIM_STOPPING_FILE target name: fraction of the atoms}: each isotope's
        element, except the isotopes with their own tables (D, T, 3He)."""
        fractions = {}
        for (Z, A), f in self.isotope_fractions.items():
            name = SRIM_ISOTOPE_TABLES.get((Z, A), atom_name(Z))
            fractions[name] = fractions.get(name, 0.0) + f
        return fractions

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


class Region:
    """One region the beam passes through: a length of one material, inside a wall.

    Args:
        material: The Material filling the region.
        length: Length along the beam [cm]; None to calculate it (see BeamTarget).
        pressure, temperature: For a gas given by pressure [Pa] and temperature [K].
            None for a solid, or for a gas given by density. A gas with a temperature but no
            pressure has its pressure calculated.
        density: For a gas given by density instead [g/cm^3]. A gas with neither pressure,
            temperature nor density has its density calculated.
        wall_radius: Radius of the wall around the region [cm]; the part of the beam outside
            it is scraped off.
        divergence: Beam divergence in the region (growth in radius per unit length) [rad].
        position_bins: Number of position bins of the output mesh.
        position_substeps: Number of steps the beam is followed in across each output
            position bin (the internal mesh); default POSITION_SUBSTEPS.
        name: Optional name, used in messages and outputs.
    """

    def __init__(
        self,
        material,
        length=None,
        *,
        pressure=None,
        temperature=None,
        density=None,
        wall_radius,
        divergence=0.0,
        position_bins,
        position_substeps=None,
        name="",
    ):
        self.material = material
        self.length = length
        self.pressure = pressure
        self.temperature = temperature
        self.density = density
        self.wall_radius = wall_radius
        self.divergence = divergence
        self.position_bins = position_bins
        self.position_substeps = (
            POSITION_SUBSTEPS if position_substeps is None else position_substeps
        )
        self.name = str(name)

    def __repr__(self):
        fields = ", ".join(
            f"{key}={getattr(self, key)!r}"
            for key in (
                "length",
                "pressure",
                "temperature",
                "density",
                "wall_radius",
                "divergence",
                "position_bins",
                "position_substeps",
                "name",
            )
            if getattr(self, key) is not None and getattr(self, key) != ""
        )
        return f"Region({getattr(self.material, 'name', self.material)!r}, {fields})"


@functools.lru_cache(maxsize=4)
def read_srim_h5(fname, particle="d"):
    """Read the stopping of one beam particle in every target from SRIM_STOPPING_FILE.

    Returns (E [keV], masses, stopping): masses maps each target name (element symbols, plus
    D, T, and He3) to its atomic mass [u]; stopping maps each phase ("gas", "solid") to
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
    are zero. The result has len(dx) + 1 entries. (Compiled: solve_region calls it many
    times.)
    """
    as_array = lambda a: np.ascontiguousarray(a, dtype=float)  # noqa: E731
    return _slow_down(
        float(E0), as_array(stopping_E), as_array(stopping_eps), as_array(atom_density),
        as_array(dx),
    )  # fmt: skip


@numba.njit(cache=True)
def _interp_or_nan(x, xp, fp):
    """np.interp(x, xp, fp, left=nan, right=nan) for one x."""
    if not (xp[0] <= x <= xp[-1]):  # Also NaN
        return np.nan
    i = np.searchsorted(xp, x, "right") - 1
    if i >= len(xp) - 1:
        return fp[-1]
    return fp[i] + (x - xp[i]) * (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i])


@numba.njit(cache=True)
def _slow_down(E0, stopping_E, stopping_eps, atom_density, dx):
    grid = np.zeros(len(dx) + 1)
    grid[0] = E0
    for i in range(len(dx)):
        E_start = grid[i]
        E_avg = grid[i]
        E_end = np.nan
        for _ in range(5):
            eps = _interp_or_nan(E_avg, stopping_E, stopping_eps)
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
    LEGINT.f95): every line is ten 8-character fields, spaces inside a field are ignored,
    and blank fields are zero. The first line holds EMIN, EMAX, EMINL, and EMAXL [MeV]; EMAXL is the
    highest energy of the Legendre table. Each record is then two lines (20 values): incident
    projectile energy [MeV], the center-of-mass differential cross section at 0 degrees [mb/sr]
    and 18 Legendre coefficients, normalized to sum to 1. Reading stops at the record at
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


@functools.cache
def get_code_version():
    """Return the version of this code, for recording in output files (looked up once per
    process: it runs git).

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
    """Format a string, number, or sequence of numbers as a TOML value. Floats are written to 7
    significant figures and always read back as floats."""
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    text = f"{float(value):.7g}"
    return text if any(c in text for c in ".enai") else text + ".0"


def _toml_wrapped(text, width=90, indent="    "):
    """Format a string as a TOML multi-line basic string wrapped at spaces, each line ending in
    a backslash (TOML drops it, the line break, and the indentation after it), so that a long
    text reads back unchanged."""
    words = text.replace("\\", "\\\\").replace('"', '\\"').split(" ")
    lines, line = [], words[0]
    for word in words[1:]:
        # Never break before a space (an empty word): TOML would drop it with the indentation
        if word and len(line) + 1 + len(word) > width:
            lines.append(line + " \\")
            line = word
        else:
            line += " " + word
    lines.append(line)
    return '"""' + ("\n" + indent).join(lines) + '"""'


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
    artifacts such as 131.90200000000002 from adding up lengths."""
    return repr(float(f"{x:.12g}"))


def _cosine_text(x):
    """Return a cosine with 15 decimal places in a fixed width of 18 characters, so columns line
    up; values within 1e-12 of zero (such as cos(90 deg), 6.1e-17 in floating point) are 0."""
    return f"{0.0 if abs(x) < 1e-12 else float(x):18.15f}"


# Matplotlib settings for every plot: serif text, upright mathtext, 200 dpi
PLOT_STYLE = {"font.family": "serif", "mathtext.default": "regular", "savefig.dpi": 200}


@functools.cache
def _pyplot():
    """Return matplotlib.pyplot, imported on first use (with the non-interactive Agg backend),
    so that runs and imports that make no plots do not pay for loading matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def _plot_label(text):
    """Return a reaction label with its mass numbers as superscripts for matplotlib, such as
    "9Be(p,xn)" as "$^{9}$Be(p,xn)": a number directly followed by an element symbol."""
    return re.sub(r"(?<![\w^])(\d+)(?=[A-Z])", r"$^{\1}$", text)


def _plot_style(method):
    """Decorator: draw the plot with PLOT_STYLE, without changing global matplotlib settings."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with _pyplot().rc_context(PLOT_STYLE):
            return method(*args, **kwargs)

    return wrapper


def _clip_polygons(P, count, level, keep_above):
    """Clip polygons P (n, m, 3), with count[i] vertices each, to z >= level (keep_above) or
    z <= level (level: a number or one per polygon), by Sutherland-Hodgman on all at once.
    Returns (polygons with up to m + 1 vertices, their vertex counts)."""
    n, m, _ = P.shape
    k = np.arange(m)[None, :]
    valid_edge = k < count[:, None]
    following = np.where(k + 1 < count[:, None], k + 1, 0)
    S = P
    E = np.take_along_axis(P, following[:, :, None].repeat(3, axis=2), axis=1)
    lev = level[:, None] if np.ndim(level) else level
    if keep_above:
        s_in, e_in = S[:, :, 2] >= lev, E[:, :, 2] >= lev
    else:
        s_in, e_in = S[:, :, 2] <= lev, E[:, :, 2] <= lev
    dz = E[:, :, 2] - S[:, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(dz != 0, (lev - S[:, :, 2]) / dz, 0.0)
    crossing = S + t[:, :, None] * (E - S)
    # Each edge gives up to two vertices: where it crosses the level, then its end
    out = np.stack([crossing, E], axis=2).reshape(n, 2 * m, 3)
    use = np.stack([valid_edge & (s_in != e_in), valid_edge & e_in], axis=2).reshape(n, 2 * m)
    order = np.argsort(~use, axis=1, kind="stable")
    out = np.take_along_axis(out, order[:, :, None].repeat(3, axis=2), axis=1)
    new_count = use.sum(axis=1)
    width = max(int(new_count.max()) if n else 0, 1)
    return out[:, :width], new_count


def _height_bands(x, y, z, levels):
    """Cut the surface through the grid points (x, y, z) (2-D arrays) into flat-shaded pieces
    that each lie between two neighboring heights in levels, so that a piece's color can
    depend on its height alone.

    Each grid cell is split into two triangles; triangles with all three corners at zero are
    left out (transparent). Returns (polygons (n, m, 3), padded with their last vertex, and
    the band of each: levels[band] <= z <= levels[band + 1])."""
    pts = np.stack([x, y, z], axis=-1)
    a, b, c, d = pts[:-1, :-1], pts[1:, :-1], pts[1:, 1:], pts[:-1, 1:]
    tri = np.concatenate(
        [np.stack([a, b, c], 2).reshape(-1, 3, 3), np.stack([a, c, d], 2).reshape(-1, 3, 3)]
    )
    tri = tri[np.any(tri[:, :, 2] > 0, axis=1)]
    num_bands = len(levels) - 1
    low = np.searchsorted(levels, tri[:, :, 2].min(axis=1), side="right") - 1
    high = np.searchsorted(levels, tri[:, :, 2].max(axis=1), side="left") - 1
    low = np.clip(low, 0, num_bands - 1)
    high = np.maximum(np.clip(high, 0, num_bands - 1), low)
    reps = high - low + 1
    which = np.repeat(np.arange(len(tri)), reps)
    band = low[which] + np.arange(len(which)) - np.repeat(np.cumsum(reps) - reps, reps)
    P, count = _clip_polygons(tri[which], np.full(len(which), 3), levels[band], True)
    P, count = _clip_polygons(P, count, levels[band + 1], False)
    keep = count >= 3
    P, count, band = P[keep], count[keep], band[keep]
    k = np.arange(P.shape[1])[None, :]
    last = np.take_along_axis(P, (count - 1)[:, None, None].repeat(3, axis=2), axis=1)
    return np.where((k < count[:, None])[:, :, None], P, last), band


def surface_density(angle_grid, energy_grid, neutrons):
    """Return (angles, energies, density) for a surface plot of a reaction's neutrons: the
    angle and energy bin middles [deg, and the unit of energy_grid] and the neutrons per unit
    energy and solid angle [n/(s sr) per that unit], over the energies the reaction gives.

    neutrons is the reaction's (position, angle, energy) array [n/s] on the meshes
    angle_grid [deg] and energy_grid (in any unit). At most SURFACE_MAX_CELLS energy cells are kept:
    finer bins are averaged in groups (which keeps the heights the same on average), because
    a surface of more cells renders slowly and its narrow features are lost in the drawing.
    """
    angle_grid, energy_grid = np.asarray(angle_grid), np.asarray(energy_grid)
    by_angle = np.sum(neutrons, 0)  # (angle, energy) [n/s]
    cosines = np.cos(np.radians(angle_grid))
    solid_angles = 2 * np.pi * (cosines[:-1] - cosines[1:])
    density = by_angle / solid_angles[:, None] / np.diff(energy_grid)[None, :]
    filled = np.flatnonzero(np.sum(by_angle, 0))
    lo, hi = (filled[0], filled[-1] + 1) if len(filled) else (0, by_angle.shape[1])
    density = density[:, lo:hi]
    energies = ((energy_grid[:-1] + energy_grid[1:]) / 2)[lo:hi]
    group = int(np.ceil(density.shape[1] / SURFACE_MAX_CELLS))
    if group > 1:
        n = density.shape[1] // group * group
        density = density[:, :n].reshape(len(density), -1, group).mean(axis=2)
        energies = energies[:n].reshape(-1, group).mean(axis=1)
    return (angle_grid[:-1] + angle_grid[1:]) / 2, energies, density


def _surface_shade(angles, energies, density):
    """Return the brightness (SURFACE_AMBIENT to 1) of the surface at each grid point, lit from
    SURFACE_LIGHT (azimuth, elevation [deg]) like a hill shade: Lambert's cosine law on the
    surface normal, worked out in the plot box (each axis scaled to its length on screen)."""
    box_x, box_y, box_z = 4.0, 4.0, 3.0  # Matplotlib's 3-D box shape
    x = (angles - angles[0]) / max(angles[-1] - angles[0], 1e-300) * box_x
    y = (energies - energies[0]) / max(energies[-1] - energies[0], 1e-300) * box_y
    z = density / max(np.max(density), 1e-300) * box_z
    dz_dx = np.gradient(z, x, axis=0) if len(x) > 1 else np.zeros_like(z)
    dz_dy = np.gradient(z, y, axis=1) if len(y) > 1 else np.zeros_like(z)
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(z)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    azimuth, elevation = np.radians(SURFACE_LIGHT)
    light = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    return SURFACE_AMBIENT + (1 - SURFACE_AMBIENT) * np.clip(normal @ light, 0, 1)


def _bilinear(angles, energies, values, x, y):
    """Interpolate values on the (angles, energies) grid (both evenly spaced) at points (x, y)."""

    def position(grid, v):
        if len(grid) < 2:
            return np.zeros(len(v), dtype=int), np.zeros(len(v))
        f = np.clip((v - grid[0]) / (grid[1] - grid[0]), 0, len(grid) - 1)
        i = np.minimum(f.astype(int), len(grid) - 2)
        return i, f - i

    i, s = position(angles, x)
    j, t = position(energies, y)
    i1 = np.minimum(i + 1, len(angles) - 1)
    j1 = np.minimum(j + 1, len(energies) - 1)
    return ((1 - s) * (1 - t) * values[i, j] + s * (1 - t) * values[i1, j]
            + (1 - s) * t * values[i, j1] + s * t * values[i1, j1])  # fmt: skip


def draw_energy_surface(fig, angles, energies, density, title, energy_unit="MeV"):
    """Draw a reaction's neutrons per unit energy and solid angle (see surface_density) as a
    surface on fig: its 3-D axes, which are returned. energy_unit names the unit of energies.

    The color follows the height alone: each facet is cut where it crosses one of
    SURFACE_COLOR_LEVELS heights, and each piece colored by its height, so no flat-colored
    facet spans a range of heights (which shows as stripes). For depth, each piece is
    darkened by a hill shade (see _surface_shade) interpolated smoothly to the middle of the
    piece, so the brightness has no jumps either. Facets with all corners at zero are left
    out (transparent), so the grid shows beneath.
    """
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax = fig.add_subplot(projection="3d")
    angle, energy = np.meshgrid(angles, energies, indexing="ij")
    top = max(np.max(density), 1e-300)
    levels = np.linspace(0, top, SURFACE_COLOR_LEVELS + 1)
    pieces, band = _height_bands(angle, energy, density, levels)
    colors = colormaps["Spectral_r"]((band + 0.5) / SURFACE_COLOR_LEVELS)
    middles = pieces.mean(axis=1)
    shade = _bilinear(
        angles, energies, _surface_shade(angles, energies, density), middles[:, 0], middles[:, 1]
    )
    colors[:, :3] *= shade[:, None]
    ax.add_collection3d(
        Poly3DCollection(
            pieces, facecolors=colors, linewidths=0, edgecolors="none", antialiased=False
        )
    )
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("Emission angle [degrees]")
    ax.set_ylim(energies[-1], energies[0])
    ax.set_ylabel(f"Neutron energy [{energy_unit}]")
    ax.set_zlim(0, top)
    ax.set_zlabel(rf"Neutrons [n/({energy_unit}$\cdot$s$\cdot$sr)]", labelpad=12)
    ax.view_init(elev=30, azim=-60)
    ax.set_title(title)
    return ax


def _save_figure(fig, fname, close=True):
    """Lay out fig, save it to fname and (unless close is False) close it."""
    fig.tight_layout()
    fig.savefig(fname)
    if close:
        _pyplot().close(fig)


class BeamTarget:
    """Neutron source term for a light-ion beam in gas or solid target regions.

    The whole calculation runs when the object is created; afterwards use write_hdf5(),
    write_mcnp_sdef(), write_text_files() and make_plots() to write the results. Summary
    values are logged at DEBUG level to the "beamtarget" logger (silent unless
    logging is configured).

    The beam passes through a list of regions (see Region), in order; the last one ends at
    the backstop. Each region holds a Material: a gas, given by pressure and temperature or
    by density, or a solid, whose density is part of the material. Neutrons come from the
    beam's reactions with the target isotopes, taken from the nuclear_data libraries in
    order: for each isotope, the first library with data for this beam on it is used (see
    select_reactions). The built-in DROSG-2000 two-body reactions (nuclear.REACTIONS, such
    as D(d,n)3He and T(d,n)4He for a deuteron beam) are one library; directories of
    evaluated files (ENDF/B, FENDL, TENDL; see endf.py) are the others. Isotopes with no data
    only slow the beam. dt_gas makes a D/T gas.

    Give either the beam current or the total neutron yield (yield_total), and the other is
    calculated: the yield is proportional to the current. Give None for the current to
    calculate it.

    Leave exactly one value to be calculated: the beam energy at the backstop
    (beam_energy_backstop None), or one region's length, gas pressure, or gas density (None
    there), which is then solved to give beam_energy_backstop (see solve_region).

    Args:
        name: Case name, used in output file names and titles.
        regions: The Regions the beam passes through, in order (one Region for a single one).
        beam_energy_entry: Beam energy where the beam enters the first region [eV] (its
            kinetic energy, not per nucleon or per charge), which must be within the data of
            each reaction used (DROSG Legendre tables, or the top energy of an evaluation).
        beam_current: Beam current [A]; None to calculate it from yield_total.
        beam_radius_entry: Beam radius where it enters the first region [cm].
        angle_bin_width: Width of the emission-angle bins of the output mesh [deg]. Must
            divide 180; 180 gives an isotropic source. (There is no internal angle mesh: each
            reaction is binned straight onto the output mesh.)
        energy_bin_width: Width of the neutron energy bins of the output mesh [eV].
        data_dir: Directory holding the data files (see data/README.md). Defaults to the
            data directory next to this file.
        stopping_power: Stopping-power set, one of STOPPING_POWERS: "srim" (default) or
            "icru49". See read_stopping_power.
        beam_energy_backstop: Beam energy wanted at the backstop, the end of the last region
            [eV]; give it to calculate a region's pressure, density, or length. None (default)
            calculates the beam energy.
        yield_total: Total neutron yield wanted [n/s]; give it, with beam_current None, to
            calculate the current. None (default) calculates the yield from the current.
        beam: Beam particle, named as in nuclear.BEAM_NAMES: "proton" or "1H", "deuteron" or
            "2H" (default), "triton" or "3H", "helion" or "3He", "alpha" or "4He".
        nuclear_data: Nuclear-data libraries, searched in order: "drosg" (DROSG) for the
            built-in two-body reactions, names of libraries defined in the user's
            configuration file, or directories of evaluated files (see config.py). None
            (default) uses the configuration's [defaults] nuclear_data, or ["drosg"].
        units: The units of the results (summary, files, and plots), as {quantity: unit} for
            any of the quantities of units.DEFAULT_UNITS (beam_energy, neutron_energy,
            current, power, pressure, temperature, divergence, length) and units of
            units.UNITS; the others are DEFAULT_UNITS. The inputs are always in the code's
            units (eV, A, Pa, K, rad, cm).
    """

    def __init__(
        self,
        name,
        regions,
        beam_energy_entry,
        beam_current,
        beam_radius_entry,
        angle_bin_width,
        energy_bin_width,
        data_dir=None,
        stopping_power="srim",
        beam_energy_backstop=None,
        yield_total=None,
        beam="deuteron",
        nuclear_data=None,
        units=None,
    ):
        t0 = time.time()

        # Input parameters
        self.name = str(name)
        self.units = dict(DEFAULT_UNITS)
        for quantity, unit in (units or {}).items():
            kind = "energy" if quantity in ("beam_energy", "neutron_energy") else quantity
            if quantity not in DEFAULT_UNITS or UNITS.get(unit, ("",))[0] != kind:
                raise ValueError(f"{self.name}: unknown units {quantity} = {unit!r}")
            self.units[quantity] = unit
        self.beam_energy_entry = float(beam_energy_entry)
        self.beam_current = None if beam_current is None else float(beam_current)
        self.requested_yield_total = None if yield_total is None else float(yield_total)
        self.beam_current_calculated = self.beam_current is None
        self.beam_radius_entry = float(beam_radius_entry)
        self.angle_bin_width = float(angle_bin_width)
        self.energy_bin_width = float(energy_bin_width)
        self.data_dir = DEFAULT_DATA_DIR if data_dir is None else str(data_dir)
        self.stopping_power = str(stopping_power)
        try:
            self.beam = beam_key(beam)  # The BEAMS key, such as "d"
        except ValueError as e:
            raise ValueError(f"{self.name}: {e}") from None
        # The entries as given (names or directories) and what each resolves to
        try:
            self.nuclear_data_given, self.nuclear_data = resolve_nuclear_data(nuclear_data)
        except ValueError as e:
            raise ValueError(f"{self.name}: {e}") from None
        self.beam_energy_backstop = (
            None if beam_energy_backstop is None else float(beam_energy_backstop)
        )

        # Regions, as one array (or list) per property, in beam order (None is NaN)
        self.regions = [regions] if isinstance(regions, Region) else list(regions)
        if not self.regions or not all(isinstance(r, Region) for r in self.regions):
            raise ValueError(f"{self.name}: regions must be a list of Region")
        self.num_regions = len(self.regions)

        def column(key):
            return np.array(
                [np.nan if getattr(r, key) is None else getattr(r, key) for r in self.regions],
                dtype=float,
            )

        self.region_names = [r.name for r in self.regions]
        self.materials = [r.material for r in self.regions]
        self.length = column("length")
        self.pressure = column("pressure")
        self.temperature = column("temperature")
        self.density = column("density")
        self.wall_radius = column("wall_radius")
        self.divergence = column("divergence")
        # Bin counts are converted to int after check_input confirms they are whole numbers
        self.position_bins = column("position_bins")
        self.position_substeps = column("position_substeps")
        self.solid = np.array([getattr(m, "phase", None) == "solid" for m in self.materials])
        # A gas region with a temperature is given by pressure; without one, by density
        self.by_pressure = ~self.solid & ~np.isnan(self.temperature)

        # The value to calculate: (quantity, region index), or ("beam_energy_backstop", None)
        self.unknowns = []
        for r in range(self.num_regions):
            if not self.solid[r]:
                if self.by_pressure[r] and np.isnan(self.pressure[r]):
                    self.unknowns.append(("pressure", r))
                elif np.all(np.isnan([self.temperature[r], self.pressure[r], self.density[r]])):
                    self.unknowns.append(("density", r))
            if np.isnan(self.length[r]):
                self.unknowns.append(("length", r))
        if self.beam_energy_backstop is None:
            self.unknowns.append(("beam_energy_backstop", None))
        self.calculated, self.calculated_region = (
            self.unknowns[0] if len(self.unknowns) == 1 else (None, None)
        )

        self.check_input()
        self.position_bins = self.position_bins.astype(int)
        self.position_substeps = self.position_substeps.astype(int)
        self.position_bins_internal = self.position_bins * self.position_substeps
        self.select_reactions()

        self.read_stopping_power()
        self.read_cross_sections()
        self.check_channels()
        if self.calculated != "beam_energy_backstop":
            self.solve_region()

        self.setup_meshes()
        self.setup_regions()

        self.calculate_beam_energy()
        self.calculate_scraping()
        self.calculate_attenuation()
        self.calculate_reactions()
        if self.beam_current_calculated:
            self.solve_current()
        self.calculate_beam_current()
        self.calculate_beam_power()
        self.bin_neutrons()
        self.check_position_mesh()

        # Recorded once, so every output of this case has the same creation time
        self.created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        logger.debug(f"{self.name}: calculation took {time.time() - t0:.1f} s")

    def check_input(self):
        """Raise ValueError if any input parameter is inconsistent or out of range."""

        def require(condition, message):
            if not condition:
                raise ValueError(f"{self.name}: {message}")

        require(
            all(isinstance(m, Material) for m in self.materials),
            "every region needs a Material",
        )
        require(self.beam_energy_entry > 0, "beam_energy_entry must be positive")
        require(len(self.nuclear_data) > 0, "nuclear_data must list at least one library")
        for r, material in enumerate(self.materials):
            where = self.region_label(r)
            given = {
                key: not np.isnan(getattr(self, key)[r])
                for key in ("pressure", "temperature", "density")
            }
            if self.solid[r]:
                require(
                    not any(given.values()),
                    f"{where} is a solid, so it takes no pressure, temperature, or density "
                    "(give None)",
                )
            elif given["temperature"]:
                require(
                    not given["density"],
                    f"{where}: give pressure and temperature, or density, not both",
                )
            else:
                require(
                    not given["pressure"],
                    f"{where}: a pressure needs a temperature (or give density instead)",
                )

        def describe(unknown):
            quantity, r = unknown
            return quantity if r is None else f"the {quantity} of {self.region_label(r)}"

        require(
            len(self.unknowns) > 0,
            "nothing is left to calculate: leave out beam_energy_backstop, or one region's "
            "length, pressure (keeping its temperature) or density",
        )
        require(
            len(self.unknowns) == 1,
            "only one value can be calculated, but these are missing: "
            + "; ".join(describe(u) for u in self.unknowns),
        )
        if self.beam_energy_backstop is not None:
            require(
                0 < self.beam_energy_backstop < self.beam_energy_entry,
                f"beam_energy_backstop must be above 0 and below the beam energy at entry "
                f"({self.beam_energy_entry / KEV:g} keV)",
            )
        require(
            (self.beam_current is None) != (self.requested_yield_total is None),
            "give exactly one of beam_current and yield_total; the other is calculated",
        )
        if self.beam_current is not None:
            require(self.beam_current > 0, "beam_current must be positive")
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
        require(np.all(temperatures > 0), "temperatures must be above 0 K")
        require(self.beam_radius_entry > 0, "beam_radius_entry must be positive")
        require(np.all(self.wall_radius > 0), "every region needs a positive wall_radius")
        require(
            self.wall_radius[0] >= self.beam_radius_entry,
            "the first region's wall_radius must not be smaller than beam_radius_entry (the beam "
            "radius at entry)",
        )
        require(np.all(self.divergence >= 0), "divergences must not be negative")
        for attr in ["position_bins", "position_substeps"]:
            counts = getattr(self, attr)
            require(np.all(counts == np.round(counts)), f"{attr} must be whole numbers")
            require(np.all(counts > 0), f"{attr} must be positive")
        require(self.angle_bin_width > 0, "angle_bin_width must be positive")
        require(self.energy_bin_width > 0, "energy_bin_width must be positive")
        require(
            self.stopping_power in STOPPING_POWERS,
            f"stopping_power must be one of {', '.join(STOPPING_POWERS)}, "
            f"got {self.stopping_power!r}",
        )
        require(_is_multiple(180, self.angle_bin_width), "angle_bin_width must divide 180")

    @property
    def beam_power_entry(self):
        """Beam power entering the first region [W]: beam energy / charge times current
        (eV x A = W)."""
        return self.beam_energy_entry / BEAMS[self.beam].charge * self.beam_current

    def region_label(self, r):
        """Return "region <n>" for region index r, with its name or material in brackets."""
        name = self.region_names[r] or getattr(self.materials[r], "name", "")
        return f"region {r + 1}" + (f" ({name})" if name else "")

    def solve_region(self):
        """Calculate the one missing region value (self.calculated in region
        self.calculated_region: its pressure, density, or length) that gives
        beam_energy_backstop at the backstop.

        The beam is slowed down through the regions before that one once, then through the
        rest on their internal meshes, exactly as the full calculation does, for trial values.
        The backstop energy falls steadily as the value grows, so it is bracketed by doubling
        and then found by bisection, to a relative precision of 1e-12. The result replaces the
        NaN in self.pressure, self.density or self.length.

        Raises ValueError if the backstop energy cannot be reached: if the beam is already at
        or below it without the region at all, or if it is below the bottom of the
        stopping-power table (where the beam is treated as stopped).
        """
        target = self.beam_energy_backstop
        k = self.calculated_region
        rest = range(k, self.num_regions)
        atom_density = self._atom_densities(self.pressure, self.density)  # NaN at k
        entry = self.beam_energy_entry
        E_in = self._slow_down_regions(entry, range(k), atom_density, self.length)[-1]

        def backstop_energy(value):
            trial = {
                "pressure": self.pressure.copy(),
                "density": self.density.copy(),
                "length": self.length.copy(),
            }
            trial[self.calculated][k] = value
            atoms = self._atom_densities(trial["pressure"], trial["density"])
            return self._slow_down_regions(E_in, rest, atoms, trial["length"])[-1]

        E_without = backstop_energy(0.0)
        if not E_without > target:
            raise ValueError(
                f"{self.name}: even without {self.region_label(k)}, the beam reaches the "
                f"backstop with {E_without / KEV:g} keV, not above beam_energy_backstop "
                f"({target / KEV:g} keV)"
            )

        # Bracket the value (Pa, g/cm3, or cm), then bisect
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
                f"{self.name}: cannot reach beam_energy_backstop = {target / KEV:g} keV: below "
                f"about {self.stopping_per_region[k][0][0] / KEV:g} keV the beam is treated as "
                "stopped"
            )

        getattr(self, self.calculated)[k] = value
        units = {"pressure": "Pa", "density": "g/cm3", "length": "cm"}[self.calculated]
        logger.debug(
            f"{self.name}: {self.region_label(k)} {self.calculated} = {value:.6g} {units} for "
            f"a beam energy of {target / KEV:g} keV at the backstop"
        )

    def setup_regions(self):
        """Set the target properties of each region and spread them over the position mesh.

        Every region is filled with a Material (self.materials) and described by its total
        atom density [1/cm^3], the fractions of its atoms that are each reaction's target
        isotope, and the stopping power per atom averaged over its composition (see
        read_stopping_power), so the energy loss per unit length is the atom density times
        that stopping.

        Sets self.atom_density_per_region, self.target_density_per_region and
        self.target_density_mids (reaction -> density of its target atoms [1/cm^3], per region
        and per internal bin) and self.divergence_mids.
        """
        self.atom_density_per_region = self._atom_densities(self.pressure, self.density)
        self.target_density_per_region = {
            rxn: self.atom_density_per_region * self._target_fractions(reaction)
            for rxn, reaction in self.reactions.items()
        }
        self.target_density_mids = {
            rxn: np.repeat(density, self.position_bins_internal)
            for rxn, density in self.target_density_per_region.items()
        }
        self.divergence_mids = np.repeat(self.divergence, self.position_bins_internal)

    def _target_fractions(self, reaction):
        """Return the fraction of each region's atoms that are the reaction's target isotope."""
        return np.array([m.isotope_fractions.get(reaction.target, 0.0) for m in self.materials])

    def _atom_densities(self, pressure, density):
        """Return the atom density [1/cm^3] of each region for the given gas pressures [Pa]
        and densities [g/cm^3]: from the ideal gas law (with self.temperature) for a gas given
        by pressure, and from the mass density for a gas given by density or a solid."""
        densities = np.zeros(self.num_regions)
        for r, material in enumerate(self.materials):
            if self.by_pressure[r]:
                molecules = get_molecule_density(pressure[r], self.temperature[r])
                densities[r] = molecules * material.atoms_per_unit
            else:
                rho = material.density if self.solid[r] else density[r]
                densities[r] = rho * NA / material.mean_mass
        return densities

    def _slow_down_regions(self, E0, regions, atom_density, length):
        """Return the beam energy [eV] at each internal grid point through the given regions
        (indices into the region arrays), starting at E0 [eV], for the given atom density
        [1/cm^3] and length [cm] of each of them. See slow_down."""
        grid = [np.array([E0])]
        for r in regions:
            n = self.position_bins_internal[r]
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
        """Build the internal and output position meshes and the output angle mesh."""

        # Angle mesh [deg], cosines, and the solid angle of each bin [sr]
        self.num_angle_bins_out = int(round(180 / self.angle_bin_width))
        self.angle_grid_out = np.linspace(0, 180, self.num_angle_bins_out + 1)
        self.angle_mids_out = (self.angle_grid_out[:-1] + self.angle_grid_out[1:]) / 2
        self.cosine_grid_out = np.cos(self.angle_grid_out * np.pi / 180)
        self.cosine_mids_out = np.cos(self.angle_mids_out * np.pi / 180)
        self.solid_angles_out = (self.cosine_grid_out[:-1] - self.cosine_grid_out[1:]) * 2 * np.pi

        # Position mesh [cm]. Each region is divided into equal bins.
        self.position_bins_internal_total = np.sum(self.position_bins_internal)
        self.position_bins_total = np.sum(self.position_bins)
        self.x_ratio_per_region = np.array(
            np.round(self.position_bins_internal / self.position_bins), dtype=int
        )
        self.dx_per_region = self.length / self.position_bins_internal
        self.dx_per_region_out = self.length / self.position_bins
        self.dx = np.repeat(self.dx_per_region, self.position_bins_internal)
        self.dx_out = np.repeat(self.dx_per_region_out, self.position_bins)
        self.x_grid = np.append(0, np.cumsum(self.dx))
        self.x_grid_out = np.append(0, np.cumsum(self.dx_out))
        self.x_mids = (self.x_grid[:-1] + self.x_grid[1:]) / 2
        self.x_mids_out = (self.x_grid_out[:-1] + self.x_grid_out[1:]) / 2

        # Number of internal bins in each output bin, and the internal grid index of each
        # output grid point
        self.x_ratio_out = np.repeat(self.x_ratio_per_region, self.position_bins)
        self.out_grid_inds = np.append(0, np.cumsum(self.x_ratio_out))

    def read_stopping_power(self):
        """Read the stopping power of the beam particle in each region's material, per atom.

        Stores self.stopping_per_region, an (E, eps) pair for each region: beam energy [eV]
        and electronic + nuclear stopping per atom [eV cm^2], averaged over the material's
        atoms (Bragg additivity), so the energy loss per unit length is eps times the atom
        density. Each isotope uses its element's table, except D, T, and 3He, which have
        their own (see Material.stopping_fractions).

        - "srim": SRIM-2012.01 stopping for the beam particle in each element
          (SRIM_STOPPING_FILE), in the material's phase (gas or solid).
        - "icru49": electronic stopping from ICRU Report 49 for protons in hydrogen gas
          (PSTAR_FILE), taken at the proton energy with the same velocity
          (E_p = E_beam * M_H1 / M_beam), plus SRIM nuclear stopping; electronic stopping
          does not depend on the hydrogen isotope. Only for gases of hydrogen isotopes (H, D,
          T), the only ICRU 49 data here, so that a case run with it uses ICRU 49 throughout.
          The table starts at 1 keV proton energy (2 keV for deuterons); below that the beam
          is treated as stopped.

        Raises ValueError if a material contains an atom with no stopping data, or if
        "icru49" is used with a solid or with any element other than hydrogen.
        """
        path = os.path.join(self.data_dir, SRIM_STOPPING_FILE)
        E, _, srim = read_srim_h5(path, particle=self.beam)
        if self.stopping_power == "icru49":
            E_p, P_elec, _ = read_pstar(os.path.join(self.data_dir, PSTAR_FILE))
            # Beam energy at the same velocity [keV]
            E_b = E_p * 1000 * BEAMS[self.beam].mass / M_H1
            P_elec = P_elec * 1000 * A_H_ICRU49 / NA  # MeV cm^2/g -> keV cm^2 per atom
            keep = (E >= E_b[0]) & (E <= E_b[-1])
            # Log-log interpolation of the ICRU 49 values onto the SRIM energy grid
            icru_elec = np.exp(np.interp(np.log(E[keep]), np.log(E_b), np.log(P_elec)))
        else:
            keep = np.ones(len(E), dtype=bool)

        cache = {}
        self.stopping_per_region = []
        for material in self.materials:
            if id(material) not in cache:
                if self.stopping_power == "icru49":
                    others = [atom_name(Z) for Z, _ in material.isotope_fractions if Z != 1]
                    others = list(dict.fromkeys(others))
                    if material.phase != "gas" or others:
                        reason = (
                            f"is a {material.phase}"
                            if material.phase != "gas"
                            else f"contains {', '.join(others)}"
                        )
                        raise ValueError(
                            f"{self.name}: stopping_power icru49 is for hydrogen-isotope gases "
                            f"only (the ICRU 49 data here are for hydrogen gas), but material "
                            f"{material.name} {reason}; use srim"
                        )
                eps = np.zeros(int(np.sum(keep)))
                for atom, fraction in material.stopping_fractions().items():
                    if atom not in srim[material.phase]:
                        raise ValueError(
                            f"{self.name}: material {material.name}: no stopping data for "
                            f"{atom!r} (elements H-U, and the isotopes D, T and 3He)"
                        )
                    elec, nuc = srim[material.phase][atom]
                    elec = elec[keep]
                    if self.stopping_power == "icru49":  # Hydrogen isotopes only (above)
                        elec = icru_elec
                    eps += fraction * (elec + nuc[keep])
                cache[id(material)] = (E[keep] * KEV, eps * KEV)  # Tables in keV
            self.stopping_per_region.append(cache[id(material)])

    def select_reactions(self):
        """Choose the data for each target isotope: the first nuclear_data library with data
        for this beam on it (like MCNP's search of its xsdir).

        - "drosg": the DROSG two-body reactions with that target (nuclear.REACTIONS).
        - A directory: the evaluated file for this beam on that target (endf.find_evaluation;
          of several TENDL variants, the most complete up to the beam energy), as one
          EndfReaction with every neutron-producing channel merged.

        Only reactions whose target isotope is present are kept: a beam has no reaction (and
        no yield in the outputs) for an isotope the targets do not hold.

        An evaluation that ends below the beam energy at entry is skipped, and the search goes
        on to the next library (check_channels warns if no library then covers the isotope).

        Sets self.reactions ({key: Reaction or EndfReaction}, DROSG reactions first),
        self.reaction_data ({key: description of the data used}) and self.skipped_data.
        """
        present = sorted(
            {za for m in self.unique_materials() for za, f in m.isotope_fractions.items() if f > 0}
        )
        drosg = [r for r in REACTIONS.values() if r.beam == self.beam]
        E_entry = self.beam_energy_entry / MEV
        claimed = {}  # (Z, A) -> DROSG or EndfReaction
        self.skipped_data = {}  # (Z, A) -> evaluations skipped for ending below the beam energy
        for za in present:
            for entry in self.nuclear_data:
                if entry == DROSG:
                    if any(r.target == za for r in drosg):
                        claimed[za] = DROSG
                        break
                else:
                    path = find_evaluation(entry, self.beam, za, E_max=E_entry)
                    if not path:
                        continue
                    reaction = EndfReaction(path, library=entry)
                    if reaction.production.emax < E_entry:  # Ends too low: try the next library
                        self.skipped_data.setdefault(za, []).append(
                            f"{path} (ends at {reaction.production.emax * MEV / KEV:g} keV)"
                        )
                        continue
                    claimed[za] = reaction
                    break

        self.reactions = {}
        self.reaction_data = {}
        for r in drosg:
            if claimed.get(r.target) == DROSG:
                self.reactions[r.key] = r
                self.reaction_data[r.key] = f"DROSG-2000 v12.01, {r.data_file}"
        for za, source in claimed.items():
            if source == DROSG:
                continue
            self.reactions[source.key] = source
            # The file, relative to its library's directory (listed in nuclear_data_dirs)
            try:
                where = os.path.relpath(source.path, source.library).replace(os.sep, "/")
            except ValueError:  # On another drive
                where = source.path
            self.reaction_data[source.key] = f"{source.evaluation} ({where})"

    def _is_endf(self, rxn):
        return isinstance(self.reactions[rxn], EndfReaction)

    def read_cross_sections(self):
        """Read the cross sections of the beam's reactions from the DROSG-2000 files (see
        read_drosg_koe), and check that the beam energy is within their range.

        The center-of-mass differential cross section is

            dsigma/dOmega_cm(E, mu) = total(E) * sum_l coeffs[l](E) * P_l(mu)

        at the energy E the data are tabulated at (see nuclear.Reaction.data_energy). Stores
        self.xs_data[rxn] with "E" [keV], "total" [b/sr] and "coeffs".

        Raises ValueError if, for a reaction whose target isotope is in some region, the beam
        energy at entry is above the end of its Legendre data. (Other neutron channels opening
        below that only give warnings, see check_channels.)
        """
        self.xs_data = {}
        files = {}
        for rxn, reaction in self.reactions.items():
            if self._is_endf(rxn):
                continue  # Evaluated data: read by EndfReaction
            if reaction.data_file not in files:
                path = os.path.join(self.data_dir, reaction.data_file)
                files[reaction.data_file] = read_drosg_koe(path)
            E, total, coeffs = files[reaction.data_file]
            self.xs_data[rxn] = {
                "E": E * MEV,  # [eV]
                "total": total / 1000,  # mb/sr -> b/sr
                "coeffs": coeffs,
            }

            if not np.any(self._target_fractions(reaction) > 0):
                continue
            E_entry = self.beam_energy_entry
            E_data_max = self.xs_data[rxn]["E"][-1]
            if reaction.data_energy(E_entry) > E_data_max:
                raise ValueError(
                    f"{self.name}: the beam energy at entry ({E_entry / KEV:g} keV) is above "
                    f"the end of the {reaction.label} Legendre data "
                    f"({E_data_max / reaction.data_energy(1.0) / KEV:.6g} keV)"
                )

    def check_channels(self):
        """Warn about neutron-producing channels that are open but not modeled.

        For every isotope in the target, the channels are: for a reaction target, the
        reaction's competing channels (see nuclear.Reaction.competing_channels); for any
        other isotope, the generic (b,n), (b,2n) and (b,np) channels of the beam b (see
        nuclear.neutron_channels). A channel is warned about if the beam energy at entry is
        above its threshold and the center-of-mass energy is at least COULOMB_FRACTION of the
        Coulomb barrier (below that, tunneling makes it negligible). The case still runs;
        those neutrons are missing from the results.

        Sets self.warnings (list of messages, also logged as warnings and written to the
        summary).
        """
        self.warnings = []
        E = self.beam_energy_entry / MEV
        beam = BEAMS[self.beam]
        where = {}  # (Z, A) -> names of the materials holding it
        for material in self.unique_materials():
            for za, fraction in material.isotope_fractions.items():
                if fraction > 0:
                    where.setdefault(za, []).append(material.name)
        targets = {}
        evaluated = {}
        for rxn, reaction in self.reactions.items():
            if self._is_endf(rxn):
                evaluated[reaction.target] = reaction
            else:
                targets.setdefault(reaction.target, []).extend(reaction.competing_channels())
        for za, reaction in sorted(evaluated.items()):
            # Every channel of an evaluation is used; warn only if some are missing from it
            missing = reaction.production.missing_fraction(E)
            if missing > ENDF_MISSING_WARNING:
                message = (
                    f"the {reaction.label} evaluation ({os.path.basename(reaction.path)}) gives "
                    f"channels for only {1 - missing:.0%} of its non-elastic cross section at "
                    f"{self.beam_energy_entry / KEV:g} keV, so the neutrons of the rest are "
                    "missing"
                )
                self.warnings.append(message)
                logger.warning(f"{self.name}: warning: {message}")
        for za in sorted(where):
            # Fission: an actinide's evaluation must give fission (MT=18) neutrons explicitly;
            # many (TENDL's, for instance) leave fission out, losing most of the neutrons
            reaction = evaluated.get(za)
            has_fission = reaction is not None and any(
                c.mt == 18 for c in reaction.production.channels
            )
            above = cm_energy(self.beam, za, E) / coulomb_barrier(beam.Z, beam.A, *za)
            if za[0] >= FISSION_MIN_Z and not has_fission and above >= COULOMB_FRACTION:
                source = (
                    f"its evaluation ({os.path.basename(reaction.path)}) gives no fission (MT=18) "
                    "neutrons"
                    if reaction is not None
                    else "there is no evaluation for it"
                )
                message = (
                    f"fission of {atom_name(*za)} ({', '.join(where[za])}) is open but not "
                    f"modeled, so its neutrons, often most of them, are missing: {source} "
                    "(FENDL-3.2c gives fission for protons on 235U and 238U)"
                )
                self.warnings.append(message)
                logger.warning(f"{self.name}: warning: {message}")
            if za in evaluated:
                continue
            if za in self.skipped_data:
                message = (
                    f"no evaluation for {BEAMS[self.beam].description}s on {atom_name(*za)} "
                    f"reaches the beam energy ({self.beam_energy_entry / KEV:g} keV); skipped: "
                    + "; ".join(self.skipped_data[za])
                )
                self.warnings.append(message)
                logger.warning(f"{self.name}: warning: {message}")
            channels = targets[za] if za in targets else neutron_channels(self.beam, za)
            barrier = coulomb_barrier(beam.Z, beam.A, *za)
            fraction = cm_energy(self.beam, za, E) / barrier
            seen = set()
            for threshold, label in sorted(channels):
                if label in seen or E <= threshold or fraction < COULOMB_FRACTION:
                    continue
                seen.add(label)
                why = (
                    f"its threshold is {threshold * MEV / KEV:.6g} keV"
                    if threshold > 0
                    else "it has no threshold"
                )
                if fraction < 1:
                    why += f"; the beam is at {fraction:.0%} of the Coulomb barrier"
                message = (
                    f"{label} is open but not modeled, so its neutrons are missing ("
                    f"{atom_name(*za)} in {', '.join(where[za])}): the beam enters at "
                    f"{self.beam_energy_entry / KEV:g} keV and {why}"
                )
                self.warnings.append(message)
                logger.warning(f"{self.name}: warning: {message}")
            if seen and za in targets:
                better = self.better_evaluation(za, E)
                if better:
                    self.warnings.append(better)
                    logger.warning(f"{self.name}: warning: {better}")

    def better_evaluation(self, za, E):
        """Return a message naming an evaluation that gives the neutrons of a built-in
        reaction's open competing channels, or None.

        For a target isotope za whose built-in (DROSG) reactions miss channels open at the
        entry energy E [MeV], look for an evaluation of this beam on it that reaches E, gives
        neutrons from more than two-body channels to discrete levels (MT=50-90) at E, and
        leaves at most ENDF_MISSING_WARNING of the non-elastic cross section uncovered: first
        in the case's nuclear_data libraries (which "drosg" comes before), then in the other
        libraries of the user's configuration file.
        """
        places = [
            (given, directory, True)
            for given, directory in zip(self.nuclear_data_given, self.nuclear_data)
            if directory != DROSG
        ]
        try:
            libraries = load_config()["libraries"]
        except ConfigError:
            libraries = {}
        listed = {os.path.normcase(os.path.abspath(d)) for _, d, _ in places}
        places += [
            (name, directory, False)
            for name, directory in libraries.items()
            if os.path.normcase(os.path.abspath(directory)) not in listed
        ]
        for name, directory, in_list in places:
            try:
                path = find_evaluation(directory, self.beam, za, E_max=E)
                if path is None:
                    continue
                production = load_evaluation(path)
            except (OSError, ValueError, NotImplementedError):
                continue
            beyond_levels = any(
                not 50 <= c.mt <= 90 and c.production(E) > 0 for c in production.channels
            )
            if (
                production.emax < E
                or not beyond_levels
                or production.missing_fraction(E) > ENDF_MISSING_WARNING
            ):
                continue
            where = (
                f'list "{name}" before "drosg" in nuclear_data'
                if in_list
                else f'add "{name}" (from your configuration file) to nuclear_data before "drosg"'
            )
            return (
                f"{name} has an evaluation of {production.label} ({os.path.basename(path)}, to "
                f"{production.emax:g} MeV) that gives these channels' neutrons: {where} to use "
                f"it for {atom_name(*za)}"
            )
        return None

    def calculate_beam_energy(self):
        """Slow the beam down through the regions.

        Sets self.beam_energy_grid, the beam energy at each internal grid point [eV], and its
        midpoint and output-mesh versions (see slow_down).
        """
        self.beam_energy_grid = self._slow_down_regions(
            self.beam_energy_entry,
            range(self.num_regions),
            self.atom_density_per_region,
            self.length,
        )
        self.beam_energy_mids = (self.beam_energy_grid[:-1] + self.beam_energy_grid[1:]) / 2
        self.beam_energy_grid_out = np.array(self.beam_energy_grid[self.out_grid_inds])

        logger.debug(
            f"{self.name}: beam energy at the end of the target = "
            f"{self.beam_energy_grid[-1] / KEV:.3f} keV"
        )

    def calculate_scraping(self, use_atan=False):
        """Grow the beam radius with divergence and work out how much is scraped.

        The beam profile is assumed flat and to expand self-similarly: a particle at a
        fraction f of the beam radius r stays at f r. It is scraped off where f r exceeds the
        wall radius, so the fraction of the beam still left at a point is m^2, where m is the
        smallest wall radius / r met so far along the path (at most 1). With one wall radius
        and a growing beam this is (wall radius / r)^2, the usual flat-beam result; walls
        that narrow and widen again (each region has its own) are handled the same way. A flat
        profile scrapes more than one peaked on the axis (such as a Gaussian), but less than a
        hollow one, with more of its current near the edge. Divergence is used as
        dr/dx = divergence [rad] (small-angle approximation) unless use_atan is True.

        Sets the beam edge radius (m r: the edge of the part left) [cm] and scrape fraction
        on the internal and output meshes.
        """
        if use_atan:
            divergence_mids = np.arctan(self.divergence_mids)
        else:
            divergence_mids = self.divergence_mids
        free_radius = self.beam_radius_entry + np.append(0, np.cumsum(divergence_mids * self.dx))
        self.free_radius_grid = free_radius  # Radius without the walls [cm]

        # Wall radius at each grid point: that of the bin ending there, so that the beam
        # scraped off by a narrower wall is scraped (and its power deposited) inside the
        # region with that wall, starting one bin in
        wall_mids = np.repeat(self.wall_radius, self.position_bins_internal)
        self.wall_grid = np.append(wall_mids[0], wall_mids)
        kept = np.minimum.accumulate(np.minimum(1.0, self.wall_grid / free_radius))
        self.scrape_grid = 1 - kept**2
        self.scrape_mids = (self.scrape_grid[:-1] + self.scrape_grid[1:]) / 2

        self.radius_grid = free_radius * kept
        rsq_mids = (self.radius_grid[:-1] ** 2 + self.radius_grid[1:] ** 2) / 2
        self.radius_mids = np.sqrt(rsq_mids)  # RMS radius in each bin

        # Output mesh: grid values are sampled, midpoint values are averaged
        def condense(vals):
            return np.add.reduceat(vals, self.out_grid_inds[:-1]) / self.x_ratio_out

        self.radius_grid_out = np.array(self.radius_grid[self.out_grid_inds])
        self.scrape_grid_out = np.array(self.scrape_grid[self.out_grid_inds])
        self.radius_mids_out = np.sqrt(condense(rsq_mids))
        self.scrape_mids_out = condense(self.scrape_mids)

    def calculate_attenuation(self):
        """Work out how much of the beam is lost to nuclear reactions on the way.

        Each beam particle that has a nuclear reaction is removed from the beam. The removal
        rate per unit length is the sum over the target isotopes with evaluated data (an
        EndfReaction) of their atom density times their non-elastic cross section at the
        beam energy (the DROSG data have no non-elastic cross sections; for a D/T gas the
        loss is negligible). Sets self.survival_grid (fraction of the beam not removed, at
        each internal grid point; 1 without evaluated data) and its midpoint and output-mesh
        versions, and self.beam_fraction_mids: the fraction of the beam still present
        (neither scraped nor removed), averaged over each internal bin.
        """
        removal = np.zeros(self.position_bins_internal_total)  # [1/cm]
        E = self.beam_energy_mids / MEV
        for rxn, reaction in self.reactions.items():
            if self._is_endf(rxn):
                sigma = reaction.production.nonelastic_cross_section(E) * 1e-24
                removal += self.target_density_mids[rxn] * sigma
        self.survival_grid = np.exp(-np.concatenate(([0.0], np.cumsum(removal * self.dx))))
        self.survival_mids = (self.survival_grid[:-1] + self.survival_grid[1:]) / 2
        self.survival_grid_out = np.array(self.survival_grid[self.out_grid_inds])
        kept = (1 - self.scrape_grid) * self.survival_grid
        self.beam_fraction_mids = (kept[:-1] + kept[1:]) / 2

    def solve_current(self):
        """Calculate the beam current [A] that gives the requested total neutron yield.

        The yield is proportional to the current, so the current is the requested yield
        divided by the yield per A. The yield per A is summed from the cross sections the
        same way bin_neutrons sums the yield, so the final yield matches the request to
        rounding error. Raises ValueError if the case produces no neutrons.
        """
        particles_per_A = 1 / (ee * BEAMS[self.beam].charge)  # [1/s]
        yield_per_A = 0.0
        for rxn in self.reactions:
            yield_per_A += np.sum(
                self.spectra[rxn]["sigma_mids"]
                * self.target_density_mids[rxn]
                * 1e-24
                * self.dx
                * particles_per_A
                * self.beam_fraction_mids
            )
        if yield_per_A == 0:
            raise ValueError(f"{self.name}: {NO_NEUTRONS}")
        self.beam_current = self.requested_yield_total / yield_per_A
        logger.debug(
            f"{self.name}: beam current = {self.beam_current / MILLIAMPERE:.6g} mA for a total "
            "neutron yield of "
            f"{self.requested_yield_total:.4e} n/s"
        )

    def calculate_beam_current(self):
        """Set the beam particle rate [1/s] and the beam current [A] after scraping and the
        loss to nuclear reactions, on the internal and output meshes."""
        self.particle_rate = self.beam_current / (ee * BEAMS[self.beam].charge)
        self.current_grid = self.beam_current * (1 - self.scrape_grid) * self.survival_grid
        self.current_mids = (self.current_grid[:-1] + self.current_grid[1:]) / 2
        self.current_grid_out = np.array(self.current_grid[self.out_grid_inds])
        self.current_mids_out = (
            np.add.reduceat(self.current_mids, self.out_grid_inds[:-1]) / self.x_ratio_out
        )

    def calculate_beam_power(self):
        """Work out where the beam power goes: into the target material, onto the wall, or to the backstop.

        Beam power is beam energy times particle rate: beam energy / charge times current
        (eV x A = W), as beam_power_entry. Along each bin, the beam
        still in it deposits the energy it loses in the target material (current x energy loss), and the
        beam scraped off hits the wall carrying its energy there (energy x current lost). Both
        use the bin averages of the other quantity, which makes the three parts add up to the
        entry power exactly. The power reaching the backstop is the remaining current times
        the final beam energy.

        Beam particles removed by nuclear reactions (see calculate_attenuation) deposit their
        energy in the target material too (self.nuclear_power, part of the deposited power).
        The current lost over a bin, w_i s_i - w_i+1 s_i+1 (w the unscraped and s the surviving
        fraction), is split exactly into the part scraped, (w_i - w_i+1) s_i, and the part
        removed, w_i+1 (s_i - s_i+1).

        Sets self.deposited_power_mids and self.wall_power_mids [W] per bin on the internal and
        output meshes (output bins are sums), and the totals self.deposited_power,
        self.wall_power, self.nuclear_power and self.backstop_power [W].
        """
        charge = BEAMS[self.beam].charge
        beam_energy_mids = (self.beam_energy_grid[:-1] + self.beam_energy_grid[1:]) / 2
        w, s = 1 - self.scrape_grid, self.survival_grid
        scraped = self.beam_current * (w[:-1] - w[1:]) * s[:-1]
        removed = self.beam_current * w[1:] * (s[:-1] - s[1:])
        nuclear_power_mids = beam_energy_mids * removed / charge
        self.nuclear_power = np.sum(nuclear_power_mids)
        self.deposited_power_mids = (
            self.current_mids * -np.diff(self.beam_energy_grid) / charge + nuclear_power_mids
        )
        self.wall_power_mids = beam_energy_mids * scraped / charge
        self.deposited_power_mids_out = np.add.reduceat(
            self.deposited_power_mids, self.out_grid_inds[:-1]
        )
        self.wall_power_mids_out = np.add.reduceat(self.wall_power_mids, self.out_grid_inds[:-1])
        self.deposited_power = np.sum(self.deposited_power_mids)
        self.wall_power = np.sum(self.wall_power_mids)
        self.backstop_power = self.beam_energy_grid[-1] * self.current_grid[-1] / charge
        kW = KILOWATT
        logger.debug(
            f"{self.name}: beam power {self.beam_power_entry / kW:.4g} kW: "
            f"{self.deposited_power / kW:.4g} kW deposited, {self.wall_power / kW:.4g} kW onto "
            f"the wall, {self.backstop_power / kW:.4g} kW to the backstop"
        )

    def calculate_reactions(self):
        """Calculate each reaction's neutrons on the output mesh, per beam particle.

        Sets self.spectra[rxn] (see _two_body_spectrum for the DROSG reactions and
        _endf_spectrum for evaluated data).
        """
        self.spectra = {}
        for rxn in self.reactions:
            if self._is_endf(rxn):
                self.spectra[rxn] = self._endf_spectrum(rxn)
            else:
                self.spectra[rxn] = self._two_body_spectrum(rxn)

    def _production_at(self, rxn, E):
        """Return the neutron production cross section [b] of a reaction at lab beam energies
        E [eV] (array), and its two-body lines as [(masses, cross section [b])]."""
        reaction = self.reactions[rxn]
        E = np.asarray(E, dtype=float)
        if self._is_endf(rxn):
            production = reaction.production
            total = np.zeros(len(E))
            lines = []
            m1, m2 = BEAMS[self.beam].mass, nuclear_mass(*reaction.target)
            m_n = nuclear_mass(0, 1)
            for channel in production.channels:
                sigma = channel.xs(E) * channel.yield_(E)  # Tabulated in eV
                total += sigma
                if channel.law == 2:
                    lines.append(((m1, m2, m_n, m1 + m2 - m_n - channel.q), sigma))
            return total, lines
        data = self.xs_data[rxn]
        E_data = reaction.data_energy(E)
        total = np.interp(E_data, data["E"], data["total"], left=0, right=np.nan)
        c0 = np.interp(E_data, data["E"], data["coeffs"][:, 0], left=0, right=np.nan)
        sigma = np.nan_to_num(4 * np.pi * total * c0)
        return sigma, [(reaction.masses(), sigma)]

    def check_position_mesh(self):
        """Warn if the internal position mesh (position_substeps steps per position bin) is
        too coarse.

        The beam energy is taken at the middle of each internal step, which can cause two
        errors:

        - Stripes in energy: at one beam energy a two-body reaction gives a line (each
          emission angle has one neutron energy), so each step adds a sharp line. If the
          lines of neighboring steps are more than MESH_LINE_SHIFT energy bins apart, the
          spectrum shows them separately instead of a smooth band, most clearly in the first
          and last angle bins. For each region and two-body line, the shift per step is the
          larger change in the line's lowest or highest neutron energy across the step, and
          a warning is given if its median over the line's neutrons is more than
          MESH_LINE_SHIFT energy bins. (Measured against much finer meshes, a median of half a
          bin leaves stripes of a few percent in the first and last angle bins; 3 bins,
          about 15%; 7 bins, about 40%.)
        - Yield: each step's neutrons are its length times the cross section at its middle
          times the average beam left over it, which is inexact if either changes much across
          the step (such as where the beam edge reaches the wall and scraping starts). The
          error is estimated against Simpson's rule on each step, with the cross section and
          the beam left at the step's ends and middle (where the beam left is worked out from
          the beam radius there); a warning is given if the region's error is more than
          MESH_YIELD_ERROR of the reaction's yield in the whole case. (Checked against much
          finer meshes, the estimate is within about a factor of 1.5 of the actual error.)

        Each warning names the position_substeps that would avoid it, from how the errors
        scale (shifts with the step length, yield errors with its square), with a margin of
        MESH_MARGIN. Appends to self.warnings.
        """
        E_grid = self.beam_energy_grid
        width = self.energy_bin_width / MEV
        start = np.concatenate(([0], np.cumsum(self.position_bins_internal)))
        # The beam left at each grid point and at each step's middle (see calculate_scraping)
        m_grid = np.sqrt(1 - self.scrape_grid)
        free_mid = (self.free_radius_grid[:-1] + self.free_radius_grid[1:]) / 2
        m_mid = np.minimum(m_grid[:-1], np.minimum(1.0, self.wall_grid[1:] / free_mid))
        left_grid = m_grid**2 * self.survival_grid
        left_mid = m_mid**2 * self.survival_mids
        # Each reaction's yield over the whole case (per beam particle), which the yield errors
        # are compared with
        case_total = {
            rxn: np.sum(
                self.target_density_mids[rxn]
                * self.dx
                * self.beam_fraction_mids
                * self._production_at(rxn, self.beam_energy_mids)[0]
            )
            for rxn in self.reactions
        }
        for r in range(self.num_regions):
            steps = slice(start[r], start[r + 1])
            grid = slice(start[r], start[r + 1] + 1)
            substeps = self.position_substeps[r]
            needed = substeps
            details = []
            for rxn, reaction in self.reactions.items():
                weight = (
                    self.target_density_mids[rxn][steps]
                    * self.dx[steps]
                    * self.beam_fraction_mids[steps]
                )
                if not np.any(weight > 0):
                    continue
                at_mids, lines = self._production_at(rxn, self.beam_energy_mids[steps])
                at_grid, _ = self._production_at(rxn, E_grid[grid])
                total = np.sum(weight * at_mids)
                if total <= 0:
                    continue
                problems = []

                # Yield: the sum used against Simpson's rule
                n_dx = self.target_density_mids[rxn][steps] * self.dx[steps]
                f_grid = at_grid * left_grid[grid]
                simpson = np.sum(
                    n_dx * (f_grid[:-1] + 4 * at_mids * left_mid[steps] + f_grid[1:]) / 6
                )
                error = abs(total - simpson) / case_total[rxn]
                if error > MESH_YIELD_ERROR:
                    factor = np.sqrt(error / MESH_YIELD_ERROR)
                    needed = max(needed, int(np.ceil(MESH_MARGIN * substeps * factor)))
                    problems.append(f"the case's yield off by about {error:.1%}")

                # Stripes: how far each line moves across a step (median over its neutrons)
                worst = 0.0
                for masses, sigma in lines:
                    produced = weight * sigma
                    if np.sum(produced) <= 0:
                        continue
                    low, high = two_body_energy_range(E_grid[grid] / MEV, *masses)
                    with np.errstate(invalid="ignore"):
                        shift = np.fmax(np.abs(np.diff(low)), np.abs(np.diff(high)))
                    shift = np.where(np.isfinite(shift), shift, 0.0)
                    order = np.argsort(shift)
                    share = np.cumsum(produced[order]) / np.sum(produced)
                    worst = max(
                        worst, shift[order[min(np.searchsorted(share, 0.5), len(order) - 1)]]
                    )
                if worst > MESH_LINE_SHIFT * width:
                    factor = worst / (MESH_LINE_SHIFT * width)
                    needed = max(needed, int(np.ceil(MESH_MARGIN * substeps * factor)))
                    problems.append(
                        f"neutron lines move {worst / width:.2g} energy bins per step, so the "
                        "energy spectrum can show stripes"
                    )
                if problems:
                    details.append(f"{reaction.label}: " + ", ".join(problems))
            if details:
                message = (
                    f"{self.region_label(r)}: the position mesh is too coarse "
                    f"(position_substeps = {substeps}); use {needed} or more. " + "; ".join(details)
                )
                self.warnings.append(message)
                logger.warning(f"{self.name}: warning: {message}")

    def _output_position_index(self):
        """Return the output position bin of each internal position bin."""
        offsets = np.cumsum(np.concatenate(([0], self.position_bins[:-1])))
        return np.concatenate(
            [
                np.floor_divide(np.arange(n), ratio) + offset
                for n, ratio, offset in zip(
                    self.position_bins_internal, self.x_ratio_per_region, offsets
                )
            ]
        ).astype(int)

    def _two_body_spectrum(self, rxn):
        """Calculate the neutrons of a DROSG two-body reaction on the output mesh, per beam
        particle.

        The center-of-mass (CM) differential cross section at the beam energy E in each
        internal bin is

            dsigma/dOmega_cm(E, mu) = total(E) * sum_l coeffs[l](E) * P_l(mu)

        with the DROSG data (see read_cross_sections) interpolated linearly to the energy the
        data are tabulated at (see nuclear.Reaction.data_energy); zero below the data. Each
        internal bin adds n_target dx times the fraction of the beam still present (see
        calculate_attenuation). The line the neutrons make at each beam energy is split where
        it crosses the output bin edges and binned exactly, with relativistic two-body
        kinematics (see nuclear.bin_two_body). For an inverse reaction (data for the target as
        the beam), the CM angle in the data is measured from the other particle, so the odd
        Legendre terms change sign.

        The neutron energy mesh spans the reaction's energies over the beam energies in the
        case, aligned to energy_bin_width (one bin from 0 if the reaction is below its
        threshold everywhere).

        Returns a dict like _endf_spectrum's.
        """
        reaction = self.reactions[rxn]
        masses = reaction.masses()
        E = self.beam_energy_mids / MEV  # The kinematics are in MeV
        width = self.energy_bin_width / MEV
        low, high = two_body_energy_range(E, *masses)
        if np.all(np.isnan(low)):
            lowest = highest = 0.0
        else:
            lowest, highest = np.nanmin(low), np.nanmax(high)
        energy_min = np.floor(lowest / width) * width
        num = max(1, int(round((np.ceil(highest / width) * width - energy_min) / width)))
        grid = np.linspace(energy_min, energy_min + num * width, num + 1)

        density = self.target_density_mids[rxn]
        idx = np.flatnonzero((density > 0) & (E > 0))
        # dsigma/dmu = 2 pi total sum_l coeffs[l] P_l(mu) [b], at each active bin
        data = self.xs_data[rxn]
        E_data = reaction.data_energy(self.beam_energy_mids[idx])
        total = np.interp(E_data, data["E"], data["total"], left=0, right=np.nan)
        coeffs = np.column_stack(
            [
                np.interp(E_data, data["E"], column, left=0, right=np.nan)
                for column in data["coeffs"].T
            ]
        ).reshape(len(idx), -1)
        # Only up to the last coefficient that is not zero anywhere (the higher ones are zero
        # at low beam energies)
        used = np.flatnonzero(np.any(coeffs != 0, axis=0))
        coeffs = coeffs[:, : used[-1] + 1] if len(used) else coeffs[:, :1]
        b = 2 * np.pi * total[:, None] * coeffs
        if reaction.inverse:
            b[:, 1::2] *= -1

        def cumulative(index, x):
            return legendre_cumulative(b[index], x)

        coef = density[idx] * 1e-24 * self.dx[idx] * self.beam_fraction_mids[idx]
        per_particle, sigma = bin_two_body(
            E[idx],
            coef,
            self._output_position_index()[idx],
            self.position_bins_total,
            self.angle_grid_out,
            grid,
            masses,
            cumulative,
        )
        sigma_mids = np.zeros(len(E))
        sigma_mids[idx] = sigma
        return {"energy_grid": grid * MEV, "per_particle": per_particle, "sigma_mids": sigma_mids}

    def _endf_spectrum(self, rxn):
        """Calculate the neutrons of an evaluated reaction on the output mesh, per beam
        particle.

        Each internal bin adds n_target dx sigma times the fraction of the beam still
        present (see calculate_attenuation). Two-body channels (discrete levels, LAW=2) are
        calculated at the beam energy of every internal bin, with its own kinematics (see
        endf.NeutronProduction.two_body_spectrum): each level makes a sharp line in lab energy
        and angle at one beam energy, so interpolating between beam energies would leave a
        band for each beam energy calculated. The other channels, whose spectra change
        smoothly with the beam energy, are calculated (see lab_spectrum) at beam energies from
        the highest to the lowest in the target and interpolated linearly to the beam energy
        in each internal bin: continua ENDF_ENERGY_STEP apart, and lab-frame tables (LAW=7),
        whose spectra have sharper features that move with the beam energy,
        ENDF_TABLE_ENERGY_STEP apart. The neutron energy mesh runs from 0
        to the highest lab energy, in steps of energy_bin_width.

        Returns {"energy_grid": the energy mesh [eV], "per_particle": neutrons per beam
        particle on the output (position, angle, energy) mesh, "sigma_mids": the neutron
        production cross section [b] at each internal bin (as interpolated)}.
        """
        production = self.reactions[rxn].production
        E = self.beam_energy_mids / MEV  # The nuclear data are in MeV
        density = self.target_density_mids[rxn]
        active = (density > 0) & (E > 0)
        width = self.energy_bin_width / MEV
        sigma_mids = np.zeros(len(E))
        if not np.any(active):
            grid = np.array([0.0, width])
            per_particle = np.zeros((self.position_bins_total, self.num_angle_bins_out, 1))
            return {
                "energy_grid": grid * MEV,
                "per_particle": per_particle,
                "sigma_mids": sigma_mids,
            }

        num = max(1, int(np.ceil(production.max_lab_energy(E[active].max()) / width)))
        grid = np.arange(num + 1) * width
        per_particle = np.zeros((self.position_bins_total, self.num_angle_bins_out, num))
        idx = np.flatnonzero(active)
        # Angle cells over which each distribution is integrated: several per output angle bin
        num_cm_angles = max(
            ENDF_MIN_ANGLE_CELLS, ENDF_ANGLE_CELLS_PER_BIN * self.num_angle_bins_out
        )
        coef = density[idx] * 1e-24 * self.dx[idx] * self.beam_fraction_mids[idx]
        rows = self._output_position_index()[idx]

        # Smooth channels, interpolated between beam energies: continua (LAW=1) and lab-frame
        # tables (LAW=7), each on its own grid of beam energies (see ENDF_ENERGY_STEP)
        laws = {c.law for c in production.channels if c.law != 2}
        groups = [
            ({law for law in laws if law != 7}, ENDF_ENERGY_STEP),
            ({7} & laws, ENDF_TABLE_ENERGY_STEP),
        ]
        for group, (rel, step_min) in groups:
            if not group:
                continue
            # Beam energies to calculate at, from the top down
            top, bottom = E[active].max(), E[active].min()
            energies = [top]
            while energies[-1] > bottom:
                energies.append(max(bottom, energies[-1] - max(rel * energies[-1], step_min)))
            energies = np.array(energies)
            # Interpolation: each active bin between energies[k] (above) and energies[k + 1]
            if len(energies) == 1:
                k, t = np.zeros(len(idx), dtype=int), np.zeros(len(idx))
            else:
                k = np.clip(
                    np.searchsorted(-energies, -E[idx], side="right") - 1, 0, len(energies) - 2
                )
                t = (energies[k] - E[idx]) / (energies[k] - energies[k + 1])
            k1 = np.minimum(k + 1, len(energies) - 1)
            totals = np.zeros(len(energies))
            needed = [
                j
                for j in np.unique(np.concatenate((k, k1)))
                if np.any((k == j) & (t < 1)) or np.any((k1 == j) & (k1 != k) & (t > 0))
            ]

            binned = {}  # Lab-frame tables binned on this mesh, shared by the beam energies

            def spectrum_at(j, group=group, binned=binned):
                return production.lab_spectrum(
                    energies[j],
                    self.angle_grid_out,
                    grid,
                    num_cm_angles=num_cm_angles,
                    two_body=False,
                    laws=group,
                    cache=binned,
                )

            # In order, so the sums do not depend on how the work is shared out
            for j, (spectrum, totals[j]) in zip(needed, in_parallel(spectrum_at, needed)):
                weight = np.where(k == j, 1 - t, 0.0) + np.where((k1 == j) & (k1 != k), t, 0.0)
                use = weight > 0
                by_row = np.bincount(
                    rows[use], coef[use] * weight[use], minlength=self.position_bins_total
                )
                touched = np.flatnonzero(by_row)
                per_particle[touched] += by_row[touched, None, None] * spectrum[None]
            sigma_mids[idx] += (1 - t) * totals[k] + t * totals[k1]
        if production.has_two_body:  # Discrete levels: at each bin's own beam energy
            two_body, sigma_two_body = production.two_body_spectrum(
                E[idx],
                coef,
                rows,
                self.position_bins_total,
                self.angle_grid_out,
                grid,
            )
            per_particle += two_body
            sigma_mids[idx] += sigma_two_body
        return {"energy_grid": grid * MEV, "per_particle": per_particle, "sigma_mids": sigma_mids}

    def bin_neutrons(self):
        """Scale each reaction's neutrons per beam particle (see calculate_reactions) by the
        beam particle rate.

        Sets, for each reaction, self.neutrons[rxn] [n/s] with shape
        (position_bins_total, num_angle_bins_out, num_energy_bins[rxn]), its energy mesh
        self.neutron_energy_grid[rxn] [eV], the yield self.yields[rxn] [n/s] and the fraction
        of neutrons lost to scraping self.neutron_scrape_fractions[rxn]. The totals over the
        reactions are self.yield_total and self.neutron_scrape_fraction.
        """
        self.num_energy_bins = {}
        self.neutron_energy_grid = {}
        self.neutron_energy_mids = {}
        self.neutrons = {}
        self.yields = {}
        self.neutron_scrape_fractions = {}

        for rxn in self.reactions:
            spectra = self.spectra[rxn]
            grid = spectra["energy_grid"]
            self.neutron_energy_grid[rxn] = grid
            self.neutron_energy_mids[rxn] = (grid[:-1] + grid[1:]) / 2
            self.num_energy_bins[rxn] = len(grid) - 1
            self.neutrons[rxn] = spectra["per_particle"] * self.particle_rate
            self.yields[rxn] = np.sum(self.neutrons[rxn])
            # Fraction lost to scraping: the yields with and without it
            unscraped = np.sum(
                spectra["sigma_mids"] * self.target_density_mids[rxn] * 1e-24 * self.dx
                * self.survival_mids
            ) * self.particle_rate  # fmt: skip
            self.neutron_scrape_fractions[rxn] = (
                1 - self.yields[rxn] / unscraped if unscraped > 0 else 0.0
            )
            logger.debug(f"{self.name}: {rxn} neutron yield = {self.yields[rxn]:.4e} n/s")

        self.yield_total = sum(self.yields.values())
        if self.yield_total == 0:
            raise ValueError(f"{self.name}: {NO_NEUTRONS}")
        self.neutron_scrape_fraction = (
            sum(self.yields[rxn] * self.neutron_scrape_fractions[rxn] for rxn in self.reactions)
            / self.yield_total
        )
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

    def in_units(self, value, quantity):
        """Return value (in the code's units) in the units of the results for quantity (see
        units), and those units' name."""
        unit = self.units[quantity]
        return from_internal(value, unit), unit

    def get_inputs(self):
        """Return the input parameters as a list of (name, value, units).

        A calculated region pressure, density, or length is given as calculated, and its
        units say so, as does a calculated current. The region values are lists, one entry
        per region in beam order. beam_energy_backstop and yield_total are
        included only if they were given, and density only if some gas is given by density.
        material lists each region's material by name (see unique_materials).
        """
        calculated = {
            name: (
                f", calculated for region {self.calculated_region + 1}"
                if self.calculated == name
                else ""
            )
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
            E, unit = self.in_units(self.beam_energy_backstop, "beam_energy")
            requested = [("beam_energy_backstop", E, f"{unit}, requested")]
        E_entry, E_unit = self.in_units(self.beam_energy_entry, "beam_energy")
        current, I_unit = self.in_units(self.beam_current, "current")
        pressure, P_unit = self.in_units(self.pressure, "pressure")
        temperature, T_unit = self.in_units(self.temperature, "temperature")
        divergence, div_unit = self.in_units(self.divergence, "divergence")
        width, width_unit = self.in_units(self.energy_bin_width, "neutron_energy")
        return (
            [
                ("beam", BEAMS[self.beam].description, "beam particle"),
                ("beam_energy_entry", E_entry, E_unit),
                (
                    "beam_current",
                    current,
                    f"{I_unit}, calculated" if self.beam_current_calculated else I_unit,
                ),
            ]
            + requested_yield
            + [
                ("region", self.region_names, "per region: names (if any)"),
                ("material", material_names, "per region; see [materials]"),
                ("pressure", pressure, P_unit + calculated["pressure"] + unused),
                ("temperature", temperature, T_unit + unused),
            ]
            + density
            + [
                ("length", self.length, "cm" + calculated["length"]),
            ]
            + requested
            + [
                ("beam_radius_entry", self.beam_radius_entry, "cm"),
                ("wall_radius", self.wall_radius, "cm"),
                ("divergence", divergence, div_unit),
                ("position_bins", self.position_bins, ""),
                ("position_substeps", self.position_substeps, "per position bin"),
                ("angle_bin_width", self.angle_bin_width, "deg"),
                ("energy_bin_width", width, width_unit),
                ("stopping_power", self.stopping_power, ""),
                ("nuclear_data", self.nuclear_data_given, "libraries, searched in order"),
                ("nuclear_data_dirs", self.nuclear_data, "where they were found"),
            ]
        )

    def write_hdf5(
        self, path, run_info=None, direction="-z", entry_position=None, backstop_position=None
    ):
        """Write the source on the output mesh to an HDF5 file at path.

        The layout is described in docs/outputs.md; the file's format_version
        (HDF5_FORMAT_VERSION) is increased whenever it changes. direction, entry_position, and
        backstop_position record where the source is placed (see sdef_positions; read by
        openmc_source.get_openmc_sources), and run_info adds file attributes, such as the input
        file.
        """
        entry, backstop = self.sdef_positions(direction, entry_position, backstop_position)
        _make_parent_dir(path)
        E_grid, E_unit = self.in_units(self.beam_energy_grid_out, "beam_energy")
        current_grid, I_unit = self.in_units(self.current_grid_out, "current")
        current_mids, _ = self.in_units(self.current_mids_out, "current")
        datasets = [
            ("x_grid", self.x_grid_out, "cm", "position mesh along the beam (bin boundaries)"),
            ("beam_energy_grid", E_grid, E_unit, "beam energy at each position grid point"),
            ("radius_grid", self.radius_grid_out, "cm",
             "beam edge radius at each position grid point, capped at the wall radius"),
            ("scrape_grid", self.scrape_grid_out, "fraction",
             "fraction of the beam scraped off at each position grid point"),
            ("current_grid", current_grid, I_unit, "beam current at each position grid point"),
            ("beam_energy_mids", (E_grid[:-1] + E_grid[1:]) / 2, E_unit,
             "beam energy at the center of each position bin"),
            ("radius_mids", self.radius_mids_out, "cm",
             "beam edge radius over each position bin (root mean square), as in the SDEF"),
            ("scrape_mids", self.scrape_mids_out, "fraction",
             "fraction of the beam scraped off, averaged over each position bin"),
            ("current_mids", current_mids, I_unit,
             "beam current, averaged over each position bin"),
            ("angle_grid", self.angle_grid_out, "deg", "emission angle mesh, from the beam direction"),
        ]  # fmt: skip
        for rxn, reaction in self.reactions.items():
            grid, unit = self.in_units(self.neutron_energy_grid[rxn], "neutron_energy")
            datasets.append((f"neutron_energy_grid_{rxn}", grid, unit,
                             f"neutron energy mesh for {reaction.label}"))  # fmt: skip
        for rxn, reaction in self.reactions.items():
            datasets.append((f"neutrons_{rxn}", self.neutrons[rxn], "n/s",
                             f"{reaction.label} neutron emission rate in each (position, angle, "
                             "energy) cell"))  # fmt: skip
        with h5py.File(path, "w") as hf:
            hf.attrs["format"] = "beam-target"
            hf.attrs["format_version"] = HDF5_FORMAT_VERSION
            hf.attrs["name"] = self.name
            hf.attrs["code_version"] = get_code_version()
            hf.attrs["created"] = self.created
            hf.attrs["warnings"] = self.warnings  # See check_channels and check_position_mesh
            hf.attrs["direction"] = direction
            hf.attrs["entry_position"] = entry
            hf.attrs["backstop_position"] = backstop
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
                g.attrs["isotopes"] = [atom_name(Z, A) for Z, A in material.isotope_fractions]
                g.attrs["isotope_fractions"] = list(material.isotope_fractions.values())
            for key, data, units, description in datasets:
                # gzip level 1 with shuffle: readable everywhere, and smaller and faster to
                # write than the default level 4 (measured: 0.8 against 1.4 s for 100 MB)
                hf.create_dataset(
                    key, data=data, compression="gzip", compression_opts=1, shuffle=True
                )
                hf[key].attrs["units"] = units
                hf[key].attrs["description"] = description
            for rxn, reaction in self.reactions.items():
                hf[f"neutrons_{rxn}"].attrs["dimensions"] = ["position", "angle", "energy"]
                hf[f"neutrons_{rxn}"].attrs["reaction"] = reaction.label
                hf[f"neutrons_{rxn}"].attrs["data"] = self.reaction_data[rxn]

    def sdef_positions(self, direction="-z", entry_position=None, backstop_position=None):
        """Return (entry, backstop): the points [cm] where the beam enters the first region
        and where it reaches the backstop, from one of them (the source's total length apart
        along the beam direction). Give entry_position or backstop_position, not both; with
        neither, the beam enters at the origin. backstop_position places the source by its
        backstop end, which stays put even when a length is calculated.
        """
        if direction not in SDEF_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {', '.join(SDEF_DIRECTIONS)}, got {direction!r}"
            )
        if entry_position is not None and backstop_position is not None:
            raise ValueError("give entry_position or backstop_position, not both")
        given = backstop_position if entry_position is None else entry_position
        name = "entry_position" if backstop_position is None else "backstop_position"
        point = np.array((0.0, 0.0, 0.0) if given is None else given, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{name} must be three numbers (x, y, z), got {given!r}")
        _, vec, _ = SDEF_DIRECTIONS[direction]
        along = self.x_grid_out[-1] * np.array(vec.split(), dtype=float)
        if backstop_position is None:
            return point, point + along
        return point - along, point

    def get_mcnp_sdef(
        self, direction="-z", run_info=None, entry_position=None, backstop_position=None
    ):
        """Return an MCNP SDEF source definition as a string.

        The source is a cylinder along the beam axis, from where the beam enters the first region
        to the backstop, headed by the case summary (see get_summary) and notes on using it;
        docs/outputs.md describes the card. Place it by its entry point (entry_position) or its
        backstop end (backstop_position); see sdef_positions. For -z the card runs along +z with
        POS at the backstop end (see SDEF_DIRECTIONS); for every other direction POS is the entry
        point.

        Args:
            direction: Beam direction, one of SDEF_DIRECTIONS ("-x", "+x", ..., "+z").
            run_info: Optional extra entries for the summary's [run] section.
            entry_position: Point where the beam enters the first region, (x, y, z) [cm].
            backstop_position: Point where the beam reaches the backstop, instead.
        """
        entry, backstop = self.sdef_positions(direction, entry_position, backstop_position)
        axs, vec, reverse_x = SDEF_DIRECTIONS[direction]
        length = self.x_grid_out[-1]
        pos = backstop if reverse_x else entry  # For -z the card runs along +z from the backstop

        def point(p):
            return " ".join(f"{v + 0.0:.12g}" for v in p)  # + 0.0 turns -0.0 into 0

        # The radius varies along the beam if it diverges or a narrower wall scrapes it
        divergent = np.ptp(self.radius_mids_out) > 0
        isotropic = self.angle_bin_width == 180
        dashed_line = "c " + "-" * 78 + "\n"

        run = {
            **(run_info or {}),
            "sdef_direction": direction,
            "sdef_entry_position": entry,
            "sdef_backstop_position": backstop,
        }
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
        ext = sum(np.sum(self.neutrons[rxn], (1, 2)) for rxn in self.reactions)
        nvals = self.position_bins
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
            dir_vals = sum(np.sum(self.neutrons[rxn], (0, 2)) for rxn in self.reactions)
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
            s += f"  si3   0 {self.radius_mids_out[0]:.{SDEF_RADIUS_DECIMALS}f}\n"
        s += "c\n"
        s += dashed_line

        # Energy distribution: one segment of bins per group of reactions whose energies
        # overlap (such as DD and DT side by side), each starting with a zero bin
        s += "c\n"
        s += "  sc4   Energy distribution\n"
        if isotropic:
            spectra = {rxn: np.sum(self.neutrons[rxn], (0, 1)) for rxn in self.reactions}
            si, sp = self._sdef_energy_bins(spectra, trim=False)
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
                spectra = {rxn: np.sum(self.neutrons[rxn][:, k, :], 0) for rxn in self.reactions}
                note = ""
                if not any(np.any(spectrum > 0) for spectrum in spectra.values()):
                    # No neutrons at these angles (sp2 gives them zero probability), but the
                    # distribution must still be valid: use the spectrum over all angles
                    spectra = {rxn: np.sum(self.neutrons[rxn], (0, 1)) for rxn in self.reactions}
                    note = "; no neutrons, never sampled"
                si, sp = self._sdef_energy_bins(spectra, trim=True)
                s += (
                    f"  sc{sid:<3d} Energy distribution "
                    f"({self.angle_grid_out[k]:.12g} - {self.angle_grid_out[k + 1]:.12g} degrees"
                    f"{note})\n"
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
        s += f"c     Source definition for {self.name}, made by beam-target\n"
        s += "c     Case summary (TOML; the same as summary.toml written by write_text_files):\n"
        s += dashed_line
        s += "c\n"
        for line in self.get_summary(run_info, title=False).splitlines():
            s += f"c {line}".rstrip() + "\n"
        s += "c\n"
        s += dashed_line
        return s

    def _sdef_energy_bins(self, spectra, trim):
        """Return the SI entries and SP values of an energy distribution.

        spectra holds {rxn: spectrum} on each reaction's own energy mesh; reactions with no
        neutrons are left out. If trim is True, empty bins at either end of each spectrum are
        dropped. Reactions whose energy ranges overlap or touch are added together on one
        segment of bins (MCNP needs increasing energies); each segment starts with a zero bin.
        """
        width = self.energy_bin_width
        ranges = []  # (first bin, end bin, spectrum) in units of energy_bin_width from 0 eV
        for rxn, spectrum in spectra.items():
            if rxn not in self.reactions_present() or not np.any(spectrum > 0):
                continue
            ind_min, ind_max = 0, len(spectrum)
            if trim:
                inds = np.nonzero(spectrum)[0]
                ind_min, ind_max = inds[0], inds[-1] + 1
            first = int(round(self.neutron_energy_grid[rxn][0] / width))
            ranges.append((first + ind_min, first + ind_max, spectrum[ind_min:ind_max]))

        groups = []
        for lo, hi, values in sorted(ranges, key=lambda r: r[0]):
            if groups and lo <= groups[-1][1]:
                groups[-1][1] = max(groups[-1][1], hi)
                groups[-1][2].append((lo, values))
            else:
                groups.append([lo, hi, [(lo, values)]])

        si = ""
        sp = []
        for lo, hi, parts in groups:
            combined = np.zeros(hi - lo)
            for start, values in parts:
                combined[start - lo : start - lo + len(values)] += values
            E_min = round(lo * width / MEV, 12)  # MCNP energies are in MeV
            E_max = round(hi * width / MEV, 12)
            si += f" {E_min} {hi - lo - 1}i {E_max}"
            sp = np.append(sp, np.append(0, combined))
        return si, sp

    def write_mcnp_sdef(
        self, path, direction="-z", run_info=None, entry_position=None, backstop_position=None
    ):
        """Write the MCNP SDEF card (see get_mcnp_sdef) to a text file at path.

        The file is ASCII with Unix (LF) line endings on every platform.
        """
        _make_parent_dir(path)
        with open(path, "w", encoding="ascii", newline="\n") as w:
            w.write(self.get_mcnp_sdef(direction, run_info, entry_position, backstop_position))

    def reactions_present(self):
        """Return the beam's reactions that produce neutrons in this case: their target
        isotope is in some region, and the beam is above their threshold there."""
        return [rxn for rxn in self.reactions if self.yields[rxn] > 0]

    def mean_neutron_energy(self, rxn=None):
        """Mean energy [eV] of the neutrons from one reaction, or (rxn None) from all of
        them, over all angles and positions; nan if there are none."""
        total = weighted = 0.0
        for r in [rxn] if rxn else self.reactions_present():
            spectrum = np.sum(self.neutrons[r], (0, 1))
            total += np.sum(spectrum)
            weighted += np.sum(spectrum * self.neutron_energy_mids[r])
        return weighted / total if total > 0 else float("nan")

    def neutrons_by_angle_and_energy(self):
        """Return the neutron source integrated over position, on one energy mesh for all the
        beam's reactions.

        Returns (neutron_energy_grid, neutrons): neutron_energy_grid [eV] runs from the lowest to the highest neutron
        energy of the reactions present, in steps of energy_bin_width (the reactions' own meshes are
        aligned with it), and neutrons[rxn] [n/s] has shape (num_angle_bins_out, len(neutron_energy_grid) - 1) for
        each of the beam's reactions (zero for a reaction that produces no neutrons here).
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
        for rxn in self.reactions:
            neutrons[rxn] = np.zeros((self.num_angle_bins_out, i_hi - i_lo))
            if rxn in present:
                by_angle = np.sum(self.neutrons[rxn], 0)
                i = first[rxn] - i_lo
                neutrons[rxn][:, i : i + by_angle.shape[1]] = by_angle
        return neutron_energy_grid, neutrons

    def get_results(self):
        """Return the key results as a list of (name, value, units or description)."""
        results = [("yield_total", self.yield_total, "n/s")]
        results += [
            (f"yield_{rxn}", self.yields[rxn], f"n/s, {reaction.label}")
            for rxn, reaction in self.reactions.items()
        ]
        results += [
            (f"fraction_{rxn}", self.yields[rxn] / self.yield_total, "share of the total yield")
            for rxn in self.reactions
        ]
        if self.reactions_present():
            mean, unit = self.in_units(self.mean_neutron_energy(), "neutron_energy")
            results.append(("mean_neutron_energy", mean, f"{unit}, all reactions and angles"))
        for rxn in self.reactions_present():
            mean, unit = self.in_units(self.mean_neutron_energy(rxn), "neutron_energy")
            results.append((f"mean_neutron_energy_{rxn}", mean, f"{unit}, over all angles"))
        region_ends = np.cumsum(self.position_bins_internal)
        E_ends, E_unit = self.in_units(self.beam_energy_grid[region_ends], "beam_energy")
        current, I_unit = self.in_units(self.current_grid[-1], "current")
        power = {
            name: self.in_units(value, "power")[0]
            for name, value in [
                ("entry", self.beam_power_entry),
                ("deposited", self.deposited_power),
                ("wall", self.wall_power),
                ("backstop", self.backstop_power),
                ("nuclear", self.nuclear_power),
            ]
        }
        W_unit = self.units["power"]
        results += [
            ("atom_density", self.atom_density_per_region, "atoms/cm3, in each region"),
            ("beam_energy_region_end", E_ends, f"{E_unit}, at the end of each region"),
            ("beam_energy_backstop", E_ends[-1], E_unit),
            ("beam_radius_backstop", self.radius_grid[-1], "cm, beam edge; capped at the wall"),
            ("beam_current_backstop", current, f"{I_unit}, after scraping and nuclear losses"),
            ("beam_power_entry", power["entry"], W_unit),
            (
                "beam_power_deposited",
                power["deposited"],
                f"{W_unit}, deposited in the target material (gas or solid)",
            ),
            ("beam_power_wall", power["wall"], f"{W_unit}, carried onto the wall by scraping"),
            ("beam_power_backstop", power["backstop"], W_unit),
            (
                "beam_power_nuclear",
                power["nuclear"],
                f"{W_unit}, of the deposited power: beam lost to nuclear reactions",
            ),
            (
                "beam_nuclear_loss",
                1 - self.survival_grid[-1],
                "fraction of the beam removed by nuclear reactions",
            ),
            ("neutron_scrape_fraction", self.neutron_scrape_fraction, "neutrons lost to scraping"),
        ]
        return results

    def get_summary(self, run_info=None, title=True):
        """Return the key inputs and results as TOML text (readable with tomllib).

        Sections: [run] (case name, code version, creation time, any run_info entries, such
        as the input file and SDEF direction, and any warnings, see check_channels), [inputs]
        (the BeamTarget arguments, see get_inputs; lists have one value per region in beam
        order) and [results] (see get_results), followed by a [materials.<name>] table for
        each material used (phase, density for a solid, and its atoms or molecules). Units are
        given in comments. title adds a first comment line saying what the file is. The "="
        signs are aligned across all sections, and so are the comments (except after values
        too long for the comment column).
        """
        sections = {
            "run": [
                ("name", self.name, ""),
                ("code_version", get_code_version(), ""),
                ("created", self.created, ""),  # ISO 8601 with "+00:00": UTC
            ]
            + [(name, value, "") for name, value in (run_info or {}).items()]
            + ([("warnings", self.warnings, "")] if self.warnings else []),
            "inputs": self.get_inputs(),
            "results": self.get_results(),
        }
        key_width = max(len(name) for entries in sections.values() for name, _, _ in entries)
        comment_column = key_width + 19  # Room for " = " and a 16-character value

        s = "# Neutron source summary (TOML; read it in Python with tomllib)\n\n" if title else ""
        for i, (section, entries) in enumerate(sections.items()):
            s += ("\n" if i else "") + f"[{section}]\n"
            for name, value, comment in entries:
                if name == "warnings":  # Long texts: one per line, wrapped
                    s += f"{name:<{key_width}} = [\n"
                    s += "".join(f"    {_toml_wrapped(w, indent='    ')},\n" for w in value)
                    s += "]\n"
                    continue
                text = f"{name:<{key_width}} = {_toml_value(value)}"
                if (
                    len(text) > SUMMARY_LINE_LENGTH
                    and isinstance(value, (list, tuple))
                    and all(isinstance(v, str) for v in value)
                ):  # A long list of strings (such as paths): one per line
                    head = f"{name:<{key_width}} = ["
                    s += f"{head:<{comment_column}} # {comment}\n" if comment else head + "\n"
                    s += "".join(f"    {_toml_value(v)},\n" for v in value) + "]\n"
                    continue
                if len(text) > SUMMARY_LINE_LENGTH and isinstance(value, str):  # Wrapped
                    text = f"{name:<{key_width}} = " + _toml_wrapped(
                        value, SUMMARY_LINE_LENGTH - key_width - 8, " " * (key_width + 6)
                    )
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
        for rxn in self.reactions_present():
            s += f"\n[reactions.{_toml_key(rxn)}]\n"
            s += f"{'reaction':<{key_width}} = {_toml_value(self.reactions[rxn].label)}\n"
            text = f"{'data':<{key_width}} = {_toml_value(self.reaction_data[rxn])}"
            if len(text) > SUMMARY_LINE_LENGTH:
                text = f"{'data':<{key_width}} = " + _toml_wrapped(
                    self.reaction_data[rxn],
                    SUMMARY_LINE_LENGTH - key_width - 8,
                    " " * (key_width + 6),
                )
            s += text + "\n"
        return s

    def write_text_files(self, directory, run_info=None):
        """Write summary.toml and the integrated profiles to directory.

        summary.toml holds the key inputs and results (see get_summary; run_info adds entries to
        its [run] section), and axial_profile.txt, energy_spectrum.txt, and
        angular_distribution.txt the source integrated over all but one variable; each profile
        sums to the neutron yields. docs/outputs.md describes the files.
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

        rxns = list(self.reactions)
        E_grid, E_unit = self.in_units(self.beam_energy_grid_out, "beam_energy")
        deposited, W_unit = self.in_units(self.deposited_power_mids_out, "power")
        wall, _ = self.in_units(self.wall_power_mids_out, "power")
        columns = [f"{rxn} neutrons [n/s] ({self.reactions[rxn].label})" for rxn in rxns]
        axial = {rxn: np.sum(self.neutrons[rxn], (1, 2)) for rxn in rxns}
        write_table(
            "axial_profile.txt",
            "Neutron source along the beam, integrated over emission angle and neutron energy",
            [
                "start of position bin [cm]",
                "end of position bin [cm]",
                f"beam energy at the bin center [{E_unit}]",
                "beam edge radius over the bin [cm] (root mean square; capped at the wall)",
                "fraction of the beam scraped off, averaged over the bin",
                *columns,
                "total neutrons [n/s]",
                f"beam power deposited in the target material over the bin [{W_unit}]",
                f"beam power carried onto the wall by scraping over the bin [{W_unit}]",
            ],
            [
                self.x_grid_out[:-1],
                self.x_grid_out[1:],
                (E_grid[:-1] + E_grid[1:]) / 2,
                self.radius_mids_out,
                self.scrape_mids_out,
                *(axial[rxn] for rxn in rxns),
                sum(axial.values()),
                deposited,
                wall,
            ],
        )

        neutron_energy_grid, by_angle_energy = self.neutrons_by_angle_and_energy()
        neutron_energy_grid, n_unit = self.in_units(neutron_energy_grid, "neutron_energy")
        energy = {rxn: np.sum(by_angle_energy[rxn], 0) for rxn in rxns}
        write_table(
            "energy_spectrum.txt",
            "Neutron energy spectrum, integrated over position and emission angle",
            [
                f"start of energy bin [{n_unit}]",
                f"end of energy bin [{n_unit}]",
                *columns,
                "total neutrons [n/s]",
            ],
            [
                neutron_energy_grid[:-1],
                neutron_energy_grid[1:],
                *(energy[rxn] for rxn in rxns),
                sum(energy.values()),
            ],
        )

        angular = {rxn: np.sum(self.neutrons[rxn], (0, 2)) for rxn in rxns}
        total = sum(angular.values())
        write_table(
            "angular_distribution.txt",
            "Neutron angular distribution (angle from the beam direction), integrated over "
            "position and neutron energy",
            [
                "start of angle bin [deg]",
                "end of angle bin [deg]",
                "solid angle of the bin [sr]",
                *columns,
                "total neutrons [n/s]",
                "total neutrons per steradian [n/s/sr]",
            ],
            [
                self.angle_grid_out[:-1],
                self.angle_grid_out[1:],
                self.solid_angles_out,
                *(angular[rxn] for rxn in rxns),
                total,
                total / self.solid_angles_out,
            ],
        )

    def make_plots(self, directory):
        """Write plots of the beam and the neutron source to directory as PNG files.

        docs/outputs.md lists them. File names do not include the case name, so use a separate
        directory per case.
        """
        os.makedirs(directory, exist_ok=True)
        self.plot_beam(directory)
        self.plot_neutron_position(directory)
        self.plot_neutron_angle(directory)
        self.plot_neutron_energy(directory)
        self.plot_neutron_energy_surface(directory)

    def _position_units(self):
        """Return (scale, unit) for position plots: micrometers if the whole source is shorter
        than 1 mm (such as a solid target), centimeters otherwise. Positions in cm are
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
        """Plot the beam energy, radius, current, and deposited power against position, in a
        2 x 2 grid."""
        scale, unit = self._position_units()
        x = self.x_grid_out * scale
        fig, axes = _pyplot().subplots(2, 2, sharex=True, figsize=(13, 9))
        (ax_energy, ax_radius), (ax_current, ax_power) = axes

        E_grid, E_unit = self.in_units(self.beam_energy_grid_out, "beam_energy")
        ax_energy.plot(x, E_grid)
        ax_energy.set_ylabel(f"Beam energy [{E_unit}]")
        ax_energy.set_ylim(bottom=0)

        ax_radius.plot(x, self.radius_grid_out, label="Beam edge")
        region_edges = np.append(0, np.cumsum(self.length)) * scale
        ax_radius.stairs(
            self.wall_radius, region_edges, baseline=None, color="k", linewidth=1.5, label="Wall"
        )
        ax_radius.set_ylabel("Radius [cm]")
        ax_radius.set_ylim(0, 1.1 * np.max(self.wall_radius))
        ax_radius.legend(loc="lower right")

        current_grid, I_unit = self.in_units(self.current_grid_out, "current")
        ax_current.plot(x, current_grid)
        ax_current.set_ylabel(f"Beam current [{I_unit}]")
        ax_current.set_ylim(0, 1.05 * self.in_units(self.beam_current, "current")[0])

        # Power per unit length [W per cm or um] in each output bin, totals in the legend
        entry, W_unit = self.in_units(self.beam_power_entry, "power")
        deposited = self.in_units(self.deposited_power, "power")[0]
        wall = self.in_units(self.wall_power, "power")[0]
        backstop = self.in_units(self.backstop_power, "power")[0]
        deposited_label = f"Deposited in the target ({deposited:.3g} {W_unit})"
        wall_label = f"Onto the wall ({wall:.3g} {W_unit})"
        per_length = 1 / (self.dx_out * scale)
        ax_power.stairs(self.deposited_power_mids_out * per_length, x, label=deposited_label)
        ax_power.stairs(self.wall_power_mids_out * per_length, x, label=wall_label)
        ax_power.set_ylabel(f"Power deposited [W/{unit}]")
        ax_power.set_ylim(bottom=0)
        ax_power.legend(
            loc="center left",
            title=f"Beam power {entry:.3g} {W_unit}\nTo the backstop: {backstop:.3g} {W_unit}",
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
            (
                _plot_label(self.reactions[rxn].label),
                np.sum(self.neutrons[rxn], (1, 2)) / (self.dx_out * scale),
            )
            for rxn in self.reactions_present()
        ]
        if len(lines) > 1:
            lines.append(("Total", sum(values for _, values in lines)))

        fig, ax = _pyplot().subplots(figsize=(8, 6))
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
            (
                _plot_label(self.reactions[rxn].label),
                np.sum(self.neutrons[rxn], (0, 2)) / self.solid_angles_out,
            )
            for rxn in self.reactions_present()
        ]
        if len(lines) > 1:
            lines.append(("Total", sum(values for _, values in lines)))

        fig, ax = _pyplot().subplots(figsize=(8, 6))
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
        spectrum at that angle, on a log color scale covering four decades below the peak;
        empty cells are left blank.

        The spectrum is integrated over angle and drawn on linear and log scales (the log
        one covers five decades below the peak). Each reaction is shown separately and drops
        to zero at the ends of its energy range; if the ranges of some overlap (such as the
        two 7Li(p,n) lines), their total is shown too. The axes follow the neutron energies,
        so the plots suit any beam energy.
        """
        from matplotlib.colors import LogNorm

        present = self.reactions_present()
        width, unit = self.in_units(self.energy_bin_width, "neutron_energy")
        for rxn in present:
            by_angle = np.sum(self.neutrons[rxn], 0)  # (angle, energy) [n/s]
            density = by_angle / self.solid_angles_out[:, None] / width
            filled = np.flatnonzero(np.sum(by_angle, 0))
            lo, hi = filled[0], filled[-1] + 1
            fig, ax = _pyplot().subplots(figsize=(8, 6))
            mesh = ax.pcolormesh(
                self.angle_grid_out,
                self.in_units(self.neutron_energy_grid[rxn][lo : hi + 1], "neutron_energy")[0],
                np.ma.masked_less_equal(density[:, lo:hi].T, 0),
                cmap="Spectral_r",
                norm=LogNorm(vmin=np.max(density) * 1e-4, vmax=np.max(density)),
            )
            fig.colorbar(mesh, ax=ax, label=rf"Neutrons [n/({unit}$\cdot$s$\cdot$sr)]")
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, 30))
            ax.set_xlabel("Emission angle from the beam direction [degrees]")
            ax.set_ylabel(f"Neutron energy [{unit}]")
            ax.grid()
            label = _plot_label(self.reactions[rxn].label)
            ax.set_title(f"{label} neutron energy against emission angle ({self.name})")
            _save_figure(fig, os.path.join(directory, f"neutron_energy_angle_{rxn}.png"))

        energy_grid, on_common_grid = self.neutrons_by_angle_and_energy()
        energy_grid = self.in_units(energy_grid, "neutron_energy")[0]
        peak = 0.0
        fig, ax = _pyplot().subplots(figsize=(8, 6))
        spectra = {rxn: np.sum(on_common_grid[rxn], 0) / width for rxn in present}
        lines = [(_plot_label(self.reactions[rxn].label), spectra[rxn], {}) for rxn in present]
        overlap = np.sum([spectrum > 0 for spectrum in spectra.values()], 0)
        if np.any(overlap > 1):
            # Wide and gray, underneath the reactions, so that they stay visible
            style = {"color": "0.75", "linewidth": 3}
            lines.insert(0, ("Total", sum(spectra.values()), style))
        for label, spectrum, style in lines:
            # Draw each line over its own energy range, dropping to zero at both ends
            filled = np.flatnonzero(spectrum)
            lo, hi = filled[0], filled[-1] + 1
            ax.stairs(spectrum[lo:hi], energy_grid[lo : hi + 1], baseline=0, label=label, **style)
            peak = max(peak, np.max(spectrum))
        ax.set_xlim(0, energy_grid[-1] + 0.05 * (energy_grid[-1] - energy_grid[0]))
        ax.set_ylim(0, 1.05 * peak)
        ax.set_xlabel(f"Neutron energy [{unit}]")
        ax.set_ylabel(rf"Neutrons per unit energy [n/({unit}$\cdot$s)]")
        ax.set_title(f"Neutron energy spectrum ({self.name})")
        ax.grid()
        ax.legend()
        _save_figure(fig, os.path.join(directory, "neutron_energy.png"), close=False)

        ax.set_yscale("log")
        ax.set_ylim(peak * 10**-5.25, peak * 10**0.25)
        _save_figure(fig, os.path.join(directory, "neutron_energy_log.png"))

    @_plot_style
    def plot_neutron_energy_surface(self, directory):
        """Plot neutron energy against emission angle for each reaction as a surface, with a
        linear vertical axis: the same neutrons per unit energy and solid angle as the maps of
        plot_neutron_energy (see surface_density and draw_energy_surface). python -m
        beamtarget.view opens the same plot from source.h5 in a window, to turn it around.
        """
        for rxn in self.reactions_present():
            grid, unit = self.in_units(self.neutron_energy_grid[rxn], "neutron_energy")
            angles, energies, density = surface_density(
                self.angle_grid_out, grid, self.neutrons[rxn]
            )
            label = _plot_label(self.reactions[rxn].label)
            fig = _pyplot().figure(figsize=(9, 7))
            draw_energy_surface(
                fig,
                angles,
                energies,
                density,
                f"{label} neutron energy against emission angle ({self.name})",
                unit,
            )
            _save_figure(fig, os.path.join(directory, f"neutron_energy_angle_surface_{rxn}.png"))
