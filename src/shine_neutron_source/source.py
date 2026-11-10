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

PACKAGE_NAME = "shine-neutron-source"
logger = logging.getLogger(__name__)
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_DATA_DIR = os.path.join(CODE_DIR, "data")
# DROSG-2000 cross-section file for each reaction (in data/)
DROSG_FILES = {"DD": "d2hags.koe", "DT": "d3hags.koe"}

# Stopping-power data files (in data/) and the available stopping-power sets
SRIM_FILES = {"D": "srim_deuteron_in_D_gas.txt", "T": "srim_deuteron_in_T_gas.txt"}
PSTAR_FILE = "pstar_hydrogen.txt"
STOPPING_POWERS = {
    "srim": "SRIM-2012.01, deuterons in hydrogen-isotope gas",
    "icru49": "ICRU 49 electronic (protons in H2 gas, same velocity) + SRIM nuclear",
}

# Decimal places for the beam radii in the SDEF card [cm]
SDEF_RADIUS_DECIMALS = 4

# Version of the HDF5 output layout; increase it whenever the layout changes
HDF5_FORMAT_VERSION = 2

# Highest allowed beam voltage [kV]. The DROSG Legendre tables in data/ end at 40 MeV; the
# rows above that are not angular distributions (see data/README.md).
MAX_VOLTAGE = 40000.0

# Repository root when running from a source checkout (src/shine_neutron_source/ -> root)
REPO_DIR = os.path.normcase(os.path.dirname(os.path.dirname(CODE_DIR)))

# Physical constants (exact SI values)
NA = 6.02214076e23  # Avogadro constant [1/mol]
ee = 1.602176634e-19  # Elementary charge [C]
kB = 1.380649e-23  # Boltzmann constant [J/K]

# Atomic masses (neutral atoms, including electrons) of the hydrogen isotopes [u], from NIST
# Atomic Weights and Isotopic Compositions (AME2020). Used to convert SRIM's per-mass stopping
# of the neutral target gas to stopping per atom.
HYDROGEN_ATOM_MASSES = {"H": 1.00782503223, "D": 2.01410177812, "T": 3.0160492779}
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


def get_atom_density(pressure, temperature):
    """Return the atom density [1/cm^3] of a diatomic ideal gas.

    pressure is in torr and temperature in degrees C. Works element-wise on arrays.
    """
    P_Pa = pressure * 101325 / 760  # Pressure [Pa]
    T_K = temperature + 273.15  # Temperature [K]
    mdens = P_Pa / (kB * T_K) / 1e6  # Molecule density [1/cm^3]
    adens = mdens * 2  # Atom density [1/cm^3] (D2, DT and T2 are diatomic)
    return adens


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


def read_srim_stopping(fname):
    """Read a SRIM stopping-power table (SRIM's "Stopping Units = MeV/(mg/cm2)" output).

    The per-mass values are converted to stopping per target atom, so the result does not
    depend on which hydrogen isotope the SRIM target was. The isotope is identified from the
    header line "Density = <rho> g/cm3 = <n> atoms/cm3" and its exact atomic mass is used (the
    header values are rounded to 5 significant figures, too coarse to use directly).

    Returns (E [keV], S_elec, S_nuc [keV cm^2 per atom], isotope) where isotope is "H", "D"
    or "T". Raises ValueError if the target is not a hydrogen isotope.
    """
    with open(fname, "r", encoding="ascii") as f:
        lines = f.read().splitlines()
    m = re.search(r"Density\s*=\s*(\S+)\s*g/cm3\s*=\s*(\S+)\s*atoms/cm3", "\n".join(lines[:20]))
    if not m:
        raise ValueError(f"{fname}: no 'Density = ... g/cm3 = ... atoms/cm3' header line")
    header_amu = float(m.group(1)) / float(m.group(2)) * NA  # Target atom mass [amu]
    isotope = min(HYDROGEN_ATOM_MASSES, key=lambda k: abs(HYDROGEN_ATOM_MASSES[k] - header_amu))
    if abs(HYDROGEN_ATOM_MASSES[isotope] / header_amu - 1) > 1e-3:
        raise ValueError(
            f"{fname}: target atom mass {header_amu:.4f} amu is not a hydrogen isotope"
        )
    atom_mass = HYDROGEN_ATOM_MASSES[isotope] / NA  # [g]

    # The data table sits between the first two lines that start with "-"
    unit_conv = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}  # To keV
    rows = []
    in_data = False
    for line in lines:
        if line.startswith("-"):
            in_data = not in_data
            continue
        if in_data:
            tokens = line.split()
            E = float(f"{float(tokens[0]) * unit_conv[tokens[1]]:.5g}")  # [keV], 5 sig. figs
            rows.append([E, float(tokens[2]), float(tokens[3])])
    data = np.array(rows)
    to_per_atom = 1e6 * atom_mass  # MeV/(mg/cm^2) -> keV cm^2/g -> keV cm^2 per atom
    return data[:, 0], data[:, 1] * to_per_atom, data[:, 2] * to_per_atom, isotope


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


def _save_figure(fname):
    """Save the current matplotlib figure to fname."""
    plt.tight_layout()
    plt.savefig(fname)


class ShineSource:
    """Neutron source term for a deuteron beam in a D/T gas target.

    The whole calculation runs when the object is created; afterwards use write_hdf5(),
    write_mcnp_sdef(), write_text_files() and make_plots() to write the results. Summary
    values are logged at DEBUG level to the "shine_neutron_source" logger (silent unless
    logging is configured).

    pressure, temperature, length, divergence, num_x_bins and num_x_bins_out take one value
    per gas region, in the order the beam passes through them; a scalar is treated as a
    single region. The last region is the target chamber; earlier regions are normally
    upstream differential-pumping stages.

    Args:
        name: Case name, used in output file names and titles.
        T_frac: Atom fraction of tritium in the gas (0 = pure D, 1 = pure T).
        voltage: Beam voltage [kV]. The beam starts with this energy in keV. At most
            MAX_VOLTAGE (40 MeV), the upper limit of the Legendre cross-section data.
        current: Beam current [mA].
        pressure: Gas pressure in each region [torr].
        temperature: Gas temperature in each region [degC].
        length: Length of each region along the beam [cm].
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
    """

    def __init__(
        self,
        name,
        T_frac,
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
    ):
        t0 = time.time()

        # Input parameters
        self.name = str(name)
        self.T_frac = float(T_frac)
        self.voltage = float(voltage)
        self.current = float(current)
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

        self.num_regions = len(self.pressure)
        self.check_input()
        self.num_x_bins = self.num_x_bins.astype(int)
        self.num_x_bins_out = self.num_x_bins_out.astype(int)

        # Deuteron source rate [1/s]
        self.deuteron_rate = self.current / 1000 / ee

        self.setup_meshes()

        # Gas atom density in each region [1/cm^3]
        self.atom_density_per_region = get_atom_density(self.pressure, self.temperature)

        # Region properties on the internal and output position meshes
        self.divergence_mids = np.repeat(self.divergence, self.num_x_bins)
        self.atom_density_mids = np.repeat(self.atom_density_per_region, self.num_x_bins)

        self.read_stopping_power()
        self.read_cross_sections()

        self.calculate_beam_energy()
        self.calculate_scraping()
        self.calculate_beam_power()
        self.calculate_ddxs()
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
            "temperature",
            "length",
            "divergence",
            "num_x_bins",
            "num_x_bins_out",
        ]:
            require(
                len(getattr(self, attr)) == self.num_regions,
                f"{attr} has {len(getattr(self, attr))} entries, expected {self.num_regions}",
            )
        require(0 <= self.T_frac <= 1, "T_frac must be between 0 and 1")
        require(
            0 < self.voltage <= MAX_VOLTAGE,
            f"voltage must be between 0 and {MAX_VOLTAGE:g} kV (the upper limit of the "
            "Legendre cross-section data, see data/README.md)",
        )
        require(self.current > 0, "current must be positive")
        require(np.all(self.pressure > 0), "pressures must be positive")
        require(np.all(self.temperature > -273.15), "temperatures must be above 0 K")
        require(np.all(self.length > 0), "lengths must be positive")
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
        """Read the stopping power of deuterons in the gas, per atom.

        Stores self.stopping_data with "E" (deuteron energy [keV]) and "eps" (electronic +
        nuclear stopping per gas atom [keV cm^2]); the energy loss per unit length is eps times
        the atom density. The D/T mixture is handled per atom (Bragg additivity): the stopping is
        (1 - T_frac) times that in deuterium plus T_frac times that in tritium.

        - "srim": electronic and nuclear stopping from the SRIM tables for deuterons in D and
          T gas (SRIM_FILES).
        - "icru49": electronic stopping from ICRU Report 49 for protons in hydrogen gas
          (PSTAR_FILE), taken at the proton energy with the same velocity
          (E_p = E_d * M_H1 / M_H2), plus the mixed SRIM nuclear stopping, because the ICRU 49
          nuclear stopping is for protons. Electronic stopping does not depend on the
          hydrogen isotope. The table covers deuteron energies from about 2 keV upwards;
          below that the beam is treated as stopped.

        Raises ValueError if a SRIM file's target is not the expected isotope, or if the two
        SRIM files have different energy grids or electronic stopping per atom that differs
        by more than 1% (it should be the same for D and T).
        """
        tables = {}
        for isotope, fname in SRIM_FILES.items():
            path = os.path.join(self.data_dir, fname)
            *table, found = read_srim_stopping(path)
            if found != isotope:
                raise ValueError(f"{path}: SRIM target is {found}, expected {isotope}")
            tables[isotope] = table
        (E, S_elec_D, S_nuc_D), (E_T, S_elec_T, S_nuc_T) = tables["D"], tables["T"]
        if not np.array_equal(E, E_T):
            raise ValueError("The SRIM tables for D and T have different energy grids")
        if np.max(np.abs(S_elec_D / S_elec_T - 1)) > 0.01:
            raise ValueError("The SRIM tables for D and T disagree on electronic stopping")
        f_D, f_T = 1 - self.T_frac, self.T_frac
        S_elec = f_D * S_elec_D + f_T * S_elec_T
        S_nuc = f_D * S_nuc_D + f_T * S_nuc_T

        if self.stopping_power == "icru49":
            E_p, P_elec, _ = read_pstar(os.path.join(self.data_dir, PSTAR_FILE))
            E_d = E_p * 1000 * M_H2 / M_H1  # Deuteron energy at the same velocity [keV]
            P_elec = P_elec * 1000 * A_H_ICRU49 / NA  # MeV cm^2/g -> keV cm^2 per atom
            keep = (E >= E_d[0]) & (E <= E_d[-1])
            # Log-log interpolation of the ICRU 49 values onto the SRIM energy grid
            S_elec = np.exp(np.interp(np.log(E[keep]), np.log(E_d), np.log(P_elec)))
            E, S_nuc = E[keep], S_nuc[keep]
        self.stopping_data = {"E": E, "eps": S_elec + S_nuc}

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
        """Slow the beam down through the gas.

        Sets self.beam_energy_grid, the beam energy at each internal grid point [keV], and its
        midpoint and output-mesh versions. The energy loss in each bin uses the
        stopping power at the bin's average energy, found by fixed-point iteration.
        If the beam stops or leaves the stopping-power table, the remaining energies stay
        at zero.
        """
        self.beam_energy_grid = np.zeros(self.x_grid.shape)
        self.beam_energy_grid[0] = self.voltage
        for i in range(self.num_x_bins_total):
            E_start = self.beam_energy_grid[i]
            E_avg = self.beam_energy_grid[i]
            for _ in range(5):
                eps = np.interp(
                    E_avg,
                    self.stopping_data["E"],
                    self.stopping_data["eps"],
                    left=np.nan,
                    right=np.nan,
                )  # Stopping per atom [keV cm^2]
                dE = eps * self.atom_density_mids[i] * self.dx[i]
                E_end = E_start - dE
                E_avg = (E_start + E_end) / 2
            if np.isnan(E_end) or E_end < 0:
                break
            self.beam_energy_grid[i + 1] = E_end
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

        Sets the beam radius [cm], scrape fraction and current [mA] on the internal and
        output meshes. Radii are capped at wall_radius after the scrape fraction is found.
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

        self.current_grid = self.current * (1 - self.scrape_grid)
        self.current_mids = (self.current_grid[:-1] + self.current_grid[1:]) / 2

        # Output mesh: grid values are sampled, midpoint values are averaged
        def condense(vals):
            return np.add.reduceat(vals, self.out_grid_inds[:-1]) / self.x_ratio_out

        self.radius_grid_out = np.array(self.radius_grid[self.out_grid_inds])
        self.scrape_grid_out = np.array(self.scrape_grid[self.out_grid_inds])
        self.current_grid_out = np.array(self.current_grid[self.out_grid_inds])
        self.radius_mids_out = np.sqrt(condense(rsq_mids))
        self.scrape_mids_out = condense(self.scrape_mids)
        self.current_mids_out = condense(self.current_mids)

    def calculate_beam_power(self):
        """Work out where the beam power goes: into the gas, onto the wall or to the backstop.

        Beam power is beam energy times current (keV x mA = W). Along each bin, the beam
        still in it deposits the energy it loses in the gas (current x energy loss), and the
        beam scraped off hits the wall carrying its energy there (energy x current lost). Both
        use the bin averages of the other quantity, which makes the three parts add up to the
        entry power exactly. The power reaching the backstop is the remaining current times
        the final beam energy.

        Sets self.gas_power_mids and self.wall_power_mids [kW] per bin on the internal and
        output meshes (output bins are sums), and the totals self.gas_power,
        self.wall_power and self.backstop_power [kW].
        """
        beam_energy_mids = (self.beam_energy_grid[:-1] + self.beam_energy_grid[1:]) / 2
        self.gas_power_mids = self.current_mids * -np.diff(self.beam_energy_grid) / 1000
        self.wall_power_mids = beam_energy_mids * -np.diff(self.current_grid) / 1000
        self.gas_power_mids_out = np.add.reduceat(self.gas_power_mids, self.out_grid_inds[:-1])
        self.wall_power_mids_out = np.add.reduceat(self.wall_power_mids, self.out_grid_inds[:-1])
        self.gas_power = np.sum(self.gas_power_mids)
        self.wall_power = np.sum(self.wall_power_mids)
        self.backstop_power = self.beam_energy_grid[-1] * self.current_grid[-1] / 1000
        logger.debug(
            f"{self.name}: beam power {self.voltage * self.current / 1000:.4g} kW: "
            f"{self.gas_power:.4g} kW into the gas, {self.wall_power:.4g} kW onto the wall, "
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

        for rxn, frac in [("DD", 1 - self.T_frac), ("DT", self.T_frac)]:
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
            factor = (self.atom_density_mids * 1e-24 * self.dx * frac * self.deuteron_rate)[:, None]
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
            raise ValueError(
                f"{self.name}: no neutrons are produced. Check that the beam voltage is high "
                "enough (the cross-section data start at 1.8 keV for DD and 10 keV for DT) and "
                "that the beam is not scraped away entirely."
            )
        self.neutron_scrape_fraction = (
            self.yields["DD"] * self.neutron_scrape_fractions["DD"]
            + self.yields["DT"] * self.neutron_scrape_fractions["DT"]
        ) / self.yield_total
        logger.debug(
            f"{self.name}: total neutron yield = {self.yield_total:.4e} n/s, "
            f"{self.neutron_scrape_fraction * 100:.3f} % lost to beam scraping"
        )

    def get_inputs(self):
        """Return the input parameters as a list of (name, value, units)."""
        return [
            ("T_frac", self.T_frac, "fraction"),
            ("voltage", self.voltage, "kV"),
            ("current", self.current, "mA"),
            ("pressure", self.pressure, "torr"),
            ("temperature", self.temperature, "degC"),
            ("length", self.length, "cm"),
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

    def write_hdf5(self, path, run_info=None):
        """Write the source on the output mesh to an HDF5 file at path.

        File attributes: format ("shine-neutron-source") and format_version (HDF5_FORMAT_VERSION,
        increased whenever the layout changes), name (case name), code_version (see
        get_code_version), created (UTC time) and any run_info entries (such as the input
        file). The inputs group holds every input parameter (see get_inputs) and the results
        group the key results (see get_results), one dataset each with a "units" attribute.

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
                    group.create_dataset(key, data=value)
                    group[key].attrs["units"] = units
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
        has_DD = self.T_frac < 1
        has_DT = self.T_frac > 0
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
        """Return the reactions the gas can produce: "DD" if it contains deuterium, "DT" if it
        contains tritium."""
        return [rxn for rxn, frac in (("DD", 1 - self.T_frac), ("DT", self.T_frac)) if frac > 0]

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
            (
                "beam_energy_region_end",
                self.beam_energy_grid[region_ends],
                "keV, at the end of each region",
            ),
            ("beam_energy_backstop", self.beam_energy_grid[-1], "keV"),
            ("beam_radius_backstop", self.radius_grid[-1], "cm, beam edge; capped at the wall"),
            ("beam_current_backstop", self.current_grid[-1], "mA, after scraping"),
            ("beam_power_entry", self.voltage * self.current / 1000, "kW"),
            ("beam_power_gas", self.gas_power, "kW, deposited in the gas"),
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
        get_results). Units are given in comments. title adds a first comment line saying what
        the file is. The "=" signs are aligned across all sections, and so are the comments
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
        return s

    def write_text_files(self, directory, run_info=None):
        """Write a summary and integrated neutron profiles as text files in directory.

        - summary.toml: key inputs and results (see get_summary; run_info adds entries to
          its [run] section).
        - axial_profile.txt: neutrons per output position bin, integrated over angle and
          energy, with the beam energy, radius and scraped fraction, and the beam power
          deposited in the gas and carried onto the wall in each bin.
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
                "beam power deposited in the gas over the bin [kW]",
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
                self.gas_power_mids_out,
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

        File names do not include the case name, so use a separate directory per case.
        """
        os.makedirs(directory, exist_ok=True)
        self.plot_beam(directory)
        self.plot_neutron_position(directory)
        self.plot_neutron_angle(directory)
        self.plot_neutron_energy(directory)

    def plot_beam(self, directory):
        """Plot beam energy, current and power against position."""
        xvals = np.array(self.x_grid_out)
        yvals_erg = np.array(self.beam_energy_grid_out)
        yvals_cur = np.array(self.current_grid_out)
        yvals_cur_tot = np.full(xvals.shape, self.current)
        yvals_pow = yvals_erg * yvals_cur / 1000
        yvals_pow_tot = yvals_erg * yvals_cur_tot / 1000

        plots = [
            ("energy", "Beam energy [keV]", [(yvals_erg, None)]),
            (
                "current",
                "Beam current [mA]",
                [(yvals_cur_tot, "No scraping"), (yvals_cur, "With scraping")],
            ),
            (
                "power",
                "Beam power [kW]",
                [(yvals_pow_tot, "No scraping"), (yvals_pow, "With scraping")],
            ),
        ]
        for quantity, ylabel, lines in plots:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 6)
            for yvals, label in lines:
                ax.plot(xvals, yvals, label=label)
            ax.plot(0, 0, "")  # Keep the origin in view
            ax.set_xlabel("Distance into target chamber [cm]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Beam {quantity} profile ({self.name})")
            ax.grid()
            if len(lines) > 1:
                ax.legend(loc="lower left")
            _save_figure(os.path.join(directory, f"beam_{quantity}.png"))
            plt.close()

    def plot_neutron_position(self, directory):
        """Plot the neutron source per unit length against position (linear and log)."""
        xvals = np.array(self.x_grid_out)
        xvals = np.append(xvals, xvals[-1])
        yvals_DD = np.sum(self.neutrons["DD"], (1, 2)) / self.dx_out
        yvals_DT = np.sum(self.neutrons["DT"], (1, 2)) / self.dx_out
        yvals_tot = yvals_DD + yvals_DT

        # Pad with zeros so the step outline drops to zero at both ends
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        for yvals, label in [(yvals_DD, "DD"), (yvals_DT, "DT"), (yvals_tot, "Total")]:
            ax.step(xvals, np.append(np.append(0, yvals), 0), label=label)
        ax.set_xlabel("Distance into target chamber [cm]")
        ax.set_ylabel("Linear neutron source [n/cm-s]")
        ax.set_title(f"Neutron source position distribution ({self.name})")
        ax.grid()
        ax.legend()
        _save_figure(os.path.join(directory, "neutron_position.png"))

        ax.set_yscale("log")
        _save_figure(os.path.join(directory, "neutron_position_log.png"))
        plt.close()

    def plot_neutron_angle(self, directory):
        """Polar plot of the neutron source per unit solid angle, normalised to its mean."""
        # Mirror 0-180 degrees onto 180-360 degrees for a full polar plot
        xvals = self.angle_grid_out * np.pi / 180
        xvals = np.append(xvals, 2 * np.pi - np.flip(xvals[:-1]))
        yvals_DD = np.sum(self.neutrons["DD"], (0, 2)) / self.solid_angles_out
        yvals_DT = np.sum(self.neutrons["DT"], (0, 2)) / self.solid_angles_out
        yvals = yvals_DD + yvals_DT
        yvals = np.append(yvals, np.flip(yvals))
        yvals = yvals / np.mean(yvals)
        yvals = np.append(yvals, yvals[-1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        fig.set_size_inches(6, 6)
        ax.plot(xvals, yvals)
        ax.set_rlabel_position(270)
        ax.set_title(f"Neutron source angular distribution ({self.name})")
        _save_figure(os.path.join(directory, "neutron_angle.png"))
        plt.close()

    def plot_neutron_energy(self, directory):
        """Plot the neutron energy spectrum stacked by emission angle.

        Writes one plot each for DD and DT (when present) and one for the total. The axis
        ranges follow the neutron energies, so the plots suit any beam energy.
        """
        present = self.reactions_present()
        spectra = {rxn: np.sum(self.neutrons[rxn], 0) / self.energy_bin_width for rxn in present}
        energy_grid_tot, on_common_grid = self.neutrons_by_angle_and_energy()
        energy_mids_tot = (energy_grid_tot[:-1] + energy_grid_tot[1:]) / 2
        spectrum_tot = (on_common_grid["DD"] + on_common_grid["DT"]) / self.energy_bin_width

        cvals = np.linspace(0, 0.95, self.num_angle_bins_out)
        colors = [plt.cm.nipy_spectral(x) for x in cvals]

        # case: (energy mids, spectrum, x min, x max, legend location). Each reaction's plot
        # spans its energy range plus a 10 % margin.
        cases = {}
        for rxn in present:
            lo, hi = self.neutron_energy_grid[rxn][0], self.neutron_energy_grid[rxn][-1]
            margin = 0.1 * (hi - lo)
            cases[rxn] = (
                self.neutron_energy_mids[rxn],
                spectra[rxn],
                max(0, lo - margin),
                hi + margin,
                "upper right",
            )
        cases["total"] = (energy_mids_tot, spectrum_tot, 0.0, energy_grid_tot[-1], "upper center")

        # Label about 13 of the angle bins in the legend
        labelled = np.round(np.linspace(0, self.num_angle_bins_out - 1, 13))

        for case, (neutron_energy_mids, spectrum, xmin, xmax, lloc) in cases.items():
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 6)
            ax.set_prop_cycle("color", colors)
            total = np.zeros(spectrum.shape[1])
            for i in range(self.num_angle_bins_out):
                if i in labelled:
                    label = rf"{self.angle_grid_out[i]:g}-{self.angle_grid_out[i + 1]:g}$\degree$"
                else:
                    label = ""
                ax.fill_between(neutron_energy_mids, total, total + spectrum[i, :], label=label)
                total += spectrum[i, :]
            ax.set_xlim(xmin, xmax)
            if case == "total":
                # Show five decades below the peak
                ax.set_yscale("log")
                ymax = np.log10(np.max(total))
                ymin = ymax - 5
                ax.set_ylim(10 ** (ymin - 0.25), 10 ** (ymax + 0.25))
            ax.set_xlabel("Neutron energy [MeV]")
            ax.set_ylabel("Neutron source [n/MeV-s]")
            ax.set_title(
                f"{case.capitalize() if case == 'total' else case} neutron source energy distribution ({self.name})"
            )
            ax.grid()
            ax.legend(loc=lloc)
            _save_figure(os.path.join(directory, f"neutron_energy_{case}.png"))
            plt.close()
