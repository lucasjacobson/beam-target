#!/usr/bin/env python3
"""Beam-target fusion neutron source term for Monte Carlo transport codes.

A deuteron beam passes through one or more gas regions containing a deuterium/tritium
mixture. For each region the user gives a pressure, temperature, length and beam
divergence. The calculation:

1. Slows the beam down along its path using SRIM stopping powers.
2. Grows the beam radius with its divergence and removes the fraction of the beam that
   strikes the target wall (beam scraping).
3. Evaluates the D(d,n)3He and T(d,n)4He double-differential cross sections at each
   position from tabulated Legendre data, transformed from the centre-of-mass frame to
   the lab frame, and the neutron energy at each emission angle from two-body kinematics.
4. Tallies neutrons on a fine internal (position, angle) mesh and condenses them onto a
   coarser output (position, angle, energy) mesh.

The result can be written as an HDF5 file, as an MCNP SDEF source card, and as plots.

Naming conventions used throughout:

- ``*_grid`` arrays hold values on mesh boundaries (length n + 1) and ``*_mids`` arrays
  hold values at interval midpoints (length n).
- A ``_C`` suffix marks the condensed (output) mesh; no suffix is the fine internal mesh.
- ``*_list`` arrays have one entry per gas region; ``*_vals`` arrays have one entry per
  position interval.
- ``EB`` is beam (deuteron) energy in keV; ``EN`` is neutron energy in MeV.
"""

import os
import textwrap
import time

import h5py
import matplotlib
import numpy as np
from scipy.special import legendre

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # Must follow matplotlib.use
from matplotlib import ticker as mticker

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Physical constants
NA = 6.02214076e23  # Avogadro constant [1/mol]
ee = 1.602176634e-19  # Elementary charge [C]
kB = 1.380649e-23  # Boltzmann constant [J/K]
mD = 2.0141017778  # Mass of deuterium atom [amu]
mT = 3.0160492777  # Mass of tritium atom [amu]

# Nuclear rest-mass energies [MeV]
M_N = 939.56542194  # Neutron (n-1)
M_D = 1875.612945  # Deuteron (H-2)
M_H = 2808.39161112  # Helion (He-3)
M_T = 2808.92113668  # Triton (H-3)
M_A = 3727.3794118  # Alpha (He-4)

# Reaction Q values [MeV]
Q_DD = (M_D + M_D) - (M_N + M_H)  #  3.26885694 MeV
Q_DT = (M_D + M_T) - (M_N + M_A)  # 17.58924794 MeV

# Two-body kinematics for each reaction, 1 + 2 -> 3 + 4:
# (beam M1, target M2, neutron M3, residual M4, Q value)
REACTIONS = {
    "DD": (M_D, M_D, M_N, M_H, Q_DD),  # D(d,n)3He
    "DT": (M_D, M_T, M_N, M_A, Q_DT),  # T(d,n)4He
}

# MCNP source direction -> (AXS vector, VEC vector, reverse the position distributions).
# For "-z" the extent runs along +z and the position-dependent distributions are written
# in reverse order instead of using AXS = 0 0 -1, which ADVANTG does not handle correctly.
# As a result, for "-z" POS is the downstream (backstop) end of the source; for every other
# direction POS is the point where the beam enters the first gas region.
SDEF_DIRECTIONS = {
    "-x": ("-1 0 0", "-1 0 0", False),
    "+x": ("1 0 0", "1 0 0", False),
    "-y": ("0 -1 0", "0 -1 0", False),
    "+y": ("0 1 0", "0 1 0", False),
    "-z": ("0 0 1", "0 0 -1", True),
    "+z": ("0 0 1", "0 0 1", False),
}


def get_atom_density(pressure, temperature):
    """Return the atom density [1/cm^3] of a diatomic ideal gas.

    pressure is in torr and temperature in degrees C. Works element-wise on arrays.
    """
    P_Pa = pressure * 101325 / 760  # Pressure [Pa]
    T_K = temperature + 273.15  # Temperature [K]
    mdens = P_Pa / (kB * T_K) / 1e6  # Molecule density [1/cm^3]
    adens = mdens * 2  # Atom density [1/cm^3] (D2, DT and T2 are diatomic)
    return adens


def _is_multiple(a, b, rtol=1e-9):
    """Return True if a / b is an integer, allowing for floating-point round-off."""
    ratio = a / b
    return abs(ratio - round(ratio)) <= rtol * max(1.0, abs(ratio))


def _format_values(values, fmt, per_line, indent="\n       "):
    """Format values as an MCNP data list, wrapping after every per_line entries."""
    s = ""
    for i, x in enumerate(values):
        if i > 0 and i % per_line == 0:
            s += indent
        s += " " + format(x, fmt)
    return s


def _save_figure(fname):
    """Save the current matplotlib figure to fname."""
    plt.tight_layout()
    plt.savefig(fname)


class SHINE_Source:
    """Neutron source term for a deuteron beam in a D/T gas target.

    The whole calculation runs when the object is created; afterwards use write_hdf5(),
    write_mcnp_sdef() and make_plots() to write the results.

    Arguments with a ``_list`` name take one value per gas region, in the order the beam
    passes through them; a scalar is treated as a single region. The last region is the
    target chamber; earlier regions are normally upstream differential-pumping stages.

    Args:
        name: Case name, used in output file names and titles.
        T_frac: Atom fraction of tritium in the gas (0 = pure D, 1 = pure T).
        voltage: Beam voltage [kV]. The beam starts with this energy in keV.
        current: Beam current [mA].
        pressure_list: Gas pressure in each region [torr].
        temperature_list: Gas temperature in each region [C].
        length_list: Length of each region along the beam [cm].
        start_radius: Beam radius at the entrance of the first region [cm].
        max_radius: Radius of the target wall [cm]. Beam outside this radius is scraped.
        divergence_list: Beam divergence (growth in radius per unit length) in each
            region [mrad].
        num_x_ints_list: Number of position intervals in each region, internal mesh.
        num_x_ints_list_C: Number of position intervals in each region, output mesh.
            Must divide num_x_ints_list.
        angle_int: Emission angle interval of the internal mesh [deg].
        angle_int_C: Emission angle interval of the output mesh [deg]. Must divide 180
            and be a multiple of angle_int. 180 gives an isotropic source.
        EN_int: Neutron energy interval of the output mesh [MeV].
        data_dir: Directory holding SP_DT.txt, xs_DD.csv and xs_DT.csv. Defaults to the
            data directory next to this file.
        verbose: Print progress and summary values.
    """

    def __init__(
        self,
        name,
        T_frac,
        voltage,
        current,
        pressure_list,
        temperature_list,
        length_list,
        start_radius,
        max_radius,
        divergence_list,
        num_x_ints_list,
        num_x_ints_list_C,
        angle_int,
        angle_int_C,
        EN_int,
        data_dir=None,
        verbose=False,
    ):
        t0 = time.time()

        # Input parameters
        self.name = str(name)
        self.T_frac = float(T_frac)
        self.voltage = float(voltage)
        self.current = float(current)
        self.pressure_list = np.atleast_1d(np.array(pressure_list, dtype=float))
        self.temperature_list = np.atleast_1d(np.array(temperature_list, dtype=float))
        self.length_list = np.atleast_1d(np.array(length_list, dtype=float))
        self.start_radius = float(start_radius)
        self.max_radius = float(max_radius)
        self.divergence_list = np.atleast_1d(np.array(divergence_list, dtype=float))
        self.num_x_ints_list = np.atleast_1d(np.array(num_x_ints_list, dtype=int))
        self.num_x_ints_list_C = np.atleast_1d(np.array(num_x_ints_list_C, dtype=int))
        self.angle_int = float(angle_int)
        self.angle_int_C = float(angle_int_C)
        self.EN_int = float(EN_int)
        self.data_dir = DEFAULT_DATA_DIR if data_dir is None else str(data_dir)
        self.verbose = bool(verbose)

        self.num_regions = len(self.pressure_list)
        self.check_input()

        # Deuteron source rate [1/s]
        self.dps = self.current / 1000 / ee

        self.setup_meshes()

        # Gas properties in each region
        self.adens_list = get_atom_density(self.pressure_list, self.temperature_list)

        # Mass density [g/cm^3] the gas would have if every atom were tritium. The SRIM
        # table is for deuterons in pure tritium, so multiplying its stopping power (per
        # unit mass) by this density gives the energy loss per unit length. Electronic
        # stopping per atom is the same for D and T (one electron each), so this holds for
        # any D/T mix. It is equivalent to scaling the stopping power by m_T / m_D for
        # deuterium, and neglects the small isotope dependence of nuclear stopping.
        self.rho_T_equiv_list = self.adens_list * mT / NA

        # Region properties on the internal and output position meshes
        self.divergence_vals = np.repeat(self.divergence_list, self.num_x_ints_list)
        self.adens_vals = np.repeat(self.adens_list, self.num_x_ints_list)
        self.adens_vals_C = np.repeat(self.adens_list, self.num_x_ints_list_C)
        self.rho_T_equiv_vals = np.repeat(self.rho_T_equiv_list, self.num_x_ints_list)

        self.read_stopping_power()
        self.read_cross_sections()

        self.calculate_energy_grid()
        self.calculate_scraping()
        self.calculate_ddxs()
        self.bin_neutrons()

        if self.verbose:
            print(f"  {self.name:<16} : Finished running in {time.time() - t0:.2f} seconds")

    def check_input(self):
        """Raise ValueError if any input parameter is inconsistent or out of range."""

        def require(condition, message):
            if not condition:
                raise ValueError(f"{self.name}: {message}")

        for attr in [
            "temperature_list",
            "length_list",
            "divergence_list",
            "num_x_ints_list",
            "num_x_ints_list_C",
        ]:
            require(
                len(getattr(self, attr)) == self.num_regions,
                f"{attr} has {len(getattr(self, attr))} entries, expected {self.num_regions}",
            )
        require(0 <= self.T_frac <= 1, "T_frac must be between 0 and 1")
        require(0 < self.voltage <= 100000, "voltage must be between 0 and 100000 kV")
        require(self.current > 0, "current must be positive")
        require(np.all(self.pressure_list > 0), "pressures must be positive")
        require(np.all(self.temperature_list > -273.15), "temperatures must be above 0 K")
        require(np.all(self.length_list > 0), "lengths must be positive")
        require(self.start_radius > 0, "start_radius must be positive")
        require(self.max_radius > 0, "max_radius must be positive")
        require(np.all(self.divergence_list >= 0), "divergences must not be negative")
        require(np.all(self.num_x_ints_list > 0), "num_x_ints_list must be positive")
        require(np.all(self.num_x_ints_list_C > 0), "num_x_ints_list_C must be positive")
        require(self.angle_int > 0, "angle_int must be positive")
        require(self.angle_int_C > 0, "angle_int_C must be positive")
        require(self.EN_int > 0, "EN_int must be positive")
        require(_is_multiple(180, self.angle_int_C), "angle_int_C must divide 180")
        require(
            _is_multiple(self.angle_int_C, self.angle_int),
            "angle_int_C must be a multiple of angle_int",
        )
        require(
            np.all(self.num_x_ints_list % self.num_x_ints_list_C == 0),
            "num_x_ints_list_C must divide num_x_ints_list in every region",
        )

    def setup_meshes(self):
        """Build the internal and output meshes in position and emission angle."""

        # Angle mesh [deg], cosines, and the solid angle of each interval [sr]
        self.num_a_ints = int(round(180 / self.angle_int))
        self.num_a_ints_C = int(round(180 / self.angle_int_C))
        self.angle_C_ratio = int(round(self.num_a_ints / self.num_a_ints_C))
        self.angle_grid = np.linspace(0, 180, self.num_a_ints + 1)
        self.angle_grid_C = np.linspace(0, 180, self.num_a_ints_C + 1)
        self.angle_mids = (self.angle_grid[:-1] + self.angle_grid[1:]) / 2
        self.angle_mids_C = (self.angle_grid_C[:-1] + self.angle_grid_C[1:]) / 2
        self.cosine_grid = np.cos(self.angle_grid * np.pi / 180)
        self.cosine_grid_C = np.cos(self.angle_grid_C * np.pi / 180)
        self.cosine_mids = np.cos(self.angle_mids * np.pi / 180)
        self.cosine_mids_C = np.cos(self.angle_mids_C * np.pi / 180)
        self.angle_areas = (self.cosine_grid[:-1] - self.cosine_grid[1:]) * 2 * np.pi
        self.angle_areas_C = (self.cosine_grid_C[:-1] - self.cosine_grid_C[1:]) * 2 * np.pi

        # Position mesh [cm]. Each region is divided into equal intervals.
        self.num_x_ints_total = np.sum(self.num_x_ints_list)
        self.num_x_ints_total_C = np.sum(self.num_x_ints_list_C)
        self.x_C_ratio_list = np.array(
            np.round(self.num_x_ints_list / self.num_x_ints_list_C), dtype=int
        )
        self.dx_list = self.length_list / self.num_x_ints_list
        self.dx_list_C = self.length_list / self.num_x_ints_list_C
        self.dx_vals = np.repeat(self.dx_list, self.num_x_ints_list)
        self.dx_vals_C = np.repeat(self.dx_list_C, self.num_x_ints_list_C)
        self.x_grid = np.append(0, np.cumsum(self.dx_vals))
        self.x_grid_C = np.append(0, np.cumsum(self.dx_vals_C))
        self.x_mids = (self.x_grid[:-1] + self.x_grid[1:]) / 2
        self.x_mids_C = (self.x_grid_C[:-1] + self.x_grid_C[1:]) / 2

        # Number of internal intervals in each output interval, and the internal grid
        # index of each output grid point
        self.x_C_ratio_vals = np.repeat(self.x_C_ratio_list, self.num_x_ints_list_C)
        self.inds_C = np.append(0, np.cumsum(self.x_C_ratio_vals))

    def read_stopping_power(self):
        """Read SRIM stopping powers for deuterons in tritium gas from SP_DT.txt.

        Stores self.sp_data["DT"] with "E" [keV] and "dEdx" (electronic + nuclear)
        [keV-cm^2/g]. See the note on rho_T_equiv_list for how it is used for D/T mixes.
        """
        unit_conv = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}  # To keV
        fname = os.path.join(self.data_dir, "SP_DT.txt")
        with open(fname, "r") as r:
            lines = r.readlines()

        # The data table sits between the first two lines that start with "-"
        E_vals = []
        dEdx_vals = []
        in_data = False
        for line in lines:
            if line[0] == "-":
                in_data = not in_data
                continue
            if not in_data:
                continue
            tokens = line.split()
            E = float(f"{float(tokens[0]) * unit_conv[tokens[1]]:.5g}")  # [keV], 5 sig. figs
            dEdx_e = float(tokens[2]) * 1e6  # Electronic, MeV/(mg/cm^2) -> keV-cm^2/g
            dEdx_n = float(tokens[3]) * 1e6  # Nuclear, MeV/(mg/cm^2) -> keV-cm^2/g
            E_vals.append(E)
            dEdx_vals.append(dEdx_e + dEdx_n)
        self.sp_data = {"DT": {"E": np.array(E_vals), "dEdx": np.array(dEdx_vals)}}

    def read_cross_sections(self):
        """Read the DD and DT cross sections from xs_DD.csv and xs_DT.csv.

        Each row gives the deuteron energy [MeV], the centre-of-mass differential cross
        section at 0 degrees [mb/sr] and Legendre coefficients C1, C2, ... of the angular
        distribution, normalised so that sum(C_l) = 1:

            dsigma/dOmega_cm(E, mu) = Total(E) * sum_l C_(l+1)(E) * P_l(mu)

        Stores self.xs_data[rxn] with "E" [keV], "total" [b/sr] and "coeffs".
        """
        self.xs_data = {}
        for rxn in REACTIONS:
            fname = os.path.join(self.data_dir, f"xs_{rxn}.csv")
            data = np.loadtxt(fname, delimiter=",", skiprows=1, ndmin=2)
            self.xs_data[rxn] = {
                "E": data[:, 0] * 1000,  # MeV -> keV
                "total": data[:, 1] / 1000,  # mb/sr -> b/sr
                "coeffs": data[:, 2:],
            }

    def calculate_energy_grid(self):
        """Slow the beam down through the gas.

        Sets self.EB_grid, the beam energy at each internal grid point [keV], and its
        midpoint and output-mesh versions. The energy loss in each interval uses the
        stopping power at the interval's average energy, found by fixed-point iteration.
        If the beam stops or leaves the stopping-power table, the remaining energies stay
        at zero.
        """
        self.EB_grid = np.zeros(self.x_grid.shape)
        self.EB_grid[0] = self.voltage
        for i in range(self.num_x_ints_total):
            E_start = self.EB_grid[i]
            E_avg = self.EB_grid[i]
            for _ in range(5):
                dEdx = np.interp(
                    E_avg,
                    self.sp_data["DT"]["E"],
                    self.sp_data["DT"]["dEdx"],
                    left=np.nan,
                    right=np.nan,
                )  # [keV-cm^2/g]
                dE = dEdx * self.rho_T_equiv_vals[i] * self.dx_vals[i]
                E_end = E_start - dE
                E_avg = (E_start + E_end) / 2
            if np.isnan(E_end) or E_end < 0:
                break
            self.EB_grid[i + 1] = E_end
        self.EB_mids = (self.EB_grid[:-1] + self.EB_grid[1:]) / 2
        self.EB_grid_C = np.array(self.EB_grid[self.inds_C])

        if self.verbose:
            print(f"  {self.name:<16} : End beam energy     = {self.EB_grid[-1]:.9f} keV")

    def calculate_scraping(self, use_atan=False):
        """Grow the beam radius with divergence and work out how much is scraped.

        The beam profile is assumed flat, so the fraction scraped at radius r > max_radius
        is 1 - (max_radius / r)^2. This is conservative. Divergence is used as
        dr/dx = divergence / 1000 (small-angle approximation) unless use_atan is True.

        Sets the beam radius [cm], scrape fraction and current [mA] on the internal and
        output meshes. Radii are capped at max_radius after the scrape fraction is found.
        """
        if use_atan:
            divergence_vals = np.arctan(self.divergence_vals / 1000)
        else:
            divergence_vals = self.divergence_vals / 1000
        self.radius_grid = self.start_radius + np.append(
            0, np.cumsum(divergence_vals * self.dx_vals)
        )

        self.scrape_grid = 1 - (self.max_radius / self.radius_grid) ** 2
        self.scrape_grid[self.scrape_grid < 0] = 0
        self.scrape_mids = (self.scrape_grid[:-1] + self.scrape_grid[1:]) / 2

        self.radius_grid[self.radius_grid > self.max_radius] = self.max_radius
        rsq_mids = (self.radius_grid[:-1] ** 2 + self.radius_grid[1:] ** 2) / 2
        self.radius_mids = np.sqrt(rsq_mids)  # RMS radius in each interval

        self.current_grid = self.current * (1 - self.scrape_grid)
        self.current_mids = (self.current_grid[:-1] + self.current_grid[1:]) / 2

        # Output mesh: grid values are sampled, midpoint values are averaged
        def condense(vals):
            return np.add.reduceat(vals, self.inds_C[:-1]) / self.x_C_ratio_vals

        self.radius_grid_C = np.array(self.radius_grid[self.inds_C])
        self.scrape_grid_C = np.array(self.scrape_grid[self.inds_C])
        self.current_grid_C = np.array(self.current_grid[self.inds_C])
        self.radius_mids_C = np.sqrt(condense(rsq_mids))
        self.scrape_mids_C = condense(self.scrape_mids)
        self.current_mids_C = condense(self.current_mids)

    def calculate_ddxs(self):
        """Calculate lab-frame cross sections and neutron energies.

        For each reaction, sets self.xs[rxn], the double-differential cross section in the
        lab frame [b/sr], and self.EN[rxn], the neutron energy [MeV], on the internal
        (position, angle) mesh. Both use the beam energy at the interval midpoint.
        """
        self.xs = {}
        self.EN = {}

        for rxn, (M1, M2, M3, M4, Q) in REACTIONS.items():
            # gamma = (neutron speed of the CM frame) / (neutron speed in the CM frame).
            # Converts lab angles to CM angles: theta_cm = theta_lab + asin(gamma sin theta_lab)
            term1 = M1 * M3 * self.EB_mids / 1000
            term2 = M4 * (M1 + M2) * (Q + self.EB_mids / 1000 * M2 / (M1 + M2))
            gamma = np.sqrt(term1 / term2)
            lab_mids = np.radians(self.angle_mids[None, :])
            cosine_cms_mids = np.cos(lab_mids + np.arcsin(gamma[:, None] * np.sin(lab_mids)))

            # Legendre polynomials for every coefficient in the data file
            num_leg = self.xs_data[rxn]["coeffs"].shape[1]
            legendre_ = [legendre(x) for x in range(num_leg)]

            # Double-differential cross sections [b/sr]: the CM angular distribution times
            # the CM-to-lab solid angle factor
            self.xs[rxn] = np.zeros((self.num_x_ints_total, self.num_a_ints))
            for i, E in enumerate(self.EB_mids):
                total = np.interp(
                    E, self.xs_data[rxn]["E"], self.xs_data[rxn]["total"], left=0, right=np.nan
                )
                coeffs = np.zeros(num_leg)
                L_vals = np.zeros((self.num_a_ints, num_leg))
                for j in range(num_leg):
                    coeffs[j] = np.interp(
                        E,
                        self.xs_data[rxn]["E"],
                        self.xs_data[rxn]["coeffs"][:, j],
                        left=0,
                        right=np.nan,
                    )
                    L_vals[:, j] = legendre_[j](cosine_cms_mids[i, :])
                # Jacobian dOmega_cm / dOmega_lab, evaluated at the CM angle
                mu_cm = cosine_cms_mids[i, :]
                factors = (1 + 2 * gamma[i] * mu_cm + gamma[i] ** 2) ** 1.5 / (1 + gamma[i] * mu_cm)
                self.xs[rxn][i, :] = total * np.sum(coeffs[None, :] * L_vals, 1) * factors

            # Neutron energy from non-relativistic two-body kinematics [MeV]
            self.EN[rxn] = np.zeros((self.num_x_ints_total, self.num_a_ints))
            for i, E_D in enumerate(self.EB_mids / 1000):
                term1 = M1 * M3 * E_D
                term2 = (M3 + M4) * (M4 * Q + (M4 - M1) * E_D)
                term3 = np.sqrt(term1) * self.cosine_mids
                term4 = np.sqrt(term1 * self.cosine_mids**2 + term2)
                self.EN[rxn][i, :] = ((term3 + term4) / (M3 + M4)) ** 2

    def bin_neutrons(self):
        """Tally neutrons onto the output (position, angle, energy) mesh.

        The neutron rate from each internal (position, angle) cell is
        sigma [b/sr] * 1e-24 [cm^2/b] * n_target [1/cm^3] * dx [cm] * dOmega [sr]
        * deuteron rate [1/s], reduced by the scrape fraction. It is added to the output
        cell containing that position, angle and neutron energy.

        Sets, for each reaction, self.neutrons[rxn] [n/s] with shape
        (num_x_ints_total_C, num_a_ints_C, num_EN_ints[rxn]), its energy mesh
        self.EN_grid[rxn] [MeV], the yield self.nyield[rxn] [n/s] and the fraction of
        neutrons lost to scraping self.nscrape[rxn]. Totals are in self.nyield_tot and
        self.nscrape_tot.
        """
        self.num_EN_ints = {}
        self.EN_grid = {}
        self.EN_mids = {}
        self.neutrons = {}
        self.nyield = {}
        self.nscrape = {}

        for rxn, frac in [("DD", 1 - self.T_frac), ("DT", self.T_frac)]:
            # Neutron energy mesh: spans this reaction's energies, aligned to EN_int
            EN_min = np.floor(np.min(self.EN[rxn]) / self.EN_int) * self.EN_int
            EN_max = np.ceil(np.max(self.EN[rxn]) / self.EN_int) * self.EN_int
            self.num_EN_ints[rxn] = int(round((EN_max - EN_min) / self.EN_int))
            self.EN_grid[rxn] = np.linspace(EN_min, EN_max, self.num_EN_ints[rxn] + 1)
            self.EN_mids[rxn] = (self.EN_grid[rxn][:-1] + self.EN_grid[rxn][1:]) / 2

            # Output-mesh indices (iC, jC, kC) of every internal (position, angle) cell
            offsets = np.cumsum(np.concatenate(([0], self.num_x_ints_list_C[:-1])))
            iC_all = np.concatenate(
                [
                    np.floor_divide(np.arange(num_x_ints)[:, None], x_C_ratio) + offset
                    for num_x_ints, x_C_ratio, offset in zip(
                        self.num_x_ints_list, self.x_C_ratio_list, offsets
                    )
                ],
                axis=0,
            )
            jC_all = np.floor_divide(np.arange(self.num_a_ints)[None, :], self.angle_C_ratio)
            iC_all = np.repeat(iC_all, self.num_a_ints, axis=1).astype(int).ravel()
            jC_all = np.repeat(jC_all, self.num_x_ints_total, axis=0).astype(int).ravel()
            kC_all = np.searchsorted(self.EN_grid[rxn], self.EN[rxn].ravel()) - 1

            # Neutron rate in each internal cell [n/s], with and without scraping
            areas = np.tile(self.angle_areas, self.num_x_ints_total)
            factor = (self.adens_vals * 1e-24 * self.dx_vals * frac * self.dps)[:, None]
            vals_scr = (self.xs[rxn] * factor * (1 - self.scrape_mids[:, None])).ravel() * areas
            vals_tot = (self.xs[rxn] * factor).ravel() * areas

            self.neutrons[rxn] = np.zeros(
                (self.num_x_ints_total_C, self.num_a_ints_C, self.num_EN_ints[rxn])
            )
            np.add.at(self.neutrons[rxn], (iC_all, jC_all, kC_all), vals_scr)

            self.nyield[rxn] = np.sum(self.neutrons[rxn])
            if self.verbose:
                print(f"  {self.name:<16} : {rxn} neutron yield    = {self.nyield[rxn]:.5e} n/s")

            if np.sum(vals_tot) == 0:
                self.nscrape[rxn] = 0.0
            else:
                self.nscrape[rxn] = 1 - np.sum(vals_scr) / np.sum(vals_tot)

        self.nyield_tot = self.nyield["DD"] + self.nyield["DT"]
        self.nscrape_tot = (
            self.nyield["DD"] * self.nscrape["DD"] + self.nyield["DT"] * self.nscrape["DT"]
        ) / self.nyield_tot
        if self.verbose:
            print(f"  {self.name:<16} : Total neutron yield = {self.nyield_tot:.5e} n/s")
            print(f"  {self.name:<16} : Scrape fraction     = {self.nscrape_tot * 100:.5f}%")

    def write_hdf5(self, output_dir="hdf5"):
        """Write the source on the output mesh to <output_dir>/<name>.h5.

        Datasets (each has a "units" attribute):

        - x_grid, EB_grid, radius_grid, scrape_grid, current_grid: position mesh and the
          beam energy, radius, scrape fraction and current at each position grid point.
        - angle_grid: emission angle mesh, relative to the beam direction.
        - EN_grid_DD, EN_grid_DT: neutron energy mesh for each reaction.
        - neutrons_DD, neutrons_DT: neutron emission rate in each
          (position, angle, energy) cell, with shape
          (len(x_grid) - 1, len(angle_grid) - 1, len(EN_grid_<rxn>) - 1).
        """
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{self.name}.h5")
        datasets = [
            ("x_grid", self.x_grid_C, "cm"),
            ("EB_grid", self.EB_grid_C, "keV"),
            ("radius_grid", self.radius_grid_C, "cm"),
            ("scrape_grid", self.scrape_grid_C, "fraction"),
            ("current_grid", self.current_grid_C, "mA"),
            ("angle_grid", self.angle_grid_C, "deg"),
            ("EN_grid_DD", self.EN_grid["DD"], "MeV"),
            ("EN_grid_DT", self.EN_grid["DT"], "MeV"),
            ("neutrons_DD", self.neutrons["DD"], "n/s"),
            ("neutrons_DT", self.neutrons["DT"], "n/s"),
        ]
        with h5py.File(fname, "w") as hf:
            for key, data, units in datasets:
                hf.create_dataset(key, data=data, compression="gzip")
                hf[key].attrs["units"] = units

    def get_mcnp_sdef(self, direction="-z"):
        """Return an MCNP SDEF source definition as a string.

        The source is a cylinder along the beam axis, starting at POS = 0 0 0 (see
        SDEF_DIRECTIONS for the -z case). Distributions:

        - d1: position along the axis (EXT).
        - d2: emission direction relative to VEC (omitted for an isotropic source).
        - d3: radius, dependent on position (via distributions 101, 102, ...) when the
          beam diverges.
        - d4: neutron energy, dependent on emission angle (via distributions 501, 502,
          ...) unless the source is isotropic.

        Args:
            direction: Beam direction, one of SDEF_DIRECTIONS ("-x", "+x", ..., "+z").
        """
        if direction not in SDEF_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {', '.join(SDEF_DIRECTIONS)}, got {direction!r}"
            )
        axs, vec, reverse_x = SDEF_DIRECTIONS[direction]
        divergent = not np.all(self.divergence_list == 0)
        isotropic = self.angle_int_C == 180
        has_DD = self.T_frac < 1
        has_DT = self.T_frac > 0
        dashed_line = "c " + "-" * 78 + "\n"

        s = self._sdef_header(dashed_line)

        # SDEF card
        s += "c\n"
        s += "  sdef  par = n\n"
        s += "        pos = 0 0 0\n"
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
        nvals = self.num_x_ints_list_C
        Lvals = self.length_list
        if reverse_x:
            ext = ext[::-1]
            nvals = nvals[::-1]
            Lvals = Lvals[::-1]
        si1 = "0"
        x = 0.0
        for n, L in zip(nvals, Lvals):
            x += L
            if n == 2:
                si1 += " i"
            elif n > 2:
                si1 += f" {n - 1}i"
            si1 += f" {x}"
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
            s += "  si2  " + _format_values(self.cosine_grid_C[::-1], "11.8f", 6) + "\n"
            s += "  sp2  " + _format_values(np.append(0, dir_vals[::-1]), "11.5e", 6) + "\n"
            s += "c\n"
            s += dashed_line

        # Radial distribution: uniform in area out to the RMS beam radius of each interval
        s += "c\n"
        s += "  sc3   Radial distribution\n"
        if divergent:
            rmids = self.radius_mids_C[::-1] if reverse_x else self.radius_mids_C
            smids = self.scrape_mids_C[::-1] if reverse_x else self.scrape_mids_C
            s += "  ds3 S" + _format_values(101 + np.arange(len(rmids)), "3d", 18) + "\n"
            s += "c\n"
            for i, (r, scraped) in enumerate(zip(rmids, smids)):
                s += f"  si{101 + i:<3d} 0 {r:10.8f}"
                if scraped > 0:
                    s += f"  $ ({scraped * 100:8.5f}% scraped)"
                s += "\n"
        else:
            s += f"  si3   0 {self.start_radius}\n"
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
            sids = 501 + np.arange(self.num_a_ints_C)
            s += "  ds4 S" + _format_values(sids, "3d", 18) + "\n"
            s += "c\n"
            for i, sid in enumerate(sids):
                # Distribution i covers angle bin i counted from 180 degrees, matching si2
                spectra = [np.sum(self.neutrons[rxn][:, -i - 1, :], 0) for rxn in ["DD", "DT"]]
                si, sp = self._sdef_energy_bins(spectra, has_DD, has_DT, trim=True)
                s += (
                    f"  sc{sid:<3d} Energy distribution "
                    f"({self.angle_grid_C[i]:.12g} - {self.angle_grid_C[i + 1]:.12g} degrees)\n"
                )
                s += f"  si{sid:<3d}{si}\n"
                s += f"  sp{sid:<3d}" + _format_values(sp, ".5e", 6) + "\n"
                s += "c\n"
        s += dashed_line

        return s

    def _sdef_header(self, dashed_line):
        """Return the comment block describing the case at the top of the SDEF card."""
        s = dashed_line
        s += "c     Source definition\n"
        s += dashed_line
        s += "c\n"
        # The target chamber is the last region
        s += f"c     Tritium fraction   : {self.T_frac * 100:.12g}%\n"
        s += f"c     Beam voltage       : {self.voltage:.12g} kV\n"
        s += f"c     Beam current       : {self.current:.12g} mA\n"
        s += f"c     Target pressure    : {self.pressure_list[-1]:.12g} torr\n"
        s += f"c     Target temperature : {self.temperature_list[-1]:.12g} C\n"
        s += f"c     Target length      : {self.length_list[-1]:.12g} cm\n"
        s += f"c     Beam start_radius  : {self.start_radius:.12g} cm\n"
        s += f"c     Beam max_radius    : {self.max_radius:.12g} cm\n"
        s += f"c     Beam divergence    : {self.divergence_list[-1]:.12g} mrad\n"
        s += "c\n"
        s += dashed_line
        s += "c\n"
        row = "c     {:>2} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n"
        s += row.format("#", "Pressure", "Temp", "Length", "Diverge", "Ints", "Ints", "E_Final")
        s += row.format("", "[torr]", "[C]", "[cm]", "[mrad]", "output", "internal", "[keV]")
        for i in range(self.num_regions):
            E_final = self.EB_grid[np.sum(self.num_x_ints_list[: i + 1])]
            s += (
                f"c     {i + 1:2d} {self.pressure_list[i]:8.7g} {self.temperature_list[i]:8.7g}"
                f" {self.length_list[i]:8.7g} {self.divergence_list[i]:8.7g}"
                f" {self.num_x_ints_list_C[i]:8d} {self.num_x_ints_list[i]:8d} {E_final:8.7g}\n"
            )
        s += "c\n"
        s += dashed_line
        s += "c\n"
        s += f"c     # of distance intervals                 : {self.num_x_ints_total_C:.12g}\n"
        s += f"c     # of distance intervals (internal calc) : {self.num_x_ints_total:.12g}\n"
        s += f"c     Angle interval                          : {self.angle_int_C:.12g} deg\n"
        s += f"c     Angle interval (internal calc)          : {self.angle_int:.12g} deg\n"
        s += f"c     Neutron energy interval                 : {self.EN_int:.12g} MeV\n"
        s += "c\n"
        s += dashed_line
        s += "c\n"
        for rxn in ["DD", "DT"]:
            pct = self.nyield[rxn] / self.nyield_tot * 100
            s += f"c     {rxn}    neutron yield : {self.nyield[rxn]:.5e} n/s ({pct:9.5f}%)\n"
        s += f"c     Total neutron yield : {self.nyield_tot:.5e} n/s\n"
        s += "c\n"
        s += f"c     Beam energy at backstop : {self.EB_grid[-1]:.6f} keV\n"
        s += "c\n"
        s += f"c     Fraction of neutrons lost to scraping : {self.nscrape_tot * 100:.5f}%\n"
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
            E_min = round(self.EN_grid[rxn][ind_min], 12)
            E_max = round(self.EN_grid[rxn][ind_max], 12)
            si += f" {E_min} {ind_max - ind_min - 1}i {E_max}"
            sp = np.append(sp, np.append(0, spectrum[ind_min:ind_max]))
        return si, sp

    def write_mcnp_sdef(self, direction="-z", output_dir="mcnp_sdef"):
        """Write the MCNP SDEF card to <output_dir>/<name>_source.txt."""
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{self.name}_source.txt")
        with open(fname, "w") as w:
            w.write(self.get_mcnp_sdef(direction))

    def make_plots(self, output_dir="images"):
        """Write plots of the beam and neutron source to output_dir as PNG files."""
        os.makedirs(output_dir, exist_ok=True)
        self.plot_beam(output_dir)
        self.plot_neutron_position(output_dir)
        self.plot_neutron_angle(output_dir)
        self.plot_neutron_energy(output_dir)

    def plot_beam(self, output_dir="images"):
        """Plot beam energy, current and power against position."""
        xvals = np.array(self.x_grid_C)
        yvals_erg = np.array(self.EB_grid_C)
        yvals_cur = np.array(self.current_grid_C)
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
            ax.set_title(f"SHINE beam {quantity} profile ({self.name})")
            ax.grid()
            if len(lines) > 1:
                ax.legend(loc="lower left")
            _save_figure(os.path.join(output_dir, f"{self.name}_beam_{quantity}.png"))
            plt.close()

    def plot_neutron_position(self, output_dir="images"):
        """Plot the neutron source per unit length against position (linear and log)."""
        xvals = np.array(self.x_grid_C)
        xvals = np.append(xvals, xvals[-1])
        yvals_DD = np.sum(self.neutrons["DD"], (1, 2)) / self.dx_vals_C
        yvals_DT = np.sum(self.neutrons["DT"], (1, 2)) / self.dx_vals_C
        yvals_tot = yvals_DD + yvals_DT

        # Pad with zeros so the step outline drops to zero at both ends
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        for yvals, label in [(yvals_DD, "DD"), (yvals_DT, "DT"), (yvals_tot, "Total")]:
            ax.step(xvals, np.append(np.append(0, yvals), 0), label=label)
        ax.set_xlabel("Distance into target chamber [cm]")
        ax.set_ylabel("Linear neutron source [n/cm-s]")
        ax.set_title(f"SHINE neutron source position distribution ({self.name})")
        ax.grid()
        ax.legend()
        _save_figure(os.path.join(output_dir, f"{self.name}_neutron_position.png"))

        ax.set_yscale("log")
        _save_figure(os.path.join(output_dir, f"{self.name}_neutron_position_log.png"))
        plt.close()

    def plot_neutron_angle(self, output_dir="images"):
        """Polar plot of the neutron source per unit solid angle, normalised to its mean."""
        # Mirror 0-180 degrees onto 180-360 degrees for a full polar plot
        xvals = self.angle_grid_C * np.pi / 180
        xvals = np.append(xvals, 2 * np.pi - np.flip(xvals[:-1]))
        yvals_DD = np.sum(self.neutrons["DD"], (0, 2)) / self.angle_areas_C
        yvals_DT = np.sum(self.neutrons["DT"], (0, 2)) / self.angle_areas_C
        yvals = yvals_DD + yvals_DT
        yvals = np.append(yvals, np.flip(yvals))
        yvals = yvals / np.mean(yvals)
        yvals = np.append(yvals, yvals[-1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        fig.set_size_inches(6, 6)
        ax.plot(xvals, yvals)
        ax.set_rlabel_position(270)
        ax.set_title(f"SHINE neutron source angular distribution ({self.name})")
        _save_figure(os.path.join(output_dir, f"{self.name}_neutron_angle.png"))
        plt.close()

    def plot_neutron_energy(self, output_dir="images"):
        """Plot the neutron energy spectrum stacked by emission angle.

        Writes one plot each for DD and DT (when present) and one for the total. The axis
        ranges are fixed to suit beam energies of a few hundred keV.
        """
        # Common energy mesh for the total spectrum
        EN_min_tot = 0.0
        EN_max_tot = 16.0
        num_EN_ints_tot = int(round((EN_max_tot - EN_min_tot) / self.EN_int))
        EN_grid_tot = np.linspace(EN_min_tot, EN_max_tot, num_EN_ints_tot + 1)
        EN_mids_tot = (EN_grid_tot[:-1] + EN_grid_tot[1:]) / 2

        spectrum_DD = np.sum(self.neutrons["DD"], 0) / self.EN_int
        spectrum_DT = np.sum(self.neutrons["DT"], 0) / self.EN_int
        i_DD = int(round(self.EN_grid["DD"][0] / self.EN_int))
        i_DT = int(round(self.EN_grid["DT"][0] / self.EN_int))
        spectrum_tot = np.zeros((self.num_a_ints_C, num_EN_ints_tot))
        spectrum_tot[:, i_DD : i_DD + spectrum_DD.shape[1]] += spectrum_DD
        spectrum_tot[:, i_DT : i_DT + spectrum_DT.shape[1]] += spectrum_DT

        cvals = np.linspace(0, 0.95, self.num_a_ints_C)
        colors = [plt.cm.nipy_spectral(x) for x in cvals]

        # case: (energy mids, spectrum, x min, x max, x tick interval, legend location)
        cases = {
            "DD": (self.EN_mids["DD"], spectrum_DD, 1.8, 3.4, 0.2, "upper right"),
            "DT": (self.EN_mids["DT"], spectrum_DT, 12.9, 15.5, 0.2, "upper right"),
            "total": (EN_mids_tot, spectrum_tot, 0.0, 16.0, 1.0, "upper center"),
        }
        if self.T_frac == 1:
            del cases["DD"]
        if self.T_frac == 0:
            del cases["DT"]

        # Label about 13 of the angle bins in the legend
        labelled = np.round(np.linspace(0, self.num_a_ints_C - 1, 13))

        for case, (EN_mids, spectrum, xmin, xmax, xint, lloc) in cases.items():
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 6)
            ax.set_prop_cycle("color", colors)
            total = np.zeros(spectrum.shape[1])
            for i in range(self.num_a_ints_C):
                if i in labelled:
                    label = rf"{self.angle_grid_C[i]:g}-{self.angle_grid_C[i + 1]:g}$\degree$"
                else:
                    label = ""
                ax.fill_between(EN_mids, total, total + spectrum[i, :], label=label)
                total += spectrum[i, :]
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(base=xint))
            if case == "total":
                # Show five decades below the peak
                ax.set_yscale("log")
                ymax = np.log10(np.max(total))
                ymin = ymax - 5
                ax.set_ylim(10 ** (ymin - 0.25), 10 ** (ymax + 0.25))
            ax.set_xlabel("Neutron energy [MeV]")
            ax.set_ylabel("Neutron source [n/MeV-s]")
            ax.set_title(f"SHINE {case} neutron source energy distribution ({self.name})")
            ax.grid()
            ax.legend(loc=lloc)
            _save_figure(os.path.join(output_dir, f"{self.name}_neutron_energy_{case}.png"))
            plt.close()
