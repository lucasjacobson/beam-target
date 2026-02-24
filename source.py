#!/usr/bin/python3

import os
import time
import numpy as np
from itertools import product
from scipy.special import legendre

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker

try: profile
except:
    def profile(func): return func

# Physical constants
NA  = 6.02214076e23    # Avogadro constant [1/mol]
ee  = 1.602176634e-19  # Elementary charge [C]
kB  = 1.380649E-23     # Boltzmann constant [J/K]
mD  = 2.0141017778     # Mass of deuterium atom [amu]
mT  = 3.0160492777     # Mass of tritium atom [amu]
M_N = 939.56542194     # Mass-energy of neutron  (n-1)
M_D = 1875.612945      # Mass-energy of deuteron (H-2)
M_H = 2808.39161112    # Mass-energy of helion   (He-3)
M_T = 2808.92113668    # Mass-energy of triton   (H-3)
M_A = 3727.3794118     # Mass-energy of alpha    (He-4)

# Calculate reaction Q values
Q_DD = (M_D + M_D) - (M_N + M_H)  #  3.26885694 MeV
Q_DT = (M_D + M_T) - (M_N + M_A)  # 17.58924794 MeV

@profile
class SHINE_Source():

    def __init__(self, _rxn, _T_frac, _voltage, _current, _pressure, _temperature, _length, _num_x_ints, _angle_int, _EN_int):

        t0 = time.time()

        # Input parameters
        self.rxn           = str(_rxn        )  # Fusion reaction (DD or DT)
        self.T_frac      = float(_T_frac     )  # Tritium fraction
        self.voltage     = float(_voltage    )  # Beam voltage [kV]
        self.current     = float(_current    )  # Beam current [mA]
        self.pressure    = float(_pressure   )  # Target pressure [torr]
        self.temperature = float(_temperature)  # Target temperature [C]
        self.length      = float(_length     )  # Target length [cm]
        self.num_x_ints  =   int(_num_x_ints )  # Number of position intervals
        self.angle_int   = float(_angle_int  )  # Angle interval [deg]
        self.EN_int      = float(_EN_int     )  # Neutron energy interval

        # Fusion reaction kinematics
        if   self.rxn == 'DD': self.M1 = M_D; self.M2 = M_D; self.M3 = M_N; self.M4 = M_H; self.Q = Q_DD
        elif self.rxn == 'DT': self.M1 = M_D; self.M2 = M_T; self.M3 = M_N; self.M4 = M_A; self.Q = Q_DT

        # Calculated values
        self.D_frac = 1 - self.T_frac                      # Deuterium fraction
        self.dps    = self.current / 1000 / ee             # Deuteron source [1/s]
        self.P_Pa   = self.pressure * 101325 / 760         # Target pressure [Pa]
        self.T_K    = self.temperature + 273.15            # Target temperature [K]
        self.amass  = mT * self.T_frac + mD * self.D_frac  # Average atomic mass [g/mol]
        self.mdens  = self.P_Pa / (kB * self.T_K) / 1e6    # Molecule density [1/cm^3]
        self.adens  = self.mdens * 2                       # Atom density [1/cm^3]
        self.rho    = self.adens * self.amass / NA         # Mass density [g/cm^3]

        # Setup position mesh [cm]
        self.dx     = self.length / self.num_x_ints  # Interval length [cm]
        self.x_grid = np.linspace(0, self.length, self.num_x_ints + 1)
        self.x_mids = (self.x_grid[:-1] + self.x_grid[1:]) / 2

        # Setup angle mesh [degrees]
        self.num_a_ints  = int(round(180 / self.angle_int))  # Number of angle intervals
        self.angle_grid  = np.linspace(0, 180, self.num_a_ints + 1)
        self.angle_mids  = (self.angle_grid[:-1] + self.angle_grid[1:]) / 2
        self.cosine_grid = np.cos(self.angle_grid * np.pi / 180)
        self.cosine_mids = np.cos(self.angle_mids * np.pi / 180)
        self.angle_areas = (self.cosine_grid[:-1] - self.cosine_grid[1:]) * 2 * np.pi

        # Setup position mesh (condensed)
        self.num_x_ints_C = 137
        self.dx_C         = self.length / self.num_x_ints_C
        self.x_grid_C     = np.linspace(0, self.length, self.num_x_ints_C + 1)
        self.x_mids_C     = (self.x_grid_C[:-1] + self.x_grid_C[1:]) / 2

        # Setup angle mesh (condensed)
        self.angle_int_C   = 3.0
        self.num_a_ints_C  = int(round(180 / self.angle_int_C))
        self.angle_grid_C  = np.linspace(0, 180, self.num_a_ints_C + 1)
        self.angle_mids_C  = (self.angle_grid_C[:-1] + self.angle_grid_C[1:]) / 2
        self.cosine_grid_C = np.cos(self.angle_grid_C * np.pi / 180)
        self.cosine_mids_C = np.cos(self.angle_mids_C * np.pi / 180)
        self.angle_areas_C = (self.cosine_grid_C[:-1] - self.cosine_grid_C[1:]) * 2 * np.pi

        # Ratios between condensed intervals and calculation intervals
        self.x_C_ratio     = self.num_x_ints / self.num_x_ints_C
        self.angle_C_ratio = self.num_a_ints / self.num_a_ints_C
        assert(self.x_C_ratio     == int(round(self.x_C_ratio    )))
        assert(self.angle_C_ratio == int(round(self.angle_C_ratio)))
        self.x_C_ratio     = int(round(self.x_C_ratio    ))
        self.angle_C_ratio = int(round(self.angle_C_ratio))

        # Read tabular data
        self.read_stopping_power()
        self.read_cross_sections()

        # Do work
        self.calculate_energy_grid()
        self.calculate_ddxs()
        self.calculate_neutron_energies()
        self.setup_neutron_mesh()
        self.bin_neutrons()

        t1 = time.time()
        print('Ran in %.2f seconds' % (t1 - t0))

        # Make plots
        self.plot_extent()
        self.plot_angle()
        self.plot_energy()

    def read_stopping_power(self):

        # Read stopping power data from file
        self.sp_data = {}
        unit_conv = {'eV': 1e-3, 'keV': 1.0, 'MeV': 1e3}
        for rxn in ['DD', 'DT']:
            fname = os.path.join('data', 'SP_%s.txt' % (rxn))
            with open(fname, 'r') as r: lines = r.readlines()
            self.sp_data[rxn] = {'E': [], 'dEdx': []}
            in_data = False
            for line in lines:
                if line[0] == '-': in_data = not in_data; continue
                if not in_data: continue
                tokens = line.split()
                E      = float('%.5g' % (float(tokens[0]) * unit_conv[tokens[1]]))  # keV
                dEdx_e = float(tokens[2]) * 1e6  # Elec [keV-cm^2/g]
                dEdx_n = float(tokens[3]) * 1e6  # Nucl [keV-cm^2/g]
                dEdx   = dEdx_e + dEdx_n         # Total
                self.sp_data[rxn]['E'   ].append(E   )
                self.sp_data[rxn]['dEdx'].append(dEdx)
            self.sp_data[rxn]['E'   ] = np.array(self.sp_data[rxn]['E'   ])
            self.sp_data[rxn]['dEdx'] = np.array(self.sp_data[rxn]['dEdx'])

    def read_cross_sections(self):

        # Read cross section data from file
        self.xs_data = {}
        for rxn in ['DD', 'DT']:
            fname = os.path.join('data', 'xs_%s.csv' % (rxn))
            with open(fname, 'r') as r: lines = r.readlines()
            self.xs_data[rxn] = {'E': [], 'total': [], 'coeffs': []}
            E_vals = []
            totals = []
            coeffs = []
            for line in lines[1:]:
                tokens = line.split(',')
                self.xs_data[rxn]['E'     ].append(float(tokens[0]) * 1000)  # keV
                self.xs_data[rxn]['total' ].append(float(tokens[1]) / 1000)  # b/sr
                self.xs_data[rxn]['coeffs'].append([float(x) for x in tokens[2:]])
            self.xs_data[rxn]['E'     ] = np.array(self.xs_data[rxn]['E'     ])
            self.xs_data[rxn]['total' ] = np.array(self.xs_data[rxn]['total' ])
            self.xs_data[rxn]['coeffs'] = np.array(self.xs_data[rxn]['coeffs'])

    def calculate_energy_grid(self):

        # Calculate beam energy at each grid point [keV]
        self.EB_grid    = np.zeros(self.x_grid.shape)
        self.EB_grid[0] = self.voltage
        for i in range(self.num_x_ints):
            E_start = self.EB_grid[i]  # Energy at start of interval
            E_avg   = self.EB_grid[i]  # Initialize value for average energy in interval
            for _ in range(5):         # Do 5 iterations
                dEdx_DD = np.interp(E_avg, self.sp_data['DD']['E'], self.sp_data['DD']['dEdx'], left=np.nan, right=np.nan)  # DD stopping power [keV-cm^2/g]
                dEdx_DT = np.interp(E_avg, self.sp_data['DT']['E'], self.sp_data['DT']['dEdx'], left=np.nan, right=np.nan)  # DT stopping power [keV-cm^2/g]
                dEdx    = (mT * self.T_frac * dEdx_DT + mD * self.D_frac * dEdx_DD) / self.amass  # Combined stopping power [keV-cm^2/g]
                dE      = dEdx * self.rho * self.dx  # Energy loss in interval
                E_end   = E_start - dE               # Energy at end of interval
                E_avg   = (E_start + E_end) / 2      # Average energy in interval
            if np.isnan(E_end): break
            self.EB_grid[i + 1] = E_end
        self.EB_int = (self.EB_grid[:-1] + self.EB_grid[1:]) / 2

        print('End beam energy = %.6f keV' % (self.EB_grid[-1]))

    def calculate_ddxs(self):

        # Convert angles from lab frame to CMS
        term1 = self.M1 * self.M3 * self.EB_int / 1000
        term2 = self.M4 * (self.M1 + self.M2) * (self.Q + self.EB_int / 1000 * self.M2 / (self.M1 + self.M2))
        gamma = np.sqrt(term1 / term2)
        lab_mids = np.radians(self.angle_mids[None, :])
        cosine_cms_mids = np.cos(lab_mids + np.asin(gamma[:, None] * np.sin(lab_mids)))

        # Pre-calculate Legendre polynomials
        self.num_leg = 7
        legendre_ = [legendre(x) for x in range(self.num_leg)]

        # Calculate double-differential cross sections [b/sr]
        self.xs = np.zeros((self.num_x_ints, self.num_a_ints))
        for i, E in enumerate(self.EB_int):
            total  = np.interp(E, self.xs_data[self.rxn]['E'], self.xs_data[self.rxn]['total'])
            coeffs = np.zeros(self.num_leg)
            L_vals = np.zeros((self.num_a_ints, self.num_leg))
            for j in range(self.num_leg):
                coeffs[   j] = np.interp(E, self.xs_data[self.rxn]['E'], self.xs_data[self.rxn]['coeffs'][:, j])
                L_vals[:, j] = legendre_[j](cosine_cms_mids[i, :])
            factors = (1 + 2 * gamma[i] * self.cosine_mids + gamma[i]**2)**1.5 / (1 + gamma[i] * self.cosine_mids)
            self.xs[i, :] = total * np.sum(coeffs[None, :] * L_vals, 1) * factors

    def calculate_neutron_energies(self):

        # Calculate neutron energies
        self.EN = np.zeros((self.num_x_ints, self.num_a_ints))
        for i, E_D in enumerate(self.EB_int / 1000):
            term1 = self.M1 * self.M3 * E_D
            term2 = (self.M3 + self.M4) * (self.M4 * self.Q + (self.M4 - self.M1) * E_D)
            term3 = np.sqrt(term1) * self.cosine_mids
            term4 = np.sqrt(term1 * self.cosine_mids**2 + term2)
            self.EN[i, :] = ((term3 + term4) / (self.M3 + self.M4))**2

    def setup_neutron_mesh(self):

        # Setup neutron energy mesh
        EN_min           = np.floor(np.min(self.EN) / self.EN_int) * self.EN_int
        EN_max           = np.ceil (np.max(self.EN) / self.EN_int) * self.EN_int
        self.num_EN_ints = int(round((EN_max - EN_min) / self.EN_int))
        self.EN_grid     = np.linspace(EN_min, EN_max, self.num_EN_ints + 1)
        self.EN_mids     = (self.EN_grid[:-1] + self.EN_grid[1:]) / 2

    def bin_neutrons(self):

        # Bin neutrons
        self.neutrons = np.zeros((self.num_x_ints_C, self.num_a_ints_C, self.num_EN_ints))
        iC_all  = np.floor_divide(np.arange(self.num_x_ints)[:, None], self.x_C_ratio    )
        jC_all  = np.floor_divide(np.arange(self.num_a_ints)[None, :], self.angle_C_ratio)
        iC_all  = np.repeat(iC_all, self.num_a_ints, axis=1).astype(int)
        jC_all  = np.repeat(jC_all, self.num_x_ints, axis=0).astype(int)
        kC_all  = np.searchsorted(self.EN_grid, self.EN.ravel()) - 1
        areas   = np.tile(self.angle_areas, self.num_x_ints)
        vals    = self.xs.ravel() * areas
        np.add.at(self.neutrons, (iC_all.ravel(), jC_all.ravel(), kC_all), vals)

        # Convert to units of n/s
        if   self.rxn == 'DD': frac = self.D_frac
        elif self.rxn == 'DT': frac = self.T_frac
        self.neutrons *= self.adens * 1e-24 * self.dx * frac * self.dps

        # Calculate total neutron yield
        self.nyield = np.sum(self.neutrons)
        print('%s neutron yield = %.5e n/s' % (self.rxn, self.nyield))

    def plot_extent(self):

        xvals = np.array(self.x_grid_C)
        xvals = np.append(np.array(xvals), xvals[-1])
        yvals = np.sum(self.neutrons, (1, 2))
        yvals = np.append(np.append(0, yvals), 0)

        # Plot vertical (extent) profile
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        ax.step(xvals, yvals)
        ax.set_xlabel('Distance from top of target chamber [cm]')
        ax.set_ylabel('Linear neutron source [n/cm-s]')
        ax.set_title('SHINE NDAS vertical neutron source distribution (%s)' % (self.rxn))
        ax.grid()
        plt.tight_layout()
        if not os.path.exists('images'): os.makedirs('images')
        fname = os.path.join('images', 'profile_vertical_%s.png' % (self.rxn))
        print('Writing %s' % (fname))
        plt.savefig(fname)
        plt.close()

    def plot_angle(self):

        xvals = self.angle_grid_C * np.pi / 180
        xvals = np.append(xvals, 2 * np.pi - np.flip(xvals[:-1]))
        yvals = np.sum(self.neutrons, (0, 2)) / self.angle_areas_C
        yvals = np.append(yvals, np.flip(yvals))
        yvals = yvals / np.mean(yvals)
        yvals = np.append(yvals, yvals[-1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        fig.set_size_inches(6, 6)
        ax.plot(xvals, yvals)
        ax.set_rlabel_position(270)
        ax.set_title('SHINE NDAS neutron angular distribution (%s)' % (self.rxn))
        plt.tight_layout()
        if not os.path.exists('images'): os.makedirs('images')
        fname = os.path.join('images', 'profile_angular_%s.png' % (self.rxn))
        print('Writing %s' % (fname))
        plt.savefig(fname)
        plt.close()

    def plot_energy(self):

        spectrum = np.sum(self.neutrons, 0) / self.EN_int
        if   self.rxn == 'DD': xmin, xmax =  1.8,  3.4
        elif self.rxn == 'DT': xmin, xmax = 12.9, 15.5
        cvals  = np.linspace(0, 0.95, self.num_a_ints_C)
        colors = [plt.cm.nipy_spectral(x) for x in cvals]

        # Plot angular-dependent energy distribution
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        ax.set_prop_cycle('color', colors)
        total = np.zeros(spectrum.shape[1])
        for i in range(self.num_a_ints_C):
            if i in np.round(np.linspace(0, self.num_a_ints_C - 1, 13)):
                label = r'%u-%u$\degree$' % (self.angle_grid_C[i], self.angle_grid_C[i + 1])
            else: label = ''
            ax.fill_between(self.EN_mids, total, total + spectrum[i, :], label=label)
            total += spectrum[i, :]
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=0.2))
        ax.set_xlabel('Neutron energy [MeV]')
        ax.set_ylabel('Neutron source [n/MeV-s]')
        ax.set_title('SHINE NDAS angular-dependent neutron energy distribution (%s)' % (self.rxn))
        ax.grid()
        ax.legend(loc='upper right')
        plt.tight_layout()
        if not os.path.exists('images'): os.makedirs('images')
        fname = os.path.join('images', 'profile_energy_angle_%s.png' % (self.rxn))
        print('Writing %s' % (fname))
        plt.savefig(fname)
