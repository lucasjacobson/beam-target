#!/usr/bin/python3

import os
import h5py
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

def get_atom_density(pressure, temperature):

    P_Pa  = pressure * 101325 / 760  # Target pressure [Pa]
    T_K   = temperature + 273.15     # Target temperature [K]
    mdens = P_Pa / (kB * T_K) / 1e6  # Molecule density [1/cm^3]
    adens = mdens * 2                # Atom density [1/cm^3]
    return adens

@profile
class SHINE_Source():

    def __init__(self, _name, _T_frac, _voltage, _current, _adens, _length, _start_radius, _max_radius, _divergence,
                 _num_x_ints, _num_x_ints_C, _angle_int, _angle_int_C, _EN_int):

        t0 = time.time()

        # Input parameters
        self.name         =   str(_name        )
        self.T_frac       = float(_T_frac      )  # Tritium fraction
        self.voltage      = float(_voltage     )  # Beam voltage [kV]
        self.current      = float(_current     )  # Beam current [mA]
        self.adens        = float(_adens       )  # Target atom density [1/cm^3]
        self.length       = float(_length      )  # Target length [cm]
        self.start_radius = float(_start_radius)  # Beam start radius [cm]
        self.max_radius   = float(_max_radius  )  # Beam max radius [cm]
        self.divergence   = float(_divergence  )  # Beam divergence [mrad]
        self.num_x_ints   =   int(_num_x_ints  )  # Number of position intervals (internal calculation)
        self.num_x_ints_C =   int(_num_x_ints_C)  # Number of position intervals (output)
        self.angle_int    = float(_angle_int   )  # Angle interval [deg] (internal calculation)
        self.angle_int_C  = float(_angle_int_C )  # Angle interval [deg] (output)
        self.EN_int       = float(_EN_int      )  # Neutron energy interval

        # Make sure angle interval fits into 180 degrees evenly
        assert(180 / self.angle_int_C == int(180 / self.angle_int_C))

        # Calculated values
        self.D_frac = 1 - self.T_frac           # Deuterium fraction
        self.dps    = self.current / 1000 / ee  # Deuteron source [1/s]
        self.rho_DT = self.adens * mT / NA      # Mass density of pure T target [g/cm^3]

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
        self.dx_C         = self.length / self.num_x_ints_C
        self.x_grid_C     = np.linspace(0, self.length, self.num_x_ints_C + 1)
        self.x_mids_C     = (self.x_grid_C[:-1] + self.x_grid_C[1:]) / 2

        # Setup angle mesh (condensed)
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
        self.calculate_scraping()
        self.calculate_ddxs()
        self.bin_neutrons()

        # Write to HDF5
        self.write_hdf5()

        t1 = time.time()
        print('Ran in %.2f seconds' % (t1 - t0))

    def read_stopping_power(self):

        # Read DT stopping power data from file
        # Note: DD stopping power is calculated as SP_DT * (mass_T / mass_D)
        self.sp_data = {}
        unit_conv = {'eV': 1e-3, 'keV': 1.0, 'MeV': 1e3}
        for rxn in ['DT']:
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
                dEdx  = np.interp(E_avg, self.sp_data['DT']['E'], self.sp_data['DT']['dEdx'],
                                  left=np.nan, right=np.nan)  # Stopping power [keV-cm^2/g]
                dE    = dEdx * self.rho_DT * self.dx  # Energy loss in interval
                E_end = E_start - dE                  # Energy at end of interval
                E_avg = (E_start + E_end) / 2         # Average energy in interval
            if np.isnan(E_end): break
            self.EB_grid[i + 1] = E_end
        self.EB_mids = (self.EB_grid[:-1] + self.EB_grid[1:]) / 2

        # Get beam energy on condensed grid
        self.EB_grid_C = np.array(self.EB_grid[::self.x_C_ratio])

        print('  End beam energy     = %.6f keV' % (self.EB_grid[-1]))

    def calculate_scraping(self):

        # Calculate beam radius at each grid point [cm]
        self.radius_grid = self.start_radius + self.x_grid * self.divergence / 1000
        self.radius_mids = (self.radius_grid[:-1] + self.radius_grid[1:]) / 2

        # Calculate beam scrape fraction at each grid point (assume flat beam profile, this is conservative)
        self.scrape_grid = 1 - (self.max_radius / self.radius_grid)**2
        self.scrape_grid[self.scrape_grid < 0] = 0
        self.scrape_mids = (self.scrape_grid[:-1] + self.scrape_grid[1:]) / 2

        # Calculate beam current at each grid point [mA]
        self.current_grid = self.current * (1 - self.scrape_grid)

        # Get beam current on condensed grid
        self.current_grid_C = np.array(self.current_grid[::self.x_C_ratio])

    def calculate_ddxs(self):

        self.xs = {}
        self.EN = {}

        for rxn in ['DD', 'DT']:

            # Fusion reaction kinematics
            if   rxn == 'DD': M1 = M_D; M2 = M_D; M3 = M_N; M4 = M_H; Q = Q_DD
            elif rxn == 'DT': M1 = M_D; M2 = M_T; M3 = M_N; M4 = M_A; Q = Q_DT

            # Convert angles from lab frame to CMS
            term1 = M1 * M3 * self.EB_mids / 1000
            term2 = M4 * (M1 + M2) * (Q + self.EB_mids / 1000 * M2 / (M1 + M2))
            gamma = np.sqrt(term1 / term2)
            lab_mids = np.radians(self.angle_mids[None, :])
            cosine_cms_mids = np.cos(lab_mids + np.asin(gamma[:, None] * np.sin(lab_mids)))

            # Pre-calculate Legendre polynomials
            self.num_leg = 7
            legendre_ = [legendre(x) for x in range(self.num_leg)]

            # Calculate double-differential cross sections [b/sr]
            self.xs[rxn] = np.zeros((self.num_x_ints, self.num_a_ints))
            for i, E in enumerate(self.EB_mids):
                total  = np.interp(E, self.xs_data[rxn]['E'], self.xs_data[rxn]['total'])
                coeffs = np.zeros(self.num_leg)
                L_vals = np.zeros((self.num_a_ints, self.num_leg))
                for j in range(self.num_leg):
                    coeffs[   j] = np.interp(E, self.xs_data[rxn]['E'], self.xs_data[rxn]['coeffs'][:, j])
                    L_vals[:, j] = legendre_[j](cosine_cms_mids[i, :])
                factors = (1 + 2 * gamma[i] * self.cosine_mids + gamma[i]**2)**1.5 / (1 + gamma[i] * self.cosine_mids)
                self.xs[rxn][i, :] = total * np.sum(coeffs[None, :] * L_vals, 1) * factors

            # Calculate neutron energies [MeV]
            self.EN[rxn] = np.zeros((self.num_x_ints, self.num_a_ints))
            for i, E_D in enumerate(self.EB_mids / 1000):
                term1 = M1 * M3 * E_D
                term2 = (M3 + M4) * (M4 * Q + (M4 - M1) * E_D)
                term3 = np.sqrt(term1) * self.cosine_mids
                term4 = np.sqrt(term1 * self.cosine_mids**2 + term2)
                self.EN[rxn][i, :] = ((term3 + term4) / (M3 + M4))**2

    def bin_neutrons(self):

        self.num_EN_ints = {}
        self.EN_grid     = {}
        self.EN_mids     = {}
        self.neutrons    = {}
        self.nyield      = {}
        self.nscrape     = {}

        for rxn, frac in [('DD', self.D_frac), ('DT', self.T_frac)]:

            # Setup neutron energy mesh
            EN_min                = np.floor(np.min(self.EN[rxn]) / self.EN_int) * self.EN_int
            EN_max                = np.ceil (np.max(self.EN[rxn]) / self.EN_int) * self.EN_int
            self.num_EN_ints[rxn] = int(round((EN_max - EN_min) / self.EN_int))
            self.EN_grid    [rxn] = np.linspace(EN_min, EN_max, self.num_EN_ints[rxn] + 1)
            self.EN_mids    [rxn] = (self.EN_grid[rxn][:-1] + self.EN_grid[rxn][1:]) / 2

            # Bin neutrons
            self.neutrons[rxn] = np.zeros((self.num_x_ints_C, self.num_a_ints_C, self.num_EN_ints[rxn]))
            iC_all   = np.floor_divide(np.arange(self.num_x_ints)[:, None], self.x_C_ratio    )
            jC_all   = np.floor_divide(np.arange(self.num_a_ints)[None, :], self.angle_C_ratio)
            iC_all   = np.repeat(iC_all, self.num_a_ints, axis=1).astype(int).ravel()
            jC_all   = np.repeat(jC_all, self.num_x_ints, axis=0).astype(int).ravel()
            kC_all   = np.searchsorted(self.EN_grid[rxn], self.EN[rxn].ravel()) - 1
            areas    = np.tile(self.angle_areas, self.num_x_ints)
            vals_scr = (self.xs[rxn] * (1 - self.scrape_mids[:, None])).ravel() * areas
            vals_tot =  self.xs[rxn]                                   .ravel() * areas
            np.add.at(self.neutrons[rxn], (iC_all, jC_all, kC_all), vals_scr)

            # Convert to units of n/s
            self.neutrons[rxn] *= self.adens * 1e-24 * self.dx * frac * self.dps

            # Calculate total neutron yield
            self.nyield[rxn] = np.sum(self.neutrons[rxn])
            print('  %s neutron yield    = %.5e n/s' % (rxn, self.nyield[rxn]))

            # Calculate amount of beam scraping
            self.nscrape[rxn] = 1 - np.sum(vals_scr) / np.sum(vals_tot)

        self.nyield_tot  = self.nyield['DD'] + self.nyield['DT']
        self.nscrape_tot = (self.nyield['DD'] * self.nscrape['DD'] +
                            self.nyield['DT'] * self.nscrape['DT']) / self.nyield_tot
        print('  Total neutron yield = %.5e n/s' % (self.nyield_tot))
        print('  Scrape fraction     = %.5f%%'   % (self.nscrape[rxn] * 100))

    def write_hdf5(self):

        if not os.path.exists('hdf5'): os.makedirs('hdf5')
        fname = os.path.join('hdf5', '%s.h5' % (self.name))
        print('Writing %s' % (fname))

        params = ['voltage', 'current', 'adens', 'length', 'start_radius', 'max_radius', 'divergence',
                  'num_x_ints', 'num_x_ints_C', 'angle_int', 'angle_int_C', 'EN_int']
        with h5py.File(fname, 'w') as hf:
            for param in params: hf.create_dataset(param, data=getattr(self, param))
            hf.create_dataset('x_grid'      , data=self.x_grid_C      , compression='gzip')  # (138         )
            hf.create_dataset('EB_grid'     , data=self.EB_grid_C     , compression='gzip')  # (138         )
            hf.create_dataset('current_grid', data=self.current_grid_C, compression='gzip')  # (138         )
            hf.create_dataset('angle_grid'  , data=self.angle_grid_C  , compression='gzip')  # (     61     )
            hf.create_dataset('EN_grid_DD'  , data=self.EN_grid ['DD'], compression='gzip')  # (         130)
            hf.create_dataset('EN_grid_DT'  , data=self.EN_grid ['DT'], compression='gzip')  # (         242)
            hf.create_dataset('neutrons_DD' , data=self.neutrons['DD'], compression='gzip')  # (137, 60, 129)
            hf.create_dataset('neutrons_DT' , data=self.neutrons['DT'], compression='gzip')  # (137, 60, 241)

    def make_plots(self):

        if not os.path.exists('images'): os.makedirs('images')
        self.plot_beam            ()
        self.plot_neutron_position()
        self.plot_neutron_angle   ()
        self.plot_neutron_energy  ()

    def plot_beam(self):

        xvals         = np.array(self.x_grid_C      )
        yvals_erg     = np.array(self.EB_grid_C     )
        yvals_cur     = np.array(self.current_grid_C)
        yvals_cur_tot = np.full(xvals.shape, self.current)
        yvals_pow     = yvals_erg * yvals_cur     / 1000
        yvals_pow_tot = yvals_erg * yvals_cur_tot / 1000

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        ax.plot(xvals, yvals_erg)
        ax.plot(0, 0, '')
        ax.set_xlabel('Distance into target chamber [cm]')
        ax.set_ylabel('Beam energy [keV]')
        ax.set_title('SHINE beam energy profile (%s)' % (self.name))
        ax.grid()
        fname = os.path.join('images', '%s_beam_energy.png' % (self.name))
        print('Writing %s' % (fname))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        ax.plot(xvals, yvals_cur_tot, label='No scraping'  )
        ax.plot(xvals, yvals_cur    , label='With scraping')
        ax.plot(0, 0, '')
        ax.set_xlabel('Distance into target chamber [cm]')
        ax.set_ylabel('Beam current [mA]')
        ax.set_title('SHINE beam current profile (%s)' % (self.name))
        ax.grid()
        ax.legend(loc='lower left')
        fname = os.path.join('images', '%s_beam_current.png' % (self.name))
        print('Writing %s' % (fname))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        ax.plot(xvals, yvals_pow_tot, label='No scraping'  )
        ax.plot(xvals, yvals_pow    , label='With scraping')
        ax.plot(0, 0, '')
        ax.set_xlabel('Distance into target chamber [cm]')
        ax.set_ylabel('Beam power [kW]')
        ax.set_title('SHINE beam power profile (%s)' % (self.name))
        ax.grid()
        ax.legend(loc='lower left')
        fname = os.path.join('images', '%s_beam_power.png' % (self.name))
        print('Writing %s' % (fname))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def plot_neutron_position(self):

        xvals     = np.array(self.x_grid_C)
        xvals     = np.append(np.array(xvals), xvals[-1])
        yvals_DD  = np.sum(self.neutrons['DD'], (1, 2))
        yvals_DT  = np.sum(self.neutrons['DT'], (1, 2))
        yvals_tot = yvals_DD + yvals_DT
        yvals_DD  = np.append(np.append(0, yvals_DD ), 0)
        yvals_DT  = np.append(np.append(0, yvals_DT ), 0)
        yvals_tot = np.append(np.append(0, yvals_tot), 0)

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        ax.step(xvals, yvals_DD , label='DD'   )
        ax.step(xvals, yvals_DT , label='DT'   )
        ax.step(xvals, yvals_tot, label='Total')
        ax.set_xlabel('Distance into target chamber [cm]')
        ax.set_ylabel('Linear neutron source [n/cm-s]')
        ax.set_title('SHINE neutron source position distribution (%s)' % (self.name))
        ax.grid()
        ax.legend(loc='upper right')
        fname = os.path.join('images', '%s_neutron_position.png' % (self.name))
        print('Writing %s' % (fname))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def plot_neutron_angle(self):

        xvals    = self.angle_grid_C * np.pi / 180
        xvals    = np.append(xvals, 2 * np.pi - np.flip(xvals[:-1]))
        yvals_DD = np.sum(self.neutrons['DD'], (0, 2)) / self.angle_areas_C
        yvals_DT = np.sum(self.neutrons['DT'], (0, 2)) / self.angle_areas_C
        yvals    = yvals_DD + yvals_DT
        yvals    = np.append(yvals, np.flip(yvals))
        yvals    = yvals / np.mean(yvals)
        yvals    = np.append(yvals, yvals[-1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        fig.set_size_inches(6, 6)
        ax.plot(xvals, yvals)
        ax.set_rlabel_position(270)
        ax.set_title('SHINE neutron source angular distribution (%s)' % (self.name))
        fname = os.path.join('images', '%s_neutron_angle.png' % (self.name))
        print('Writing %s' % (fname))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def plot_neutron_energy(self):

        EN_min_tot      =  0.0
        EN_max_tot      = 16.0
        num_EN_ints_tot = int(round((EN_max_tot - EN_min_tot) / self.EN_int))
        EN_grid_tot     = np.linspace(EN_min_tot, EN_max_tot, num_EN_ints_tot + 1)
        EN_mids_tot     = (EN_grid_tot[:-1] + EN_grid_tot[1:]) / 2

        spectrum_DD     = np.sum(self.neutrons['DD'], 0) / self.EN_int
        spectrum_DT     = np.sum(self.neutrons['DT'], 0) / self.EN_int
        i_DD            = int(round(self.EN_grid['DD'][0] / self.EN_int))
        i_DT            = int(round(self.EN_grid['DT'][0] / self.EN_int))
        spectrum_tot    = np.zeros((self.num_a_ints_C, num_EN_ints_tot))
        spectrum_tot[:, i_DD:i_DD + spectrum_DD.shape[1]] += spectrum_DD
        spectrum_tot[:, i_DT:i_DT + spectrum_DT.shape[1]] += spectrum_DT

        cvals  = np.linspace(0, 0.95, self.num_a_ints_C)
        colors = [plt.cm.nipy_spectral(x) for x in cvals]

        for case in ['DD', 'DT', 'total']:

            if case == 'DD' and self.D_frac == 0: continue
            if case == 'DT' and self.T_frac == 0: continue

            if   case == 'DD'   : EN_mids = self.EN_mids['DD']; spectrum = spectrum_DD ; xmin =  1.8; xmax =  3.4; xint = 0.2; lloc = 'upper right'
            elif case == 'DT'   : EN_mids = self.EN_mids['DT']; spectrum = spectrum_DT ; xmin = 12.9; xmax = 15.5; xint = 0.2; lloc = 'upper right'
            elif case == 'total': EN_mids =      EN_mids_tot  ; spectrum = spectrum_tot; xmin =  0.0; xmax = 16.0; xint = 1.0; lloc = 'upper center'

            # Plot angular-dependent energy distribution
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 6)
            ax.set_prop_cycle('color', colors)
            total = np.zeros(spectrum.shape[1])
            for i in range(self.num_a_ints_C):
                if i in np.round(np.linspace(0, self.num_a_ints_C - 1, 13)):
                    label = r'%u-%u$\degree$' % (self.angle_grid_C[i], self.angle_grid_C[i + 1])
                else: label = ''
                ax.fill_between(EN_mids, total, total + spectrum[i, :], label=label)
                total += spectrum[i, :]
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(base=xint))
            if case == 'total':
                ax.set_yscale('log')
                ymax = np.log10(np.max(total))
                ymin = ymax - 5
                ax.set_ylim(10**(ymin - 0.25), 10**(ymax + 0.25))
            ax.set_xlabel('Neutron energy [MeV]')
            ax.set_ylabel('Neutron source [n/MeV-s]')
            ax.set_title('SHINE %s neutron source energy distribution (%s)' % (case, self.name))
            ax.grid()
            ax.legend(loc=lloc)
            fname = os.path.join('images', '%s_neutron_energy_%s.png' % (self.name, case))
            print('Writing %s' % (fname))
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
