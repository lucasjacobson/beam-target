#!/usr/bin/python3

import os
import h5py
import time
import textwrap
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
    adens = mdens * 2                # Atom density [1/cm^3] (the factor of 2 is because the molecules are diatomic)
    return adens

@profile
class SHINE_Source():

    def __init__(self, _name, _T_frac, _voltage, _current, _pressure_list, _temperature_list, _length_list,
                 _start_radius, _max_radius, _divergence_list, _num_x_ints_list, _num_x_ints_list_C, _angle_int,
                 _angle_int_C, _EN_int, _data_dir=None, _verbose=True):

        t0 = time.time()

        pressure_list     = np.atleast_1d(np.array(_pressure_list    , dtype=float))
        temperature_list  = np.atleast_1d(np.array(_temperature_list , dtype=float))
        length_list       = np.atleast_1d(np.array(_length_list      , dtype=float))
        divergence_list   = np.atleast_1d(np.array(_divergence_list  , dtype=float))
        num_x_ints_list   = np.atleast_1d(np.array(_num_x_ints_list  , dtype=int  ))
        num_x_ints_list_C = np.atleast_1d(np.array(_num_x_ints_list_C, dtype=int  ))

        # Input parameters
        self.name              =   str(_name   )
        self.T_frac            = float(_T_frac )       # Tritium fraction
        self.voltage           = float(_voltage)       # Beam voltage [kV]
        self.current           = float(_current)       # Beam current [mA]
        self.pressure_list     = pressure_list         # Target pressure [torr]
        self.temperature_list  = temperature_list      # Target temperature [C]
        self.length_list       = length_list           # Target length [cm]
        self.start_radius      = float(_start_radius)  # Beam start radius [cm]
        self.max_radius        = float(_max_radius  )  # Beam max radius [cm]
        self.divergence_list   = divergence_list       # Beam divergence [mrad]
        self.num_x_ints_list   = num_x_ints_list       # Number of position intervals (internal calculation)
        self.num_x_ints_list_C = num_x_ints_list_C     # Number of position intervals (output)
        self.angle_int         = float(_angle_int  )   # Angle interval [deg] (internal calculation)
        self.angle_int_C       = float(_angle_int_C)   # Angle interval [deg] (output)
        self.EN_int            = float(_EN_int     )   # Neutron energy interval
        self.verbose           =  bool(_verbose    )

        # Data directory
        if _data_dir is None: self.data_dir = 'data'
        else:                 self.data_dir = str(_data_dir)

        # Number of material regions
        self.num_regions = len(self.pressure_list)

        # Check input parameters for errors
        self.check_input()

        # Deuteron source [1/s]
        self.dps = self.current / 1000 / ee

        # Number of angle intervals
        self.num_a_ints   = int(round(180 / self.angle_int  ))
        self.num_a_ints_C = int(round(180 / self.angle_int_C))

        # Ratio between condensed and calculation angle intervals
        self.angle_C_ratio = int(round(self.num_a_ints / self.num_a_ints_C))

        # Setup angle mesh
        self.angle_grid    = np.linspace(0, 180, self.num_a_ints   + 1)
        self.angle_grid_C  = np.linspace(0, 180, self.num_a_ints_C + 1)
        self.angle_mids    = (self.angle_grid  [:-1] + self.angle_grid  [1:]) / 2
        self.angle_mids_C  = (self.angle_grid_C[:-1] + self.angle_grid_C[1:]) / 2
        self.cosine_grid   = np.cos(self.angle_grid   * np.pi / 180)
        self.cosine_grid_C = np.cos(self.angle_grid_C * np.pi / 180)
        self.cosine_mids   = np.cos(self.angle_mids   * np.pi / 180)
        self.cosine_mids_C = np.cos(self.angle_mids_C * np.pi / 180)
        self.angle_areas   = (self.cosine_grid  [:-1] - self.cosine_grid  [1:]) * 2 * np.pi
        self.angle_areas_C = (self.cosine_grid_C[:-1] - self.cosine_grid_C[1:]) * 2 * np.pi

        # Total number of position intervals in all material regions
        self.num_x_ints_total   = np.sum(self.num_x_ints_list  )
        self.num_x_ints_total_C = np.sum(self.num_x_ints_list_C)

        # Ratio between condensed and calculation position intervals in each material region
        self.x_C_ratio_list = np.array(np.round(self.num_x_ints_list / self.num_x_ints_list_C), dtype=int)

        # Interval length in each material region [cm]
        self.dx_list   = self.length_list / self.num_x_ints_list
        self.dx_list_C = self.length_list / self.num_x_ints_list_C

        # Setup position mesh [cm]
        self.dx_vals   = np.repeat(self.dx_list  , self.num_x_ints_list  )
        self.dx_vals_C = np.repeat(self.dx_list_C, self.num_x_ints_list_C)
        self.x_grid    = np.append(0, np.cumsum(self.dx_vals  ))
        self.x_grid_C  = np.append(0, np.cumsum(self.dx_vals_C))
        self.x_mids    = (self.x_grid  [:-1] + self.x_grid  [1:]) / 2
        self.x_mids_C  = (self.x_grid_C[:-1] + self.x_grid_C[1:]) / 2

        # Arrays to help with condensing data into the condensed position mesh
        self.x_C_ratio_vals = np.repeat(self.x_C_ratio_list, self.num_x_ints_list_C)
        self.inds_C         = np.append(0, np.cumsum(self.x_C_ratio_vals))

        # Calculate target atom density in each material regiono [1/cm^3]
        self.adens_list = get_atom_density(self.pressure_list, self.temperature_list)

        # Calculate mass density of pure T target in each material region [g/cm^3]
        self.rho_DT_list = self.adens_list * mT / NA

        # Create some fine grids
        self.divergence_vals = np.repeat(self.divergence_list, self.num_x_ints_list  )
        self.adens_vals      = np.repeat(self.adens_list     , self.num_x_ints_list  )
        self.adens_vals_C    = np.repeat(self.adens_list     , self.num_x_ints_list_C)
        self.rho_DT_vals     = np.repeat(self.rho_DT_list    , self.num_x_ints_list  )

        # Read tabular data
        self.read_stopping_power()
        self.read_cross_sections()

        # Do work
        self.calculate_energy_grid()
        self.calculate_scraping()
        self.calculate_ddxs()
        self.bin_neutrons()

        t1 = time.time()
        if self.verbose: print('Ran in %.2f seconds' % (t1 - t0))

    def check_input(self):

        # Check input parameters for errors
        assert(len(self.pressure_list    ) == self.num_regions)
        assert(len(self.temperature_list ) == self.num_regions)
        assert(len(self.length_list      ) == self.num_regions)
        assert(len(self.divergence_list  ) == self.num_regions)
        assert(len(self.num_x_ints_list  ) == self.num_regions)
        assert(len(self.num_x_ints_list_C) == self.num_regions)
        assert(self.T_frac >= 0 and self.T_frac <= 1)
        assert(self.voltage > 0 and self.voltage <= 100000)
        assert(self.current > 0)
        for x in self.pressure_list   : assert(x > 0)
        for x in self.temperature_list: assert(x > -273.15)
        for x in self.length_list:      assert(x > 0)
        assert(self.start_radius > 0)
        assert(self.max_radius   > 0)
        for x in self.divergence_list:   assert(x >= 0)
        for x in self.num_x_ints_list:   assert(x >  0)
        for x in self.num_x_ints_list_C: assert(x >  0)
        assert(self.angle_int   > 0)
        assert(self.angle_int_C > 0)
        assert(self.EN_int      > 0)
        assert(180              / self.angle_int_C  == int(180              / self.angle_int_C ))
        assert(self.angle_int_C / self.angle_int    == int(self.angle_int_C / self.angle_int   ))
        for x, y in zip(self.num_x_ints_list, self.num_x_ints_list_C): assert(x / y == int(x / y))

    def read_stopping_power(self):

        # Read DT stopping power data from file
        # Note: DD stopping power is calculated as SP_DT * (mass_T / mass_D)
        self.sp_data = {}
        unit_conv = {'eV': 1e-3, 'keV': 1.0, 'MeV': 1e3}
        for rxn in ['DT']:
            fname = os.path.join(self.data_dir, 'SP_%s.txt' % (rxn))
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
            fname = os.path.join(self.data_dir, 'xs_%s.csv' % (rxn))
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
        for i in range(self.num_x_ints_total):
            E_start = self.EB_grid[i]  # Energy at start of interval
            E_avg   = self.EB_grid[i]  # Initialize value for average energy in interval
            for _ in range(5):         # Do 5 iterations
                dEdx  = np.interp(E_avg, self.sp_data['DT']['E'], self.sp_data['DT']['dEdx'],
                                  left=np.nan, right=np.nan)  # Stopping power [keV-cm^2/g]
                dE    = dEdx * self.rho_DT_vals[i] * self.dx_vals[i]  # Energy loss in interval
                E_end = E_start - dE           # Energy at end of interval
                E_avg = (E_start + E_end) / 2  # Average energy in interval
            if np.isnan(E_end): break
            if E_end < 0:       break
            self.EB_grid[i + 1] = E_end
        self.EB_mids = (self.EB_grid[:-1] + self.EB_grid[1:]) / 2

        # Get beam energy on condensed grid
        self.EB_grid_C = np.array(self.EB_grid[self.inds_C])

        if self.verbose: print('  End beam energy     = %.9f keV' % (self.EB_grid[-1]))

    def calculate_scraping(self, use_atan=False):

        # Calculate beam radius at each grid point [cm]
        if use_atan: divergence_vals = np.arctan(self.divergence_vals / 1000)
        else:        divergence_vals =           self.divergence_vals / 1000
        self.radius_grid = self.start_radius + np.append(0, np.cumsum(divergence_vals * self.dx_vals))

        # Calculate beam scrape fraction at each grid point (assume flat beam profile, this is conservative)
        self.scrape_grid = 1 - (self.max_radius / self.radius_grid)**2
        self.scrape_grid[self.scrape_grid < 0] = 0
        self.scrape_mids = (self.scrape_grid[:-1] + self.scrape_grid[1:]) / 2

        # Replace radii that are larger than the max radius
        self.radius_grid[self.radius_grid > self.max_radius] = self.max_radius
        self.radius_mids = np.sqrt((self.radius_grid[:-1]**2 + self.radius_grid[1:]**2) / 2)

        # Calculate beam current at each grid point [mA]
        self.current_grid = self.current * (1 - self.scrape_grid)
        self.current_mids = (self.current_grid[:-1] + self.current_grid[1:]) / 2

        # Get quantities on condensed grid
        self.radius_grid_C  = np.array(self.radius_grid [self.inds_C])
        self.scrape_grid_C  = np.array(self.scrape_grid [self.inds_C])
        self.current_grid_C = np.array(self.current_grid[self.inds_C])
        rsq_mids = (self.radius_grid [:-1]**2 + self.radius_grid [1:]**2) / 2
        s_mids   = (self.scrape_grid [:-1]    + self.scrape_grid [1:]   ) / 2
        c_mids   = (self.current_grid[:-1]    + self.current_grid[1:]   ) / 2
        self.radius_mids_C  = np.sqrt(np.add.reduceat(rsq_mids, self.inds_C[:-1]) / self.x_C_ratio_vals)
        self.scrape_mids_C  =         np.add.reduceat(s_mids  , self.inds_C[:-1]) / self.x_C_ratio_vals
        self.current_mids_C =         np.add.reduceat(c_mids  , self.inds_C[:-1]) / self.x_C_ratio_vals

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
            cosine_cms_mids = np.cos(lab_mids + np.arcsin(gamma[:, None] * np.sin(lab_mids)))

            # Pre-calculate Legendre polynomials
            self.num_leg = 7
            legendre_ = [legendre(x) for x in range(self.num_leg)]

            # Calculate double-differential cross sections [b/sr]
            self.xs[rxn] = np.zeros((self.num_x_ints_total, self.num_a_ints))
            for i, E in enumerate(self.EB_mids):
                total  = np.interp(E, self.xs_data[rxn]['E'], self.xs_data[rxn]['total'], left=0, right=np.nan)
                coeffs = np.zeros(self.num_leg)
                L_vals = np.zeros((self.num_a_ints, self.num_leg))
                for j in range(self.num_leg):
                    coeffs[   j] = np.interp(E, self.xs_data[rxn]['E'], self.xs_data[rxn]['coeffs'][:, j], left=0, right=np.nan)
                    L_vals[:, j] = legendre_[j](cosine_cms_mids[i, :])
                factors = (1 + 2 * gamma[i] * self.cosine_mids + gamma[i]**2)**1.5 / (1 + gamma[i] * self.cosine_mids)
                self.xs[rxn][i, :] = total * np.sum(coeffs[None, :] * L_vals, 1) * factors

            # Calculate neutron energies [MeV]
            self.EN[rxn] = np.zeros((self.num_x_ints_total, self.num_a_ints))
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

        for rxn, frac in [('DD', 1 - self.T_frac), ('DT', self.T_frac)]:

            # Setup neutron energy mesh
            EN_min                = np.floor(np.min(self.EN[rxn]) / self.EN_int) * self.EN_int
            EN_max                = np.ceil (np.max(self.EN[rxn]) / self.EN_int) * self.EN_int
            self.num_EN_ints[rxn] = int(round((EN_max - EN_min) / self.EN_int))
            self.EN_grid    [rxn] = np.linspace(EN_min, EN_max, self.num_EN_ints[rxn] + 1)
            self.EN_mids    [rxn] = (self.EN_grid[rxn][:-1] + self.EN_grid[rxn][1:]) / 2

            # Bin neutrons and convert to units of n/s
            self.neutrons[rxn] = np.zeros((self.num_x_ints_total_C, self.num_a_ints_C, self.num_EN_ints[rxn]))
            offsets  = np.cumsum(np.concatenate(([0], self.num_x_ints_list_C[:-1])))
            iC_all   = np.concatenate([
                np.floor_divide(np.arange(num_x_ints)[:, None], x_C_ratio) + offset
                for num_x_ints, x_C_ratio, offset in zip(self.num_x_ints_list, self.x_C_ratio_list, offsets)
            ], axis=0)
            jC_all   = np.floor_divide(np.arange(self.num_a_ints)[None, :], self.angle_C_ratio)
            iC_all   = np.repeat(iC_all, self.num_a_ints      , axis=1).astype(int).ravel()
            jC_all   = np.repeat(jC_all, self.num_x_ints_total, axis=0).astype(int).ravel()
            kC_all   = np.searchsorted(self.EN_grid[rxn], self.EN[rxn].ravel()) - 1
            areas    = np.tile(self.angle_areas, self.num_x_ints_total)
            factor   = (self.adens_vals * 1e-24 * self.dx_vals * frac * self.dps)[:, None]
            vals_scr = (self.xs[rxn] * factor * (1 - self.scrape_mids[:, None])).ravel() * areas
            vals_tot = (self.xs[rxn] * factor                                  ).ravel() * areas
            np.add.at(self.neutrons[rxn], (iC_all, jC_all, kC_all), vals_scr)

            # Calculate total neutron yield
            self.nyield[rxn] = np.sum(self.neutrons[rxn])
            if self.verbose: print('  %s neutron yield    = %.5e n/s' % (rxn, self.nyield[rxn]))

            # Calculate amount of beam scraping
            if np.sum(np.sum(vals_tot)) == 0: self.nscrape[rxn] = 0.0
            else: self.nscrape[rxn] = 1 - np.sum(vals_scr) / np.sum(vals_tot)

        self.nyield_tot  = self.nyield['DD'] + self.nyield['DT']
        self.nscrape_tot = (self.nyield['DD'] * self.nscrape['DD'] +
                            self.nyield['DT'] * self.nscrape['DT']) / self.nyield_tot
        if self.verbose: print('  Total neutron yield = %.5e n/s' % (self.nyield_tot))
        if self.verbose: print('  Scrape fraction     = %.5f%%'   % (self.nscrape[rxn] * 100))

    def write_hdf5(self):

        if not os.path.exists('hdf5'): os.makedirs('hdf5')
        fname = os.path.join('hdf5', '%s.h5' % (self.name))
        if self.verbose: print('Writing %s' % (fname))

        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('x_grid'      , data=self.x_grid_C      , compression='gzip')  # (138         )
            hf.create_dataset('EB_grid'     , data=self.EB_grid_C     , compression='gzip')  # (138         )
            hf.create_dataset('radius_grid' , data=self.radius_grid_C , compression='gzip')  # (138         )
            hf.create_dataset('scrape_grid' , data=self.scrape_grid_C , compression='gzip')  # (138         )
            hf.create_dataset('current_grid', data=self.current_grid_C, compression='gzip')  # (138         )
            hf.create_dataset('angle_grid'  , data=self.angle_grid_C  , compression='gzip')  # (     61     )
            hf.create_dataset('EN_grid_DD'  , data=self.EN_grid ['DD'], compression='gzip')  # (         130)
            hf.create_dataset('EN_grid_DT'  , data=self.EN_grid ['DT'], compression='gzip')  # (         242)
            hf.create_dataset('neutrons_DD' , data=self.neutrons['DD'], compression='gzip')  # (137, 60, 129)
            hf.create_dataset('neutrons_DT' , data=self.neutrons['DT'], compression='gzip')  # (137, 60, 241)

    def get_mcnp_sdef(self, direction='-z'):

        if np.all(self.divergence_list == 0): divergent = False
        else:                                 divergent = True
        if self.angle_int_C == 180:           isotropic = True
        else:                                 isotropic = False

        has_DD = False; has_DT = False
        if self.T_frac < 1: has_DD = True
        if self.T_frac > 0: has_DT = True

        # Header
        dashed_line = 'c %s\n' % ('-' * 78)
        s = ''
        s += dashed_line
        s += 'c     Source definition\n'
        s += dashed_line
        s += 'c\n'
        s += 'c     Tritium fraction   : %.12g%%\n'    % (self.T_frac * 100        )
        s += 'c     Beam voltage       : %.12g kV\n'   % (self.voltage             )
        s += 'c     Beam current       : %.12g mA\n'   % (self.current             )
        s += 'c     Target pressure    : %.12g torr\n' % (self.pressure_list   [-1])
        s += 'c     Target temperature : %.12g C\n'    % (self.temperature_list[-1])
        s += 'c     Target length      : %.12g cm\n'   % (self.length_list     [-1])
        s += 'c     Beam start_radius  : %.12g cm\n'   % (self.start_radius        )
        s += 'c     Beam max_radius    : %.12g cm\n'   % (self.max_radius          )
        s += 'c     Beam divergence    : %.12g mrad\n' % (self.divergence_list[-1] )
        s += 'c\n'
        s += dashed_line
        s += 'c\n'
        s += 'c     %2s %8s %8s %8s %8s %8s %8s %8s\n' % ('#', 'Pressure', 'Temp', 'Length', 'Diverge', 'Ints'   , 'Ints'    , 'E_Final')
        s += 'c     %2s %8s %8s %8s %8s %8s %8s %8s\n' % ('' , '[torr]'  , '[C]' , '[cm]'  , '[mrad]' , 'output' , 'internal', '[keV]'  )
        for i in range(self.num_regions): s += 'c     %2u %8.7g %8.7g %8.7g %8.7g %8u %8u %8.7g\n' % (
            i + 1, self.pressure_list[i], self.temperature_list[i], self.length_list[i], self.divergence_list[i],
            self.num_x_ints_list_C[i], self.num_x_ints_list[i], self.EB_grid[np.sum(self.num_x_ints_list[:i + 1])])
        s += 'c\n'
        s += dashed_line
        s += 'c\n'
        s += 'c     # of distance intervals                 : %.12g\n'     % (self.num_x_ints_total  )
        s += 'c     # of distance intervals (internal calc) : %.12g\n'     % (self.num_x_ints_total_C)
        s += 'c     Angle interval                          : %.12g deg\n' % (self.angle_int_C       )
        s += 'c     Angle interval (internal calc)          : %.12g deg\n' % (self.angle_int         )
        s += 'c     Neutron energy interval                 : %.12g MeV\n' % (self.EN_int            )
        s += 'c\n'
        s += dashed_line
        s += 'c\n'
        s += 'c     DD    neutron yield : %.5e n/s (%9.5f%%)\n' % (self.nyield['DD'], self.nyield['DD'] / self.nyield_tot * 100)
        s += 'c     DT    neutron yield : %.5e n/s (%9.5f%%)\n' % (self.nyield['DT'], self.nyield['DT'] / self.nyield_tot * 100)
        s += 'c     Total neutron yield : %.5e n/s\n'           % (self.nyield_tot  )
        s += 'c\n'
        s += 'c     Beam energy at backstop : %.6f keV\n' % (self.EB_grid[-1])
        s += 'c\n'
        s += 'c     Fraction of neutrons lost to scraping : %.5f%%\n' % (self.nscrape_tot * 100)
        s += 'c\n'
        s += dashed_line

        # SDEF card
        if   direction == '-x': axs = '-1  0  0'; vec = '-1  0  0'; reverse_x = False
        elif direction == '+x': axs = ' 1  0  0'; vec = ' 1  0  0'; reverse_x = False
        elif direction == '-y': axs = ' 0 -1  0'; vec = ' 0 -1  0'; reverse_x = False
        elif direction == '+y': axs = ' 0  1  0'; vec = ' 0  1  0'; reverse_x = False
        elif direction == '-z': axs = ' 0  0  1'; vec = ' 0  0 -1'; reverse_x = True  # ADVANTG bug means -z case has to be reversed
        elif direction == '+z': axs = ' 0  0  1'; vec = ' 0  0  1'; reverse_x = False
        axs = ' '.join(axs.strip().split())
        vec = ' '.join(vec.strip().split())
        s += 'c\n'
        s += '  sdef  par = n\n'
        s += '        pos = 0 0 0\n'
        s += '        axs = %s\n' % (axs)
        s += '        vec = %s\n' % (vec)
        s += '        ext = d1\n'
        if isotropic: pass
        else:         s += '        dir = d2\n'
        if divergent: s += '        rad = fext = d3\n'
        else:         s += '        rad = d3\n'
        if isotropic: s += '        erg = d4\n'
        else:         s += '        erg = fdir = d4\n'

        s += 'c\n'
        s += dashed_line

        # Extent distribution
        ext = np.sum(self.neutrons['DD'], (1, 2)) + np.sum(self.neutrons['DT'], (1, 2))
        if reverse_x: ext = ext[::-1]
        si1 = '0'; x = 0.0
        nvals = self.num_x_ints_list_C; Lvals = self.length_list
        if reverse_x: nvals = np.array(nvals)[::-1]; Lvals = np.array(Lvals)[::-1]
        for n, L in zip(nvals, Lvals):
            x += L
            if   n == 1: pass
            elif n == 2: si1 += ' i'
            else: si1 += ' %ui' % (n - 1)
            si1 += ' %s' % (x)
        si1 = textwrap.fill(si1, width=72).replace('\n', '\n        ')
        s += 'c\n'
        s += '  sc1   Extent distribution\n'
        s += '  si1   %s\n' % (si1)
        s += '  sp1  '
        for i, x in enumerate(np.append(0, ext)):
            if i > 0 and i % 6 == 0: s += '\n       '
            s += ' %.5e' % (x)
        s += '\n'
        s += 'c\n'
        s += dashed_line

        # Angular distribution
        if not isotropic:
            dir = (np.sum(self.neutrons['DD'], (0, 2)) + np.sum(self.neutrons['DT'], (0, 2)))[::-1]
            s += 'c\n'
            s += '  sc2   Angular distribution\n'
            s += '  si2  '
            for i, x in enumerate(self.cosine_grid_C[::-1]):
                if i > 0 and i % 6 == 0: s += '\n       '
                s += ' %11.8f' % (x)
            s += '\n'
            s += '  sp2  '
            for i, x in enumerate(np.append(0, dir)):
                if i > 0 and i % 6 == 0: s += '\n       '
                s += ' %11.5e' % (x)
            s += '\n'
            s += 'c\n'
            s += dashed_line

        # Radial distribution
        s += 'c\n'
        s += '  sc3   Radial distribution\n'
        if divergent:
            rmids = self.radius_mids_C[::-1] if reverse_x else self.radius_mids_C
            smids = self.scrape_mids_C[::-1] if reverse_x else self.scrape_mids_C
            s += '  ds3 S'
            for i in range(self.num_x_ints_list_C[-1]):  ##### Fix
                if i > 0 and i % 18 == 0: s += '\n       '
                s += ' %3u' % (101 + i)
            s += '\n'
            s += 'c\n'
            for i, x in enumerate(rmids):
                s += '  si%-3u 0 %10.8f' % (101 + i, x)
                if smids[i] > 0: s += '  $ (%8.5f%% scraped)' % (smids[i] * 100)
                s += '\n'
        else: s += '  si3   0 %s\n' % (self.start_radius)
        s += 'c\n'
        s += dashed_line

        # Energy distribution
        s += 'c\n'
        s += '  sc4   Energy distribution\n'
        if isotropic:
            if has_DD:
                DD_min = round(self.EN_grid['DD'][ 0], 12)
                DD_max = round(self.EN_grid['DD'][-1], 12)
                num_DD = self.EN_grid['DD'].shape[0] - 2
                erg_DD = np.sum(self.neutrons['DD'], (0, 1))
            if has_DT:
                DT_min = round(self.EN_grid['DT'][ 0], 12)
                DT_max = round(self.EN_grid['DT'][-1], 12)
                num_DT = self.EN_grid['DT'].shape[0] - 2
                erg_DT = np.sum(self.neutrons['DT'], (0, 1))
            s += '  si4  '
            if has_DD: s += ' %s %ui %s' % (DD_min, num_DD, DD_max)
            if has_DT: s += ' %s %ui %s' % (DT_min, num_DT, DT_max)
            s += '\n'
            s += '  sp4  '
            vals = []
            if has_DD: vals = np.append(vals, np.append(0, erg_DD))
            if has_DT: vals = np.append(vals, np.append(0, erg_DT))
            for i, x in enumerate(vals):
                if i > 0 and i % 6 == 0: s += '\n       '
                s += ' %.5e' % (x)
            s += '\n'
            s += 'c\n'
        else:
            s += '  ds4 S'
            for i in range(self.num_a_ints_C):
                if i > 0 and i % 18 == 0: s += '\n       '
                s += ' %3u' % (501 + i)
            s += '\n'
            s += 'c\n'
            for i in range(self.num_a_ints_C):
                sid = 501 + i
                if has_DD:
                    erg_DD  = np.sum(self.neutrons['DD'][:, -i - 1, :], 0)
                    inds    = np.nonzero(erg_DD)[0]
                    ind_min = inds[ 0]
                    ind_max = inds[-1] + 1
                    erg_DD  = erg_DD[ind_min:ind_max]
                    DD_min  = round(self.EN_grid['DD'][ind_min], 12)
                    DD_max  = round(self.EN_grid['DD'][ind_max], 12)
                    num_DD  = ind_max - ind_min - 1
                if has_DT:
                    erg_DT  = np.sum(self.neutrons['DT'][:, -i - 1, :], 0)
                    inds    = np.nonzero(erg_DT)[0]
                    ind_min = inds[ 0]
                    ind_max = inds[-1] + 1
                    erg_DT  = erg_DT[ind_min:ind_max]
                    DT_min  = round(self.EN_grid['DT'][ind_min], 12)
                    DT_max  = round(self.EN_grid['DT'][ind_max], 12)
                    num_DT  = ind_max - ind_min - 1
                s += '  sc%-3u Energy distribution (%.12g - %.12g degrees)\n' % (sid, self.angle_grid_C[i], self.angle_grid_C[i + 1])
                s += '  si%-3u' % (sid)
                if has_DD: s += ' %s %ui %s' % (DD_min, num_DD, DD_max)
                if has_DT: s += ' %s %ui %s' % (DT_min, num_DT, DT_max)
                s += '\n'
                s += '  sp%-3u' % (sid)
                vals = []
                if has_DD: vals = np.append(vals, np.append(0, erg_DD))
                if has_DT: vals = np.append(vals, np.append(0, erg_DT))
                for j, x in enumerate(vals):
                    if j > 0 and j % 6 == 0: s += '\n       '
                    s += ' %.5e' % (x)
                s += '\n'
                s += 'c\n'
        s += dashed_line

        return s

    def write_mcnp_sdef(self, direction='-z'):

        if not os.path.exists('mcnp_sdef'): os.makedirs('mcnp_sdef')
        fname = os.path.join('mcnp_sdef', '%s_source.txt' % (self.name))
        if self.verbose: print('Writing %s' % (fname))
        with open(fname, 'w') as w: w.write(self.get_mcnp_sdef(direction))

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
        if self.verbose: print('Writing %s' % (fname))
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
        if self.verbose: print('Writing %s' % (fname))
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
        if self.verbose: print('Writing %s' % (fname))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def plot_neutron_position(self):

        xvals     = np.array(self.x_grid_C)
        xvals     = np.append(np.array(xvals), xvals[-1])
        yvals_DD  = np.sum(self.neutrons['DD'], (1, 2)) / self.dx_vals_C
        yvals_DT  = np.sum(self.neutrons['DT'], (1, 2)) / self.dx_vals_C
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
        ax.legend()
        fname = os.path.join('images', '%s_neutron_position.png' % (self.name))
        if self.verbose: print('Writing %s' % (fname))
        plt.tight_layout()
        plt.savefig(fname)

        ax.set_yscale('log')
        fname = os.path.join('images', '%s_neutron_position_log.png' % (self.name))
        if self.verbose: print('Writing %s' % (fname))
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
        if self.verbose: print('Writing %s' % (fname))
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

            if case == 'DD' and self.T_frac == 1: continue
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
            if self.verbose: print('Writing %s' % (fname))
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
