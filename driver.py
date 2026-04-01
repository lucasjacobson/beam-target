#!/usr/bin/python3

# Note: code runs out of memory at 0.025 degree interval

import argparse
import multiprocessing as mp
import numpy as np

from shine_source_gen import SHINE_Source

class Case():
    def __init__(self, T_frac, voltage, current, pressure, temperature, length,
                 start_radius, max_radius, divergence, include_upstream):
        self.T_frac           = T_frac
        self.voltage          = voltage
        self.current          = current
        self.pressure         = pressure
        self.temperature      = temperature
        self.length           = length
        self.start_radius     = start_radius
        self.max_radius       = max_radius
        self.divergence       = divergence
        self.include_upstream = include_upstream

def run_case(name, case):

    T_frac       = case.T_frac        # Tritium fraction
    voltage      = case.voltage       # Beam voltage [kV]
    current      = case.current       # Beam current [mA]
    pressure     = case.pressure      # Target pressure [torr]
    temperature  = case.temperature   # Target temperature [C]
    length       = case.length        # Target length [cm]
    start_radius = case.start_radius  # Beam start radius [cm]
    max_radius   = case.max_radius    # Beam max radius [cm]
    divergence   = case.divergence    # Beam divergence [mrad]
    num_x_ints   = 1370               # Number of position intervals (internal calculation)
    num_x_ints_C = 137                # Number of position intervals (output)
    angle_int    = 0.01               # Angle interval [deg] (internal calculation)
    angle_int_C  = 3.0                # Angle interval [deg] (output)
    EN_int       = 0.01               # Neutron energy interval [MeV]

    if case.include_upstream:
        # Upstream stage values are not stored in this repository
        pressure_list     = np.array([np.nan, pressure])
        length_list       = np.array([np.nan, length  ])
        num_regions       = len(pressure_list)
        temperature_list  = np.full(num_regions, temperature)
        divergence_list   = np.append(np.full(num_regions - 1, 0.0), divergence  )
        num_x_ints_list   = np.append(np.full(num_regions - 1, 10 ), num_x_ints  )
        num_x_ints_list_C = np.append(np.full(num_regions - 1,  1 ), num_x_ints_C)
    else:
        pressure_list     = pressure
        length_list       = length
        temperature_list  = temperature
        divergence_list   = divergence
        num_x_ints_list   = num_x_ints
        num_x_ints_list_C = num_x_ints_C

    SS = SHINE_Source(name, T_frac, voltage, current, pressure_list, temperature_list, length_list, start_radius,
                      max_radius, divergence_list, num_x_ints_list, num_x_ints_list_C, angle_int, angle_int_C, EN_int)

    SS.write_hdf5()
    SS.write_mcnp_sdef()
    SS.make_plots()

    return SS

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int, default=mp.cpu_count(), help='number of CPUs')
    args = parser.parse_args()
    return args

def worker(item):
    name, case = item
    return name, run_case(name, case)

def main():

    cases = {  #          T_Frac Voltage Current   Pressure  Temp    Length    Radius RadMax  Diverge Upstream
        'DD'       : Case(0.0 ,  280.0,  38.39596, 38.39728, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
        'DT01pct'  : Case(0.01,  280.0,  38.39596, 38.39738, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
        'DT90pct'  : Case(0.9 ,  280.0,  38.39596, 38.39738, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
        'DT'       : Case(1.0 ,  280.0,  38.39596, 38.39738, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
    }

    jobs = get_args().jobs
    if jobs > 1:
        with mp.Pool(processes=jobs) as pool: sources = dict(dict(pool.map(worker, list(cases.items()))))
    else: sources = {name: run_case(name, case) for name, case in cases.items()}

if __name__ == '__main__': main()
