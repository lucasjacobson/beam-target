#!/usr/bin/env python3
"""Generate neutron source terms for a set of SHINE beam-target cases.

For each case in CASES this writes:

- hdf5/<case>.h5: the full source distribution
- mcnp_sdef/<case>_source.txt: an MCNP SDEF source card
- images/<case>_*.png: plots of the beam and neutron source

Output directories are created in the current working directory.

Usage: python driver.py [-j JOBS] [-v]
"""

import argparse
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

from shine_neutron_source import ShineSource

# Mesh resolution used for every case. Memory use scales with
# (total internal position intervals) x (180 / ANGLE_INT). At these settings the peak is
# about 2.5 GB per case (measured); with -j N, up to N cases run at once.
NUM_X_INTS = 1370  # Position intervals in the target chamber (internal calculation)
NUM_X_INTS_C = 137  # Position intervals in the target chamber (output)
ANGLE_INT = 0.01  # Angle interval [deg] (internal calculation)
ANGLE_INT_C = 3.0  # Angle interval [deg] (output)
EN_INT = 0.01  # Neutron energy interval [MeV]

# Upstream differential-pumping stages, in the order the beam passes through them. Used
# when a case has include_upstream=True. The beam does not diverge in these regions, and
# each uses 10 internal and 1 output position intervals.
# Upstream stage values are not stored in this repository.
UPSTREAM_PRESSURES = [np.nan]  # [torr]
UPSTREAM_LENGTHS = [np.nan]  # [cm]


@dataclass
class Case:
    """Operating conditions for one source calculation."""

    T_frac: float  # Atom fraction of tritium in the gas
    voltage: float  # Beam voltage [kV]
    current: float  # Beam current [mA]
    pressure: float  # Target chamber pressure [torr]
    temperature: float  # Gas temperature [C], used in every region
    length: float  # Target chamber length [cm]
    start_radius: float  # Beam radius at the entrance [cm]
    max_radius: float  # Target chamber wall radius [cm]
    divergence: float  # Beam divergence in the target chamber [mrad]
    include_upstream: bool  # Add the upstream pumping stages before the target chamber


# fmt: off
CASES = {  #          T_Frac Voltage Current   Pressure  Temp    Length    Radius RadMax  Diverge Upstream
    "DD"       : Case(0.0 ,  280.0,  38.39596, 38.39728, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
    "DT01pct"  : Case(0.01,  280.0,  38.39596, 38.39738, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
    "DT90pct"  : Case(0.9 ,  280.0,  38.39596, 38.39738, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
    "DT"       : Case(1.0 ,  280.0,  38.39596, 38.39738, 1800.0, 137.3025, 0.2,   4.1783, 36.0,   False),
}
# fmt: on


def run_case(name, case, verbose=False):
    """Calculate the source for one case, write its outputs and return the source."""
    if case.include_upstream:
        num_upstream = len(UPSTREAM_PRESSURES)
        pressure_list = UPSTREAM_PRESSURES + [case.pressure]
        length_list = UPSTREAM_LENGTHS + [case.length]
        divergence_list = [0.0] * num_upstream + [case.divergence]
        num_x_ints_list = [10] * num_upstream + [NUM_X_INTS]
        num_x_ints_list_C = [1] * num_upstream + [NUM_X_INTS_C]
    else:
        pressure_list = [case.pressure]
        length_list = [case.length]
        divergence_list = [case.divergence]
        num_x_ints_list = [NUM_X_INTS]
        num_x_ints_list_C = [NUM_X_INTS_C]

    source = ShineSource(
        name=name,
        T_frac=case.T_frac,
        voltage=case.voltage,
        current=case.current,
        pressure_list=pressure_list,
        temperature_list=np.full(len(pressure_list), case.temperature),
        length_list=length_list,
        start_radius=case.start_radius,
        max_radius=case.max_radius,
        divergence_list=divergence_list,
        num_x_ints_list=num_x_ints_list,
        num_x_ints_list_C=num_x_ints_list_C,
        angle_int=ANGLE_INT,
        angle_int_C=ANGLE_INT_C,
        EN_int=EN_INT,
        verbose=verbose,
    )
    source.write_hdf5()
    source.write_mcnp_sdef()
    source.make_plots()
    return source


def _run_case_in_worker(args):
    """Run a case in a worker process. Returns nothing, so the large source object is not
    copied back to the parent process."""
    name, case, verbose = args
    run_case(name, case, verbose)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=mp.cpu_count(),
        help="number of cases to run in parallel (default: number of CPUs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    return parser.parse_args()


def main():
    args = get_args()
    jobs = min(args.jobs, len(CASES))
    if jobs > 1:
        work = [(name, case, args.verbose) for name, case in CASES.items()]
        with mp.Pool(processes=jobs) as pool:
            pool.map(_run_case_in_worker, work)
    else:
        for name, case in CASES.items():
            run_case(name, case, args.verbose)


if __name__ == "__main__":
    main()
