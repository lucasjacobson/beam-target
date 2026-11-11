"""Beam-target fusion neutron source terms for Monte Carlo transport codes.

- ShineSource (in source.py) calculates the source for one case and writes it as HDF5, an
  MCNP SDEF card and plots.
- load_input and run_case (in cli.py) read cases from a TOML input file and run them; the
  shine-neutron-source command (or python -m shine_neutron_source) does this from the
  command line.
- openmc_source.get_openmc_sources turns a case's source.h5 into OpenMC sources (needs the
  openmc package, which is not imported otherwise).

Progress and results are logged to the "shine_neutron_source" logger, which is silent unless
logging is configured (the command line shows it; see setup_logging).
"""

import logging

from .cli import InputError, load_input, main, run_case, setup_logging
from .source import (
    Material,
    Region,
    ShineSource,
    dt_gas,
    get_atom_density,
    get_code_version,
    get_package_version,
    parse_formula,
)

__version__ = get_package_version()

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "InputError",
    "Material",
    "Region",
    "ShineSource",
    "__version__",
    "dt_gas",
    "get_atom_density",
    "get_code_version",
    "load_input",
    "main",
    "parse_formula",
    "run_case",
    "setup_logging",
]
