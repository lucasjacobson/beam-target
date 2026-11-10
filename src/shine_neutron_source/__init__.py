"""Beam-target fusion neutron source terms for Monte Carlo transport codes.

- ShineSource (in source.py) calculates the source for one case and writes it as HDF5, an
  MCNP SDEF card and plots.
- load_input and run_case (in cli.py) read cases from a TOML input file and run them; the
  shine-neutron-source command (or python -m shine_neutron_source) does this from the
  command line.

Progress and results are logged to the "shine_neutron_source" logger, which is silent unless
logging is configured (the command line shows it; see setup_logging).
"""

import logging

from .cli import InputError, load_input, main, run_case, setup_logging
from .source import ShineSource, get_atom_density, get_code_version, get_package_version

__version__ = get_package_version()

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "InputError",
    "ShineSource",
    "__version__",
    "get_atom_density",
    "get_code_version",
    "load_input",
    "main",
    "run_case",
    "setup_logging",
]
