"""Units: the code works in eV (beam and neutron energies), A (current), Pa (pressure), K
(temperature), rad (divergence), W (power), cm (lengths), g/cm3 (density), and degrees
(emission angles). UNITS gives the other units it reports in, with their factors to those."""

EV, KEV, MEV, GEV = 1.0, 1e3, 1e6, 1e9  # Energies [eV]
AMPERE, MILLIAMPERE, MICROAMPERE = 1.0, 1e-3, 1e-6  # Currents [A]
TORR = 101325 / 760  # [Pa]
MILLIRADIAN = 1e-3  # [rad]
KILOWATT = 1e3  # [W]
ZERO_CELSIUS = 273.15  # [K]

# Each unit: (quantity, factor, offset), so that the value in the code's unit is
# value * factor + offset
UNITS = {
    "eV": ("energy", EV, 0.0),
    "keV": ("energy", KEV, 0.0),
    "MeV": ("energy", MEV, 0.0),
    "A": ("current", AMPERE, 0.0),
    "mA": ("current", MILLIAMPERE, 0.0),
    "Pa": ("pressure", 1.0, 0.0),
    "torr": ("pressure", TORR, 0.0),
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, ZERO_CELSIUS),
    "rad": ("divergence", 1.0, 0.0),
    "mrad": ("divergence", MILLIRADIAN, 0.0),
    "W": ("power", 1.0, 0.0),
    "kW": ("power", KILOWATT, 0.0),
    "cm": ("length", 1.0, 0.0),
}

# The units BeamTarget reports its results in unless told otherwise (see BeamTarget's units)
DEFAULT_UNITS = {
    "beam_energy": "keV",
    "neutron_energy": "MeV",
    "current": "mA",
    "power": "kW",
    "pressure": "torr",
    "temperature": "degC",
    "divergence": "mrad",
    "length": "cm",
}


def to_internal(value, unit):
    """Return value (a number or an array) given in unit in the code's units."""
    _, factor, offset = UNITS[unit]
    return value * factor + offset


def from_internal(value, unit):
    """Return value (a number or an array) in the code's units in unit."""
    _, factor, offset = UNITS[unit]
    return (value - offset) / factor
