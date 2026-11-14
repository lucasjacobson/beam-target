"""Units: the code works in eV (beam and neutron energies), A (current), Pa (pressure), K
(temperature), rad (divergence), W (power), cm (lengths), g/cm3 (density), and degrees
(emission angles). UNITS lists the units input files may use and results are reported in,
with their factors to those; parse reads a value with its unit, such as "280 keV"."""

import re

EV, KEV, MEV, GEV = 1.0, 1e3, 1e6, 1e9  # Energies [eV]
AMPERE, MILLIAMPERE, MICROAMPERE, NANOAMPERE = 1.0, 1e-3, 1e-6, 1e-9  # Currents [A]
TORR = 101325 / 760  # [Pa]
MILLIRADIAN = 1e-3  # [rad]
KILOWATT = 1e3  # [W]
ZERO_CELSIUS = 273.15  # [K]

# Each unit: (quantity, factor, offset), so that the value in the code's unit is
# value * factor + offset. Names are case-sensitive; "u" and "µ" (micro) are both accepted.
UNITS = {
    "eV": ("energy", EV, 0.0),
    "keV": ("energy", KEV, 0.0),
    "MeV": ("energy", MEV, 0.0),
    "GeV": ("energy", GEV, 0.0),
    "A": ("current", AMPERE, 0.0),
    "mA": ("current", MILLIAMPERE, 0.0),
    "uA": ("current", MICROAMPERE, 0.0),
    "nA": ("current", NANOAMPERE, 0.0),
    "Pa": ("pressure", 1.0, 0.0),
    "kPa": ("pressure", 1e3, 0.0),
    "torr": ("pressure", TORR, 0.0),
    "Torr": ("pressure", TORR, 0.0),
    "mtorr": ("pressure", TORR / 1000, 0.0),
    "mTorr": ("pressure", TORR / 1000, 0.0),
    "mbar": ("pressure", 100.0, 0.0),
    "bar": ("pressure", 1e5, 0.0),
    "atm": ("pressure", 101325.0, 0.0),
    "psi": ("pressure", 6894.757293168361, 0.0),  # lbf/in2: 0.45359237 kg x g / 0.0254^2 m2
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, ZERO_CELSIUS),
    "C": ("temperature", 1.0, ZERO_CELSIUS),
    "degF": ("temperature", 5 / 9, ZERO_CELSIUS - 32 * 5 / 9),
    "F": ("temperature", 5 / 9, ZERO_CELSIUS - 32 * 5 / 9),
    "rad": ("divergence", 1.0, 0.0),
    "mrad": ("divergence", MILLIRADIAN, 0.0),
    "urad": ("divergence", 1e-6, 0.0),
    "deg": ("divergence", 0.017453292519943295, 0.0),  # pi / 180
    "W": ("power", 1.0, 0.0),
    "kW": ("power", KILOWATT, 0.0),
    "m": ("length", 100.0, 0.0),
    "cm": ("length", 1.0, 0.0),
    "mm": ("length", 0.1, 0.0),
    "um": ("length", 1e-4, 0.0),
    "nm": ("length", 1e-7, 0.0),
    "in": ("length", 2.54, 0.0),
    "inch": ("length", 2.54, 0.0),
    "ft": ("length", 30.48, 0.0),
    "g/cm3": ("density", 1.0, 0.0),
    "kg/m3": ("density", 1e-3, 0.0),
    "g/L": ("density", 1e-3, 0.0),
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
    "length": "cm",  # Along the beam
    "radius": "cm",  # Beam and wall radii
    "density": "g/cm3",
}

_VALUE_WITH_UNIT = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\s\d.+-]\S*)\s*$"
)


def unit_name(unit):
    """Return the UNITS name of unit: the same, with the micro sign "µ" read as "u"."""
    return unit.replace("µ", "u").replace("μ", "u")


def quantity_of(unit):
    """Return the quantity a unit is for ("energy", "current", ...); KeyError if unknown."""
    return UNITS[unit_name(unit)][0]


def units_of(quantity):
    """Return the names of the units of a quantity, in UNITS order."""
    return [name for name, (q, _, _) in UNITS.items() if q == quantity]


def parse(text, quantity):
    """Read a value with its unit, such as "280 keV" or "5um", for a quantity of UNITS.

    Returns (value in the code's unit, unit name). Raises ValueError if text is not a number
    followed by one of the quantity's units."""
    accepted = ", ".join(units_of(quantity))
    match = _VALUE_WITH_UNIT.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(
            f'expected a number and a unit ({accepted}), such as "{example(quantity)}"'
        )
    unit = unit_name(match.group(2))
    if UNITS.get(unit, ("",))[0] != quantity:
        raise ValueError(f"unknown {quantity} unit {match.group(2)!r} (use {accepted})")
    return to_internal(float(match.group(1)), unit), unit


def example(quantity):
    """A value with a unit for a quantity, for messages."""
    return {
        "energy": "280 keV",
        "current": "10 mA",
        "pressure": "5 torr",
        "temperature": "25 degC",
        "divergence": "30 mrad",
        "length": "50 cm",
        "density": "1.848 g/cm3",
    }[quantity]


def to_internal(value, unit):
    """Return value (a number or an array) given in unit in the code's units."""
    _, factor, offset = UNITS[unit_name(unit)]
    return value * factor + offset


def from_internal(value, unit):
    """Return value (a number or an array) in the code's units in unit."""
    _, factor, offset = UNITS[unit_name(unit)]
    return (value - offset) / factor
