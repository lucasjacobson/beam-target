"""Input files and the command-line interface.

Run the cases in a TOML input file with either of:

    shine-neutron-source examples/generic.toml [-c CASE] [-o OUTPUT_DIR] [-j JOBS] [-v]
    python -m shine_neutron_source examples/generic.toml ...

See load_input for the file format and examples/generic.toml for an example.
"""

import argparse
import logging
import multiprocessing as mp
import os
import re
import time
import tomllib

from .source import PHASES, SDEF_DIRECTIONS, STOPPING_POWERS, Material, ShineSource

logger = logging.getLogger(__name__)

# Case settings: name -> required type. Every case needs all of them, either directly or
# from [defaults].
CASE_SETTINGS = {
    "material": str,  # Name of the [materials.<name>] table filling the target chamber
    "voltage": float,  # Beam voltage [kV]
    "start_radius": float,  # Beam radius where it enters the first region [cm]
    "wall_radius": float,  # Target chamber wall radius [cm]
    "divergence": float,  # Beam divergence in the target chamber [mrad]
    "num_x_bins": int,  # Position bins in the target chamber (internal mesh)
    "num_x_bins_out": int,  # Position bins in the target chamber (output mesh)
    "angle_step": float,  # Width of the emission-angle bins [deg] (internal mesh)
    "angle_step_out": float,  # Width of the emission-angle bins [deg] (output mesh)
    "energy_bin_width": float,  # Width of the neutron energy bins [MeV] (output mesh)
}

# Gas temperature [degC], used in every gas region given by pressure. Needed only by cases
# with such a region.
TEMPERATURE_SETTING = "temperature"

# The gas state (pressure, temperature, density) is set in each case, never in [defaults], so
# that what a case gives, and so what it calculates, is all in the case itself
CASE_ONLY_SETTINGS = ("pressure", "temperature", "density")

# Target chamber settings. A gas target chamber needs two of its state (pressure, which
# needs a temperature, or density), its length and beam_energy_backstop, and the third is
# calculated; with no state given, the calculated state is a pressure if the case has a
# temperature and a density otherwise. The state is set in each case (CASE_ONLY_SETTINGS).
# A solid target chamber has no state: it needs one of its length and beam_energy_backstop.
TARGET_SETTINGS = {
    "pressure": float,  # Target chamber pressure [torr]
    "density": float,  # Target chamber gas density [g/cm3], instead of pressure
    "length": float,  # Target chamber length [cm]
    "beam_energy_backstop": float,  # Beam energy at the backstop [keV]
}
STATE_SETTINGS = ("pressure", "density")

# Beam settings: every case needs exactly one of these (directly or from [defaults]), and
# the other is calculated
BEAM_SETTINGS = {
    "current": float,  # Beam current [mA]
    "yield_total": float,  # Total neutron yield [n/s]
}

# Optional case settings and their defaults
OPTIONAL_CASE_SETTINGS = {
    "direction": "-z",  # MCNP SDEF beam direction, see SDEF_DIRECTIONS
    # Name of an [upstream.<name>] table to put before the target chamber; false for none (to
    # override one set in [defaults])
    "upstream": None,
    "stopping_power": "srim",  # Stopping-power set, see STOPPING_POWERS
    "entry_position": [0.0, 0.0, 0.0],  # cm, where the beam enters (MCNP coordinates)
}

# Material settings: name -> type. A solid needs atoms and density, a gas molecules (see
# Material). A gas's state (pressure and temperature, or density) belongs to the region it
# fills.
MATERIAL_SETTINGS = {
    "phase": str,  # "gas" or "solid"
    "atoms": dict,  # Solid: {atom: amount}, atom ratios
    "molecules": dict,  # Gas: {formula: amount}, mole fractions of its molecules
    "density": float,  # Mass density of a solid [g/cm3]
}

# Upstream stage settings: name -> (type, default). None means required. Each stage holds a
# gas given by pressure (with the case's temperature) or by density: an upstream table has
# exactly one of the two lists. material is the name of a gas [materials.<name>] table, or a
# list with one per stage. num_x_bins, num_x_bins_out and divergence can be one value for
# every stage or a list with one per stage.
UPSTREAM_SETTINGS = {
    "material": (str, None),  # Gas filling each stage
    "length": (list, None),  # Length of each stage [cm]
    "pressure": (list, ""),  # Pressure of each stage [torr] (or density)
    "density": (list, ""),  # Gas density of each stage [g/cm3] (or pressure)
    "num_x_bins": (int, 10),  # Position bins per stage (internal mesh)
    "num_x_bins_out": (int, 1),  # Position bins per stage (output mesh)
    "divergence": (float, 0.0),  # Beam divergence in each stage [mrad]
}

CASE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class InputError(ValueError):
    """An input file is missing a setting or has an invalid one."""


def _check_type(where, key, value, expected):
    """Raise InputError unless value has the expected type (int is accepted for float)."""
    if isinstance(value, bool):
        ok = expected is bool
    elif expected is float:
        ok = isinstance(value, (int, float))
    else:
        ok = isinstance(value, expected)
    if not ok:
        raise InputError(f"{where}: {key} must be {expected.__name__}, got {value!r}")


def _check_keys(where, table, allowed, what="setting"):
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise InputError(f"{where}: unknown {what}(s) {', '.join(unknown)}")


def _per_stage(where, key, value, num_stages, item_type):
    """Return value as a list with one entry per upstream stage."""
    values = value if isinstance(value, list) else [value] * num_stages
    if len(values) != num_stages:
        raise InputError(f"{where}: {key} has {len(values)} entries, expected {num_stages}")
    for v in values:
        _check_type(where, key, v, item_type)
    return values


def _read_upstream(where, table):
    """Validate an [upstream.<name>] table and return its stages as lists (one of pressure
    and density is None for every stage)."""
    _check_keys(where, table, UPSTREAM_SETTINGS)
    for key in ("material", "length"):
        if key not in table:
            raise InputError(f"{where}: missing {key}")
    states = [key for key in STATE_SETTINGS if key in table]
    if len(states) != 1:
        raise InputError(f"{where}: give pressure or density for the stages (one of them)")
    state = states[0]
    num_stages = len(table[state]) if isinstance(table[state], list) else 0
    if num_stages == 0:
        raise InputError(f"{where}: {state} must be a non-empty list")
    stages = {}
    for key, (item_type, default) in UPSTREAM_SETTINGS.items():
        if key in STATE_SETTINGS and key != state:
            stages[key] = [None] * num_stages
            continue
        item_type = float if item_type is list else item_type
        stages[key] = _per_stage(where, key, table.get(key, default), num_stages, item_type)
    return stages


def _read_material(where, name, table):
    """Validate a [materials.<name>] table and return its Material."""
    _check_keys(where, table, MATERIAL_SETTINGS)
    if "phase" not in table:
        raise InputError(f"{where}: missing phase")
    for key, expected in MATERIAL_SETTINGS.items():
        if key in table:
            _check_type(where, key, table[key], expected)
    if table["phase"] not in PHASES:
        raise InputError(f"{where}: phase must be one of {', '.join(PHASES)}")
    for key in ("atoms", "molecules"):
        for formula, amount in table.get(key, {}).items():
            _check_type(where, f"{key} {formula}", amount, float)
    try:
        return Material(
            name,
            table["phase"],
            atoms=table.get("atoms"),
            molecules=table.get("molecules"),
            density=table.get("density"),
        )
    except ValueError as e:
        raise InputError(f"{where}: {str(e).split(': ', 1)[-1]}") from None


def _choose(where, s, case, keys, needed, what):
    """Return the keys of s to use when a case must give `needed` of `keys`: if s (case plus
    [defaults]) has too many, the ones the case sets itself are used."""
    given = [key for key in keys if key in s]
    if len(given) > needed:
        given = [key for key in keys if key in case]
        if len(given) > needed:
            raise InputError(f"{where}: {', '.join(given)} are all set; give {what}")
    if len(given) != needed:
        raise InputError(f"{where}: give {what}; got {', '.join(given) or 'none'}")
    return given


def load_input(path):
    """Read a TOML input file and return {case name: settings}.

    The file has a [cases.<name>] table for each case, an optional [defaults] table whose
    settings apply to every case unless the case overrides them, optional [upstream.<name>]
    tables describing gas regions (such as differential-pumping stages) that the beam passes
    through before the target chamber, and [materials.<name>] tables describing the materials
    that fill them. See CASE_SETTINGS, TEMPERATURE_SETTING, TARGET_SETTINGS, BEAM_SETTINGS,
    OPTIONAL_CASE_SETTINGS, UPSTREAM_SETTINGS and MATERIAL_SETTINGS for the settings and their
    units, and examples/generic.toml for an example.

    A gas target chamber needs two of its state (pressure with temperature, or density),
    length and beam_energy_backstop; the third is calculated (for a solid target chamber, one
    of length and beam_energy_backstop). The state is set in each case, never in [defaults]:
    with none given, a case with a temperature calculates its pressure and one without
    calculates its density. If a case and [defaults] together give too many, the ones the
    case sets itself are used. Likewise each case needs one of current and yield_total, and
    if both are given, the one the case sets itself is used.

    Each returned settings dict has "source", the keyword arguments for ShineSource
    (without name), and "direction", the MCNP SDEF beam direction.
    """
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise InputError(f"{path}: {e}") from None
    _check_keys(path, data, ["defaults", "upstream", "materials", "cases"], what="table")
    allowed = (
        list(CASE_SETTINGS)
        + [TEMPERATURE_SETTING]
        + list(TARGET_SETTINGS)
        + list(BEAM_SETTINGS)
        + list(OPTIONAL_CASE_SETTINGS)
    )

    defaults = data.get("defaults", {})
    _check_keys(f"{path} [defaults]", defaults, allowed)
    in_defaults = [key for key in CASE_ONLY_SETTINGS if key in defaults]
    if in_defaults:
        raise InputError(
            f"{path} [defaults]: {', '.join(in_defaults)} must be set in each case (the gas "
            "state is never a default)"
        )

    materials = {
        name: _read_material(f"{path} [materials.{name}]", name, table)
        for name, table in data.get("materials", {}).items()
    }
    upstreams = {
        name: _read_upstream(f"{path} [upstream.{name}]", table)
        for name, table in data.get("upstream", {}).items()
    }
    for name, up in upstreams.items():
        for material in up["material"]:
            if material not in materials:
                raise InputError(f"{path} [upstream.{name}]: no [materials.{material}] table")
            if materials[material].phase != "gas":
                raise InputError(
                    f"{path} [upstream.{name}]: material {material} is a solid; upstream "
                    "stages hold gas"
                )

    cases = data.get("cases", {})
    if not cases:
        raise InputError(f"{path}: no [cases.<name>] tables")

    resolved = {}
    for name, case in cases.items():
        where = f"{path} [cases.{name}]"
        if not CASE_NAME_PATTERN.match(name):
            raise InputError(f"{where}: case names may only use letters, digits, _ . and -")
        _check_keys(where, case, allowed)
        s = {**OPTIONAL_CASE_SETTINGS, **defaults, **case}
        missing = [key for key in CASE_SETTINGS if key not in s]
        if missing:
            raise InputError(f"{where}: missing {', '.join(missing)}")
        for key, expected in CASE_SETTINGS.items():
            _check_type(where, key, s[key], expected)
        # Target chamber material and state
        if s["material"] not in materials:
            raise InputError(f"{where}: no [materials.{s['material']}] table")
        target_material = materials[s["material"]]
        solid = target_material.phase == "solid"
        case_state = [key for key in STATE_SETTINGS if key in case]
        if len(case_state) > 1:
            raise InputError(f"{where}: give pressure or density, not both")
        if solid:
            for key in STATE_SETTINGS:
                if key in case:
                    raise InputError(f"{where}: {key} does not apply to a solid target chamber")
            keys = ["length", "beam_energy_backstop"]
            what = "one of length and beam_energy_backstop (the target is a solid)"
            target_state = None
        else:
            if case_state:
                target_state = case_state[0]
            else:  # State calculated: a pressure if the case has a temperature
                target_state = "pressure" if TEMPERATURE_SETTING in case else "density"
            if target_state == "density" and TEMPERATURE_SETTING in case:
                raise InputError(
                    f"{where}: temperature goes with pressure; a gas given by density has none"
                )
            keys = [target_state, "length", "beam_energy_backstop"]
            if case_state:
                what = f"two of {target_state}, length and beam_energy_backstop"
            else:
                what = (
                    "two of the gas state (pressure with temperature, or density), length and "
                    "beam_energy_backstop"
                )
        given = _choose(where, s, case, keys, len(keys) - 1, what + "; the rest is calculated")
        for key in given:
            _check_type(where, key, s[key], TARGET_SETTINGS[key])
        target = {key: s[key] if key in given else None for key in TARGET_SETTINGS}
        given = _choose(where, s, case, list(BEAM_SETTINGS), 1, "one of current and yield_total")
        _check_type(where, given[0], s[given[0]], float)
        beam = {key: s[key] if key in given else None for key in BEAM_SETTINGS}
        if s["direction"] not in SDEF_DIRECTIONS:
            raise InputError(
                f"{where}: direction must be one of {', '.join(SDEF_DIRECTIONS)}, "
                f"got {s['direction']!r}"
            )
        entry = s["entry_position"]
        if not (
            isinstance(entry, list)
            and len(entry) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in entry)
        ):
            raise InputError(
                f"{where}: entry_position must be a list of three numbers [x, y, z], got {entry!r}"
            )
        if s["stopping_power"] not in STOPPING_POWERS:
            raise InputError(
                f"{where}: stopping_power must be one of {', '.join(STOPPING_POWERS)}, "
                f"got {s['stopping_power']!r}"
            )

        # Regions: upstream stages (if any), then the target chamber
        up = {key: [] for key in UPSTREAM_SETTINGS}
        if s["upstream"] is not None and s["upstream"] is not False:
            if not isinstance(s["upstream"], str):
                raise InputError(
                    f"{where}: upstream must be the name of an [upstream.<name>] table, or "
                    f"false for none; got {s['upstream']!r}"
                )
            if s["upstream"] not in upstreams:
                raise InputError(f"{where}: no [upstream.{s['upstream']}] table")
            up = upstreams[s["upstream"]]
        region_materials = [materials[m] for m in up["material"]] + [target_material]
        pressure = up["pressure"] + [target["pressure"]]
        density = up["density"] + [target["density"]]
        by_pressure = [p is not None for p in up["pressure"]] + [target_state == "pressure"]

        # The temperature, where a gas is given by pressure
        if any(by_pressure):
            if TEMPERATURE_SETTING not in s:
                raise InputError(f"{where}: missing temperature (for a gas given by pressure)")
            _check_type(where, TEMPERATURE_SETTING, s[TEMPERATURE_SETTING], float)
        elif TEMPERATURE_SETTING in case:
            raise InputError(f"{where}: temperature does not apply (no gas given by pressure)")
        temperature = [s[TEMPERATURE_SETTING] if b else None for b in by_pressure]
        resolved[name] = {
            "source": {
                "voltage": s["voltage"],
                "current": beam["current"],
                "pressure": pressure,
                "temperature": temperature,
                "density": density,
                "length": up["length"] + [target["length"]],
                "start_radius": s["start_radius"],
                "wall_radius": s["wall_radius"],
                "divergence": up["divergence"] + [s["divergence"]],
                "num_x_bins": up["num_x_bins"] + [s["num_x_bins"]],
                "num_x_bins_out": up["num_x_bins_out"] + [s["num_x_bins_out"]],
                "angle_step": s["angle_step"],
                "angle_step_out": s["angle_step_out"],
                "energy_bin_width": s["energy_bin_width"],
                "stopping_power": s["stopping_power"],
                "beam_energy_backstop": target["beam_energy_backstop"],
                "yield_total": beam["yield_total"],
                "material": region_materials,
            },
            "direction": s["direction"],
            "entry_position": [float(v) for v in entry],
        }
    return resolved


def setup_logging(verbose=False):
    """Show this package's log messages on stderr: progress and one line per case, plus each
    case's settings and detailed results if verbose. Safe to call more than once."""
    package_logger = logging.getLogger("shine_neutron_source")
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(getattr(h, "_shine_cli", False) for h in package_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._shine_cli = True
        package_logger.addHandler(handler)


def run_case(name, settings, output_dir="output", input_file=None):
    """Calculate one case from load_input() and write its outputs to <output_dir>/<name>/.

    Writes source.h5, source.sdef, plots/*.png, summary.toml and the integrated profiles
    axial_profile.txt, energy_spectrum.txt and angular_distribution.txt (see
    ShineSource.write_text_files), and returns the ShineSource. input_file, if given, is
    recorded in the summary and the SDEF header.
    """
    t0 = time.time()
    s = settings["source"]
    if s["current"] is None:
        beam = f"yield {s['yield_total']:.4e} n/s"
    else:
        beam = f"{s['current']:g} mA"
    logger.debug(
        f"{name}: starting: {s['voltage']:g} kV, {beam}, {len(s['pressure'])} region(s), "
        f"target {s['material'][-1].name}, stopping power {s['stopping_power']}"
    )
    source = ShineSource(name=name, **s)
    case_dir = os.path.join(output_dir, name)
    run_info = {} if input_file is None else {"input_file": str(input_file)}
    direction, entry = settings["direction"], settings["entry_position"]
    source.write_hdf5(os.path.join(case_dir, "source.h5"), run_info)
    source.write_mcnp_sdef(os.path.join(case_dir, "source.sdef"), direction, run_info, entry)
    source.make_plots(os.path.join(case_dir, "plots"))
    sdef_info = {"sdef_direction": direction, "sdef_entry_position": entry}
    source.write_text_files(case_dir, {**run_info, **sdef_info})
    logger.info(
        f"{name}: {source.yield_total:.3e} n/s in {time.time() - t0:.1f} s "
        f"-> {os.path.join(case_dir, '')}"  # Trailing separator: a directory
    )
    return source


def _run_case_in_worker(args):
    """Run a case in a worker process. Returns nothing, so the large source object is not
    copied back to the parent process."""
    run_case(*args)


def main(argv=None):
    """Command-line interface: run the cases in an input file."""
    parser = argparse.ArgumentParser(
        prog="shine-neutron-source",
        description="Calculate neutron source terms for the cases in a TOML input file. "
        "Each case is written to <output-dir>/<case>/: source.h5, source.sdef, plots/, "
        "summary.toml, axial_profile.txt, energy_spectrum.txt and angular_distribution.txt. "
        "Memory use is roughly 2.5 GB per case at 1370 position "
        "bins and a 0.01 degree angle step, and scales with both.",
    )
    parser.add_argument("input", help="TOML input file (see examples/generic.toml)")
    parser.add_argument(
        "-c",
        "--case",
        action="append",
        dest="cases",
        metavar="NAME",
        help="run only this case (repeat for several; default: all cases)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="output", help="directory for results (default: output)"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=mp.cpu_count(),
        help="number of cases to run in parallel (default: number of CPUs)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also show each case's settings and detailed results",
    )
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    args = parser.parse_args(argv)

    try:
        cases = load_input(args.input)
    except (InputError, OSError) as e:
        parser.exit(2, f"error: {e}\n")
    if args.list:
        print("\n".join(cases))
        return
    if args.cases:
        unknown = [name for name in args.cases if name not in cases]
        if unknown:
            parser.exit(2, f"error: no case(s) named {', '.join(unknown)} in {args.input}\n")
        cases = {name: cases[name] for name in args.cases}

    setup_logging(args.verbose)
    t0 = time.time()
    jobs = min(args.jobs, len(cases))
    logger.info(
        f"Running {len(cases)} case(s) from {args.input} with {jobs} process(es); "
        f"results in {os.path.join(args.output_dir, '')}"  # Trailing separator: a directory
    )
    if jobs > 1:
        work = [(name, s, args.output_dir, args.input) for name, s in cases.items()]
        # Each worker sets up logging; close() and join() let workers exit normally (the
        # context manager alone would terminate them, which can lose their last output)
        with mp.Pool(jobs, initializer=setup_logging, initargs=(args.verbose,)) as pool:
            pool.map(_run_case_in_worker, work)
            pool.close()
            pool.join()
    else:
        for name, s in cases.items():
            run_case(name, s, args.output_dir, args.input)
    logger.info(f"Done: {len(cases)} case(s) in {time.time() - t0:.1f} s")
