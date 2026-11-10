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

from .source import SDEF_DIRECTIONS, STOPPING_POWERS, ShineSource

logger = logging.getLogger(__name__)

# Case settings: name -> required type. Every case needs all of them, either directly or
# from [defaults].
CASE_SETTINGS = {
    "T_frac": float,  # Atom fraction of tritium in the gas
    "voltage": float,  # Beam voltage [kV]
    "temperature": float,  # Gas temperature [degC], used in every region
    "start_radius": float,  # Beam radius where it enters the first region [cm]
    "wall_radius": float,  # Target chamber wall radius [cm]
    "divergence": float,  # Beam divergence in the target chamber [mrad]
    "num_x_bins": int,  # Position bins in the target chamber (internal mesh)
    "num_x_bins_out": int,  # Position bins in the target chamber (output mesh)
    "angle_step": float,  # Width of the emission-angle bins [deg] (internal mesh)
    "angle_step_out": float,  # Width of the emission-angle bins [deg] (output mesh)
    "energy_bin_width": float,  # Width of the neutron energy bins [MeV] (output mesh)
}

# Target chamber settings: every case needs exactly two of these (directly or from
# [defaults]), and the third is calculated
TARGET_SETTINGS = {
    "pressure": float,  # Target chamber pressure [torr]
    "length": float,  # Target chamber length [cm]
    "beam_energy_backstop": float,  # Beam energy at the backstop [keV]
}

# Beam settings: every case needs exactly one of these (directly or from [defaults]), and
# the other is calculated
BEAM_SETTINGS = {
    "current": float,  # Beam current [mA]
    "yield_total": float,  # Total neutron yield [n/s]
}

# Optional case settings and their defaults
OPTIONAL_CASE_SETTINGS = {
    "direction": "-z",  # MCNP SDEF beam direction, see SDEF_DIRECTIONS
    "upstream": None,  # Name of an [upstream.<name>] table to put before the target chamber
    "stopping_power": "srim",  # Stopping-power set, see STOPPING_POWERS
    "entry_position": [0.0, 0.0, 0.0],  # cm, where the beam enters (MCNP coordinates)
}

# Upstream stage settings: name -> (type, default). None means required. num_x_bins,
# num_x_bins_out and divergence can be one value for every stage or a list with one per stage.
UPSTREAM_SETTINGS = {
    "pressure": (list, None),  # Pressure of each stage [torr]
    "length": (list, None),  # Length of each stage [cm]
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
    """Validate an [upstream.<name>] table and return its stages as lists."""
    _check_keys(where, table, UPSTREAM_SETTINGS)
    for key, (_, default) in UPSTREAM_SETTINGS.items():
        if default is None and key not in table:
            raise InputError(f"{where}: missing {key}")
    num_stages = len(table["pressure"]) if isinstance(table["pressure"], list) else 0
    if num_stages == 0:
        raise InputError(f"{where}: pressure must be a non-empty list")
    stages = {}
    for key, (item_type, default) in UPSTREAM_SETTINGS.items():
        item_type = float if item_type is list else item_type
        stages[key] = _per_stage(where, key, table.get(key, default), num_stages, item_type)
    return stages


def load_input(path):
    """Read a TOML input file and return {case name: settings}.

    The file has a [cases.<name>] table for each case, an optional [defaults] table whose
    settings apply to every case unless the case overrides them, and optional
    [upstream.<name>] tables describing gas regions (such as differential-pumping stages)
    that the beam passes through before the target chamber. See CASE_SETTINGS,
    TARGET_SETTINGS, BEAM_SETTINGS, OPTIONAL_CASE_SETTINGS and UPSTREAM_SETTINGS for the
    settings and their units, and examples/generic.toml for an example.

    Each case needs two of pressure, length and beam_energy_backstop; the third is
    calculated. If a case and [defaults] together give all three, the two the case sets
    itself are used. Likewise each case needs one of current and yield_total, and if both
    are given, the one the case sets itself is used.

    Each returned settings dict has "source", the keyword arguments for ShineSource
    (without name), and "direction", the MCNP SDEF beam direction.
    """
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise InputError(f"{path}: {e}") from None
    _check_keys(path, data, ["defaults", "upstream", "cases"], what="table")
    allowed = (
        list(CASE_SETTINGS)
        + list(TARGET_SETTINGS)
        + list(BEAM_SETTINGS)
        + list(OPTIONAL_CASE_SETTINGS)
    )

    defaults = data.get("defaults", {})
    _check_keys(f"{path} [defaults]", defaults, allowed)

    upstreams = {
        name: _read_upstream(f"{path} [upstream.{name}]", table)
        for name, table in data.get("upstream", {}).items()
    }

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
        given = [key for key in TARGET_SETTINGS if key in s]
        if len(given) == 3:  # The case's own two override the third from [defaults]
            given = [key for key in TARGET_SETTINGS if key in case]
            if len(given) != 2:
                raise InputError(
                    f"{where}: pressure, length and beam_energy_backstop are all set; give "
                    "two of them (the third is calculated)"
                )
        if len(given) != 2:
            raise InputError(
                f"{where}: give two of pressure, length and beam_energy_backstop (the third "
                f"is calculated); got {', '.join(given) or 'none'}"
            )
        for key in given:
            _check_type(where, key, s[key], TARGET_SETTINGS[key])
        target = {key: s[key] if key in given else None for key in TARGET_SETTINGS}
        given = [key for key in BEAM_SETTINGS if key in s]
        if len(given) == 2:  # The case's own setting overrides the other from [defaults]
            given = [key for key in BEAM_SETTINGS if key in case]
            if len(given) != 1:
                raise InputError(
                    f"{where}: current and yield_total are both set; give one of them (the "
                    "other is calculated)"
                )
        if len(given) != 1:
            raise InputError(
                f"{where}: give one of current and yield_total (the other is calculated)"
            )
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

        # Gas regions: upstream stages (if any), then the target chamber
        up = {key: [] for key in UPSTREAM_SETTINGS}
        if s["upstream"] is not None:
            if s["upstream"] not in upstreams:
                raise InputError(f"{where}: no [upstream.{s['upstream']}] table")
            up = upstreams[s["upstream"]]
        pressure = up["pressure"] + [target["pressure"]]
        resolved[name] = {
            "source": {
                "T_frac": s["T_frac"],
                "voltage": s["voltage"],
                "current": beam["current"],
                "pressure": pressure,
                "temperature": [s["temperature"]] * len(pressure),
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
        f"{name}: starting: T_frac {s['T_frac']:g}, {s['voltage']:g} kV, {beam}, "
        f"{len(s['pressure'])} gas region(s), stopping power {s['stopping_power']}"
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
