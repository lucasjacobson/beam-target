"""Input files and the command-line interface.

Run the cases in a TOML input file with either of:

    beam-target examples/generic.toml [-c CASE] [-o OUTPUT_DIR] [-j JOBS] [-v]
    python -m beamtarget examples/generic.toml ...

See load_input for the file format and examples/generic.toml for an example.
"""

import argparse
import logging
import multiprocessing as mp
import os
import re
import time
import tomllib

from .config import DROSG, ConfigError, load_config, resolve_nuclear_data
from .nuclear import BEAM_NAMES, BEAMS, beam_key
from .parallel import set_parallel
from .source import (
    PHASES,
    POSITION_SUBSTEPS,
    SDEF_DIRECTIONS,
    STOPPING_POWERS,
    BeamTarget,
    Material,
    Region,
)

logger = logging.getLogger(__name__)

# Case settings: name -> required type. Every case needs all of them, either directly or
# from [defaults].
CASE_SETTINGS = {
    "regions": list,  # The regions the beam passes through, in order (see REGION_SETTINGS)
    "beam_energy_entry": float,  # Beam energy where it enters the first region [keV]
    "beam_radius_entry": float,  # Beam radius where it enters the first region [cm]
    "angle_bin_width": float,  # Width of the emission-angle bins [deg] (output mesh)
    "energy_bin_width": float,  # Width of the neutron energy bins [MeV] (output mesh)
}

# Settings that were renamed: old name -> what to do. An error, not a warning: the old
# angle_bin_width was a fine internal mesh (such as 0.01 deg), which must not be taken for the
# output bin width.
RENAMED_SETTINGS = {
    "angle_bin_width_out": "it is now angle_bin_width; the old angle_bin_width (the internal "
    "angle mesh) is no longer used, as two-body lines are binned exactly onto the output mesh: "
    "remove it, then rename angle_bin_width_out to angle_bin_width",
    "position_bins_out": "it is now position_bins (the output bins), and the internal mesh is "
    "set by position_substeps (steps per output bin, default 10): remove the old "
    "position_bins (the internal count), then rename position_bins_out to position_bins and, "
    "if the old ratio was not 10, set position_substeps to it",
}

# Beam settings: every case needs exactly one of these (directly or from [defaults]), and
# the other is calculated
BEAM_SETTINGS = {
    "beam_current": float,  # Beam current [mA]
    "yield_total": float,  # Total neutron yield [n/s]
}

# Optional case settings and their defaults
OPTIONAL_CASE_SETTINGS = {
    # Beam particle, see nuclear.BEAM_NAMES: "proton" or "1H", "deuteron" or "2H", "triton" or
    # "3H", "helion" or "3He", "alpha" or "4He"
    "beam": "deuteron",
    # Beam energy at the backstop [keV]; give it to calculate the one region value left out
    "beam_energy_backstop": None,
    "beam_direction": "-z",  # Beam direction in MCNP and OpenMC, see SDEF_DIRECTIONS
    "stopping_power": "srim",  # Stopping-power set, see STOPPING_POWERS
    # Nuclear-data libraries, searched in order for each target isotope: "drosg" (the built-in
    # two-body reactions), names of libraries defined in the user's configuration file (see
    # config.py), or directories of evaluated files (relative to the input file's directory).
    # None: the configuration's [defaults] nuclear_data, or ["drosg"].
    "nuclear_data": None,
    # Steps the beam is followed in across each position bin, for every region that does not
    # set its own position_substeps
    "position_substeps": POSITION_SUBSTEPS,
}

# MCNP SDEF placement: give at most one (with neither, the beam enters at the origin). If a
# case and [defaults] give both, the one the case sets itself is used.
PLACEMENT_SETTINGS = {
    "entry_position": list,  # [x, y, z] cm, where the beam enters the first region
    "backstop_position": list,  # [x, y, z] cm, where it reaches the backstop, instead
}

# Region settings: name -> (type, default); None means required. A region is a
# [regions.<name>] table, or an inline table in a case's regions list, which may start from
# a named region with use = "<name>" and change some of its settings (setting = false leaves
# one of the named region's settings out, for example to calculate it). A gas gives pressure
# with temperature, or density; a solid none of them. Leave out one region's length,
# pressure (keeping its temperature) or density, and give beam_energy_backstop, to have it
# calculated.
REGION_SETTINGS = {
    "material": (str, None),  # Name of the [materials.<name>] table filling the region
    "wall_radius": (float, None),  # Radius of the wall around the region [cm]
    "position_bins": (int, None),  # Position bins (output mesh)
    "length": (float, ""),  # Length along the beam [cm]
    "pressure": (float, ""),  # Gas pressure [torr], with temperature
    "temperature": (float, ""),  # Gas temperature [degC], with pressure
    "density": (float, ""),  # Gas density [g/cm3], instead of pressure and temperature
    "divergence": (float, 0.0),  # Beam divergence [mrad]
    # Steps the beam is followed in across each position bin (the internal mesh)
    "position_substeps": (int, POSITION_SUBSTEPS),
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
    for key in sorted(set(table) & set(RENAMED_SETTINGS)):
        raise InputError(f"{where}: {key}: {RENAMED_SETTINGS[key]}")
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise InputError(f"{where}: unknown {what}(s) {', '.join(unknown)}")


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


def _make_region(where, settings, materials, name="", substeps=POSITION_SUBSTEPS):
    """Validate region settings (a dict) and return the Region; substeps is its
    position_substeps unless it sets its own."""
    _check_keys(where, settings, REGION_SETTINGS)
    values = {}
    for key, (expected, default) in REGION_SETTINGS.items():
        if key == "position_substeps":
            default = substeps
        if key not in settings:
            if default is None:
                raise InputError(f"{where}: missing {key}")
            values[key] = None if default == "" else default
            continue
        _check_type(where, key, settings[key], expected)
        values[key] = settings[key]
    if values["material"] not in materials:
        raise InputError(f"{where}: no [materials.{values['material']}] table")
    values["material"] = materials[values["material"]]
    return Region(**values, name=name)


def _read_regions(where, items, named, materials, substeps=POSITION_SUBSTEPS):
    """Return the Regions of a case's regions list: names of [regions.<name>] tables, or inline
    tables (optionally starting from one with use = "<name>"). substeps is the case's
    position_substeps, for the regions that do not set their own."""
    if not items:
        raise InputError(f"{where}: regions must list at least one region")
    regions = []
    for i, item in enumerate(items):
        here = f"{where} regions[{i}]"
        if isinstance(item, str):
            if item not in named:
                raise InputError(f"{here}: no [regions.{item}] table")
            regions.append(_make_region(f"{here} ({item})", named[item], materials, item, substeps))
        elif isinstance(item, dict):
            settings = dict(item)
            base, name = {}, ""
            if "use" in settings:
                name = settings.pop("use")
                if not isinstance(name, str) or name not in named:
                    raise InputError(
                        f"{here}: use = {name!r}: no [regions.<name>] table of that name"
                    )
                base = named[name]
            # false leaves out a setting of the named region (to calculate it, for example)
            merged = {**base, **settings}
            merged = {k: v for k, v in merged.items() if not (v is False and k in base)}
            regions.append(_make_region(here, merged, materials, name, substeps))
        else:
            raise InputError(
                f"{here}: each region is the name of a [regions.<name>] table or an inline "
                f"table, got {item!r}"
            )
    return regions


def _read_nuclear_data(where, entries, input_path, config):
    """Check a nuclear_data list (None: the configuration's default) and return the entries
    to give BeamTarget: "drosg" and library names (from the configuration) as they are,
    directories made absolute (relative ones are taken from the input file's directory)."""
    if entries is None:
        return None
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, list) or not entries or not all(isinstance(e, str) for e in entries):
        raise InputError(
            f'{where}: nuclear_data must be a list of "drosg", library names and directories, '
            f"got {entries!r}"
        )
    base = os.path.dirname(os.path.abspath(input_path))
    try:
        _, directories = resolve_nuclear_data(entries, base_dir=base, config=config)
    except ConfigError as e:
        raise InputError(f"{where}: {e}") from None
    return [
        entry if entry == DROSG or entry in config["libraries"] else directory
        for entry, directory in zip(entries, directories)
    ]


def load_input(path):
    """Read a TOML input file and return {case name: settings}.

    The file has a [cases.<name>] table for each case, an optional [defaults] table whose
    settings apply to every case unless the case sets them itself, [materials.<name>] tables
    describing materials (see MATERIAL_SETTINGS) and optional [regions.<name>] tables
    describing regions that cases can use by name (see REGION_SETTINGS). Each case lists the
    regions the beam passes through, in order, in its regions setting: names of
    [regions.<name>] tables, or inline tables (which may start from a named one with
    use = "<name>"). See CASE_SETTINGS, BEAM_SETTINGS, and OPTIONAL_CASE_SETTINGS for the case
    settings and their units, and examples/generic.toml for an example.

    Leave exactly one value to be calculated: beam_energy_backstop, or one region's length,
    pressure (keeping its temperature) or density, with beam_energy_backstop given. Each case
    also needs one of beam_current and yield_total; if the case and [defaults] give both, the one
    the case sets itself is used.

    Each returned settings dict has "source", the keyword arguments for BeamTarget
    (without name), "beam_direction", the beam direction in MCNP and OpenMC, and
    "entry_position" and "backstop_position" (one of them None, or both for the default
    placement).
    """
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise InputError(f"{path}: {e}") from None
    _check_keys(path, data, ["defaults", "materials", "regions", "cases"], what="table")
    allowed = (
        list(CASE_SETTINGS)
        + list(BEAM_SETTINGS)
        + list(OPTIONAL_CASE_SETTINGS)
        + list(PLACEMENT_SETTINGS)
    )

    defaults = data.get("defaults", {})
    _check_keys(f"{path} [defaults]", defaults, allowed)

    materials = {
        name: _read_material(f"{path} [materials.{name}]", name, table)
        for name, table in data.get("materials", {}).items()
    }
    named_regions = data.get("regions", {})
    for name, table in named_regions.items():  # Check each once, even if no case uses it
        _make_region(f"{path} [regions.{name}]", table, materials, name)

    cases = data.get("cases", {})
    if not cases:
        raise InputError(f"{path}: no [cases.<name>] tables")

    try:
        config = load_config()
    except ConfigError as e:
        raise InputError(str(e)) from None
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
        _check_type(where, "position_substeps", s["position_substeps"], int)
        regions = _read_regions(
            where, s["regions"], named_regions, materials, s["position_substeps"]
        )

        given = [key for key in BEAM_SETTINGS if key in s]
        if len(given) > 1:  # The case's own setting overrides the other from [defaults]
            given = [key for key in BEAM_SETTINGS if key in case]
            if len(given) != 1:
                raise InputError(f"{where}: beam_current and yield_total are both set; give one")
        if len(given) != 1:
            raise InputError(
                f"{where}: give one of beam_current and yield_total (the other is calculated)"
            )
        _check_type(where, given[0], s[given[0]], float)
        rate = {key: s[key] if key in given else None for key in BEAM_SETTINGS}

        if s["beam"] not in BEAM_NAMES:
            names = ", ".join(f'"{n}"' for n in BEAM_NAMES)
            raise InputError(f"{where}: beam must be one of {names}, got {s['beam']!r}")
        if s["beam_energy_backstop"] is not None:
            _check_type(where, "beam_energy_backstop", s["beam_energy_backstop"], float)
        if s["beam_direction"] not in SDEF_DIRECTIONS:
            raise InputError(
                f"{where}: beam_direction must be one of {', '.join(SDEF_DIRECTIONS)}, "
                f"got {s['beam_direction']!r}"
            )
        placement = [key for key in PLACEMENT_SETTINGS if key in s]
        if len(placement) > 1:  # The case's own setting overrides the other from [defaults]
            placement = [key for key in PLACEMENT_SETTINGS if key in case]
            if len(placement) != 1:
                raise InputError(f"{where}: give entry_position or backstop_position, not both")
        positions = {key: None for key in PLACEMENT_SETTINGS}
        for key in placement:
            point = s[key]
            if not (
                isinstance(point, list)
                and len(point) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point)
            ):
                raise InputError(
                    f"{where}: {key} must be a list of three numbers [x, y, z], got {point!r}"
                )
            positions[key] = [float(v) for v in point]
        if s["stopping_power"] not in STOPPING_POWERS:
            raise InputError(
                f"{where}: stopping_power must be one of {', '.join(STOPPING_POWERS)}, "
                f"got {s['stopping_power']!r}"
            )
        nuclear_data = _read_nuclear_data(where, s["nuclear_data"], path, config)
        resolved[name] = {
            "source": {
                "regions": regions,
                "beam": s["beam"],
                "beam_energy_entry": s["beam_energy_entry"],
                "beam_current": rate["beam_current"],
                "yield_total": rate["yield_total"],
                "beam_radius_entry": s["beam_radius_entry"],
                "angle_bin_width": s["angle_bin_width"],
                "energy_bin_width": s["energy_bin_width"],
                "stopping_power": s["stopping_power"],
                "beam_energy_backstop": s["beam_energy_backstop"],
                "nuclear_data": nuclear_data,
            },
            "beam_direction": s["beam_direction"],
            **positions,
        }
    return resolved


def setup_logging(verbose=False):
    """Show this package's log messages on stderr: progress and one line per case, plus each
    case's settings and detailed results if verbose. Safe to call more than once."""
    package_logger = logging.getLogger("beamtarget")
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(getattr(h, "_beamtarget_cli", False) for h in package_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._beamtarget_cli = True
        package_logger.addHandler(handler)


def run_case(name, settings, output_dir="output", input_file=None, name_width=0, plots=True):
    """Calculate one case from load_input() and write its outputs to <output_dir>/<name>/.

    Writes source.h5, source.sdef, summary.toml, the profiles, and (unless plots is False)
    plots/*.png (see docs/outputs.md), and returns the BeamTarget. input_file, if given, is recorded in the
    summary and the SDEF header. The case's result line pads its name to name_width
    characters (the longest name of the run), so the lines line up.
    """
    t0 = time.time()
    s = settings["source"]
    if s["beam_current"] is None:
        beam = f"yield {s['yield_total']:.4e} n/s"
    else:
        beam = f"{s['beam_current']:g} mA"
    logger.debug(
        f"{name}: starting: {BEAMS[beam_key(s['beam'])].description} beam, "
        f"{s['beam_energy_entry']:g} keV, {beam}, "
        f"{len(s['regions'])} region(s), "
        f"last {s['regions'][-1].material.name}, stopping power {s['stopping_power']}"
    )
    source = BeamTarget(name=name, **s)
    case_dir = os.path.join(output_dir, name)
    run_info = {} if input_file is None else {"input_file": str(input_file)}
    direction = settings["beam_direction"]
    entry, backstop = source.sdef_positions(
        direction, settings["entry_position"], settings["backstop_position"]
    )
    source.write_hdf5(os.path.join(case_dir, "source.h5"), run_info, direction, entry)
    source.write_mcnp_sdef(os.path.join(case_dir, "source.sdef"), direction, run_info, entry)
    if plots:
        source.make_plots(os.path.join(case_dir, "plots"))
    sdef_info = {
        "sdef_direction": direction,
        "sdef_entry_position": entry,
        "sdef_backstop_position": backstop,
    }
    source.write_text_files(case_dir, {**run_info, **sdef_info})
    # Fixed widths, so the lines of a run line up
    logger.info(
        f"{name + ':':<{name_width + 1}} {source.yield_total:.3e} n/s, mean energy "
        f"{source.mean_neutron_energy():5.2f} MeV, in {time.time() - t0:6.1f} s "
        f"-> {os.path.join(case_dir, '')}"  # Trailing separator: a directory
    )
    return source


def _start_worker(verbose, cores, tokens):
    """Set up a worker process: logging, and up to cores threads within each case, sharing
    the cores with the other workers through tokens (see parallel.set_parallel)."""
    setup_logging(verbose)
    set_parallel(cores, tokens)


def _expected_cost(settings):
    """Rough relative cost of a case, to start the slowest first: the beam energy (evaluated
    continua, the slow part, cost more at higher energies; the built-in reactions are
    cheap and run at keV energies)."""
    return settings["source"]["beam_energy_entry"]


def _run_case_in_worker(args):
    """Run a case in a worker process. Returns nothing, so the large source object is not
    copied back to the parent process."""
    run_case(*args)


def main(argv=None):
    """Command-line interface: run the cases in an input file."""
    parser = argparse.ArgumentParser(
        prog="beam-target",
        description="Calculate neutron source terms for the cases in a TOML input file. "
        "Each case is written to <output-dir>/<case>/: source.h5, source.sdef, plots/, "
        "summary.toml, axial_profile.txt, energy_spectrum.txt and angular_distribution.txt.",
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
        help="number of CPU cores to use (default: all of them). Cases run in parallel, up "
        "to this many at once, and share the cores: each uses several for its evaluated "
        "continua, and a case still running when others finish takes over their cores",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also show each case's settings and detailed results",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="skip the plots (about 2 s per case)"
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
    cores = max(1, args.jobs)
    processes = min(cores, len(cases))
    logger.info(
        f"Running {len(cases)} case(s) from {args.input} on {cores} core(s), "
        f"{processes} at a time; results in {os.path.join(args.output_dir, '')}"
    )
    width = max(len(name) for name in cases)
    # The likely slowest first, so that none starts last and holds up the end
    order = sorted(cases, key=lambda name: -_expected_cost(cases[name]))
    if processes > 1:
        work = [
            (name, cases[name], args.output_dir, args.input, width, not args.no_plots)
            for name in order
        ]
        # One token per core, shared by the workers (see parallel.set_parallel). Each worker
        # sets up logging; close() and join() let workers exit normally (the context manager
        # alone would terminate them, which can lose their last output)
        tokens = mp.Semaphore(cores)
        with mp.Pool(
            processes, initializer=_start_worker, initargs=(args.verbose, cores, tokens)
        ) as pool:
            pool.map(_run_case_in_worker, work, chunksize=1)
            pool.close()
            pool.join()
    else:
        set_parallel(cores)
        for name in order:
            run_case(name, cases[name], args.output_dir, args.input, width, not args.no_plots)
    logger.info(f"Done: {len(cases)} case(s) in {time.time() - t0:.1f} s")
