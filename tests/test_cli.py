"""Input files (load_input) and the command-line interface (main)."""

import os
import re
import tomllib

import h5py
import pytest
from conftest import EXAMPLE_INPUT

from beamtarget import BeamTarget, InputError, load_input, main
from beamtarget.units import KEV, MILLIAMPERE, MILLIRADIAN, TORR

# Settings for a small, fast case; the tests below add cases and tables to these. [defaults]
# comes last, so that text added to BASE goes into it.
BASE = """
[materials.D2]
phase = "gas"
molecules = { D2 = 1 }

[materials.DT50]
phase = "gas"
molecules = { D2 = 0.5, T2 = 0.5 }

[materials."TiD1.5"]
phase = "solid"
density = "3.9 g/cm3"
atoms = { Ti = 1, D = 1.5 }

[regions.gas]
material = "DT50"
length = "50 cm"
pressure = "5 torr"
temperature = "25 degC"
wall_radius = "1 cm"
divergence = "30 mrad"
position_bins = 10
position_substeps = 2

[regions.stage]
material = "DT50"
length = "10 cm"
pressure = "1e-3 torr"
temperature = "25 degC"
wall_radius = "0.5 cm"
position_bins = 1
position_substeps = 4

[defaults]
regions = ["gas"]
angle_bin_width = "10 deg"
energy_bin_width = "0.01 MeV"
beam_energy_entry = "280 keV"
beam_current = "10 mA"
beam_radius_entry = "0.2 cm"
"""


def write_input(tmp_path, text, name="input.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def regions_of(path, case="a"):
    return load_input(path)[case]["source"]["regions"]


def test_position_substeps_from_defaults_case_or_region(tmp_path):
    """position_substeps can be set in [defaults] or a case, for the regions that do not set
    their own; a region's own setting wins."""
    text = BASE.replace("position_substeps = 2\n", "", 1)  # The gas region's own setting
    text = text.replace("[defaults]\n", "[defaults]\nposition_substeps = 5\n")
    text += (
        "\n[cases.a]\n"
        "\n[cases.b]\nposition_substeps = 3\n"
        '\n[cases.c]\nposition_substeps = 3\nregions = [{ use = "gas", position_substeps = 2 }]\n'
    )
    cases = load_input(write_input(tmp_path, text))
    steps = {name: [r.position_substeps for r in case["source"]["regions"]]
             for name, case in cases.items()}  # fmt: skip
    assert steps == {"a": [5], "b": [3], "c": [2]}
    assert "position_substeps" not in cases["a"]["source"]


@pytest.mark.parametrize("where", ["defaults", "case"])
def test_renamed_setting_is_an_error(tmp_path, where):
    """angle_bin_width_out is now angle_bin_width. An old file (with the old internal
    angle_bin_width, such as 0.01 deg, too) is an error that says what to change, not a run
    with the internal width taken for the output one."""
    text = BASE + "\n[cases.a]\n"
    old = "angle_bin_width_out = 10.0\n"
    if where == "defaults":
        text = text.replace("[defaults]\n", "[defaults]\n" + old)
        match = r"\[defaults\]: angle_bin_width_out: it is now angle_bin_width"
    else:
        text += old
        match = r"\[cases.a\]: angle_bin_width_out: it is now angle_bin_width"
    with pytest.raises(InputError, match=match):
        load_input(write_input(tmp_path, text))


# --- The example input -----------------------------------------------------------------------

EXAMPLE_CASES = [
    "DD",
    "DT",
    "DT50",
    "DT50_argon",
    "DT50_by_density",
    "TiD_thick",
    "TiD_find_thickness",
    "p_Li_thin",
    "t_D2",
    "p_Be_thick",
    "DT50_srim",
    "DT50_beam_stops",
    "DT50_find_length",
    "DT50_find_pressure",
    "DT50_find_current",
    "DT50_upstream",
    "DT50_aperture",
    "DT50_isotropic",
]


def test_example_input():
    cases = load_input(EXAMPLE_INPUT)
    assert list(cases) == EXAMPLE_CASES
    dd = cases["DD"]
    assert dd["beam_direction"] == "-z"
    assert dd["source"]["stopping_power"] == "icru49"
    assert dd["source"]["beam"] == "deuteron"
    assert dd["source"]["angle_bin_width"] == 3.0
    (chamber,) = dd["source"]["regions"]
    assert chamber.name == "chamber" and chamber.material.name == "D2"
    assert chamber.position_bins == 140 and chamber.position_substeps == 10
    assert chamber.pressure == pytest.approx(40 * TORR, rel=1e-15)


@pytest.fixture(scope="module")
def example_sources():
    """Every example case, run on a coarse mesh (one internal bin per output bin)."""
    sources = {}
    for name, settings in load_input(EXAMPLE_INPUT).items():
        s = settings["source"]
        for region in s["regions"]:
            region.position_substeps = 1
        sources[name] = BeamTarget(name=name, **s)
    return sources


def test_example_cases_show_their_features(example_sources):
    """Each example case does what its comment in examples/generic.toml says."""
    s = example_sources
    assert list(s["DD"].reactions) == ["DD"] and s["DD"].yields["DD"] > 0
    assert list(s["DT"].reactions) == ["DT"] and s["DT"].yields["DT"] > 0
    assert s["DT50_argon"].materials[0].atom_fractions["Ar"] > 0
    assert s["DT50_argon"].yield_total < s["DT50"].yield_total
    assert s["DT50_by_density"].yield_total == pytest.approx(s["DT50"].yield_total, rel=1e-5)
    assert s["TiD_thick"].materials[0].phase == "solid"
    assert s["TiD_thick"].beam_energy_grid[-1] == 0 and s["TiD_thick"].reactions_present() == ["DD"]
    assert s["TiD_find_thickness"].calculated == "length"
    assert s["TiD_find_thickness"].beam_energy_grid[-1] == pytest.approx(100 * KEV, rel=1e-6)
    assert s["p_Li_thin"].reactions_present() == ["p7Li_n0", "p7Li_n1"]
    assert s["t_D2"].reactions_present() == ["tD"]
    assert s["p_Be_thick"].reactions_present() == ["p9Be_xn"]
    assert s["p_Be_thick"].beam_energy_grid[-1] == 0  # The protons stop in the beryllium
    assert s["DT50_srim"].stopping_power == "srim"
    assert s["DT50_beam_stops"].beam_energy_grid[-1] == 0
    assert s["DT50_beam_stops"].backstop_power == 0
    assert s["DT50_find_length"].calculated == "length"
    assert s["DT50_find_length"].beam_energy_grid[-1] == pytest.approx(30 * KEV, rel=1e-6)
    assert s["DT50_find_pressure"].calculated == "pressure"
    assert s["DT50_find_pressure"].beam_energy_grid[-1] == pytest.approx(30 * KEV, rel=1e-6)
    assert s["DT50_find_current"].beam_current_calculated
    assert s["DT50_find_current"].yield_total == pytest.approx(1e13, rel=1e-9)
    assert s["DT50_upstream"].num_regions == 3
    # The aperture scrapes more of the beam than the chamber wall alone
    assert s["DT50_aperture"].current_grid[-1] < s["DT50"].current_grid[-1]
    assert s["DT50_isotropic"].wall_power == 0
    assert s["DT50_isotropic"].num_angle_bins_out == 1
    for name in ["DT50", "DT50_upstream", "DT50_isotropic"]:
        assert s[name].calculated == "beam_energy_backstop"
        assert not s[name].beam_current_calculated


# --- Regions ---------------------------------------------------------------------------------


def test_regions_by_name_inline_and_use(tmp_path):
    """Regions come from named tables, inline tables, or inline tables that use a named one;
    false there leaves one of its settings out."""
    path = write_input(
        tmp_path,
        BASE
        + """
[cases.named]
regions = ["stage", "gas"]

[cases.inline]
regions = [{ material = "D2", length = "5 cm", density = "1e-6 g/cm3", wall_radius = "2 cm", position_bins = 2 }]

[cases.use]
regions = [{ use = "gas", material = "D2", length = false }]
beam_energy_backstop = "200 keV"

[cases.defaults]
""",
    )
    cases = {name: c["source"]["regions"] for name, c in load_input(path).items()}
    stage, gas = cases["named"]
    assert (stage.name, stage.pressure, stage.wall_radius, stage.divergence) == (
        "stage",
        pytest.approx(1e-3 * TORR, rel=1e-15),
        0.5,
        0.0,
    )
    assert gas.name == "gas" and gas.material.name == "DT50"
    assert gas.divergence == pytest.approx(30 * MILLIRADIAN, rel=1e-15)
    (inline,) = cases["inline"]
    assert inline.name == "" and inline.density == 1e-6 and inline.pressure is None
    (used,) = cases["use"]
    assert used.name == "gas" and used.material.name == "D2" and used.length is None
    assert used.pressure == pytest.approx(5 * TORR, rel=1e-15)  # The rest from the named region
    assert [r.name for r in cases["defaults"]] == ["gas"]


@pytest.mark.parametrize(
    "case, message",
    [
        ('regions = ["nope"]', r"no \[regions.nope\] table"),
        ('regions = [{ use = "nope" }]', r"use = 'nope': no \[regions.<name>\] table"),
        ("regions = [3]", "each region is the name of"),
        ("regions = []", "regions must list at least one region"),
        ('regions = "gas"', "regions must be list"),
        ('regions = [{ use = "gas", color = "red" }]', r"unknown setting\(s\) color"),
        ('regions = [{ use = "gas", material = "nope" }]', r"no \[materials.nope\] table"),
        ('regions = [{ use = "gas", length = "long" }]', "length = 'long': expected a number"),
        ('regions = [{ use = "gas", density = false }]', "density must be a number and a unit"),
        ('regions = [{ material = "DT50", length = "1 cm" }]', "missing position_bins"),
    ],
    ids=[
        "unknown name",
        "unknown use",
        "item type",
        "empty",
        "not a list",
        "unknown setting",
        "unknown material",
        "wrong type",
        "false for a setting not in the named region",
        "missing setting",
    ],
)
def test_invalid_regions_rejected(tmp_path, case, message):
    with pytest.raises(InputError, match=message):
        load_input(write_input(tmp_path, BASE + f"\n[cases.a]\n{case}\n"))


def test_named_region_checked_even_if_unused(tmp_path):
    text = BASE + '\n[regions.bad]\nmaterial = "DT50"\n\n[cases.a]\n'
    with pytest.raises(InputError, match=r"\[regions.bad\]: missing position_bins"):
        load_input(write_input(tmp_path, text))


# --- Materials -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table, message",
    [
        ('phase = "gas"\natoms = { D = 1 }', "give a gas's composition as molecules"),
        ('phase = "gas"\nmolecules = { D = 1 }', "D on its own is a lone atom"),
        (
            'phase = "solid"\ndensity = "4 g/cm3"\nmolecules = { TiD2 = 1 }',
            "give a solid's composition",
        ),
        ("atoms = { D = 1 }", "missing phase"),
        ('phase = "solid"\natoms = { Ti = 1 }', "a solid needs a positive density"),
        ('phase = "gas"\nmolecules = { D2 = "a" }', "molecules D2 must be float"),
        (
            'phase = "solid"\ndensity = "1 g/cm3"\natoms = { d = 1 }',
            "cannot read the chemical formula",
        ),
        ('phase = "plasma"\natoms = { D = 1 }', "phase must be one of"),
    ],
)
def test_invalid_materials_rejected(tmp_path, table, message):
    text = BASE + f"\n[materials.m]\n{table}\n\n[cases.a]\n"
    with pytest.raises(InputError, match=message):
        load_input(write_input(tmp_path, text))


def test_solid_region(tmp_path):
    path = write_input(
        tmp_path,
        BASE
        + """
[cases.a]
regions = [{ material = "TiD1.5", length = "5e-4 cm", wall_radius = "1 cm", position_bins = 10, position_substeps = 4 }]
""",
    )
    (film,) = regions_of(path)
    assert film.material.phase == "solid" and film.pressure is None and film.temperature is None


# --- Case settings ---------------------------------------------------------------------------


def test_defaults_and_overrides(tmp_path):
    path = write_input(
        tmp_path,
        BASE + '\n[cases.a]\n\n[cases.b]\nbeam_direction = "+x"\nstopping_power = "srim"\n',
    )
    cases = load_input(path)
    assert cases["a"]["beam_direction"] == "-z"  # Built-in default
    assert cases["a"]["source"]["stopping_power"] == "icru49"  # Built-in default
    assert cases["a"]["source"]["beam_energy_backstop"] is None
    assert cases["b"]["beam_direction"] == "+x"
    assert cases["b"]["source"]["stopping_power"] == "srim"


def test_current_or_yield(tmp_path):
    """A case gives the current or the yield; one set in the case overrides the other from
    [defaults]."""
    path = write_input(
        tmp_path, BASE + "\n[cases.by_current]\n\n[cases.by_yield]\nyield_total = 1e10\n"
    )
    cases = {name: c["source"] for name, c in load_input(path).items()}
    assert cases["by_current"]["beam_current"] == pytest.approx(10 * MILLIAMPERE, rel=1e-15)
    assert cases["by_current"]["yield_total"] is None
    assert cases["by_yield"]["beam_current"] is None  # The current from [defaults] is overridden
    assert cases["by_yield"]["yield_total"] == 1e10


def test_beam_setting(tmp_path):
    """beam picks the beam particle, a deuteron by default; a proton beam on lithium."""
    lithium = (
        '\n[materials.Li]\nphase = "solid"\ndensity = "0.534 g/cm3"\natoms = { 7Li = 1 }\n'
        '\n[cases.d]\n\n[cases.p]\nbeam = "proton"\nbeam_energy_entry = "2.5 MeV"\n'
        'regions = [{ material = "Li", length = "1e-3 cm", wall_radius = "1 cm", position_bins = 2, '
        "position_substeps = 2 }]\n"
    )
    cases = load_input(write_input(tmp_path, BASE + lithium))
    assert cases["d"]["source"]["beam"] == "deuteron"
    assert cases["p"]["source"]["beam"] == "proton"
    source = BeamTarget(name="p", **cases["p"]["source"])
    assert source.reactions_present() == ["p7Li_n0", "p7Li_n1"]


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            '[cases.a]\nbeam_current = "5 mA"\nyield_total = 1e10\n',
            "beam_current and yield_total are both set",
        ),
        ('[cases.a]\nyield_total = "many"\n', "yield_total must be float"),
        ('[cases.a]\nbeam_energy_backstop = "low"\n', "beam_energy_backstop = 'low': expected"),
        ("[cases.a]\nbeam_energy = 3\n", "unknown setting"),
        ('[cases.a]\npressure = "5 torr"\n', r"unknown setting\(s\) pressure"),
        ('[cases.a]\nbeam_energy_entry = "high"\n', "beam_energy_entry = 'high': expected"),
        (
            "[cases.a]\nbeam_energy_entry = 280\n",
            'beam_energy_entry needs a unit, such as "280 keV"',
        ),
        ('[cases.a]\nbeam_energy_entry = "280 kev"\n', r"unknown energy unit 'kev' \(use eV, keV"),
        ('[cases.a]\nbeam_current = "10 keV"\n', "unknown current unit 'keV'"),
        ("[cases.a]\nbeam_current = 10\n", "beam_current needs a unit"),
        ("[cases.a]\nbeam_energy_backstop = 10\n", "beam_energy_backstop needs a unit"),
        ("[cases.a]\nenergy_bin_width = 0.01\n", "energy_bin_width needs a unit"),
        ('[cases.a]\nregions = [{ use = "gas", pressure = 5 }]\n', "pressure needs a unit"),
        ('[cases.a]\nregions = [{ use = "gas", temperature = 25 }]\n', "temperature needs a unit"),
        ('[cases.a]\nregions = [{ use = "gas", divergence = 30 }]\n', "divergence needs a unit"),
        ('[cases.a]\nregions = [{ use = "gas", length = 50 }]\n', "length needs a unit"),
        ('[cases.a]\nregions = [{ use = "gas", wall_radius = 1 }]\n', "wall_radius needs a unit"),
        ("[cases.a]\nbeam_radius_entry = 0.2\n", "beam_radius_entry needs a unit"),
        ("[cases.a]\nangle_bin_width = 3\n", 'angle_bin_width needs a unit, such as "3 deg"'),
        (
            '[cases.a]\nregions = [{ use = "gas", temperature = false, density = 1e-6 }]\n',
            "density needs a unit",
        ),
        ('[cases.a]\nregions = [{ use = "gas", length = "5 furlongs" }]\n', "unknown length unit"),
        ("[cases.a]\nbeam_radius_entry = [0.2]\n", "beam_radius_entry must be a number and a unit"),
        ('[cases.a]\nbeam_direction = "z"\n', "beam_direction must be one of"),
        ('[cases.a]\nstopping_power = "pstar"\n', "stopping_power must be one of"),
        ('[cases.a]\nbeam = "p"\n', 'beam must be one of "1H", "proton"'),
        ('[cases.a]\nentry_position = ["1 cm", "2 cm"]\n', "entry_position must be a list"),
        ('[cases.a]\nentry_position = ["1 cm", "2 cm", "z"]\n', r"entry_position\[2\] = 'z'"),
        ("[cases.a]\nentry_position = [1, 2, 3]\n", r"entry_position\[0\] needs a unit"),
        ('[cases."a/b"]\n', "case names may only use"),
        ("", r"no \[cases.<name>\] tables"),
        ("[case.a]\n", "unknown table"),
        ("[upstream.a]\n", "unknown table"),
        ("[cases.a\n", "Expected"),
    ],
    ids=[
        "current and yield",
        "yield type",
        "backstop energy type",
        "misspelt",
        "region setting in a case",
        "wrong type",
        "plain beam energy",
        "unit case",
        "unit of another quantity",
        "plain current",
        "plain backstop energy",
        "plain energy bin",
        "plain pressure",
        "plain temperature",
        "plain divergence",
        "plain length",
        "plain wall radius",
        "plain beam radius",
        "plain angle bin width",
        "plain density",
        "unknown length unit",
        "length type",
        "direction",
        "stopping power",
        "beam",
        "entry position length",
        "entry position type",
        "plain entry position",
        "case name",
        "no cases",
        "unknown table",
        "no more upstream tables",
        "TOML syntax",
    ],
)
def test_invalid_input_rejected(tmp_path, extra, message):
    text = extra if extra.startswith("[cases.a\n") else BASE + extra
    with pytest.raises(InputError, match=message):
        load_input(write_input(tmp_path, text))


# --- Command line ----------------------------------------------------------------------------

TWO_CASES = BASE + '\n[cases.mix]\n\n[cases.pureD]\nregions = [{ use = "gas", material = "D2" }]\n'


def test_main_runs_cases(tmp_path):
    path = write_input(tmp_path, TWO_CASES)
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-j", "1"])
    for case in ("mix", "pureD"):
        case_dir = out / case
        assert (case_dir / "source.sdef").is_file()
        for fname in [
            "summary.toml",
            "axial_profile.txt",
            "energy_spectrum.txt",
            "angular_distribution.txt",
            "surface.html",
            "openmc_source.py",
        ]:
            assert (case_dir / fname).is_file()
        assert not (case_dir / "input.toml").exists()  # The summary records the inputs instead
        assert len(list((case_dir / "plots").glob("*.png"))) in (9, 11)
        with open(case_dir / "summary.toml", "rb") as f:
            summary = tomllib.load(f)
        run = summary["run"]
        assert run["name"] == case
        assert run["input_file"] == str(path)
        assert run["sdef_direction"] == "-z"
        assert run["sdef_entry_position"] == [0.0, 0.0, 0.0]  # Default
        assert summary["inputs"]["region"] == ["gas"]
        assert re.search(r"^c input_file +=", (case_dir / "source.sdef").read_text(), re.M)
        with h5py.File(case_dir / "source.h5") as f:
            assert f.attrs["name"] == case
            assert f.attrs["created"] == run["created"]  # One creation time for every output
            assert f.attrs["input_file"] == str(path)


def test_entry_position_reaches_sdef(tmp_path):
    text = TWO_CASES.replace(
        "[cases.mix]\n",
        '[cases.mix]\nentry_position = ["5 cm", "0 cm", "105 mm"]\nbeam_direction = "+z"\n',
    )
    path = write_input(tmp_path, text)
    assert load_input(path)["mix"]["entry_position"] == [5.0, 0.0, 10.5]
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-j", "1", "-c", "mix"])
    assert "        pos = 5 0 10.5\n" in (out / "mix" / "source.sdef").read_text()
    with open(out / "mix" / "summary.toml", "rb") as f:
        assert tomllib.load(f)["run"]["sdef_entry_position"] == [5.0, 0.0, 10.5]


def test_main_without_plots(tmp_path):
    out = tmp_path / "out"
    main([str(write_input(tmp_path, TWO_CASES)), "-o", str(out), "-c", "mix", "--no-plots"])
    assert (out / "mix" / "source.h5").is_file() and not (out / "mix" / "plots").exists()
    assert not (out / "mix" / "surface.html").exists()


def test_main_selected_case_in_parallel(tmp_path):
    path = write_input(tmp_path, TWO_CASES)
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-c", "pureD", "-c", "mix", "-j", "2"])
    assert sorted(p.name for p in out.iterdir()) == ["mix", "pureD"]


def test_main_list(tmp_path, capsys):
    main([str(write_input(tmp_path, TWO_CASES)), "--list"])
    assert capsys.readouterr().out.split() == ["mix", "pureD"]


def test_main_default_messages(tmp_path, caplog):
    """By default: a start line, one line per case, and a closing line; no details."""
    main([str(write_input(tmp_path, TWO_CASES)), "-o", str(tmp_path / "out"), "-j", "1"])
    # (The coarse test mesh gets a position-mesh warning, which is not checked here)
    messages = [
        r.getMessage()
        for r in caplog.records
        if "position mesh is too coarse" not in r.getMessage()
    ]
    assert messages[0].startswith("Running 2 case(s)")
    assert messages[0].endswith(f"results in {tmp_path / 'out'}{os.sep}")
    assert [m.split(":")[0] for m in messages[1:3]] == ["mix", "pureD"]
    assert all(" n/s, mean energy " in m and " MeV, in " in m for m in messages[1:3])
    # The case lines line up: names padded to the longest, numbers at fixed widths
    for column in (" n/s", " MeV", " s ->"):
        assert len({m.index(column) for m in messages[1:3]}) == 1, column
    assert messages[1].endswith(f"-> {tmp_path / 'out' / 'mix'}{os.sep}")
    assert messages[3].startswith("Done: 2 case(s)")
    assert len(messages) == 4


def test_main_verbose_messages(tmp_path, caplog):
    """With -v: also each case's settings and detailed results."""
    main([str(write_input(tmp_path, TWO_CASES)), "-o", str(tmp_path / "out"), "-j", "1", "-v"])
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert (
        "mix: starting: deuteron beam, 280 keV, 10 mA, 1 region(s), last DT50, stopping power icru49"
        in text
    )
    for detail in ["beam energy at the end of the target", "DT neutron yield", "total neutron"]:
        assert f"mix: {detail}" in text
    assert "% lost to beam scraping" in text


@pytest.mark.parametrize(
    "args, message",
    [(["-c", "nope"], "no case(s) named nope"), (["--list"], "missing position_bins")],
    ids=["unknown case", "bad input file"],
)
def test_main_errors(tmp_path, capsys, args, message):
    text = TWO_CASES if "-c" in args else BASE + '\n[cases.a]\nregions = [{ material = "D2" }]\n'
    with pytest.raises(SystemExit) as exit_info:
        main([str(write_input(tmp_path, text))] + args)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_simple_example():
    """examples/simple.toml, the smallest input file, holds one case that runs."""
    cases = load_input(EXAMPLE_INPUT.with_name("simple.toml"))
    assert list(cases) == ["DT90"]
    s = cases["DT90"]["source"]
    (chamber,) = s["regions"]
    assert chamber.material.atom_fractions["T"] == pytest.approx(0.9)
    assert chamber.pressure == pytest.approx(40 * TORR, rel=1e-15)
    assert s["beam_current"] == pytest.approx(50 * MILLIAMPERE, rel=1e-15)
    assert s["beam"] == "deuteron"
    chamber.position_substeps = 1  # A coarse mesh, for speed
    source = BeamTarget(name="DT90", **s)
    assert source.calculated == "beam_energy_backstop"
    assert source.yields["DT"] > 10 * source.yields["DD"] > 0


# --- SDEF placement --------------------------------------------------------------------------


def test_backstop_position(tmp_path):
    """backstop_position places the source by its backstop end, even when its length is
    calculated; entry_position set in a case replaces it from [defaults]."""
    text = BASE.replace(
        "[defaults]\n", '[defaults]\nbackstop_position = ["0 cm", "0 cm", "0 cm"]\n'
    ) + (
        '\n[cases.fixed]\n\n[cases.find_length]\nregions = [{ use = "gas", length = false }]\n'
        'beam_energy_backstop = "200 keV"\n\n'
        '[cases.by_entry]\nentry_position = ["0 cm", "0 cm", "7 cm"]\n'
    )
    path = write_input(tmp_path, text)
    cases = load_input(path)
    assert cases["fixed"]["backstop_position"] == [0.0, 0.0, 0.0]
    assert cases["fixed"]["entry_position"] is None
    assert cases["by_entry"]["entry_position"] == [0.0, 0.0, 7.0]
    assert cases["by_entry"]["backstop_position"] is None
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-j", "1", "-c", "find_length"])
    with open(out / "find_length" / "summary.toml", "rb") as f:
        summary = tomllib.load(f)
    length = summary["inputs"]["length"][0]
    assert length != 50.0  # Calculated
    run = summary["run"]
    assert run["sdef_backstop_position"] == [0.0, 0.0, 0.0]
    assert run["sdef_entry_position"] == pytest.approx([0.0, 0.0, length], rel=1e-6)
    sdef = (out / "find_length" / "source.sdef").read_text()
    assert "        pos = 0 0 0\n" in sdef  # -z: POS is the backstop end
    # The HDF5 file records the same placement (read by openmc_source)
    with h5py.File(out / "find_length" / "source.h5") as f:
        assert f.attrs["direction"] == "-z"
        assert list(f.attrs["backstop_position"]) == [0.0, 0.0, 0.0]
        assert list(f.attrs["entry_position"]) == pytest.approx(run["sdef_entry_position"])


@pytest.mark.parametrize(
    "case, message",
    [
        (
            'entry_position = ["0 cm", "0 cm", "1 cm"]\nbackstop_position = ["0 cm", "0 cm", "0 cm"]\n',
            "give entry_position or backstop_position, not both",
        ),
        (
            'backstop_position = ["0 cm", "0 cm"]\n',
            "backstop_position must be a list of three lengths",
        ),
    ],
)
def test_placement_rejected(tmp_path, case, message):
    with pytest.raises(InputError, match=message):
        load_input(write_input(tmp_path, BASE + f"\n[cases.a]\n{case}"))


@pytest.mark.parametrize(
    "direction, backstop, entry",
    [
        ("-z", (0, 0, 0), (0, 0, 50)),
        ("+x", (10, 0, 0), (-40, 0, 0)),
        ("+y", (1, 2, 3), (1, -48, 3)),
    ],
)
def test_sdef_positions(mixed_source, direction, backstop, entry):
    """Either end gives the other, the source's length (50 cm) apart along the beam."""
    e, b = mixed_source.sdef_positions(direction, backstop_position=backstop)
    assert list(e) == pytest.approx(entry) and list(b) == pytest.approx(backstop)
    e2, b2 = mixed_source.sdef_positions(direction, entry_position=entry)
    assert list(b2) == pytest.approx(backstop)
    with pytest.raises(ValueError, match="not both"):
        mixed_source.sdef_positions(direction, entry_position=entry, backstop_position=backstop)


def test_region_without_wall(tmp_path):
    """wall_radius may be left out (or left out of a named region with false): no wall."""
    text = BASE + '\n[cases.a]\nregions = [{ use = "gas", wall_radius = false }]\n'
    (region,) = load_input(write_input(tmp_path, text))["a"]["source"]["regions"]
    assert region.wall_radius is None


@pytest.mark.parametrize("width", ["3 deg", "0.05235987755982988 rad", "52.35987755982988 mrad"])
def test_angle_bin_width_in_any_angle_unit(tmp_path, width):
    """angle_bin_width takes any angle unit; the code works in degrees, exactly 3 here."""
    text = BASE + f'\n[cases.a]\nangle_bin_width = "{width}"\n'
    assert load_input(write_input(tmp_path, text))["a"]["source"]["angle_bin_width"] == 3.0


def test_plain_material_density_rejected(tmp_path):
    text = BASE + '\n[materials.Be]\nphase = "solid"\ndensity = 1.848\natoms = { Be = 1 }\n'
    with pytest.raises(InputError, match=r"\[materials.Be\]: density needs a unit"):
        load_input(write_input(tmp_path, text + "\n[cases.a]\n"))
