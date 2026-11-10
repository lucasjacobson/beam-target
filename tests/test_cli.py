"""Input files (load_input) and the command-line interface (main)."""

import h5py
import pytest
from conftest import EXAMPLE_INPUT

from shine_neutron_source import InputError, load_input, main

# Settings for a small, fast case; the tests below add cases and tables to these
BASE = """
[defaults]
num_x_ints = 20
num_x_ints_C = 10
angle_int = 1.0
angle_int_C = 10.0
EN_int = 0.01
T_frac = 0.5
voltage = 280.0
current = 10.0
temperature = 25.0
length = 50.0
start_radius = 0.2
max_radius = 1.0
divergence = 30.0
"""


def write_input(tmp_path, text, name="input.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


# --- load_input ------------------------------------------------------------------------------


def test_example_input():
    cases = load_input(EXAMPLE_INPUT)
    assert list(cases) == ["T000pct", "T001pct", "T050pct", "T090pct", "T100pct"]
    dd = cases["T000pct"]
    assert dd["direction"] == "-z"
    assert dd["source"]["T_frac"] == 0.0
    assert dd["source"]["stopping_power"] == "srim"
    assert dd["source"]["num_x_ints_list"] == [1370]
    assert dd["source"]["angle_int"] == 0.01


def test_defaults_and_overrides(tmp_path):
    path = write_input(
        tmp_path,
        BASE
        + """
[cases.a]
pressure = 5.0

[cases.b]
pressure = 5.0
T_frac = 1.0
direction = "+x"
stopping_power = "icru49"
""",
    )
    cases = load_input(path)
    assert cases["a"]["source"]["T_frac"] == 0.5
    assert cases["a"]["direction"] == "-z"  # Built-in default
    assert cases["a"]["source"]["stopping_power"] == "srim"  # Built-in default
    assert cases["b"]["source"]["T_frac"] == 1.0
    assert cases["b"]["direction"] == "+x"
    assert cases["b"]["source"]["stopping_power"] == "icru49"


def test_upstream_stages(tmp_path):
    path = write_input(
        tmp_path,
        BASE
        + """
[upstream.stages]
pressure = [1e-3, 1e-2]
length = [10.0, 20.0]
num_x_ints = [4, 6]

[cases.a]
pressure = 5.0
upstream = "stages"
""",
    )
    source = load_input(path)["a"]["source"]
    assert source["pressure_list"] == [1e-3, 1e-2, 5.0]
    assert source["length_list"] == [10.0, 20.0, 50.0]
    assert source["num_x_ints_list"] == [4, 6, 20]
    assert source["num_x_ints_list_C"] == [1, 1, 10]  # Default: 1 per stage
    assert source["divergence_list"] == [0.0, 0.0, 30.0]  # Default: 0 per stage
    assert source["temperature_list"] == [25.0, 25.0, 25.0]


@pytest.mark.parametrize(
    "extra, message",
    [
        ("[cases.a]\n", "missing pressure"),
        ("[cases.a]\npressure = 5.0\nvoltge = 3\n", "unknown setting"),
        ('[cases.a]\npressure = "high"\n', "pressure must be float"),
        ("[cases.a]\npressure = 5.0\nnum_x_ints = 2.5\n", "num_x_ints must be int"),
        ('[cases.a]\npressure = 5.0\nupstream = "nope"\n', r"no \[upstream.nope\] table"),
        (
            "[upstream.u]\npressure = [1e-3, 1e-2]\nlength = [10.0]\n[cases.a]\npressure = 5.0\n"
            'upstream = "u"\n',
            "length has 1 entries, expected 2",
        ),
        ('[cases.a]\npressure = 5.0\ndirection = "z"\n', "direction must be one of"),
        ('[cases.a]\npressure = 5.0\nstopping_power = "pstar"\n', "stopping_power must be one of"),
        ('[cases."a/b"]\npressure = 5.0\n', "case names may only use"),
        ("", r"no \[cases.<name>\] tables"),
        ("[case.a]\npressure = 5.0\n", "unknown table"),
        ("[cases.a\n", "Expected"),
    ],
    ids=[
        "missing",
        "misspelt",
        "wrong type",
        "float for int",
        "unknown upstream",
        "stage count",
        "direction",
        "stopping power",
        "case name",
        "no cases",
        "unknown table",
        "TOML syntax",
    ],
)
def test_invalid_input_rejected(tmp_path, extra, message):
    text = extra if extra.startswith("[cases.a\n") else BASE + extra
    with pytest.raises(InputError, match=message):
        load_input(write_input(tmp_path, text))


# --- Command line ----------------------------------------------------------------------------

TWO_CASES = BASE + "\n[cases.mix]\npressure = 5.0\n\n[cases.pureD]\npressure = 5.0\nT_frac = 0.0\n"


def test_main_runs_cases(tmp_path):
    path = write_input(tmp_path, TWO_CASES)
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-j", "1"])
    for case in ("mix", "pureD"):
        case_dir = out / case
        assert (case_dir / "source.sdef").is_file()
        assert (case_dir / "input.toml").read_bytes() == path.read_bytes()
        assert len(list((case_dir / "plots").glob("*.png"))) in (8, 9)
        with h5py.File(case_dir / "source.h5") as f:
            assert f.attrs["name"] == case


def test_main_selected_case_in_parallel(tmp_path):
    path = write_input(tmp_path, TWO_CASES)
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-c", "pureD", "-c", "mix", "-j", "2"])
    assert sorted(p.name for p in out.iterdir()) == ["mix", "pureD"]


def test_main_list(tmp_path, capsys):
    main([str(write_input(tmp_path, TWO_CASES)), "--list"])
    assert capsys.readouterr().out.split() == ["mix", "pureD"]


def test_main_default_messages(tmp_path, caplog):
    """By default: a start line, one line per case and a closing line; no details."""
    main([str(write_input(tmp_path, TWO_CASES)), "-o", str(tmp_path / "out"), "-j", "1"])
    messages = [r.getMessage() for r in caplog.records]
    assert messages[0].startswith("Running 2 case(s)")
    assert [m.split(":")[0] for m in messages[1:3]] == ["mix", "pureD"]
    assert all(" n/s in " in m for m in messages[1:3])
    assert messages[3].startswith("Done: 2 case(s)")
    assert len(messages) == 4


def test_main_verbose_messages(tmp_path, caplog):
    """With -v: also each case's settings and detailed results."""
    main([str(write_input(tmp_path, TWO_CASES)), "-o", str(tmp_path / "out"), "-j", "1", "-v"])
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "mix: starting: T_frac 0.5, 280 kV, 10 mA, 1 gas region(s), stopping power srim" in text
    for detail in ["beam energy at the end of the target", "DT neutron yield", "total neutron"]:
        assert f"mix: {detail}" in text
    assert "% lost to beam scraping" in text


@pytest.mark.parametrize(
    "args, message",
    [(["-c", "nope"], "no case(s) named nope"), (["--list"], "missing pressure")],
    ids=["unknown case", "bad input file"],
)
def test_main_errors(tmp_path, capsys, args, message):
    text = TWO_CASES if "-c" in args else BASE + "[cases.a]\n"
    with pytest.raises(SystemExit) as exit_info:
        main([str(write_input(tmp_path, text))] + args)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
