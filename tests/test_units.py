"""Units: reading values with units, and results reported in the units an input file uses."""

import tomllib

import h5py
import numpy as np
import pytest
from conftest import make_source
from test_cli import BASE, write_input

from beamtarget import load_input, main
from beamtarget.units import DEFAULT_UNITS, UNITS, from_internal, parse, to_internal


@pytest.mark.parametrize(
    "text, quantity, value, unit",
    [
        ("280 keV", "energy", 280e3, "keV"),
        ("70MeV", "energy", 70e6, "MeV"),
        ("1.2 GeV", "energy", 1.2e9, "GeV"),
        ("5e-3 eV", "energy", 5e-3, "eV"),
        ("1 uA", "current", 1e-6, "uA"),
        ("1 µA", "current", 1e-6, "uA"),  # Micro sign
        ("1 μA", "current", 1e-6, "uA"),  # Greek mu
        ("760 torr", "pressure", 101325.0, "torr"),
        ("1 atm", "pressure", 101325.0, "atm"),
        ("1 psi", "pressure", 6894.757293168361, "psi"),
        ("10 mbar", "pressure", 1000.0, "mbar"),
        ("25 degC", "temperature", 298.15, "degC"),
        ("25 C", "temperature", 298.15, "C"),
        ("212 degF", "temperature", 373.15, "degF"),
        ("300 K", "temperature", 300.0, "K"),
        ("30 mrad", "divergence", 0.03, "mrad"),
        ("1 deg", "divergence", np.pi / 180, "deg"),
        ("1 ft", "length", 30.48, "ft"),
        ("2 in", "length", 5.08, "in"),
        ("5 um", "length", 5e-4, "um"),
        ("1 m", "length", 100.0, "m"),
        ("1000 kg/m3", "density", 1.0, "kg/m3"),
    ],
)
def test_parse(text, quantity, value, unit):
    got, name = parse(text, quantity)
    assert got == pytest.approx(value, rel=1e-12) and name == unit


@pytest.mark.parametrize(
    "text, quantity, message",
    [
        ("280", "energy", "expected a number and a unit"),
        ("keV", "energy", "expected a number and a unit"),
        ("280 kev", "energy", "unknown energy unit 'kev'"),  # Case-sensitive
        ("10 mA", "energy", "unknown energy unit 'mA'"),
        ("1 2 keV", "energy", "expected a number and a unit"),
    ],
)
def test_parse_errors(text, quantity, message):
    with pytest.raises(ValueError, match=message):
        parse(text, quantity)


@pytest.mark.parametrize("unit", list(UNITS))
def test_conversions_round_trip(unit):
    assert from_internal(to_internal(12.5, unit), unit) == pytest.approx(12.5, rel=1e-12)


def test_default_units_are_known():
    for quantity, unit in DEFAULT_UNITS.items():
        kind = {"beam_energy": "energy", "neutron_energy": "energy", "radius": "length"}.get(
            quantity, quantity
        )
        assert UNITS[unit][0] == kind


def test_unknown_result_units_rejected():
    with pytest.raises(ValueError, match="unknown units current = 'keV'"):
        make_source(units={"current": "keV"})
    with pytest.raises(ValueError, match="unknown units speed"):
        make_source(units={"speed": "m/s"})


# The same case as BASE's (280 keV, 10 mA, 5 torr, 25 degC, 30 mrad, 50 cm, 0.01 MeV bins), in
# other units
OTHER_UNITS = """
[regions.other]
material = "DT50"
length = "0.5 m"
pressure = "666.6118421052631 Pa"
temperature = "298.15 K"
wall_radius = "10 mm"
divergence = "0.030 rad"
position_bins = 10
position_substeps = 2

[cases.other]
beam_energy_entry = "0.28 MeV"
beam_current = "10000 uA"
energy_bin_width = "10 keV"
beam_radius_entry = "2 mm"
regions = ["other"]

[cases.base]
"""


def test_results_follow_the_input_units(tmp_path):
    """A case written in other units gives the same source, with its results in its units;
    the HDF5 file records them, and the MCNP SDEF stays in cm and MeV."""
    path = write_input(tmp_path, BASE + OTHER_UNITS)
    cases = load_input(path)
    assert cases["other"]["source"]["units"] == {
        "beam_energy": "MeV",
        "neutron_energy": "keV",
        "radius": "mm",
        "length": "m",
        "current": "uA",
        "pressure": "Pa",
        "temperature": "K",
        "divergence": "rad",
    }
    assert cases["base"]["source"]["units"]["beam_energy"] == "keV"
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-j", "1", "--no-plots"])

    def summary(name):
        with open(out / name / "summary.toml", "rb") as f:
            return tomllib.load(f)

    other, base = summary("other"), summary("base")
    assert other["inputs"]["beam_energy_entry"] == pytest.approx(0.28)
    assert other["inputs"]["beam_current"] == pytest.approx(10000)
    assert other["inputs"]["pressure"] == pytest.approx([666.6118], rel=1e-6)
    assert other["inputs"]["temperature"] == pytest.approx([298.15])
    assert other["inputs"]["length"] == pytest.approx([0.5])
    assert other["inputs"]["wall_radius"] == pytest.approx([10.0])
    assert other["inputs"]["energy_bin_width"] == pytest.approx(10.0)
    results, expected = other["results"], base["results"]
    assert results["yield_total"] == pytest.approx(expected["yield_total"], rel=1e-6)
    assert results["mean_neutron_energy"] == pytest.approx(
        1000 * expected["mean_neutron_energy"], rel=1e-6
    )
    assert results["beam_energy_backstop"] == pytest.approx(
        expected["beam_energy_backstop"] / 1000, rel=1e-6
    )
    assert results["beam_current_backstop"] == pytest.approx(
        1000 * expected["beam_current_backstop"], rel=1e-6
    )
    assert results["beam_radius_backstop"] == pytest.approx(
        10 * expected["beam_radius_backstop"], rel=1e-6
    )
    text = (out / "other" / "summary.toml").read_text()
    assert "# MeV, at the end of each region" in text and "# uA, after scraping" in text

    axial = (out / "other" / "axial_profile.txt").read_text()
    assert "start of position bin [m]" in axial and "over the bin [mm]" in axial
    assert "beam energy at the bin center [MeV]" in axial
    spectrum = (out / "other" / "energy_spectrum.txt").read_text()
    assert "start of energy bin [keV]" in spectrum

    with h5py.File(out / "other" / "source.h5") as f, h5py.File(out / "base" / "source.h5") as g:
        assert f["x_grid"].attrs["units"] == "m" and f["radius_mids"].attrs["units"] == "mm"
        assert f["neutron_energy_grid_DT"].attrs["units"] == "keV"
        np.testing.assert_allclose(f["x_grid"][()], g["x_grid"][()] / 100, rtol=1e-12)
        np.testing.assert_allclose(f["radius_mids"][()], 10 * g["radius_mids"][()], rtol=1e-12)
        np.testing.assert_allclose(f["neutrons_DT"][()], g["neutrons_DT"][()], rtol=1e-6)

    def sdef_cards(name):
        lines = (out / name / "source.sdef").read_text().splitlines()
        return [line for line in lines if not line.startswith("c ")]

    assert sdef_cards("other") == sdef_cards("base")  # Always cm and MeV


def test_openmc_reads_any_units(tmp_path):
    """get_openmc_sources reads the units of the HDF5 file, so the sources are the same."""
    openmc_source = pytest.importorskip("beamtarget.openmc_source")
    pytest.importorskip("openmc")
    path = write_input(tmp_path, BASE + OTHER_UNITS)
    out = tmp_path / "out"
    main([str(path), "-o", str(out), "-j", "1", "--no-plots"])
    other = openmc_source.get_openmc_sources(out / "other" / "source.h5")
    base = openmc_source.get_openmc_sources(out / "base" / "source.h5")
    assert len(other) == len(base)
    for a, b in zip(other, base):
        assert a.strength == pytest.approx(b.strength, rel=1e-6)
        assert (a.space.z.a, a.space.z.b) == pytest.approx((b.space.z.a, b.space.z.b), rel=1e-12)
        np.testing.assert_allclose(a.energy.x, b.energy.x, rtol=1e-12)
