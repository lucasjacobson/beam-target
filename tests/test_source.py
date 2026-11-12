"""Whole calculations with ShineSource: physics checks, input checks, and outputs."""

import re
import tomllib

import h5py
import numpy as np
import pytest
from conftest import SMALL_CASE, make_source

from shine_neutron_source.nuclear import REACTIONS
from shine_neutron_source.source import HDF5_FORMAT_VERSION, SDEF_RADIUS_DECIMALS

# Highest voltage for SMALL_CASE (D and T): where T(d,np)T opens [kV]
MAX_VOLTAGE = REACTIONS["DT"].complete_up_to()[0] * 1000

# --- Physics checks ------------------------------------------------------------------------


@pytest.mark.parametrize("rxn", ["DD", "DT"])
def test_cross_section_integrates_to_total(rxn):
    """At every position, the CM cross section summed over the CM angle mesh equals the total
    cross section 4 pi sigma_0 C_0 from the data, and every neutron lands in an output bin
    (none are lost converting to lab angles and energies)."""
    source = make_source(angle_step=0.25)
    xs = source.xs_data[rxn]
    total = (
        4
        * np.pi
        * np.interp(source.beam_energy_mids, xs["E"], xs["total"])
        * np.interp(source.beam_energy_mids, xs["E"], xs["coeffs"][:, 0])
    )
    cm = np.sum(source.cm_xs[rxn] * source.solid_angles[None, :], axis=1)
    np.testing.assert_allclose(cm, total, rtol=1e-4)
    per_bin = cm * source.target_density_mids[rxn] * 1e-24 * source.dx * source.particle_rate
    expected = np.sum(per_bin * (1 - source.scrape_mids))
    assert source.yields[rxn] == pytest.approx(expected, rel=1e-4)


def test_yields_add_up(mixed_source):
    s = mixed_source
    assert s.yields["DD"] == pytest.approx(s.neutrons["DD"].sum(), rel=1e-12)
    assert s.yields["DT"] == pytest.approx(s.neutrons["DT"].sum(), rel=1e-12)
    assert s.yield_total == s.yields["DD"] + s.yields["DT"]
    assert 0 < s.neutron_scrape_fraction < 1
    for rxn in ("DD", "DT"):
        assert np.all(s.neutrons[rxn] >= 0)


@pytest.mark.parametrize("T_frac, absent", [(0.0, "DT"), (1.0, "DD")])
def test_pure_gases(T_frac, absent):
    """A reaction whose target isotope is absent is left out altogether, outputs included."""
    source = make_source(T_frac=T_frac)
    assert absent not in source.reactions and absent not in source.yields
    assert f"yield_{absent}" not in tomllib.loads(source.get_summary())["results"]
    assert source.yield_total > 0


def test_yield_scales_with_tritium_fraction():
    """At fixed atom density, the DT yield is proportional to T_frac (stopping hardly depends on
    the mix), and the DD yield to 1 - T_frac."""
    pure_D, pure_T, mix = make_source(T_frac=0.0), make_source(T_frac=1.0), make_source(T_frac=0.5)
    assert mix.yields["DT"] / pure_T.yields["DT"] == pytest.approx(0.5, rel=2e-3)
    assert mix.yields["DD"] / pure_D.yields["DD"] == pytest.approx(0.5, rel=2e-3)


def test_beam_slows_down(mixed_source):
    EB = mixed_source.beam_energy_grid
    assert EB[0] == 280.0
    assert np.all(np.diff(EB) < 0)
    assert 150 < EB[-1] < 250  # A thin target: the beam passes through


@pytest.mark.parametrize("stopping_power", ["srim", "icru49"])
@pytest.mark.parametrize("T_frac", [0.0, 1.0])
def test_beam_stops_inside_target(T_frac, stopping_power):
    """When the beam stops well before the end of the target, everything beyond the stopping
    point is exactly zero: the beam energy, the cross sections, and the neutrons. There must be
    no low-level tail of neutrons all the way to the backstop."""
    source = make_source(
        T_frac=T_frac,
        stopping_power=stopping_power,
        pressure=[40.0],
        length=[300.0],
        num_x_bins=[600],
        num_x_bins_out=[60],
        wall_radius=10.0,
    )
    EB = source.beam_energy_grid
    assert EB[-1] == 0, f"The beam never stops: it reaches the backstop with {EB[-1]} keV"
    i_stop = np.flatnonzero(EB > 0)[-1]  # Last grid point the beam reaches with energy
    x_stop = source.x_grid[i_stop + 1]
    assert 15 < x_stop < 30  # Stops within the first tenth of the 300 cm target

    rxn = "DT" if T_frac else "DD"
    assert np.all(EB[i_stop + 1 :] == 0)
    assert np.all(source.beam_energy_mids[i_stop + 1 :] == 0)
    assert np.all(source.cm_xs[rxn][i_stop + 1 :] == 0)
    neutrons_per_bin = source.neutrons[rxn].sum(axis=(1, 2))
    beyond = source.x_grid_out[:-1] >= x_stop  # Output bins entirely past the stop
    assert beyond.sum() > 50
    assert np.all(neutrons_per_bin[beyond] == 0)
    assert source.yields[rxn] > 0
    assert source.beam_energy_grid_out[-1] == 0  # Beam energy at the backstop
    assert source.backstop_power == 0  # All the beam power stays in the target


# --- Two of pressure, length, and backstop energy -------------------------------------------


@pytest.mark.parametrize("calculated", ["pressure", "length"])
@pytest.mark.parametrize("stopping_power", ["srim", "icru49"])
def test_calculate_pressure_or_length(calculated, stopping_power):
    """Giving the backstop energy of a forward run recovers that run's pressure or length,
    and the rest of the results with it."""
    forward = make_source(stopping_power=stopping_power)
    E_end = forward.beam_energy_grid[-1]
    s = make_source(
        stopping_power=stopping_power, **{calculated: [None]}, beam_energy_backstop=E_end
    )
    assert s.calculated == calculated
    assert getattr(s, calculated)[-1] == pytest.approx(SMALL_CASE[calculated][-1], rel=1e-9)
    assert s.beam_energy_grid[-1] == pytest.approx(E_end, rel=1e-9)
    assert s.yield_total == pytest.approx(forward.yield_total, rel=1e-8)
    inputs = {name: (value, units) for name, value, units in s.get_inputs()}
    assert inputs["beam_energy_backstop"] == (E_end, "keV, requested")
    assert inputs[calculated][1].endswith("calculated for region 1")
    assert "beam_energy_backstop" not in [name for name, _, _ in forward.get_inputs()]
    assert forward.calculated == "beam_energy_backstop"


@pytest.mark.parametrize("calculated", ["pressure", "length"])
def test_calculate_with_upstream_regions(calculated):
    """Only the target chamber's value is calculated; the upstream region slows the beam
    first."""
    regions = dict(
        pressure=[0.5, 5.0],
        temperature=[25.0, 25.0],
        length=[20.0, 50.0],
        divergence=[0.0, 30.0],
        num_x_bins=[4, 40],
        num_x_bins_out=[1, 10],
    )
    forward = make_source(**regions)
    values = {calculated: [regions[calculated][0], None]}
    s = make_source(**{**regions, **values}, beam_energy_backstop=forward.beam_energy_grid[-1])
    np.testing.assert_allclose(getattr(s, calculated), regions[calculated], rtol=1e-9)


def test_target_chamber_needs_two_of_three():
    for overrides in [
        dict(beam_energy_backstop=100.0),  # All three given
        dict(pressure=None, length=None, beam_energy_backstop=100.0),  # Only one
        dict(pressure=None),  # Only length
    ]:
        with pytest.raises(ValueError, match="nothing is left|only one value can be"):
            make_source(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(length=None, beam_energy_backstop=280.0), "must be above 0 and below"),
        (dict(length=None, beam_energy_backstop=0.0), "must be above 0 and below"),
        (dict(length=None, beam_energy_backstop=1e-6), "cannot reach beam_energy_backstop"),
        (
            dict(
                pressure=[None, 5.0],
                temperature=[25.0, 25.0],
                length=[20.0, None],
                divergence=[0.0, 30.0],
                num_x_bins=[4, 40],
                num_x_bins_out=[1, 10],
                beam_energy_backstop=100.0,
            ),
            "only one value can be calculated.*the pressure of region 1",
        ),
        (
            dict(
                pressure=[50.0, None],
                temperature=[25.0, 25.0],
                length=[100.0, 50.0],
                divergence=[0.0, 30.0],
                num_x_bins=[40, 40],
                num_x_bins_out=[1, 10],
                beam_energy_backstop=100.0,
            ),
            "even without region 2 .*the beam reaches the backstop with",
        ),
    ],
    ids=["at entry energy", "zero", "below the stopping table", "upstream missing", "upstream"],
)
def test_backstop_energy_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_source(**overrides)


# --- Current or yield ------------------------------------------------------------------------


def test_calculate_current(mixed_source):
    """Giving a forward run's yield recovers its current, and everything else with it."""
    s = make_source(current=None, yield_total=mixed_source.yield_total)
    assert s.current_calculated and not mixed_source.current_calculated
    assert s.current == pytest.approx(mixed_source.current, rel=1e-12)
    assert s.yield_total == pytest.approx(mixed_source.yield_total, rel=1e-12)
    np.testing.assert_allclose(s.current_grid, mixed_source.current_grid, rtol=1e-12)
    assert s.deposited_power == pytest.approx(mixed_source.deposited_power, rel=1e-12)
    inputs = {name: (value, units) for name, value, units in s.get_inputs()}
    assert inputs["current"][1] == "mA, calculated"
    assert inputs["yield_total"] == (mixed_source.yield_total, "n/s, requested")
    assert "yield_total" not in [name for name, _, _ in mixed_source.get_inputs()]


def test_yield_scales_current():
    """Doubling the requested yield doubles the current."""
    one = make_source(current=None, yield_total=1e10)
    two = make_source(current=None, yield_total=2e10)
    assert two.current == pytest.approx(2 * one.current, rel=1e-12)
    assert two.yield_total == pytest.approx(2e10, rel=1e-12)


def test_calculate_current_and_length(mixed_source):
    """Both kinds of calculation together."""
    s = make_source(
        current=None,
        yield_total=mixed_source.yield_total,
        length=[None],
        beam_energy_backstop=mixed_source.beam_energy_grid[-1],
    )
    assert s.length[-1] == pytest.approx(50.0, rel=1e-9)
    assert s.current == pytest.approx(mixed_source.current, rel=1e-8)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(yield_total=1e10), "give exactly one of current and yield_total"),
        (dict(current=None), "give exactly one of current and yield_total"),
        (dict(current=None, yield_total=0.0), "yield_total must be positive"),
        (dict(current=None, yield_total=1e10, voltage=1.0), "no neutrons are produced"),
    ],
    ids=["both", "neither", "zero yield", "no neutrons"],
)
def test_current_or_yield_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_source(**overrides)


def test_beam_power_adds_up(mixed_source):
    """The beam power splits into gas, wall, and backstop with nothing lost or double counted."""
    s = mixed_source
    entry = s.voltage * s.current / 1000
    assert s.wall_power > 0 and s.deposited_power > 0 and s.backstop_power > 0
    assert s.deposited_power + s.wall_power + s.backstop_power == pytest.approx(entry, rel=1e-12)
    assert np.all(s.deposited_power_mids >= 0) and np.all(s.wall_power_mids >= 0)
    assert s.deposited_power_mids_out.sum() == pytest.approx(s.deposited_power, rel=1e-12)
    assert s.wall_power_mids_out.sum() == pytest.approx(s.wall_power, rel=1e-12)


def test_beam_power_without_scraping():
    s = make_source(divergence=[0.0])
    assert s.wall_power == 0
    assert s.deposited_power + s.backstop_power == pytest.approx(s.voltage * s.current / 1000)


def test_neutron_energies(mixed_source):
    """Neutron energies lie in the expected bands for a 280 keV beam."""
    for rxn, lo, hi in [("DD", 1.9, 3.3), ("DT", 12.9, 15.5)]:
        neutron_energy = mixed_source.neutron_energy[rxn]
        assert lo < neutron_energy.min() and neutron_energy.max() < hi
        assert np.all(np.diff(neutron_energy, axis=1) < 0)  # Faster forward than backward


# Reference results for SMALL_CASE: (DD yield [n/s], DT yield [n/s], beam energy at the end
# [keV], fraction of neutrons lost to scraping). These pin the physics: if a change moves them,
# check that it was meant to and update them.
REFERENCE = {
    (0.0, "srim"): (3.6281319841e10, 0.0, 194.2685042747, 1.7222154825e-01),
    (0.0, "icru49"): (3.6135512429e10, 0.0, 191.6598797222, 1.7142870030e-01),
    (0.5, "srim"): (1.8140958175e10, 7.8054183702e11, 194.2802627760, 2.3018646410e-01),
    (0.5, "icru49"): (1.8068057129e10, 7.8737785819e11, 191.6716694943, 2.3159260225e-01),
    (1.0, "srim"): (0.0, 1.5610281756e12, 194.2920207672, 2.3152708373e-01),
    (1.0, "icru49"): (0.0, 1.5746989030e12, 191.6834643214, 2.3296652994e-01),
}


@pytest.mark.parametrize("T_frac, stopping_power", REFERENCE)
def test_reference_results(T_frac, stopping_power):
    source = make_source(T_frac=T_frac, stopping_power=stopping_power)
    yield_DD, yield_DT, E_end, scrape = REFERENCE[(T_frac, stopping_power)]
    # A reaction with no target isotope is absent; its reference yield is 0
    assert source.yields.get("DD", 0.0) == pytest.approx(yield_DD, rel=1e-9)
    assert source.yields.get("DT", 0.0) == pytest.approx(yield_DT, rel=1e-9)
    assert source.beam_energy_grid[-1] == pytest.approx(E_end, rel=1e-9)
    assert source.neutron_scrape_fraction == pytest.approx(scrape, rel=1e-8)


# --- Input checks ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(voltage=0.0), "voltage must be positive"),
        (dict(T_frac=1.5), "T_frac must be between 0 and 1"),
        (dict(current=0.0), "current must be positive"),
        (dict(pressure=[0.0]), "pressures must be positive"),
        (dict(angle_step_out=7.0), "angle_step_out must divide 180"),
        (dict(angle_step=4.0), "multiple of angle_step"),
        (dict(num_x_bins_out=[7]), "must divide num_x_bins"),
        (dict(divergence=[-1.0]), "divergences must not be negative"),
        (dict(angle_step=200.0, angle_step_out=200.0), "angle_step_out must divide 180"),
        (dict(num_x_bins=[0], num_x_bins_out=[1]), "num_x_bins must be positive"),
        (dict(num_x_bins=[2.5], num_x_bins_out=[1]), "must be whole numbers"),
        (dict(num_x_bins_out=[float("nan")]), "must be whole numbers"),
        (dict(wall_radius=0.1), "must not be smaller than start_radius"),
    ],
)
def test_invalid_inputs_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_source(**overrides)


def test_voltage_limit_accepted():
    """Up to where the next DT channel opens there is no warning; above, a warning (in the
    summary and the HDF5 file too), but the case runs."""
    assert make_source(voltage=MAX_VOLTAGE, num_x_bins=[2], num_x_bins_out=[1]).warnings == []
    s = make_source(voltage=MAX_VOLTAGE + 1, num_x_bins=[2], num_x_bins_out=[1])
    assert [w.split(" is open")[0] for w in s.warnings] == ["T(d,np)T"]
    assert tomllib.loads(s.get_summary())["run"]["warnings"] == s.warnings


def test_warnings_in_hdf5(tmp_path):
    s = make_source(voltage=MAX_VOLTAGE + 1, num_x_bins=[2], num_x_bins_out=[1])
    s.write_hdf5(tmp_path / "s.h5")
    with h5py.File(tmp_path / "s.h5") as f:
        assert list(f.attrs["warnings"]) == s.warnings


@pytest.mark.parametrize(
    "overrides",
    [
        dict(
            angle_step=0.1, angle_step_out=0.3
        ),  # 0.3 / 0.1 = 2.9999999999999996 in floating point
        dict(angle_step=0.01, angle_step_out=3.0, num_x_bins=[4], num_x_bins_out=[2]),
        dict(num_x_bins=[40.0], num_x_bins_out=[10.0]),  # Whole numbers given as floats
        dict(wall_radius=0.2, divergence=[0.0]),  # Wall exactly at the beam radius
    ],
)
def test_valid_edge_inputs_accepted(overrides):
    assert make_source(**overrides).yield_total > 0


@pytest.mark.parametrize("overrides", [dict(voltage=1.0), dict(voltage=5.0, T_frac=1.0)])
def test_no_neutrons_rejected(overrides):
    """A beam below the cross-section data (1.8 keV DD, 10 keV DT) gives a clear error."""
    with pytest.raises(ValueError, match="no neutrons are produced"):
        make_source(**overrides)


def test_multiple_regions():
    """Upstream regions before the target chamber: the beam passes through all of them."""
    source = make_source(
        pressure=[1e-3, 5.0],
        temperature=[25.0, 25.0],
        length=[20.0, 50.0],
        divergence=[0.0, 30.0],
        num_x_bins=[4, 40],
        num_x_bins_out=[1, 10],
    )
    assert len(source.x_grid_out) == 12
    assert source.x_grid_out[-1] == pytest.approx(70.0)
    assert source.beam_energy_grid[4] < 280.0  # Energy lost in the upstream region


# --- Outputs ---------------------------------------------------------------------------------


def test_sdef_file(mixed_source, tmp_path):
    path = tmp_path / "sub" / "source.sdef"
    mixed_source.write_mcnp_sdef(path)
    raw = path.read_bytes()
    assert b"\r" not in raw  # Unix line endings on every platform
    raw.decode("ascii")  # Plain ASCII
    text = raw.decode()
    for card in ["sdef  par = n", "si1", "sp1", "si2", "sp2", "ds3 S", "ds4 S", "si501", "sp501"]:
        assert card in text
    assert text == mixed_source.get_mcnp_sdef()
    assert max(len(line) for line in text.splitlines()) <= 128  # MCNP's line-length limit

    # The comment block at the top is the case summary as TOML, plus the SDEF direction
    dashed = "c " + "-" * 78 + "\n"
    block = text.split(dashed)[2]
    header = tomllib.loads("\n".join(line[2:] for line in block.splitlines()))
    run = {
        "sdef_direction": "-z",
        "sdef_entry_position": [0.0, 0.0, 0.0],
        "sdef_backstop_position": [0.0, 0.0, -50.0],  # 50 cm along -z from the entry
    }
    summary = tomllib.loads(mixed_source.get_summary(run))
    assert header == summary
    assert header["run"]["sdef_direction"] == "-z"
    assert header["inputs"]["stopping_power"] == "srim"
    # Then notes on using the source
    notes = text.split(dashed)[3]
    assert "multiply them by yield_total" in notes
    assert "not on position along the" in notes


@pytest.mark.parametrize("direction", ["-x", "+x", "-y", "+y", "-z", "+z"])
def test_sdef_directions(mixed_source, direction):
    text = mixed_source.get_mcnp_sdef(direction)
    assert "axs = " in text and "vec = " in text


@pytest.mark.parametrize(
    "direction, pos, backstop",
    [
        ("+x", "1 2 3", "51 2 3"),
        ("-x", "1 2 3", "-49 2 3"),
        ("+y", "1 2 3", "1 52 3"),
        ("-y", "1 2 3", "1 -48 3"),
        ("+z", "1 2 3", "1 2 53"),
        ("-z", "1 2 -47", "1 2 -47"),  # The -z card runs along +z from the backstop end
    ],
)
def test_sdef_entry_position(mixed_source, direction, pos, backstop):
    """For every direction, the beam enters at entry_position and travels the 50 cm source
    length along the direction to the backstop."""
    text = mixed_source.get_mcnp_sdef(direction, entry_position=(1, 2, 3))
    assert f"        pos = {pos}\n" in text
    assert "The beam enters at (1 2 3) cm" in text
    assert f"reaches the backstop at ({backstop}) cm, 50 cm further on" in text
    assert ("POS is the backstop end" in text) == (direction == "-z")


def test_sdef_default_entry_position(mixed_source):
    assert "        pos = 0 0 0\n" in mixed_source.get_mcnp_sdef("+z")
    assert "        pos = 0 0 -50\n" in mixed_source.get_mcnp_sdef("-z")


@pytest.mark.parametrize("entry", [(1, 2), (1, 2, float("nan")), (1, 2, 3, 4)])
def test_sdef_invalid_entry_position(mixed_source, entry):
    with pytest.raises(ValueError, match="entry_position must be three numbers"):
        mixed_source.get_mcnp_sdef("-z", entry_position=entry)


def test_sdef_invalid_direction(mixed_source):
    with pytest.raises(ValueError, match="direction must be one of"):
        mixed_source.get_mcnp_sdef("z")


def test_isotropic_source_sdef():
    """angle_step_out = 180 gives an isotropic source: no direction distribution."""
    text = make_source(angle_step_out=180.0).get_mcnp_sdef()
    assert "dir = d2" not in text and "si2" not in text


def _sdef_entries(text, card):
    """Return the entries of an SDEF data card (e.g. "si2"), including continuation lines."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"  {card} "))
    entries = lines[start].split()[1:]
    for line in lines[start + 1 :]:
        if not line.startswith("        "):
            break
        entries += line.split()
    return entries


def test_sdef_energy_labels_match_angles(mixed_source):
    """Each energy distribution's comment names the angle bin its data are for: si2 runs from
    180 to 0 degrees, so distribution 501 is 170-180 degrees and has the lowest energies."""
    text = mixed_source.get_mcnp_sdef()
    num = mixed_source.num_angle_bins_out
    assert "sc501 Energy distribution (170 - 180 degrees)" in text
    assert f"sc{500 + num} Energy distribution (0 - 10 degrees)" in text
    # Highest DT neutron energy in each distribution rises from backward to forward bins
    top = [float(_sdef_entries(text, f"si{500 + i}")[-1]) for i in range(1, num + 1)]
    assert all(b > a for a, b in zip(top, top[1:]))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(angle_step_out=180.0),  # Isotropic
        dict(T_frac=0.0),
        dict(
            pressure=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 5.0],
            temperature=[25.0] * 6,
            length=[12.345, 23.456, 34.567, 45.678, 56.789, 50.0],
            divergence=[0.0] * 5 + [30.0],
            num_x_bins=[10] * 5 + [40],
            num_x_bins_out=[1] * 5 + [10],
        ),
    ],
    ids=["mixed", "isotropic", "pure D", "six regions"],
)
def test_sdef_data_lines_within_80_columns(overrides):
    """No input line extends past column 80, apart from comment lines ("c") and "$" comments."""
    for line in make_source(**overrides).get_mcnp_sdef().splitlines():
        if line == "c" or line.startswith("c "):
            continue
        assert len(line.split("$")[0].rstrip()) <= 80, line


def test_sdef_number_formats(mixed_source):
    text = mixed_source.get_mcnp_sdef()
    # Cosines with 15 decimal places (cos 90 degrees written as 0), increasing for MCNP
    entries = _sdef_entries(text, "si2")
    assert all(re.fullmatch(r"-?\d\.\d{15}", v) for v in entries)
    cosines = [float(v) for v in entries]
    np.testing.assert_allclose(cosines, mixed_source.cosine_grid_out[::-1], rtol=0, atol=5e-16)
    assert all(b > a for a, b in zip(cosines, cosines[1:]))
    # Radii to SDEF_RADIUS_DECIMALS decimal places; scraped percentages to 2
    radii = re.findall(r"^  si1\d\d +0 (\S+)", text, re.M)
    assert radii and all(re.fullmatch(rf"\d+\.\d{{{SDEF_RADIUS_DECIMALS}}}", r) for r in radii)
    scraped = re.findall(r"\$ \((\S+)% scraped\)", text)
    assert scraped and all(re.fullmatch(r"\d+\.\d{2}", p) for p in scraped)


def test_sdef_extent_without_float_artefacts():
    """Region boundaries are written as stored, without artifacts from adding lengths
    (0.1 + 0.2 is 0.30000000000000004 in floating point)."""
    source = make_source(
        pressure=[5.0, 5.0],
        temperature=[25.0, 25.0],
        length=[0.1, 0.2],
        divergence=[30.0, 30.0],
        num_x_bins=[2, 4],
        num_x_bins_out=[1, 2],
    )
    assert _sdef_entries(source.get_mcnp_sdef("+z"), "si1") == ["0", "0.1", "i", "0.3"]


def test_hdf5_file(mixed_source, tmp_path):
    s = mixed_source
    path = tmp_path / "source.h5"
    s.write_hdf5(path, {"input_file": "cases.toml"})
    with h5py.File(path) as f:
        assert f.attrs["format"] == "shine-neutron-source"
        assert f.attrs["format_version"] == HDF5_FORMAT_VERSION
        assert f.attrs["name"] == "test"
        assert f.attrs["code_version"]
        assert f.attrs["created"] == s.created
        assert f.attrs["input_file"] == "cases.toml"
        assert f["inputs/material"][0].decode() == "D/T gas"
        assert f["inputs/stopping_power"][()] == b"srim"
        assert f["inputs/voltage"].attrs["units"] == "kV"
        for name, value, units in s.get_results():  # The same results as summary.toml
            np.testing.assert_allclose(f[f"results/{name}"][()], value, rtol=1e-12)
            assert f[f"results/{name}"].attrs["units"] == units
        for key in f:
            if not isinstance(f[key], h5py.Group):
                assert f[key].attrs["description"] and "units" in f[key].attrs
        # Per-bin beam values, as used in the SDEF card and the axial profile
        np.testing.assert_allclose(f["radius_mids"][()], s.radius_mids_out)
        np.testing.assert_allclose(f["scrape_mids"][()], s.scrape_mids_out)
        assert len(f["beam_energy_mids"]) == len(f["current_mids"]) == len(f["x_grid"]) - 1
        for rxn in ("DD", "DT"):
            neutrons = f[f"neutrons_{rxn}"]
            assert list(neutrons.attrs["dimensions"]) == ["position", "angle", "energy"]
            assert neutrons.attrs["units"] == "n/s"
            assert neutrons.shape == (
                len(f["x_grid"]) - 1,
                len(f["angle_grid"]) - 1,
                len(f[f"neutron_energy_grid_{rxn}"]) - 1,
            )
            assert neutrons[()].sum() == pytest.approx(mixed_source.yields[rxn], rel=1e-12)


def test_text_files(mixed_source, tmp_path):
    s = mixed_source
    s.write_text_files(tmp_path)
    for fname in [
        "summary.toml",
        "axial_profile.txt",
        "energy_spectrum.txt",
        "angular_distribution.txt",
    ]:
        raw = (tmp_path / fname).read_bytes()
        assert b"\r" not in raw  # Unix line endings
        raw.decode("ascii")
        assert not any(line.endswith(b" ") for line in raw.splitlines())

    # Each profile sums to the yields (values are written to 7 significant figures)
    def check_sums(table, dd_col, dt_col, total_col):
        assert table[:, dd_col].sum() == pytest.approx(s.yields["DD"], rel=1e-6)
        assert table[:, dt_col].sum() == pytest.approx(s.yields["DT"], rel=1e-6)
        np.testing.assert_allclose(
            table[:, total_col], table[:, dd_col] + table[:, dt_col], rtol=1e-6
        )

    axial = np.loadtxt(tmp_path / "axial_profile.txt")
    np.testing.assert_allclose(axial[:, 0], s.x_grid_out[:-1], atol=1e-9)
    np.testing.assert_allclose(axial[:, 1], s.x_grid_out[1:], rtol=1e-6)
    assert np.all(np.diff(axial[:, 2]) < 0)  # The beam slows down along the target
    np.testing.assert_allclose(axial[:, 3], s.radius_mids_out, rtol=1e-6)
    assert np.all(np.diff(axial[:, 3]) >= 0) and axial[-1, 3] <= s.wall_radius * (1 + 1e-6)
    np.testing.assert_allclose(axial[:, 4], s.scrape_mids_out, rtol=1e-6, atol=1e-12)
    assert np.all(np.diff(axial[:, 4]) >= 0) and 0 < axial[-1, 4] < 1  # Scraping grows
    check_sums(axial, 5, 6, 7)
    assert axial[:, 8].sum() == pytest.approx(s.deposited_power, rel=1e-5)
    assert axial[:, 9].sum() == pytest.approx(s.wall_power, rel=1e-5)

    energy = np.loadtxt(tmp_path / "energy_spectrum.txt")
    np.testing.assert_allclose(energy[:, 1] - energy[:, 0], s.energy_bin_width, rtol=1e-4)
    np.testing.assert_allclose(energy[1:, 0], energy[:-1, 1], rtol=1e-6)  # Contiguous bins
    assert energy[0, 0] > 1.5 and energy[-1, 1] < 16  # From the DD peak to the DT peak
    assert energy[0, 2] > 0 and energy[-1, 3] > 0  # Starts and ends on actual neutrons
    check_sums(energy, 2, 3, 4)

    angular = np.loadtxt(tmp_path / "angular_distribution.txt")
    assert angular[0, 0] == 0 and angular[-1, 1] == 180
    assert angular[:, 2].sum() == pytest.approx(4 * np.pi, rel=1e-6)
    check_sums(angular, 3, 4, 5)
    np.testing.assert_allclose(angular[:, 6], angular[:, 5] / angular[:, 2], rtol=1e-5)

    with open(tmp_path / "summary.toml", "rb") as f:
        summary = tomllib.load(f)  # The summary is valid TOML
    assert summary["run"]["name"] == "test"
    assert summary["run"]["code_version"]
    for name, value, _ in s.get_inputs():  # Every input, with the right type
        expected = value.tolist() if isinstance(value, np.ndarray) else value
        if isinstance(expected, str) or (
            isinstance(expected, list) and isinstance(expected[0], str)
        ):
            assert summary["inputs"][name] == expected
        else:
            assert summary["inputs"][name] == pytest.approx(expected, rel=1e-6)
        assert type(summary["inputs"][name]) is type(expected)
    results = summary["results"]
    assert results["yield_total"] == pytest.approx(s.yield_total, rel=1e-6)
    assert results["yield_DD"] + results["yield_DT"] == pytest.approx(s.yield_total, rel=1e-6)
    assert results["fraction_DD"] + results["fraction_DT"] == pytest.approx(1, rel=1e-6)
    assert 2.0 < results["mean_neutron_energy_DD"] < 3.0
    assert 13.5 < results["mean_neutron_energy_DT"] < 14.8
    assert results["beam_energy_backstop"] == pytest.approx(s.beam_energy_grid[-1], rel=1e-6)
    assert results["beam_energy_region_end"] == pytest.approx([s.beam_energy_grid[-1]], rel=1e-6)
    assert results["beam_power_entry"] == pytest.approx(280 * 10 / 1000)  # kV x mA -> kW
    parts = [results[f"beam_power_{p}"] for p in ("deposited", "wall", "backstop")]
    assert sum(parts) == pytest.approx(results["beam_power_entry"], rel=1e-6)
    assert results["beam_current_backstop"] < 10  # Part of the beam is scraped off
    assert results["neutron_scrape_fraction"] == pytest.approx(s.neutron_scrape_fraction, rel=1e-6)


def test_text_files_pure_deuterium(tmp_path):
    """With no tritium there is no DT reaction: the files have DD columns only, the spectrum
    covers DD energies only, and the summary gives no DT yield or mean energy."""
    make_source(T_frac=0.0).write_text_files(tmp_path)
    for name in ("energy_spectrum.txt", "axial_profile.txt", "angular_distribution.txt"):
        text = (tmp_path / name).read_text()
        assert "DD neutrons" in text and "DT neutrons" not in text
    energy = np.loadtxt(tmp_path / "energy_spectrum.txt")
    assert energy[-1, 1] < 4  # DD neutrons only
    with open(tmp_path / "summary.toml", "rb") as f:
        results = tomllib.load(f)["results"]
    assert "yield_DT" not in results and "mean_neutron_energy_DT" not in results
    assert results["yield_DD"] == pytest.approx(results["yield_total"], rel=1e-6)


@pytest.mark.parametrize("T_frac, num_plots", [(0.0, 7), (0.5, 8)])
def test_plots(tmp_path, T_frac, num_plots):
    make_source(T_frac=T_frac).make_plots(tmp_path / "plots")
    assert len(list((tmp_path / "plots").glob("*.png"))) == num_plots


@pytest.mark.parametrize(
    "overrides",
    [
        dict(voltage=1000.0),  # DT neutrons above 16 MeV
        dict(voltage=MAX_VOLTAGE, num_x_bins=[4], num_x_bins_out=[2]),
        dict(energy_bin_width=5.0),  # Energy bins coarser than the spectra
        dict(energy_bin_width=0.03),  # Bins that do not divide 16 MeV
    ],
)
def test_plots_and_sdef_at_any_energy(tmp_path, overrides):
    """The energy plots follow the neutron energies, so they work for any beam energy and
    energy bin size."""
    source = make_source(**overrides)
    source.get_mcnp_sdef()
    source.make_plots(tmp_path / "plots")
    assert len(list((tmp_path / "plots").glob("*.png"))) == 8


# --- Regions: per-region walls and values calculated in any region ----------------------------

TWO_REGIONS = dict(
    pressure=[5.0, 5.0],
    temperature=[25.0, 25.0],
    length=[25.0, 25.0],
    divergence=[30.0, 30.0],
    num_x_bins=[20, 20],
    num_x_bins_out=[5, 5],
)


def test_single_wall_scraping_formula(mixed_source):
    """With one wall radius, the scraped fraction is the flat-beam 1 - (wall / r)^2."""
    s = mixed_source
    free = s.start_radius + np.append(0, np.cumsum(s.divergence_mids / 1000 * s.dx))
    expected = np.clip(1 - (1.0 / free) ** 2, 0, None)
    np.testing.assert_allclose(s.scrape_grid, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(s.radius_grid, np.minimum(free, 1.0), rtol=1e-12)


def test_narrow_aperture_scrapes():
    """A narrow first region scrapes the beam; the wider region after it does not give the
    scraped beam back."""
    narrow = make_source(**TWO_REGIONS, wall_radius=[0.5, 1.0])
    wide = make_source(**TWO_REGIONS, wall_radius=[1.0, 1.0])
    assert np.all(np.diff(narrow.scrape_grid) >= 0)  # Scraping never goes back
    assert narrow.scrape_grid[-1] > wide.scrape_grid[-1]
    assert narrow.wall_power > wide.wall_power
    # The beam left after the aperture: (0.5 / r)^2 at the end of the first region
    r_end_first = 0.2 + 30e-3 * 25.0
    i = narrow.num_x_bins[0]
    assert narrow.scrape_grid[i] == pytest.approx(1 - (0.5 / r_end_first) ** 2, rel=1e-12)
    # The beam edge is clipped to the aperture there, then grows from it
    assert narrow.radius_grid[i] == pytest.approx(0.5, rel=1e-12)
    assert narrow.radius_grid[-1] > 0.5


def test_first_wall_must_hold_the_beam():
    with pytest.raises(ValueError, match="the first region's wall_radius must not be smaller"):
        make_source(**TWO_REGIONS, wall_radius=[0.1, 1.0])


@pytest.mark.parametrize("quantity", ["length", "pressure"])
def test_value_calculated_in_first_region(quantity):
    """The one missing value can be in any region, not only the last."""
    forward = make_source(**TWO_REGIONS)
    values = {quantity: [None, TWO_REGIONS[quantity][1]]}
    s = make_source(**{**TWO_REGIONS, **values}, beam_energy_backstop=forward.beam_energy_grid[-1])
    assert (s.calculated, s.calculated_region) == (quantity, 0)
    assert getattr(s, quantity)[0] == pytest.approx(TWO_REGIONS[quantity][0], rel=1e-8)
    inputs = {name: units for name, _, units in s.get_inputs()}
    assert inputs[quantity].endswith("calculated for region 1")


def test_only_one_value_calculated():
    with pytest.raises(ValueError, match="the length of region 1 .*the length of region 2"):
        make_source(**{**TWO_REGIONS, "length": [None, None]}, beam_energy_backstop=150.0)
