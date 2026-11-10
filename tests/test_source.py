"""Whole calculations with ShineSource: physics checks, input checks and outputs."""

import tomllib

import h5py
import numpy as np
import pytest
from conftest import make_source

from shine_neutron_source.source import HDF5_FORMAT_VERSION, MAX_VOLTAGE

# --- Physics checks ------------------------------------------------------------------------


@pytest.mark.parametrize("rxn", ["DD", "DT"])
def test_lab_cross_section_integrates_to_total(rxn):
    """At every position, the lab-frame cross section summed over all emission angles equals
    the total cross section 4 pi sigma_0 C_0 from the data (the CM-to-lab conversion conserves
    reactions)."""
    source = make_source(angle_int=0.25)
    xs = source.xs_data[rxn]
    total = (
        4
        * np.pi
        * np.interp(source.EB_mids, xs["E"], xs["total"])
        * np.interp(source.EB_mids, xs["E"], xs["coeffs"][:, 0])
    )
    lab = np.sum(source.xs[rxn] * source.angle_areas[None, :], axis=1)
    np.testing.assert_allclose(lab, total, rtol=1e-4)


def test_yields_add_up(mixed_source):
    s = mixed_source
    assert s.nyield["DD"] == pytest.approx(s.neutrons["DD"].sum(), rel=1e-12)
    assert s.nyield["DT"] == pytest.approx(s.neutrons["DT"].sum(), rel=1e-12)
    assert s.nyield_tot == s.nyield["DD"] + s.nyield["DT"]
    assert 0 < s.nscrape_tot < 1
    for rxn in ("DD", "DT"):
        assert np.all(s.neutrons[rxn] >= 0)


@pytest.mark.parametrize("T_frac, absent", [(0.0, "DT"), (1.0, "DD")])
def test_pure_gases(T_frac, absent):
    source = make_source(T_frac=T_frac)
    assert source.nyield[absent] == 0
    assert source.nyield_tot > 0


def test_yield_scales_with_tritium_fraction():
    """At fixed atom density, the DT yield is proportional to T_frac (stopping hardly depends on
    the mix), and the DD yield to 1 - T_frac."""
    pure_D, pure_T, mix = make_source(T_frac=0.0), make_source(T_frac=1.0), make_source(T_frac=0.5)
    assert mix.nyield["DT"] / pure_T.nyield["DT"] == pytest.approx(0.5, rel=2e-3)
    assert mix.nyield["DD"] / pure_D.nyield["DD"] == pytest.approx(0.5, rel=2e-3)


def test_beam_slows_down(mixed_source):
    EB = mixed_source.EB_grid
    assert EB[0] == 280.0
    assert np.all(np.diff(EB) < 0)
    assert 150 < EB[-1] < 250  # A thin target: the beam passes through


@pytest.mark.parametrize("stopping_power", ["srim", "icru49"])
@pytest.mark.parametrize("T_frac", [0.0, 1.0])
def test_beam_stops_inside_target(T_frac, stopping_power):
    """When the beam stops well before the end of the target, everything beyond the stopping
    point is exactly zero: the beam energy, the cross sections and the neutrons. There must be
    no low-level tail of neutrons all the way to the backstop."""
    source = make_source(
        T_frac=T_frac,
        stopping_power=stopping_power,
        pressure_list=[40.0],
        length_list=[300.0],
        num_x_ints_list=[600],
        num_x_ints_list_C=[60],
        max_radius=10.0,
    )
    EB = source.EB_grid
    assert EB[-1] == 0, f"The beam never stops: it reaches the backstop with {EB[-1]} keV"
    i_stop = np.flatnonzero(EB > 0)[-1]  # Last grid point the beam reaches with energy
    x_stop = source.x_grid[i_stop + 1]
    assert 15 < x_stop < 30  # Stops within the first tenth of the 300 cm target

    rxn = "DT" if T_frac else "DD"
    assert np.all(EB[i_stop + 1 :] == 0)
    assert np.all(source.EB_mids[i_stop + 1 :] == 0)
    assert np.all(source.xs[rxn][i_stop + 1 :] == 0)
    neutrons_per_interval = source.neutrons[rxn].sum(axis=(1, 2))
    beyond = source.x_grid_C[:-1] >= x_stop  # Output intervals entirely past the stop
    assert beyond.sum() > 50
    assert np.all(neutrons_per_interval[beyond] == 0)
    assert source.nyield[rxn] > 0
    assert source.EB_grid_C[-1] == 0  # Beam energy at the backstop


def test_neutron_energies(mixed_source):
    """Neutron energies lie in the expected bands for a 280 keV beam."""
    for rxn, lo, hi in [("DD", 1.9, 3.3), ("DT", 12.9, 15.5)]:
        EN = mixed_source.EN[rxn]
        assert lo < EN.min() and EN.max() < hi
        assert np.all(np.diff(EN, axis=1) < 0)  # Faster forward than backward


# Reference results for SMALL_CASE: (DD yield [n/s], DT yield [n/s], beam energy at the end
# [keV], fraction of neutrons lost to scraping). These pin the physics: if a change moves them,
# check that it was meant to and update them.
REFERENCE = {
    (0.0, "srim"): (3.6280823956e10, 0.0, 194.2617384151, 1.7221908798e-01),
    (0.0, "icru49"): (3.6135521924e10, 0.0, 191.6598774432, 1.7142869197e-01),
    (0.5, "srim"): (1.8140788955e10, 7.8055872871e11, 194.2709754563, 2.3019247174e-01),
    (0.5, "icru49"): (1.8068061908e10, 7.8737788283e11, 191.6716677698, 2.3159260331e-01),
    (1.0, "srim"): (0.0, 1.5610496159e12, 194.2802155775, 2.3153508122e-01),
    (1.0, "icru49"): (0.0, 1.5746989457e12, 191.6834631549, 2.3296653056e-01),
}


@pytest.mark.parametrize("T_frac, stopping_power", REFERENCE)
def test_reference_results(T_frac, stopping_power):
    source = make_source(T_frac=T_frac, stopping_power=stopping_power)
    yield_DD, yield_DT, E_end, scrape = REFERENCE[(T_frac, stopping_power)]
    assert source.nyield["DD"] == pytest.approx(yield_DD, rel=1e-9)
    assert source.nyield["DT"] == pytest.approx(yield_DT, rel=1e-9)
    assert source.EB_grid[-1] == pytest.approx(E_end, rel=1e-9)
    assert source.nscrape_tot == pytest.approx(scrape, rel=1e-8)


# --- Input checks ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(voltage=MAX_VOLTAGE + 0.1), "voltage must be between"),
        (dict(T_frac=1.5), "T_frac must be between 0 and 1"),
        (dict(current=0.0), "current must be positive"),
        (dict(pressure_list=[float("nan")]), "pressures must be positive"),
        (dict(temperature_list=[25.0, 25.0]), "temperature_list has 2 entries"),
        (dict(angle_int_C=7.0), "angle_int_C must divide 180"),
        (dict(angle_int=4.0), "multiple of angle_int"),
        (dict(num_x_ints_list_C=[7]), "must divide num_x_ints_list"),
        (dict(divergence_list=[-1.0]), "divergences must not be negative"),
        (dict(angle_int=200.0, angle_int_C=200.0), "angle_int_C must divide 180"),
        (dict(num_x_ints_list=[0], num_x_ints_list_C=[1]), "num_x_ints_list must be positive"),
        (dict(num_x_ints_list=[2.5], num_x_ints_list_C=[1]), "must be whole numbers"),
        (dict(num_x_ints_list_C=[float("nan")]), "must be whole numbers"),
        (dict(max_radius=0.1), "must not be smaller than start_radius"),
    ],
)
def test_invalid_inputs_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_source(**overrides)


def test_voltage_limit_accepted():
    """The highest allowed voltage (the top of the cross-section tables) is accepted."""
    assert make_source(voltage=MAX_VOLTAGE, num_x_ints_list=[2], num_x_ints_list_C=[1])


@pytest.mark.parametrize(
    "overrides",
    [
        dict(angle_int=0.1, angle_int_C=0.3),  # 0.3 / 0.1 = 2.9999999999999996 in floating point
        dict(angle_int=0.01, angle_int_C=3.0, num_x_ints_list=[4], num_x_ints_list_C=[2]),
        dict(num_x_ints_list=[40.0], num_x_ints_list_C=[10.0]),  # Whole numbers given as floats
        dict(max_radius=0.2, divergence_list=[0.0]),  # Wall exactly at the beam radius
    ],
)
def test_valid_edge_inputs_accepted(overrides):
    assert make_source(**overrides).nyield_tot > 0


@pytest.mark.parametrize("overrides", [dict(voltage=1.0), dict(voltage=5.0, T_frac=1.0)])
def test_no_neutrons_rejected(overrides):
    """A beam below the cross-section data (1.8 keV DD, 10 keV DT) gives a clear error."""
    with pytest.raises(ValueError, match="no neutrons are produced"):
        make_source(**overrides)


def test_multiple_regions():
    """Upstream regions before the target chamber: the beam passes through all of them."""
    source = make_source(
        pressure_list=[1e-3, 5.0],
        temperature_list=[25.0, 25.0],
        length_list=[20.0, 50.0],
        divergence_list=[0.0, 30.0],
        num_x_ints_list=[4, 40],
        num_x_ints_list_C=[1, 10],
    )
    assert len(source.x_grid_C) == 12
    assert source.x_grid_C[-1] == pytest.approx(70.0)
    assert source.EB_grid[4] < 280.0  # Energy lost in the upstream region


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
    run = {"sdef_direction": "-z", "sdef_entry_position": [0.0, 0.0, 0.0]}
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
    """angle_int_C = 180 gives an isotropic source: no direction distribution."""
    text = make_source(angle_int_C=180.0).get_mcnp_sdef()
    assert "dir = d2" not in text and "si2" not in text


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
        assert f["inputs/T_frac"][()] == 0.5
        assert f["inputs/stopping_power"][()] == b"srim"
        assert f["inputs/voltage"].attrs["units"] == "kV"
        for name, value, units in s.get_results():  # The same results as summary.toml
            np.testing.assert_allclose(f[f"results/{name}"][()], value, rtol=1e-12)
            assert f[f"results/{name}"].attrs["units"] == units
        for key in f:
            if not isinstance(f[key], h5py.Group):
                assert f[key].attrs["description"] and "units" in f[key].attrs
        # Per-interval beam values, as used in the SDEF card and the axial profile
        np.testing.assert_allclose(f["radius_mids"][()], s.radius_mids_C)
        np.testing.assert_allclose(f["scrape_mids"][()], s.scrape_mids_C)
        assert len(f["EB_mids"]) == len(f["current_mids"]) == len(f["x_grid"]) - 1
        for rxn in ("DD", "DT"):
            neutrons = f[f"neutrons_{rxn}"]
            assert list(neutrons.attrs["dimensions"]) == ["position", "angle", "energy"]
            assert neutrons.attrs["units"] == "n/s"
            assert neutrons.shape == (
                len(f["x_grid"]) - 1,
                len(f["angle_grid"]) - 1,
                len(f[f"EN_grid_{rxn}"]) - 1,
            )
            assert neutrons[()].sum() == pytest.approx(mixed_source.nyield[rxn], rel=1e-12)


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
        assert table[:, dd_col].sum() == pytest.approx(s.nyield["DD"], rel=1e-6)
        assert table[:, dt_col].sum() == pytest.approx(s.nyield["DT"], rel=1e-6)
        np.testing.assert_allclose(
            table[:, total_col], table[:, dd_col] + table[:, dt_col], rtol=1e-6
        )

    axial = np.loadtxt(tmp_path / "axial_profile.txt")
    np.testing.assert_allclose(axial[:, 0], s.x_grid_C[:-1], atol=1e-9)
    np.testing.assert_allclose(axial[:, 1], s.x_grid_C[1:], rtol=1e-6)
    assert np.all(np.diff(axial[:, 2]) < 0)  # The beam slows down along the target
    np.testing.assert_allclose(axial[:, 3], s.radius_mids_C, rtol=1e-6)
    assert np.all(np.diff(axial[:, 3]) >= 0) and axial[-1, 3] <= s.max_radius * (1 + 1e-6)
    np.testing.assert_allclose(axial[:, 4], s.scrape_mids_C, rtol=1e-6, atol=1e-12)
    assert np.all(np.diff(axial[:, 4]) >= 0) and 0 < axial[-1, 4] < 1  # Scraping grows
    check_sums(axial, 5, 6, 7)

    energy = np.loadtxt(tmp_path / "energy_spectrum.txt")
    np.testing.assert_allclose(energy[:, 1] - energy[:, 0], s.EN_int, rtol=1e-4)
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
        if isinstance(expected, str):
            assert summary["inputs"][name] == expected
        else:
            assert summary["inputs"][name] == pytest.approx(expected, rel=1e-6)
        assert type(summary["inputs"][name]) is type(expected)
    results = summary["results"]
    assert results["yield_total"] == pytest.approx(s.nyield_tot, rel=1e-6)
    assert results["yield_DD"] + results["yield_DT"] == pytest.approx(s.nyield_tot, rel=1e-6)
    assert results["fraction_DD"] + results["fraction_DT"] == pytest.approx(1, rel=1e-6)
    assert 2.0 < results["mean_neutron_energy_DD"] < 3.0
    assert 13.5 < results["mean_neutron_energy_DT"] < 14.8
    assert results["beam_energy_backstop"] == pytest.approx(s.EB_grid[-1], rel=1e-6)
    assert results["beam_energy_region_end"] == pytest.approx([s.EB_grid[-1]], rel=1e-6)
    assert results["beam_power_entry"] == pytest.approx(280 * 10 / 1000)  # kV x mA -> kW
    assert results["beam_current_backstop"] < 10  # Part of the beam is scraped off
    assert results["neutron_scrape_fraction"] == pytest.approx(s.nscrape_tot, rel=1e-6)


def test_text_files_pure_deuterium(tmp_path):
    """With no tritium, the DT columns are all zero, the spectrum covers DD energies only and
    the summary gives no mean DT neutron energy."""
    make_source(T_frac=0.0).write_text_files(tmp_path)
    energy = np.loadtxt(tmp_path / "energy_spectrum.txt")
    assert np.all(energy[:, 3] == 0)
    assert energy[-1, 1] < 4  # DD neutrons only
    assert np.all(np.loadtxt(tmp_path / "axial_profile.txt")[:, 6] == 0)
    with open(tmp_path / "summary.toml", "rb") as f:
        results = tomllib.load(f)["results"]
    assert results["yield_DT"] == 0 and "mean_neutron_energy_DT" not in results


@pytest.mark.parametrize("T_frac, num_plots", [(0.0, 8), (0.5, 9)])
def test_plots(tmp_path, T_frac, num_plots):
    make_source(T_frac=T_frac).make_plots(tmp_path / "plots")
    assert len(list((tmp_path / "plots").glob("*.png"))) == num_plots


@pytest.mark.parametrize(
    "overrides",
    [
        dict(voltage=1000.0),  # DT neutrons above 16 MeV
        dict(voltage=MAX_VOLTAGE, num_x_ints_list=[4], num_x_ints_list_C=[2]),
        dict(EN_int=5.0),  # Energy bins coarser than the spectra
        dict(EN_int=0.03),  # Bins that do not divide 16 MeV
    ],
)
def test_plots_and_sdef_at_any_energy(tmp_path, overrides):
    """The energy plots follow the neutron energies, so they work for any beam energy and
    energy bin size."""
    source = make_source(**overrides)
    source.get_mcnp_sdef()
    source.make_plots(tmp_path / "plots")
    assert len(list((tmp_path / "plots").glob("*.png"))) == 9
