"""Whole calculations with ShineSource: physics checks, input checks and outputs."""

import h5py
import numpy as np
import pytest
from conftest import make_source

from shine_neutron_source.source import MAX_VOLTAGE

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
    ],
)
def test_invalid_inputs_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_source(**overrides)


def test_voltage_limit_accepted():
    """The highest allowed voltage (the top of the cross-section tables) is accepted."""
    assert make_source(voltage=MAX_VOLTAGE, num_x_ints_list=[2], num_x_ints_list_C=[1])


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
    assert "Stopping power     : srim" in text
    assert text == mixed_source.get_mcnp_sdef()


@pytest.mark.parametrize("direction", ["-x", "+x", "-y", "+y", "-z", "+z"])
def test_sdef_directions(mixed_source, direction):
    text = mixed_source.get_mcnp_sdef(direction)
    assert "axs = " in text and "vec = " in text


def test_sdef_invalid_direction(mixed_source):
    with pytest.raises(ValueError, match="direction must be one of"):
        mixed_source.get_mcnp_sdef("z")


def test_isotropic_source_sdef():
    """angle_int_C = 180 gives an isotropic source: no direction distribution."""
    text = make_source(angle_int_C=180.0).get_mcnp_sdef()
    assert "dir = d2" not in text and "si2" not in text


def test_hdf5_file(mixed_source, tmp_path):
    path = tmp_path / "source.h5"
    mixed_source.write_hdf5(path)
    with h5py.File(path) as f:
        assert f.attrs["name"] == "test"
        assert f.attrs["code_version"]
        assert f["inputs/T_frac"][()] == 0.5
        assert f["inputs/stopping_power"][()] == b"srim"
        assert f["inputs/voltage"].attrs["units"] == "kV"
        for rxn in ("DD", "DT"):
            neutrons = f[f"neutrons_{rxn}"]
            assert neutrons.attrs["units"] == "n/s"
            assert neutrons.shape == (
                len(f["x_grid"]) - 1,
                len(f["angle_grid"]) - 1,
                len(f[f"EN_grid_{rxn}"]) - 1,
            )
            assert neutrons[()].sum() == pytest.approx(mixed_source.nyield[rxn], rel=1e-12)


@pytest.mark.parametrize("T_frac, num_plots", [(0.0, 8), (0.5, 9)])
def test_plots(tmp_path, T_frac, num_plots):
    make_source(T_frac=T_frac).make_plots(tmp_path / "plots")
    assert len(list((tmp_path / "plots").glob("*.png"))) == num_plots
