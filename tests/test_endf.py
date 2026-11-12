"""Evaluated nuclear data (endf.py) and reactions calculated from it."""

import os
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest

from shine_neutron_source import Material, Region, ShineSource, load_input
from shine_neutron_source.endf import (
    NeutronProduction,
    _number,
    _parse_name,
    find_evaluation,
    separation_energy,
)

DATA = Path(__file__).parent / "data" / "endf"
ENDFB = DATA / "ENDF-B-VIII.1"
TENDL21 = DATA / "TENDL-2021"
P_BE9 = ENDFB / "protons" / "p-004_Be_009.endf"
A_BE9_TENDL = TENDL21 / "a-Be009.tendl"
BE = Material("Be", "solid", atoms={"Be": 1}, density=1.848)

# MCNP: neutrons created by charged-particle reactions per beam particle in thick Be
# (MCNP 6.2 with ENDF/B-VII.0 protons, the same LANL evaluation as ENDF/B-VIII.1; MCNP 6.3
# with TENDL-2021 alphas)
MCNP_PROTONS = {10: 3.3893e-03, 18: 1.2004e-02, 30: 2.9426e-02}
MCNP_ALPHAS = {30: 3.1157e-03, 70: 1.4677e-02}


def thick_be(beam, energy_MeV, nuclear_data, length=1.0, num_x_bins=600, **kw):
    """A Be block thick enough to stop the beam."""
    charge = 2 if beam in ("h", "a") else 1
    return ShineSource(
        name=f"{beam}{energy_MeV}",
        regions=[
            Region(
                BE,
                length=length,
                wall_radius=10.0,
                num_x_bins=num_x_bins,
                num_x_bins_out=num_x_bins // 10,
            )
        ],  # fmt: skip
        voltage=energy_MeV * 1000 / charge,
        current=1.0,
        start_radius=0.1,
        angle_step=1.0,
        angle_step_out=10.0,
        energy_bin_width=0.1,
        beam=beam,
        nuclear_data=None if nuclear_data is None else [str(d) for d in nuclear_data],
        **kw,
    )


# --- Reading evaluations -------------------------------------------------------------------


def test_reads_proton_evaluation():
    x = NeutronProduction(P_BE9)
    assert (x.beam, x.target, x.emax) == ("p", (4, 9), 113.0)
    assert [(c.mt, c.law) for c in x.channels] == [(5, 1)] + [(mt, 2) for mt in range(50, 56)]
    # Below the (p,n) threshold of 2.06 MeV there are no neutrons
    assert x.production_cross_section(2.0) == 0
    assert x.production_cross_section(18.0) == pytest.approx(0.4607, abs=1e-4)
    assert x.missing_fraction(18.0) == 0


def test_reads_tendl_evaluation():
    x = NeutronProduction(A_BE9_TENDL)
    assert (x.beam, x.target, x.emax) == ("a", (4, 9), 200.0)
    # Kalbach-Mann with the slope from the systematics: no slope in the file
    dist = x.channels[0].distributions[5]
    assert dist.kind == "kalbach" and dist.params.shape[1] == 1


@pytest.mark.parametrize("path, energy", [(P_BE9, 18.0), (P_BE9, 70.0), (A_BE9_TENDL, 40.0)])
def test_lab_spectrum_conserves_the_total(path, energy):
    """Binned into lab angles and energies, the spectrum adds up to the production cross
    section exactly, and forward neutrons are faster than backward ones."""
    x = NeutronProduction(path)
    angles = np.arange(0, 181, 10.0)
    top = x.max_lab_energy(energy)
    edges = np.arange(0, np.ceil(top / 0.1) * 0.1 + 0.05, 0.1)
    spectrum, total = x.lab_spectrum(energy, angles, edges)
    assert spectrum.sum() == pytest.approx(x.production_cross_section(energy), rel=1e-10)
    assert total == pytest.approx(spectrum.sum(), rel=1e-12)
    mids = (edges[:-1] + edges[1:]) / 2
    mean = np.sum(spectrum * mids, axis=1) / spectrum.sum(axis=1)
    assert mean[0] > mean[-1]


@pytest.mark.parametrize(
    "field, value",
    [(" 1.234567+3", 1234.567), ("-2.500000-12", -2.5e-12), (" 6.01200+ 3", 6012.0),
     (" 1.00000- 5", 1e-5), ("  1.5e+2   ", 150.0), ("           ", 0.0)],
)  # fmt: skip
def test_endf_numbers(field, value):
    """ENDF numbers without the "e", including Fortran's blank after the exponent's sign."""
    assert _number(field) == pytest.approx(value, rel=1e-12)


def test_continuum_is_mixed_by_unit_base():
    """Between tabulated incident energies (16 and 18 MeV for p + 9Be), the continuum's upper
    edge moves with the beam energy. Mixed literally (INT=2), it jumped to near the 18 MeV
    edge just above 16 MeV: a stripe in a thick target, and neutrons above what the
    reaction can give."""
    x = NeutronProduction(P_BE9)
    assert all(c.unit_base for c in x.channels if c.law == 1)
    edges = np.arange(0, 20.0001, 0.05)

    def top(E):  # Highest forward (0-10 degree) neutron energy of the continuum
        spectrum, _ = x.lab_spectrum(E, [0.0, 10.0, 180.0], edges, two_body=False)
        return edges[np.flatnonzero(spectrum[0] > spectrum[0].max() * 1e-6)[-1] + 1]

    low, mid, high = top(16.0), top(17.0), top(18.0)
    assert low < mid < high
    assert mid == pytest.approx((low + high) / 2, abs=0.1)


def test_two_body_channels_at_each_beam_energy():
    """two_body_spectrum, at one beam energy, gives the two-body part of lab_spectrum."""
    x = NeutronProduction(P_BE9)
    angles = np.arange(0, 181, 10.0)
    edges = np.arange(0, 12.05, 0.1)
    full, total = x.lab_spectrum(12.0, angles, edges, num_cm_angles=360)
    rest, total_rest = x.lab_spectrum(12.0, angles, edges, num_cm_angles=360, two_body=False)
    two_body, sigma = x.two_body_spectrum(
        [12.0, 12.0], [1.0, 0.0], [0, 1], 2, angles, edges, num_cm_angles=360
    )
    assert np.allclose(two_body[0], full - rest, rtol=1e-12, atol=1e-15)
    assert not two_body[1].any()  # Its weight is zero
    assert sigma[0] == pytest.approx(total - total_rest, rel=1e-12)
    assert sigma[0] == pytest.approx(
        sum(c.production(12.0) for c in x.channels if c.law == 2), rel=1e-12
    )


def test_kalbach_slope_systematics():
    """The slope grows with the emitted energy; separation energies are sensible (a
    neutron is bound by roughly 6-8 MeV in 12C, the compound of alpha + 9Be)."""
    x = NeutronProduction(A_BE9_TENDL)
    a = x.kalbach_slope(40.0, np.array([1.0, 10.0, 30.0]))
    assert np.all(np.diff(a) > 0) and np.all(a > 0)
    assert 3 < separation_energy(6, 13, 6, 12, (0, 1)) < 10


@pytest.mark.parametrize(
    "name, parsed",
    [
        ("p-Be009-mt.tendl", ("p", "Be", 9, 0)),
        ("a-Be009-s0.tendl", ("a", "Be", 9, 4)),
        ("p-004_Be_009.endf", ("p", "Be", 9, 0)),
        ("d_0425_4-Be-9.endf", ("d", "Be", 9, 0)),
        ("n-004_Be_009.endf", None),  # Neutron data: not a beam
        ("p-Ac222m-mt.tendl", None),  # Isomer targets are not used
    ],
)
def test_library_file_names(name, parsed):
    assert _parse_name(name) == parsed


def test_find_evaluation():
    assert Path(find_evaluation(ENDFB, "p", (4, 9))) == P_BE9
    assert Path(find_evaluation(TENDL21, "a", (4, 9))) == A_BE9_TENDL
    assert find_evaluation(ENDFB, "d", (4, 9)) is None
    assert find_evaluation(DATA / "missing", "p", (4, 9)) is None


# --- Thick-target benchmarks against MCNP --------------------------------------------------


@pytest.mark.parametrize("energy", list(MCNP_PROTONS))
def test_protons_on_thick_be_match_mcnp(energy):
    s = thick_be("p", energy, ["drosg", ENDFB], length=1.0)
    assert s.beam_energy_grid[-1] == 0  # The beam stops in the target
    assert s.yield_total / s.particle_rate == pytest.approx(MCNP_PROTONS[energy], rel=0.01)
    assert s.reaction_data["p9Be_xn"].startswith("ENDF/B-VIII.1")
    assert s.warnings == []


@pytest.mark.parametrize("energy", list(MCNP_ALPHAS))
def test_alphas_fall_through_to_tendl_and_match_mcnp(energy):
    """ENDF/B-VIII.1's alpha + 9Be ends at 15 MeV, so the next library is used."""
    s = thick_be("a", energy, ["drosg", ENDFB, TENDL21], length=0.25, num_x_bins=500)
    assert s.beam_energy_grid[-1] == 0
    assert s.reaction_data["a9Be_xn"].startswith("TENDL-2021")
    assert s.yield_total / s.particle_rate == pytest.approx(MCNP_ALPHAS[energy], rel=0.02)
    assert s.warnings == []


def test_no_evaluation_reaching_the_beam_energy():
    """With only the 15 MeV evaluation there are no data for 30 MeV alphas on 9Be."""
    with pytest.raises(ValueError, match="no neutrons are produced"):
        thick_be("a", 30, ["drosg", ENDFB], length=0.1)


def test_attenuation_and_power_balance():
    """The beam loses some particles to nuclear reactions on its way; the power they carry is
    deposited in the target, and the power still adds up."""
    s = thick_be("p", 30, ["drosg", ENDFB], length=0.5)
    assert 0.03 < 1 - s.survival_grid[-1] < 0.06  # About 4% at 30 MeV
    assert np.all(np.diff(s.survival_grid) <= 0)
    total = s.deposited_power + s.wall_power + s.backstop_power
    assert total == pytest.approx(s.voltage * s.current / 1000, rel=1e-12)
    assert 0 < s.nuclear_power < s.deposited_power


def test_drosg_first_keeps_its_reactions():
    """DROSG first: a D/T gas uses DROSG even with an evaluated library after it, so the
    results do not change; a Be atom in the gas uses the library."""
    gas = Material("DT + Be", "gas", molecules={"D2": 0.5, "T2": 0.5})
    regions = [Region(gas, length=50.0, pressure=5.0, temperature=25.0, wall_radius=1.0,
                      num_x_bins=40, num_x_bins_out=10, divergence=30.0)]  # fmt: skip
    kw = dict(name="t", regions=regions, voltage=280.0, current=10.0, start_radius=0.2,
              angle_step=1.0, angle_step_out=10.0, energy_bin_width=0.01)  # fmt: skip
    plain = ShineSource(**kw)
    with_library = ShineSource(**kw, nuclear_data=["drosg", str(ENDFB)])
    assert list(with_library.reactions) == ["DD", "DT"]
    assert with_library.yield_total == plain.yield_total
    assert np.all(with_library.survival_grid == 1)


# --- Outputs -------------------------------------------------------------------------------


def test_data_recorded_in_outputs(tmp_path):
    s = thick_be("p", 18, ["drosg", ENDFB], length=0.5, num_x_bins=200)
    summary = tomllib.loads(s.get_summary())
    assert summary["inputs"]["nuclear_data"] == ["drosg", str(ENDFB)]
    assert summary["reactions"]["p9Be_xn"]["reaction"] == "9Be(p,xn)"
    data = summary["reactions"]["p9Be_xn"]["data"]
    assert "LANL" in data
    # The file relative to its library's directory, which nuclear_data_dirs lists in full
    assert data.endswith(" (protons/p-004_Be_009.endf)")
    assert str(ENDFB) not in data
    assert summary["results"]["beam_nuclear_loss"] > 0
    s.write_hdf5(tmp_path / "s.h5")
    with h5py.File(tmp_path / "s.h5") as f:
        assert f["neutrons_p9Be_xn"].attrs["data"] == s.reaction_data["p9Be_xn"]
    assert "p9Be_xn" in s.get_mcnp_sdef()  # Data recorded in the SDEF header too


INPUT = """
[materials.Be]
phase = "solid"
density = 1.848
atoms = { Be = 1 }

[cases.p18]
beam = "p"
voltage = 18000.0
current = 1.0
start_radius = 0.1
angle_step = 1.0
angle_step_out = 10.0
energy_bin_width = 0.1
nuclear_data = ["drosg", "LIBRARY"]
regions = [{ material = "Be", length = 0.5, wall_radius = 10.0, num_x_bins = 100, num_x_bins_out = 10 }]
"""


def test_input_file_nuclear_data(tmp_path):
    """nuclear_data in an input file: relative directories are taken from the input file's
    directory; a missing directory is an error; the default is DROSG alone."""
    path = tmp_path / "input.toml"
    relative = os.path.relpath(ENDFB, tmp_path).replace("\\", "/")
    path.write_text(INPUT.replace("LIBRARY", relative))
    case = load_input(path)["p18"]
    assert case["source"]["nuclear_data"] == ["drosg", os.path.normpath(str(ENDFB))]
    s = ShineSource(name="p18", **case["source"])
    assert s.yield_total / s.particle_rate == pytest.approx(MCNP_PROTONS[18], rel=0.02)

    path.write_text(INPUT.replace("LIBRARY", "nowhere"))
    with pytest.raises(ValueError, match="'nowhere' is not 'drosg', a library defined in"):
        load_input(path)
    # Not set: left to the configuration's default (none here), which is DROSG alone
    path.write_text(INPUT.replace('nuclear_data = ["drosg", "LIBRARY"]\n', ""))
    source = load_input(path)["p18"]["source"]
    assert source["nuclear_data"] is None
    with pytest.raises(ValueError, match="no neutrons are produced"):  # DROSG has no 9Be
        ShineSource(name="p18", **source)


# --- The user's configuration file: named libraries ----------------------------------------


def write_config(tmp_path, monkeypatch, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    monkeypatch.setenv("SHINE_NEUTRON_SOURCE_CONFIG", str(path))
    return path


def test_named_libraries(tmp_path, monkeypatch):
    """Input files name libraries defined in the configuration (relative directories there
    are from the configuration file's directory); outputs record the names and directories;
    the configuration's default list is used where a case sets none."""
    data = tmp_path / "data"
    data.mkdir()
    relative = os.path.relpath(ENDFB, data).replace("\\", "/")
    write_config(
        data,
        monkeypatch,
        f'[libraries]\nendfb81 = "{relative}"\ntendl21 = "{TENDL21.as_posix()}"\n\n'
        '[defaults]\nnuclear_data = ["drosg", "endfb81", "tendl21"]\n',
    )
    path = tmp_path / "input.toml"
    path.write_text(INPUT.replace('"LIBRARY"', '"endfb81"'))
    source = load_input(path)["p18"]["source"]
    assert source["nuclear_data"] == ["drosg", "endfb81"]
    s = ShineSource(name="p18", **source)
    assert s.nuclear_data == ["drosg", os.path.normpath(str(ENDFB))]
    summary = tomllib.loads(s.get_summary())["inputs"]
    assert summary["nuclear_data"] == ["drosg", "endfb81"]
    assert summary["nuclear_data_dirs"] == ["drosg", os.path.normpath(str(ENDFB))]
    assert s.yield_total / s.particle_rate == pytest.approx(MCNP_PROTONS[18], rel=0.02)

    # No nuclear_data in the case: the configuration's default list
    path.write_text(INPUT.replace('nuclear_data = ["drosg", "LIBRARY"]\n', ""))
    s = ShineSource(name="p18", **load_input(path)["p18"]["source"])
    assert s.nuclear_data_given == ["drosg", "endfb81", "tendl21"]
    assert "p9Be_xn" in s.reactions_present()


def test_unknown_library_name(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, f'[libraries]\nendfb81 = "{ENDFB.as_posix()}"\n')
    with pytest.raises(ValueError, match=r"'fendl' is not 'drosg'.*\(defined: endfb81\)"):
        thick_be("p", 18, ["drosg", "fendl"])


@pytest.mark.parametrize(
    "text, message",
    [
        ("[library]\n", "unknown table"),
        ('[libraries]\ndrosg = "x"\n', "cannot define 'drosg'"),
        ("[libraries]\nendf = 3\n", "must be a directory"),
        ('[defaults]\nnuclear_data = "drosg"\n', "must be a list"),
        ('[defaults]\nbeam = "p"\n', "unknown setting"),
    ],
)
def test_invalid_configuration(tmp_path, monkeypatch, text, message):
    write_config(tmp_path, monkeypatch, text)
    with pytest.raises(ValueError, match=message):
        thick_be("p", 18, None)


# --- Lab-frame tables (LAW=7), with a local FENDL library ----------------------------------

FENDL_D = Path(os.environ.get("SHINE_FENDL_DIR", r"C:\Users\LJ0489\Software\ENDF\FENDL-3.2c"))


@pytest.mark.skipif(not (FENDL_D / "deuteron").is_dir(), reason="no local FENDL-3.2c library")
def test_deuterons_with_lab_tables_match_mcnp():
    """FENDL-3.2c's d + 9Be (JENDL/DEU-2020) is given as lab-frame tables (LAW=7)."""
    s = thick_be("d", 18, ["drosg", FENDL_D], length=0.5)
    assert s.reactions["d9Be_xn"].production.channels[0].law == 7
    assert s.yield_total / s.particle_rate == pytest.approx(1.2896e-02, rel=0.01)


@pytest.mark.skipif(not (FENDL_D / "proton").is_dir(), reason="no local FENDL-3.2c library")
def test_every_neutron_product_of_a_section_is_read():
    """FENDL-3.2c's p + 7Li (JENDL-5) lists the MT=5 neutrons twice, below and above 10 MeV;
    reading only the first lost every neutron above 10 MeV."""
    x = NeutronProduction(FENDL_D / "proton" / "p_0328_3-Li-7.endf")
    assert [c.mt for c in x.channels].count(5) == 2
    assert x.production_cross_section(30.0) > 0.1  # b, all from the second product
    angles = np.arange(0, 181, 10.0)
    edges = np.arange(0, np.ceil(x.max_lab_energy(30.0) / 0.1) * 0.1 + 0.05, 0.1)
    spectrum, _ = x.lab_spectrum(30.0, angles, edges)
    assert spectrum.sum() == pytest.approx(x.production_cross_section(30.0), rel=1e-10)


@pytest.mark.skipif(not (FENDL_D / "proton").is_dir(), reason="no local FENDL-3.2c library")
def test_lab_tables_with_histogram_spectra():
    """FENDL-3.2c's p + 12C gives its lab-frame (LAW=7) spectra as histograms (INT=1), and
    writes some exponents with a blank after the sign ("6.01200+ 3")."""
    x = NeutronProduction(FENDL_D / "proton" / "p_0625_6-C-12.endf")
    assert set(x.channels[0].distributions[0].spectra[0].codes) == {1}
    angles = np.arange(0, 181, 10.0)
    edges = np.arange(0, np.ceil(x.max_lab_energy(70.0) / 0.1) * 0.1 + 0.05, 0.1)
    spectrum, _ = x.lab_spectrum(70.0, angles, edges)
    assert spectrum.sum() == pytest.approx(x.production_cross_section(70.0), rel=1e-10)
    mids = (edges[:-1] + edges[1:]) / 2
    mean = np.sum(spectrum * mids, axis=1) / spectrum.sum(axis=1)
    assert mean[0] > mean[-1]  # Forward neutrons are faster


# --- Fission ---------------------------------------------------------------------------------


def uranium_source(beam, energy_MeV, atoms, nuclear_data, length=0.2):
    material = Material("U target", "solid", atoms=atoms, density=18.95)
    return ShineSource(
        name=f"{beam}U",
        regions=[Region(material, length=length, wall_radius=10.0, num_x_bins=100,
                        num_x_bins_out=1)],
        voltage=energy_MeV * 1000, current=1.0, start_radius=0.1, angle_step=1.0,
        angle_step_out=10.0, energy_bin_width=0.1, beam=beam,
        nuclear_data=[str(d) for d in nuclear_data],
    )  # fmt: skip


def fission_warnings(s):
    return [w for w in s.warnings if w.startswith("fission of")]


def test_fission_warning_without_an_evaluation():
    """Uranium with no evaluation: its fission neutrons are missing, and the case says so."""
    s = uranium_source("d", 10.0, {"D": 1.5, "238U": 1}, ["drosg"], length=0.05)
    assert list(s.reactions) == ["DD"]
    [message] = fission_warnings(s)
    assert "238U" in message and "there is no evaluation" in message


def test_no_fission_warning_below_thorium():
    s = uranium_source("d", 10.0, {"D": 1.5, "Ta": 1}, ["drosg"], length=0.05)
    assert fission_warnings(s) == []


@pytest.mark.skipif(not (FENDL_D / "proton").is_dir(), reason="no local FENDL-3.2c library")
def test_fission_neutrons_from_fendl_protons():
    """FENDL-3.2c's p + 238U gives fission (MT=18) neutrons explicitly: no warning."""
    s = uranium_source("p", 30.0, {"238U": 1}, [FENDL_D])
    assert 18 in [c.mt for c in s.reactions["p238U_xn"].production.channels]
    assert fission_warnings(s) == []


@pytest.mark.skipif(not (FENDL_D / "deuteron").is_dir(), reason="no local FENDL-3.2c library")
def test_fission_warning_for_an_evaluation_without_fission():
    """FENDL-3.2c's d + 238U (TENDL-2011) has no fission channel: warned about."""
    s = uranium_source("d", 30.0, {"238U": 1}, [FENDL_D])
    [message] = fission_warnings(s)
    assert "d_9237_92-U-238.endf" in message and "no fission (MT=18)" in message
