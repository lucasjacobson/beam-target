"""Evaluated nuclear data (endf.py) and reactions calculated from it."""

import os
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import channel_warnings

from beamtarget import BeamTarget, Material, Region, endf, load_input
from beamtarget.endf import (
    NeutronProduction,
    _number,
    _parse_name,
    choose_variant,
    find_evaluation,
    separation_energy,
)

DATA = Path(__file__).parent / "data" / "endf"
ENDFB = DATA / "ENDF-B-VIII.1"
TENDL25 = DATA / "TENDL-2025"
P_BE9 = ENDFB / "protons" / "p-004_Be_009.endf"
A_BE9_TENDL = TENDL25 / "a" / "a-Be009.tendl"
BE = Material("Be", "solid", atoms={"Be": 1}, density=1.848)

# MCNP: neutrons created by charged-particle reactions per beam particle in thick Be
# (MCNP 6.2 with ENDF/B-VII.0 protons, the same LANL evaluation as ENDF/B-VIII.1)
MCNP_PROTONS = {10: 3.3893e-03, 18: 1.2004e-02, 30: 2.9426e-02}


def thick_be(beam, energy_MeV, nuclear_data, length=1.0, position_bins=60, **kw):
    """A Be block thick enough to stop the beam."""
    return BeamTarget(
        name=f"{beam}{energy_MeV}",
        regions=[
            Region(
                BE,
                length=length,
                wall_radius=10.0,
                position_bins=position_bins,
            )
        ],  # fmt: skip
        beam_energy_entry=energy_MeV * 1000,
        beam_current=1.0,
        beam_radius_entry=0.1,
        angle_bin_width=10.0,
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
    two_body, sigma = x.two_body_spectrum([12.0, 12.0], [1.0, 0.0], [0, 1], 2, angles, edges)
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
        ("p-Be009.tendl", ("p", "Be", 9, 0)),
        ("p-Be009-mt.tendl", ("p", "Be", 9, 1)),
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
    assert Path(find_evaluation(TENDL25, "a", (4, 9))) == A_BE9_TENDL
    assert find_evaluation(ENDFB, "d", (4, 9)) is None
    assert find_evaluation(DATA / "missing", "p", (4, 9)) is None


class _Variant:
    """A stand-in evaluation whose explicit channels leave `share` of the non-elastic cross
    section uncovered (None: it cannot be read)."""

    def __init__(self, share):
        self.share = share

    def missing_share(self, E_max=None):
        if self.share is None:
            raise NotImplementedError("unsupported representation")
        return self.share


@pytest.mark.parametrize(
    "shares, chosen",
    [
        ({"": 0.0, "-mt": 0.0}, ""),  # Equally complete: the standard file
        ({"": 0.005, "-mt": 0.0}, ""),  # Within VARIANT_TIE
        ({"": 0.10, "-mt": 0.0}, "-mt"),  # The standard file misses channels (d + 9Be)
        ({"": 0.0, "-mt": 0.40, "-s60": 0.08}, ""),  # -mt misses channels (above 30 MeV)
        ({"": 0.05, "-s0": 0.0}, ""),  # The lumped -s0 is not compared, though complete
        ({"-s20": 0.0, "-s0": 0.0}, "-s20"),  # Lumped files only: the first
        ({"": None, "-mt": 0.30}, "-mt"),  # A variant that cannot be read is passed over
    ],
)
def test_most_complete_tendl_variant_is_chosen(tmp_path, monkeypatch, shares, chosen):
    variants = {}
    for suffix, share in shares.items():
        path = tmp_path / f"d-Be009{suffix}.tendl"
        path.write_text("")
        variants[str(path)] = _Variant(share)
    monkeypatch.setattr(endf, "load_evaluation", lambda path: variants[str(path)])
    assert Path(find_evaluation(tmp_path, "d", (4, 9), E_max=40)).name == f"d-Be009{chosen}.tendl"


def test_choose_variant_raises_if_no_variant_can_be_read(tmp_path, monkeypatch):
    paths = [str(tmp_path / f"p-Be009{s}.tendl") for s in ("", "-mt")]
    monkeypatch.setattr(endf, "load_evaluation", lambda path: _Variant(None))
    with pytest.raises(NotImplementedError):
        choose_variant(paths, 40)


TENDL25_VARIANTS = Path(
    os.environ.get(
        "BEAM_TARGET_TENDL_VARIANTS_DIR",
        r"C:\Users\LJ0489\Software\ENDF\_downloads\tendl2025-variants",
    )
)


@pytest.mark.skipif(
    not (TENDL25_VARIANTS / "p-Be009").is_dir(), reason="no local TENDL-2025 variants"
)
def test_tendl2025_variants():
    """In TENDL-2025, -mt misses most p + 9Be channels above 30 MeV and the standard file
    misses some d + 9Be channels below it, so the standard file is used for protons and -mt
    for deuterons; -s0 is complete but not compared."""
    p = find_evaluation(TENDL25_VARIANTS, "p", (4, 9), E_max=70)
    d = find_evaluation(TENDL25_VARIANTS, "d", (4, 9), E_max=70)
    assert Path(p).name == "p-Be009.tendl"
    assert Path(d).name == "d-Be009-mt.tendl"
    mt = NeutronProduction(TENDL25_VARIANTS / "p-Be009" / "p-Be009-mt.tendl")
    assert mt.missing_share(70) > 0.2 > NeutronProduction(p).missing_share(70)
    assert NeutronProduction(TENDL25_VARIANTS / "p-Be009" / "p-Be009-s0.tendl").missing_share() == 0


# --- Thick-target benchmarks against MCNP --------------------------------------------------


@pytest.mark.parametrize("energy", list(MCNP_PROTONS))
def test_protons_on_thick_be_match_mcnp(energy):
    s = thick_be("proton", energy, ["drosg", ENDFB], length=1.0)
    assert s.beam_energy_grid[-1] == 0  # The beam stops in the target
    assert s.yield_total / s.particle_rate == pytest.approx(MCNP_PROTONS[energy], rel=0.01)
    assert s.reaction_data["p9Be_xn"].startswith("ENDF/B-VIII.1")
    assert channel_warnings(s) == []


@pytest.mark.parametrize("energy", [30, 70])
def test_alphas_fall_through_to_tendl(energy):
    """ENDF/B-VIII.1's alpha + 9Be ends at 15 MeV, so the next library is used. (MCNP 6.3
    with TENDL-2021-based tables gave 3.1157e-3 and 1.4677e-2 neutrons per alpha at 30 and
    70 MeV, which TENDL-2021 reproduced within 1.5%; TENDL-2025's evaluation gives 1.25 and
    1.17 times as many. Its results are pinned in test_regression.py.)"""
    s = thick_be("alpha", energy, ["drosg", ENDFB, TENDL25], length=0.25, position_bins=50)
    assert s.beam_energy_grid[-1] == 0
    assert s.reaction_data["a9Be_xn"].startswith("TENDL-2025")
    assert channel_warnings(s) == []


def test_no_evaluation_reaching_the_beam_energy():
    """With only the 15 MeV evaluation there are no data for 30 MeV alphas on 9Be."""
    with pytest.raises(ValueError, match="no neutrons are produced"):
        thick_be("alpha", 30, ["drosg", ENDFB], length=0.1)


def test_attenuation_and_power_balance():
    """The beam loses some particles to nuclear reactions on its way; the power they carry is
    deposited in the target, and the power still adds up."""
    s = thick_be("proton", 30, ["drosg", ENDFB], length=0.5)
    assert 0.03 < 1 - s.survival_grid[-1] < 0.06  # About 4% at 30 MeV
    assert np.all(np.diff(s.survival_grid) <= 0)
    total = s.deposited_power + s.wall_power + s.backstop_power
    assert total == pytest.approx(s.beam_power_entry, rel=1e-12)
    assert 0 < s.nuclear_power < s.deposited_power


def test_drosg_first_keeps_its_reactions():
    """DROSG first: a D/T gas uses DROSG even with an evaluated library after it, so the
    results do not change; a Be atom in the gas uses the library."""
    gas = Material("DT + Be", "gas", molecules={"D2": 0.5, "T2": 0.5})
    regions = [Region(gas, length=50.0, pressure=5.0, temperature=25.0, wall_radius=1.0,
                      position_bins=10, position_substeps=4, divergence=30.0)]  # fmt: skip
    kw = dict(name="t", regions=regions, beam_energy_entry=280.0, beam_current=10.0,
              beam_radius_entry=0.2, angle_bin_width=10.0,
              energy_bin_width=0.01)  # fmt: skip
    plain = BeamTarget(**kw)
    with_library = BeamTarget(**kw, nuclear_data=["drosg", str(ENDFB)])
    assert list(with_library.reactions) == ["DD", "DT"]
    assert with_library.yield_total == plain.yield_total
    assert np.all(with_library.survival_grid == 1)


# --- Outputs -------------------------------------------------------------------------------


def test_data_recorded_in_outputs(tmp_path):
    s = thick_be("proton", 18, ["drosg", ENDFB], length=0.5, position_bins=20)
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
beam = "proton"
beam_energy_entry = 18000.0
beam_current = 1.0
beam_radius_entry = 0.1
angle_bin_width = 10.0
energy_bin_width = 0.1
nuclear_data = ["drosg", "LIBRARY"]
regions = [{ material = "Be", length = 0.5, wall_radius = 10.0, position_bins = 10, position_substeps = 10 }]
"""


def _same_dirs(entries, expected):
    """Whether nuclear_data entries are the expected ones, comparing directories as the
    operating system does (on Windows, "c:\\..." and "C:\\..." are the same directory)."""

    def norm(entry):
        return os.path.normcase(os.path.normpath(str(entry)))

    return [norm(e) for e in entries] == [norm(e) for e in expected]


def test_input_file_nuclear_data(tmp_path):
    """nuclear_data in an input file: relative directories are taken from the input file's
    directory; a missing directory is an error; the default is DROSG alone."""
    path = tmp_path / "input.toml"
    relative = os.path.relpath(ENDFB, tmp_path).replace("\\", "/")
    path.write_text(INPUT.replace("LIBRARY", relative))
    case = load_input(path)["p18"]
    assert _same_dirs(case["source"]["nuclear_data"], ["drosg", ENDFB])
    s = BeamTarget(name="p18", **case["source"])
    assert s.yield_total / s.particle_rate == pytest.approx(MCNP_PROTONS[18], rel=0.02)

    path.write_text(INPUT.replace("LIBRARY", "nowhere"))
    with pytest.raises(ValueError, match="'nowhere' is not 'drosg', a library defined in"):
        load_input(path)
    # Not set: left to the configuration's default (none here), which is DROSG alone
    path.write_text(INPUT.replace('nuclear_data = ["drosg", "LIBRARY"]\n', ""))
    source = load_input(path)["p18"]["source"]
    assert source["nuclear_data"] is None
    with pytest.raises(ValueError, match="no neutrons are produced"):  # DROSG has no 9Be
        BeamTarget(name="p18", **source)


# --- The user's configuration file: named libraries ----------------------------------------


def write_config(tmp_path, monkeypatch, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    monkeypatch.setenv("BEAM_TARGET_CONFIG", str(path))
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
        f'[libraries]\nendfb81 = "{relative}"\ntendl25 = "{TENDL25.as_posix()}"\n\n'
        '[defaults]\nnuclear_data = ["drosg", "endfb81", "tendl25"]\n',
    )
    path = tmp_path / "input.toml"
    path.write_text(INPUT.replace('"LIBRARY"', '"endfb81"'))
    source = load_input(path)["p18"]["source"]
    assert source["nuclear_data"] == ["drosg", "endfb81"]
    s = BeamTarget(name="p18", **source)
    assert _same_dirs(s.nuclear_data, ["drosg", ENDFB])
    summary = tomllib.loads(s.get_summary())["inputs"]
    assert summary["nuclear_data"] == ["drosg", "endfb81"]
    assert _same_dirs(summary["nuclear_data_dirs"], ["drosg", ENDFB])
    assert s.yield_total / s.particle_rate == pytest.approx(MCNP_PROTONS[18], rel=0.02)

    # No nuclear_data in the case: the configuration's default list
    path.write_text(INPUT.replace('nuclear_data = ["drosg", "LIBRARY"]\n', ""))
    s = BeamTarget(name="p18", **load_input(path)["p18"]["source"])
    assert s.nuclear_data_given == ["drosg", "endfb81", "tendl25"]
    assert "p9Be_xn" in s.reactions_present()


def test_unknown_library_name(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, f'[libraries]\nendfb81 = "{ENDFB.as_posix()}"\n')
    with pytest.raises(ValueError, match=r"'fendl' is not 'drosg'.*\(defined: endfb81\)"):
        thick_be("proton", 18, ["drosg", "fendl"])


@pytest.mark.parametrize(
    "text, message",
    [
        ("[library]\n", "unknown table"),
        ('[libraries]\ndrosg = "x"\n', "cannot define 'drosg'"),
        ("[libraries]\nendf = 3\n", "must be a directory"),
        ('[defaults]\nnuclear_data = "drosg"\n', "must be a list"),
        ('[defaults]\nbeam = "proton"\n', "unknown setting"),
    ],
)
def test_invalid_configuration(tmp_path, monkeypatch, text, message):
    write_config(tmp_path, monkeypatch, text)
    with pytest.raises(ValueError, match=message):
        thick_be("proton", 18, None)


# --- Lab-frame tables (LAW=7), with a local FENDL library ----------------------------------

FENDL_D = Path(os.environ.get("BEAM_TARGET_FENDL_DIR", r"C:\Users\LJ0489\Software\ENDF\FENDL-3.2c"))


@pytest.mark.skipif(not (FENDL_D / "proton").is_dir(), reason="no local FENDL-3.2c library")
@pytest.mark.parametrize("path, energy", [(P_BE9, 18.0)])  # Smooth there in the data
def test_slow_lab_neutrons_bin_smoothly_in_angle(path, energy):
    """Neutrons that are slow in the lab come from CM energies near the CM's own speed,
    where the lab angle changes quickly with the CM energy. Pieces half an energy bin wide
    put them in separate angle bins (neighbors differing by tens of percent); the narrower
    pieces there keep the angular distribution smooth."""
    x = NeutronProduction(path)
    angles = np.arange(0, 180.5, 1.0)
    edges = np.arange(0, np.ceil(x.max_lab_energy(energy) / 0.1) * 0.1 + 0.05, 0.1)
    spectrum, _ = x.lab_spectrum(energy, angles, edges, num_cm_angles=1440, two_body=False)
    solid = 2 * np.pi * (np.cos(np.radians(angles[:-1])) - np.cos(np.radians(angles[1:])))
    mids = (edges[:-1] + edges[1:]) / 2
    for lo, hi in [(0, 0.5), (0.5, 1)]:
        a = spectrum[:, (mids > lo) & (mids < hi)].sum(axis=1) / solid
        ratio = a[1:-1] / ((a[:-2] + a[2:]) / 2)
        # Away from the edges of the angles these energies can reach
        inside = np.min([a[:-2], a[1:-1], a[2:]], axis=0) > 0.1 * a.max()
        assert np.all(np.abs(ratio[inside] - 1) < 0.05), (lo, hi, ratio[inside].min())


def test_energy_pieces_follow_the_requested_widths():
    """Pieces are equal and at most max_step wide without a width function, and as narrow as
    it asks where it asks for less; the probabilities are exact either way."""
    energies = np.array([0.0, 0.3, 1.0, 4.0])
    f0 = np.array([1.0, 2.0, 0.5, 0.0])
    plain, p_plain, _ = endf._energy_points(energies, f0, 2, 0.05)
    assert p_plain.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.max(np.diff(plain)) <= 0.05 + 1e-12
    step = lambda e: np.where((e > 0.5) & (e < 0.7), 0.001, 1.0)  # noqa: E731
    fine, p_fine, _ = endf._energy_points(energies, f0, 2, 0.05, step)
    assert p_fine.sum() == pytest.approx(1.0, rel=1e-12)
    inside = (fine > 0.51) & (fine < 0.69)
    assert np.max(np.diff(fine[inside])) < 0.0011
    # The mean energy is about the same (each piece is placed at its middle)
    assert np.sum(fine * p_fine) == pytest.approx(np.sum(plain * p_plain), rel=1e-3)


@pytest.mark.parametrize("angle_bin_width", [2.0, 3.0, 5.0])
def test_lab_tables_bin_smoothly_at_any_angle_bin(angle_bin_width):
    """FENDL-3.2c's p + 181Ta is given as lab-frame tables every 10 degrees. Cells that ignored
    the output bin edges gave some bins one cell more than their neighbors: a regular stripe
    of up to 25% with 3-degree bins. Each bin must now be within 3% of its neighbors' mean."""
    ta = Material("Ta", "solid", atoms={"181Ta": 1}, density=16.654)
    s = BeamTarget(
        name="ta", regions=[Region(ta, wall_radius=10.0, position_bins=4, position_substeps=10)],
        beam_energy_entry=30000.0, beam_current=1e-3, beam_energy_backstop=10.0,
        beam_radius_entry=0.5, angle_bin_width=angle_bin_width,
        energy_bin_width=0.5, beam="proton", nuclear_data=[str(FENDL_D)],
    )  # fmt: skip
    assert s.reactions["p181Ta_xn"].production.channels[0].law == 7
    per_sr = np.sum(s.neutrons["p181Ta_xn"], axis=(0, 2)) / s.solid_angles_out
    neighbors = (per_sr[:-2] + per_sr[2:]) / 2
    assert np.all(np.abs(per_sr[1:-1] / neighbors - 1) < 0.03)


@pytest.mark.skipif(not (FENDL_D / "deuteron").is_dir(), reason="no local FENDL-3.2c library")
def test_deuterons_with_lab_tables_match_mcnp():
    """FENDL-3.2c's d + 9Be (JENDL/DEU-2020) is given as lab-frame tables (LAW=7)."""
    s = thick_be("deuteron", 18, ["drosg", FENDL_D], length=0.5)
    assert s.reactions["d9Be_xn"].production.channels[0].law == 7
    assert s.yield_total / s.particle_rate == pytest.approx(1.2896e-02, rel=0.01)


def thin_li7(energy_MeV, nuclear_data):
    """Protons through a thin 7Li film."""
    li = Material("7Li", "solid", atoms={"7Li": 1}, density=0.534)
    return BeamTarget(
        name=f"p{energy_MeV}",
        regions=[Region(li, length=2e-4, wall_radius=1.0, position_bins=1, position_substeps=4)],
        beam_energy_entry=energy_MeV * 1000, beam_current=1e-3, beam_radius_entry=0.1,
        angle_bin_width=10.0, energy_bin_width=0.05, beam="proton",
        nuclear_data=[str(d) for d in nuclear_data],
    )  # fmt: skip


def better_warnings(s):
    return [w for w in s.warnings if "that gives these channels' neutrons" in w]


@pytest.mark.skipif(not (FENDL_D / "proton").is_dir(), reason="no local FENDL-3.2c library")
def test_better_evaluation_named_for_open_channels(tmp_path, monkeypatch):
    """Above 7Li(p,n3He)4He (3.70 MeV) the built-in 7Li(p,n) lines miss neutrons; FENDL-3.2c's
    p + 7Li gives them, so a warning says to list it before "drosg", in the case's
    nuclear_data or (if not there) from the configuration file. Below the threshold, or with
    FENDL-3.2c first (so it is used), there is no such warning."""
    (better,) = better_warnings(thin_li7(5.0, ["drosg", FENDL_D]))
    assert 'before "drosg" in nuclear_data' in better and "p_0328_3-Li-7.endf" in better
    assert better_warnings(thin_li7(3.0, ["drosg", FENDL_D])) == []
    s = thin_li7(5.0, [FENDL_D, "drosg"])
    assert better_warnings(s) == [] and "p7Li_xn" in s.reactions
    write_config(tmp_path, monkeypatch, f'[libraries]\nfendl = "{FENDL_D.as_posix()}"\n')
    (better,) = better_warnings(thin_li7(5.0, ["drosg"]))
    assert 'add "fendl" (from your configuration file)' in better


ENDFB_FULL = Path(
    os.environ.get("BEAM_TARGET_ENDFB_DIR", r"C:\Users\LJ0489\Software\ENDF\ENDF-B-VIII.1")
)


@pytest.mark.skipif(
    not find_evaluation(ENDFB_FULL, "p", (3, 7)), reason="no local ENDF/B-VIII.1 library"
)
def test_no_better_evaluation_from_two_body_levels_only():
    """ENDF/B-VIII.1's p + 7Li gives only the two levels the built-in data have (MT=50, 51),
    so it is not suggested."""
    s = thin_li7(5.0, ["drosg", ENDFB_FULL])
    assert any("7Li(p,n3He)4He is open" in w for w in s.warnings)
    assert better_warnings(s) == []


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
    return BeamTarget(
        name=f"{beam}U",
        regions=[Region(material, length=length, wall_radius=10.0, position_bins=1,
                        position_substeps=100)],
        beam_energy_entry=energy_MeV * 1000, beam_current=1.0, beam_radius_entry=0.1,
        angle_bin_width=10.0, energy_bin_width=0.1, beam=beam,
        nuclear_data=[str(d) for d in nuclear_data],
    )  # fmt: skip


def fission_warnings(s):
    return [w for w in s.warnings if w.startswith("fission of")]


def test_fission_warning_without_an_evaluation():
    """Uranium with no evaluation: its fission neutrons are missing, and the case says so."""
    s = uranium_source("deuteron", 10.0, {"D": 1.5, "238U": 1}, ["drosg"], length=0.05)
    assert list(s.reactions) == ["DD"]
    [message] = fission_warnings(s)
    assert "238U" in message and "there is no evaluation" in message


def test_no_fission_warning_below_thorium():
    s = uranium_source("deuteron", 10.0, {"D": 1.5, "Ta": 1}, ["drosg"], length=0.05)
    assert fission_warnings(s) == []


@pytest.mark.skipif(not (FENDL_D / "proton").is_dir(), reason="no local FENDL-3.2c library")
def test_fission_neutrons_from_fendl_protons():
    """FENDL-3.2c's p + 238U gives fission (MT=18) neutrons explicitly: no warning."""
    s = uranium_source("proton", 30.0, {"238U": 1}, [FENDL_D])
    assert 18 in [c.mt for c in s.reactions["p238U_xn"].production.channels]
    assert fission_warnings(s) == []


@pytest.mark.skipif(not (FENDL_D / "deuteron").is_dir(), reason="no local FENDL-3.2c library")
def test_fission_warning_for_an_evaluation_without_fission():
    """FENDL-3.2c's d + 238U (TENDL-2011) has no fission channel: warned about."""
    s = uranium_source("deuteron", 30.0, {"238U": 1}, [FENDL_D])
    [message] = fission_warnings(s)
    assert "d_9237_92-U-238.endf" in message and "no fission (MT=18)" in message
