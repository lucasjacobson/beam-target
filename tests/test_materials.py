"""Materials other than D/T gas: formulas, the Material class, solid targets and gas mixtures."""

import tomllib

import h5py
import numpy as np
import pytest
from conftest import SMALL_CASE, make_source

from shine_neutron_source.source import Material, parse_formula

TID = Material("TiD1.5", {"Ti": 1.0, "D": 1.5}, "solid", density=3.9)
TIT = Material("TiT1.5", {"Ti": 1.0, "T": 1.5}, "solid", density=4.0)

# A solid target region: no pressure or temperature, micrometres thick, no divergence
SOLID = dict(
    T_frac=None,
    pressure=[None],
    temperature=[None],
    length=[10e-4],
    divergence=[0.0],
    num_x_bins=[400],
    num_x_bins_out=[40],
)


def make_solid(material=TID, **overrides):
    return make_source(**{**SOLID, **overrides}, material=material)


# --- Formulas and the Material class ---------------------------------------------------------


@pytest.mark.parametrize(
    "formula, atoms",
    [
        ("D2", {"D": 2.0}),
        ("Ti", {"Ti": 1.0}),
        ("TiD1.5", {"Ti": 1.0, "D": 1.5}),
        ("CD4", {"C": 1.0, "D": 4.0}),
        ("3He", {"He3": 1.0}),  # The helium-3 isotope
        ("He3", {"He": 3.0}),  # Three helium atoms
        ("DT", {"D": 1.0, "T": 1.0}),
        ("Er.5D", {"Er": 0.5, "D": 1.0}),
    ],
)
def test_parse_formula(formula, atoms):
    assert parse_formula(formula) == atoms


@pytest.mark.parametrize("formula", ["", "d2", "Ti-D", "2"])
def test_parse_formula_rejects(formula):
    with pytest.raises(ValueError):
        parse_formula(formula)


def test_material_atoms():
    assert TID.atom_fractions == pytest.approx({"Ti": 0.4, "D": 0.6})
    assert TID.atoms_per_unit == 1.0  # Each species listed is a single atom
    gas = Material("D2 + Ar", {"D2": 0.98, "Ar": 0.02}, "gas")
    assert gas.atoms_per_unit == pytest.approx(1.98)  # Atoms per molecule
    assert gas.atom_fractions["D"] == pytest.approx(1.96 / 1.98)


@pytest.mark.parametrize(
    "args, message",
    [
        (({"D": 1}, "liquid"), "phase must be one of"),
        (({"Ti": 1, "D": 1.5}, "solid"), "a solid needs a positive density"),
        (({"D2": 1}, "gas", 0.1), "cannot have a density"),
        (({"D": -1}, "solid", 1.0), "must not be negative"),
        (({"D": 0}, "solid", 1.0), "must not all be zero"),
        (({}, "solid", 1.0), "empty composition"),
        (({"ti": 1}, "solid", 1.0), "cannot read the chemical formula"),
    ],
)
def test_material_rejects(args, message):
    with pytest.raises(ValueError, match=message):
        Material("m", *args)


# --- Gas materials ---------------------------------------------------------------------------


def test_dt_gas_material_matches_default_gas():
    """A gas material of D2 and T2 gives the same results as T_frac (the default D/T gas)."""
    default = make_source()
    material = Material("D/T", {"D2": 0.5, "T2": 0.5}, "gas")
    explicit = make_source(T_frac=None, material=material)
    for rxn in ("DD", "DT"):
        np.testing.assert_allclose(explicit.neutrons[rxn], default.neutrons[rxn], rtol=1e-12)


def test_gas_mixture_with_argon():
    """Argon dilutes the D/T atoms and adds stopping, so fewer neutrons come out per mA."""
    pure = make_source(T_frac=None, material=Material("D2", {"D2": 1}, "gas"))
    argon = make_source(T_frac=None, material=Material("D2 + Ar", {"D2": 0.9, "Ar": 0.1}, "gas"))
    assert argon.atom_density_per_region[0] == pytest.approx(
        pure.atom_density_per_region[0] * 1.9 / 2, rel=1e-12
    )
    assert argon.beam_energy_grid[-1] < pure.beam_energy_grid[-1]  # More stopping
    assert argon.yields["DD"] < 0.9 * pure.yields["DD"]
    assert argon.yields["DT"] == 0


def test_gas_material_upstream():
    """An upstream stage can hold a gas material while the target chamber holds D/T gas."""
    argon = Material("Ar", {"Ar": 1}, "gas")
    s = make_source(
        pressure=[0.5, 5.0],
        temperature=[25.0, 25.0],
        length=[20.0, 50.0],
        divergence=[0.0, 30.0],
        num_x_bins=[4, 40],
        num_x_bins_out=[1, 10],
        material=[argon, None],
    )
    assert s.target_density_per_region["DD"][0] == 0  # No neutrons from the argon stage
    assert s.materials[1].name == "D/T gas"


# --- Solid targets ---------------------------------------------------------------------------


def test_thick_solid_target():
    """The beam stops within a few micrometres of TiD1.5: all its power stays in the target."""
    s = make_solid()
    assert s.atom_density_per_region[0] == pytest.approx(3.9 * 6.02214076e23 / (
        (47.9 + 1.5 * 2.0141) / 2.5), rel=1e-3)  # fmt: skip
    assert s.target_density_per_region["DD"][0] == pytest.approx(0.6 * s.atom_density_per_region[0])
    E = s.beam_energy_grid
    x_stop = s.x_grid[np.flatnonzero(E > 0)[-1] + 1]
    assert 1e-4 < x_stop < 4e-4  # 1-4 um
    assert E[-1] == 0 and s.backstop_power == 0 and s.wall_power == 0
    assert s.deposited_power == pytest.approx(s.voltage * s.current / 1000)
    assert s.reactions_present() == ["DD"]
    # Thick-target DD yield, about 1.7e6 n/s per uA at 280 keV
    assert 1e6 < s.yields["DD"] / (s.current * 1000) < 3e6


def test_titanium_tritide_gives_dt():
    s = make_solid(material=TIT)
    assert s.reactions_present() == ["DT"] and s.yields["DD"] == 0
    assert s.yields["DT"] > 50 * make_solid().yields["DD"]  # DT is far more likely


def test_solid_thickness_calculated():
    """For a solid target chamber, the length follows from the backstop energy."""
    thick = make_solid(length=[None], beam_energy_backstop=100.0)
    assert thick.calculated == "length"
    assert thick.beam_energy_grid[-1] == pytest.approx(100.0, rel=1e-6)
    assert 0.5e-4 < thick.length[0] < 2e-4


def test_solid_plots_in_micrometres(tmp_path):
    s = make_solid()
    assert s._position_units() == (1e4, r"$\mu$m")
    assert make_source()._position_units() == (1.0, "cm")
    s.make_plots(tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 7  # DD only


def test_solid_outputs_record_material(tmp_path):
    s = make_solid()
    s.write_text_files(tmp_path)
    with open(tmp_path / "summary.toml", "rb") as f:
        summary = tomllib.load(f)
    assert summary["inputs"]["material"] == ["TiD1.5"]
    assert "T_frac" not in summary["inputs"]
    assert summary["materials"]["TiD1.5"] == {
        "phase": "solid",
        "density": 3.9,
        "composition": {"Ti": 1.0, "D": 1.5},
    }
    s.write_hdf5(tmp_path / "source.h5")
    with h5py.File(tmp_path / "source.h5") as f:
        assert f["inputs/material"][0].decode() == "TiD1.5"
        material = f["materials/TiD1.5"]
        assert material.attrs["phase"] == "solid" and material.attrs["density"] == 3.9
        assert list(material.attrs["atoms"]) == ["Ti", "D"]
    sdef = s.get_mcnp_sdef()
    assert '[materials."TiD1.5"]' in sdef


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(pressure=[5.0]), "is a solid .* takes no pressure"),
        (dict(temperature=[25.0]), "is a solid .* takes no pressure or temperature"),
        (dict(stopping_power="icru49"), "icru49 is for hydrogen gas only"),
        (dict(length=[None]), "give exactly one of the target chamber's length"),
    ],
)
def test_solid_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_solid(**overrides)


def test_unknown_element_rejected():
    with pytest.raises(ValueError, match="no stopping data for 'Xx'"):
        make_solid(material=Material("bad", {"Xx": 1}, "solid", density=1.0))


def test_dt_gas_needs_t_frac():
    """A region without a material holds D/T gas, so T_frac is needed."""
    with pytest.raises(ValueError, match="T_frac must be between 0 and 1"):
        make_source(T_frac=None)


def test_small_case_unchanged_by_materials():
    """The default (no materials) is D/T gas in every region."""
    s = make_source()
    assert [m.name for m in s.materials] == ["D/T gas"]
    assert s.materials[0].phase == "gas"
    assert s.atom_fractions_per_region["T"][0] == SMALL_CASE["T_frac"]
