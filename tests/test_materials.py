"""Materials other than D/T gas: formulas, the Material class, solid targets, and gas mixtures."""

import tomllib

import h5py
import numpy as np
import pytest
from conftest import make_source

from beamtarget.source import Material, dt_gas, get_atom_density, parse_formula
from beamtarget.units import KEV, MICROAMPERE, TORR, ZERO_CELSIUS

TID = Material("TiD1.5", "solid", atoms={"Ti": 1.0, "D": 1.5}, density=3.9)
TIT = Material("TiT1.5", "solid", atoms={"Ti": 1.0, "T": 1.5}, density=4.0)

# A solid target region: no pressure or temperature, micrometers thick, no divergence
SOLID = dict(
    pressure=[None],
    temperature=[None],
    length=[10e-4],
    divergence=[0.0],
    position_bins=[40],
    position_substeps=[10],
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
        ("3He", {"3He": 1.0}),  # The helium-3 isotope
        ("7Li", {"7Li": 1.0}),
        ("2H", {"D": 1.0}),  # Canonical names for the hydrogen isotopes
        ("LiH", {"Li": 1.0, "H": 1.0}),
        ("He3", {"He": 3.0}),  # Three helium atoms
        ("DT", {"D": 1.0, "T": 1.0}),
        ("Er.5D", {"Er": 0.5, "D": 1.0}),
    ],
)
def test_parse_formula(formula, atoms):
    assert parse_formula(formula) == atoms


@pytest.mark.parametrize("formula", ["", "d2", "Ti-D", "2", "Xx", "99Li"])
def test_parse_formula_rejects(formula):
    with pytest.raises(ValueError):
        parse_formula(formula)


def test_material_atoms():
    assert TID.atom_fractions == pytest.approx({"Ti": 0.4, "D": 0.6})
    assert TID.atoms_per_unit == 1.0  # Each species listed is a single atom
    gas = Material("D2 + Ar", "gas", molecules={"D2": 0.98, "Ar": 0.02})
    assert gas.atoms_per_unit == pytest.approx(1.98)  # Atoms per molecule
    assert gas.atom_fractions["D"] == pytest.approx(1.96 / 1.98)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(phase="liquid", atoms={"D": 1}), "phase must be one of"),
        (dict(phase="solid", atoms={"Ti": 1, "D": 1.5}), "a solid needs a positive density"),
        (dict(phase="solid", molecules={"TiD2": 1}, density=1.0), "give a solid's composition"),
        (dict(phase="solid", atoms={"TiD2": 1}, density=1.0), "atoms takes single atoms"),
        (dict(phase="gas", atoms={"D": 1}), "give a gas's composition as molecules"),
        (dict(phase="gas", molecules={"D2": 1}, density=0.1), "belongs to the region"),
        (dict(phase="gas", molecules={"D": 1}), "D on its own is a lone atom"),
        (dict(phase="gas", molecules={"O": 1, "Ar": 1}), "write O2"),
        (dict(phase="solid", atoms={"D": -1}, density=1.0), "must not be negative"),
        (dict(phase="solid", atoms={"D": 0}, density=1.0), "must not all be zero"),
        (dict(phase="solid", atoms={}, density=1.0), "empty composition"),
        (dict(phase="solid", atoms={"ti": 1}, density=1.0), "cannot read the chemical formula"),
    ],
)
def test_material_rejects(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Material("m", **kwargs)


# --- Gas materials ---------------------------------------------------------------------------


def test_dt_gas_material_matches_default_gas():
    """A gas material of D2 and T2 gives the same results as dt_gas."""
    default = make_source()
    material = Material("D/T", "gas", molecules={"D2": 0.5, "T2": 0.5})
    explicit = make_source(material=material)
    for rxn in ("DD", "DT"):
        np.testing.assert_allclose(explicit.neutrons[rxn], default.neutrons[rxn], rtol=1e-12)


def test_gas_mixture_with_argon():
    """Argon dilutes the D/T atoms and adds stopping, so fewer neutrons come out per mA."""
    pure = make_source(material=Material("D2", "gas", molecules={"D2": 1}))
    argon = make_source(material=Material("D2 + Ar", "gas", molecules={"D2": 0.9, "Ar": 0.1}))
    assert argon.atom_density_per_region[0] == pytest.approx(
        pure.atom_density_per_region[0] * 1.9 / 2, rel=1e-12
    )
    assert argon.beam_energy_grid[-1] < pure.beam_energy_grid[-1]  # More stopping
    assert argon.yields["DD"] < 0.9 * pure.yields["DD"]
    assert list(argon.reactions) == ["DD"]  # No tritium, so no DT reaction


def test_gas_material_upstream():
    """An upstream stage can hold a gas material while the target chamber holds D/T gas."""
    argon = Material("Ar", "gas", molecules={"Ar": 1})
    s = make_source(
        pressure=[0.5, 5.0],
        temperature=[25.0, 25.0],
        length=[20.0, 50.0],
        divergence=[0.0, 30.0],
        position_bins=[1, 10],
        position_substeps=[4, 4],
        material=[argon, dt_gas(0.5)],
    )
    assert s.target_density_per_region["DD"][0] == 0  # No neutrons from the argon stage
    assert s.materials[1].name == "D/T gas"


# --- Solid targets ---------------------------------------------------------------------------


def test_thick_solid_target():
    """The beam stops within a few micrometers of TiD1.5: all its power stays in the target."""
    s = make_solid()
    assert s.atom_density_per_region[0] == pytest.approx(3.9 * 6.02214076e23 / (
        (47.9 + 1.5 * 2.0141) / 2.5), rel=1e-3)  # fmt: skip
    assert s.target_density_per_region["DD"][0] == pytest.approx(0.6 * s.atom_density_per_region[0])
    E = s.beam_energy_grid
    x_stop = s.x_grid[np.flatnonzero(E > 0)[-1] + 1]
    assert 1e-4 < x_stop < 4e-4  # 1-4 um
    assert E[-1] == 0 and s.backstop_power == 0 and s.wall_power == 0
    assert s.deposited_power == pytest.approx(s.beam_power_entry)
    assert s.reactions_present() == ["DD"]
    # Thick-target DD yield, about 1.7e6 n/s per uA at 280 keV
    assert 1e6 < s.yields["DD"] / (s.beam_current / MICROAMPERE) < 3e6


def test_titanium_tritide_gives_dt():
    s = make_solid(material=TIT)
    assert list(s.reactions) == ["DT"]  # No deuterium, so no DD reaction
    assert s.yields["DT"] > 50 * make_solid().yields["DD"]  # DT is far more likely


def test_solid_thickness_calculated():
    """For a solid target chamber, the length follows from the backstop energy."""
    thick = make_solid(length=[None], beam_energy_backstop=100.0)
    assert thick.calculated == "length"
    assert thick.beam_energy_grid[-1] == pytest.approx(100 * KEV, rel=1e-6)
    assert 0.5e-4 < thick.length[0] < 2e-4


def test_solid_plots_in_micrometres(tmp_path):
    s = make_solid()
    assert s._position_units() == (1e4, r"$\mu$m")
    assert make_source()._position_units() == (1.0, "cm")
    s.make_plots(tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 8  # DD only


def test_solid_outputs_record_material(tmp_path):
    s = make_solid()
    s.write_text_files(tmp_path)
    with open(tmp_path / "summary.toml", "rb") as f:
        summary = tomllib.load(f)
    assert summary["inputs"]["material"] == ["TiD1.5"]
    assert "density" not in summary["inputs"]  # Only listed when a gas is given by density
    assert summary["materials"]["TiD1.5"] == {
        "phase": "solid",
        "density": 3.9,
        "atoms": {"Ti": 1.0, "D": 1.5},
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
        (dict(pressure=[5.0]), "is a solid, so it takes no pressure"),
        (dict(temperature=[25.0]), "is a solid, so it takes no pressure, temperature, or density"),
        (dict(stopping_power="icru49"), "icru49 is for hydrogen-isotope gases only.*is a solid"),
        (dict(length=[None]), "only one value can be calculated"),
    ],
)
def test_solid_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_solid(**overrides)


def test_unknown_element_rejected():
    with pytest.raises(ValueError, match="unknown element 'Xx'"):
        Material("bad", "solid", atoms={"Xx": 1}, density=1.0)


def test_isotopes_and_natural_elements():
    """A natural element is split into its isotopes; an isotope stays itself. Stopping uses
    the element's table, except for D, T, and 3He."""
    natural = Material("Li", "solid", atoms={"Li": 1}, density=0.534)
    assert natural.isotope_fractions[(3, 7)] == 0.9241
    assert natural.stopping_fractions() == {"Li": pytest.approx(1.0)}
    enriched = Material("7Li", "solid", atoms={"7Li": 1}, density=0.534)
    assert enriched.isotope_fractions == {(3, 7): 1.0}
    assert enriched.mean_mass == pytest.approx(7.016, abs=1e-3)
    gas = Material("He3", "gas", molecules={"3He": 1, "D2": 1})
    assert gas.stopping_fractions() == {"He3": pytest.approx(1 / 3), "D": pytest.approx(2 / 3)}
    hydrogen = Material("H2", "gas", molecules={"H2": 1})
    assert hydrogen.isotope_fractions[(1, 2)] > 0  # Natural hydrogen holds some deuterium


def test_material_required():
    with pytest.raises(ValueError, match="every region needs a Material"):
        make_source(material=[None])
    with pytest.raises(ValueError, match="T_frac must be between 0 and 1"):
        dt_gas(1.5)


# --- Gas given by density --------------------------------------------------------------------

# SMALL_CASE's D/T gas: 5 torr at 25 degC, as a mass density [g/cm3]
DT_MEAN_MASS = (2.01410177812 + 3.0160492779) / 2  # u, per atom (SRIM's masses)
DT_DENSITY = get_atom_density(5 * TORR, 25 + ZERO_CELSIUS) * DT_MEAN_MASS / 6.02214076e23


def test_gas_by_density_matches_pressure():
    """A gas given by its mass density gives the same results as the same gas given by the
    pressure and temperature that make that density."""
    by_pressure = make_source()
    by_density = make_source(pressure=[None], temperature=[None], density=[DT_DENSITY])
    assert by_density.atom_density_per_region[0] == pytest.approx(
        by_pressure.atom_density_per_region[0], rel=1e-6
    )
    assert by_density.yield_total == pytest.approx(by_pressure.yield_total, rel=1e-6)
    inputs = {name: value for name, value, _ in by_density.get_inputs()}
    assert inputs["density"][0] == DT_DENSITY


def test_gas_density_calculated():
    """With no temperature, the target chamber's state is its density, which can be solved."""
    forward = make_source(pressure=[None], temperature=[None], density=[DT_DENSITY])
    s = make_source(
        pressure=[None],
        temperature=[None],
        density=[None],
        beam_energy_backstop=forward.beam_energy_grid[-1] / KEV,
    )
    assert s.calculated == "density"
    assert s.density[0] == pytest.approx(DT_DENSITY, rel=1e-8)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(density=[1e-6]), "give pressure and temperature, or density, not both"),
        (dict(temperature=[None]), "a pressure needs a temperature"),
        (
            dict(pressure=[None], temperature=[None]),
            "only one value can be calculated.*the density of region 1",
        ),
        (dict(density=[-1.0], pressure=[None], temperature=[None]), "densities must be positive"),
    ],
)
def test_gas_state_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_source(**overrides)


def test_solid_takes_no_density():
    with pytest.raises(ValueError, match="takes no pressure, temperature, or density"):
        make_solid(density=[1.0])


def test_monatomic_and_molecular_gases():
    """A gas's molecules set its atoms per molecule: water vapor has 3, argon 1."""
    steam = Material("steam", "gas", molecules={"H2O": 1})
    assert steam.atoms_per_unit == 3 and steam.atom_fractions == pytest.approx(
        {"H": 2 / 3, "O": 1 / 3}
    )
    assert Material("Ar", "gas", molecules={"Ar": 1}).atoms_per_unit == 1
    mix = Material("H2 + O2", "gas", molecules={"H2": 2, "O2": 1})
    assert mix.atoms_per_unit == 2 and mix.atom_fractions == steam.atom_fractions
    assert (
        mix.kind == "molecules"
        and Material("Ti", "solid", atoms={"Ti": 1}, density=4.5).kind == "atoms"
    )


def test_small_case_unchanged_by_materials():
    """SMALL_CASE is a 50/50 D/T gas (dt_gas)."""
    s = make_source()
    assert [m.name for m in s.materials] == ["D/T gas"]
    assert s.materials[0].phase == "gas"
    assert s.materials[0].isotope_fractions == {(1, 2): 0.5, (1, 3): 0.5}
    assert s.target_density_per_region["DT"][0] == 0.5 * s.atom_density_per_region[0]


@pytest.mark.parametrize("molecules", [{"D2": 0.9, "Ar": 0.1}, {"D2": 0.5, "N2": 0.5}])
def test_icru49_refused_for_other_elements(molecules):
    """ICRU 49 data here are for hydrogen gas only, so a gas with any other element is
    refused rather than half run with SRIM."""
    gas = Material("mix", "gas", molecules=molecules)
    with pytest.raises(ValueError, match="icru49 is for hydrogen-isotope gases only.*contains"):
        make_source(material=gas, stopping_power="icru49")
    assert make_source(material=gas).yield_total > 0  # Fine with srim


def test_icru49_for_any_hydrogen_isotope_gas():
    gas = Material("H2 + D2 + T2", "gas", molecules={"H2": 0.2, "D2": 0.4, "T2": 0.4})
    assert make_source(material=gas, stopping_power="icru49").yield_total > 0
