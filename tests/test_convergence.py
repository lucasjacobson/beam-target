"""Numerical convergence: refining each numerical setting of the calculation in turn must not
move the results, on output cells as fine as users choose (1 degree, 0.1 MeV).

The regression tests pin the results, so they notice any change but not an error that was
there all along; these notice when a setting is too coarse (a jitter between neighboring
cells, say, as a coarse grid of beam energies once left in 18 MeV p + Be). Each case covers a
different part of the code: evaluated continua (Kalbach-Mann), lab-frame tables, the built-in
two-body reactions in a stopping solid, and a gas the beam spreads in. Each must run without
warnings, so these also test that a case the code does not warn about is converged (the
position mesh, for one, is checked by BeamTarget.check_position_mesh).
"""

import os

import numpy as np
import pytest
from regression_cases import BE, ENDFB, FENDL, TA, TID, unpacked_library

import beamtarget
import beamtarget.endf as endf
import beamtarget.source as source
from beamtarget import BeamTarget, Region
from beamtarget.source import dt_gas
from beamtarget.units import KEV, MEV, MILLIAMPERE, MILLIRADIAN, TORR, ZERO_CELSIUS

# The largest change allowed when a setting is refined: rms relative over the (angle, energy)
# cells holding over 1% of the peak (summed over position), and largest relative change of the
# neutrons in a position bin
CELLS_RMS = 0.015
POSITION_MAX = 0.01


def solid(material, energy_MeV, library, beam="proton", substeps=10):
    region = Region(material, wall_radius=10.0, position_bins=20, position_substeps=substeps)
    return dict(
        beam=beam, nuclear_data=[library], beam_energy_entry=energy_MeV * MEV, regions=[region],
        beam_energy_backstop=10 * KEV, angle_bin_width=1.0, energy_bin_width=0.1 * MEV,
    )  # fmt: skip


def dt_gas_chamber():
    region = Region(
        dt_gas(0.5), 50.0, pressure=10 * TORR, temperature=25 + ZERO_CELSIUS, wall_radius=1.0,
        divergence=30 * MILLIRADIAN, position_bins=20,
    )  # fmt: skip
    return dict(
        beam="deuteron", nuclear_data=["drosg"], beam_energy_entry=300 * KEV, regions=[region],
        angle_bin_width=1.0, energy_bin_width=0.01 * MEV,
    )  # fmt: skip


CASES = {
    # Continua (and two-body levels, whose lines need the finer position mesh)
    "p_Be_18_endfb81": lambda d: solid(BE, 18.0, str(unpacked_library(ENDFB, d)), substeps=25),
    "p_Ta_30_fendl32c": lambda d: solid(TA, 30.0, str(unpacked_library(FENDL, d))),  # Tables
    "d_TiD_0.3_drosg": lambda d: dict(  # Two-body lines in a stopping solid
        solid(TID, 0.3, "drosg", beam="deuteron"), energy_bin_width=0.01 * MEV
    ),
    "d_DT_gas_0.3_drosg": lambda d: dt_gas_chamber(),  # Passing through, spread and scraped
}

# Each numerical setting, and how to refine it (4 times finer)
REFINEMENTS = {
    "beam-energy interpolation": lambda m: m.setattr(
        source, "ENDF_INTERPOLATION_TOLERANCE", source.ENDF_INTERPOLATION_TOLERANCE / 4
    ),
    "continuum slices": lambda m: m.setattr(
        endf, "CONTINUUM_ANGLE_SHIFT", endf.CONTINUUM_ANGLE_SHIFT / 4
    ),
    "CM angle cells": lambda m: m.setattr(
        source, "ENDF_ANGLE_CELLS_PER_BIN", 4 * source.ENDF_ANGLE_CELLS_PER_BIN
    ),
    "position steps": None,  # position_substeps, set on the regions
}


def run(settings, substeps=None):
    if substeps is not None:
        for region in settings["regions"]:
            region.position_substeps = substeps
    return BeamTarget(name="c", beam_current=1 * MILLIAMPERE, beam_radius_entry=0.1, **settings)


@pytest.fixture(scope="module", autouse=True)
def all_cores():
    """Use every core within each case (the results do not depend on it)."""
    beamtarget.set_parallel(os.cpu_count() or 1)
    yield
    beamtarget.set_parallel(1)


@pytest.fixture(scope="module")
def default_results(tmp_path_factory):
    """Each case with the default settings, run once."""
    directory = tmp_path_factory.mktemp("libraries")
    return {name: (run(make(directory)), make, directory) for name, make in CASES.items()}


@pytest.mark.parametrize("refinement", REFINEMENTS)
@pytest.mark.parametrize("case", CASES)
def test_results_converged(default_results, case, refinement, monkeypatch):
    base, make, directory = default_results[case]
    assert base.warnings == []
    settings = make(directory)
    substeps = None
    if REFINEMENTS[refinement] is None:
        substeps = 4 * base.position_substeps[0]
    else:
        REFINEMENTS[refinement](monkeypatch)
    fine = run(settings, substeps)
    assert fine.yield_total == pytest.approx(base.yield_total, rel=1e-3)
    for rxn in base.reactions_present():
        a, b = base.neutrons[rxn], fine.neutrons[rxn]
        cells_a, cells_b = a.sum(0), b.sum(0)
        change = source._interpolation_change(cells_b, cells_a)
        assert change <= CELLS_RMS, f"{rxn}: (angle, energy) cells change {change:.2%} rms"
        positions_a, positions_b = a.sum((1, 2)), b.sum((1, 2))
        big = positions_b > 0.01 * positions_b.max()
        worst = np.max(np.abs(positions_a - positions_b)[big] / positions_b[big])
        assert worst <= POSITION_MAX, f"{rxn}: position bins change up to {worst:.2%}"


def test_unconverged_interpolation_warns(monkeypatch, tmp_path):
    """If the beam-energy grid cannot be refined enough, the case says so."""
    monkeypatch.setattr(source, "ENDF_INTERPOLATION_TOLERANCE", 1e-6)
    monkeypatch.setattr(source, "ENDF_MAX_HALVINGS", 1)
    s = run(CASES["p_Be_18_endfb81"](tmp_path))
    assert any("had not converged after 1 halvings" in w for w in s.warnings)


def test_coarse_grid_is_refined(monkeypatch, tmp_path):
    """The check works: starting from the coarse grid that once left a jitter of up to 25%,
    the result is the same as from a fine one."""
    settings = CASES["p_Be_18_endfb81"](tmp_path)
    coarse = run(settings)
    monkeypatch.setattr(source, "ENDF_ENERGY_STEP", (0.005, 0.00125))
    monkeypatch.setattr(source, "ENDF_MAX_HALVINGS", 1)
    monkeypatch.setattr(source, "ENDF_INTERPOLATION_TOLERANCE", 1.0)  # One halving: 0.25%
    fine = run(CASES["p_Be_18_endfb81"](tmp_path))
    a, b = coarse.neutrons["p9Be_xn"].sum(0), fine.neutrons["p9Be_xn"].sum(0)
    assert source._interpolation_change(b, a) <= 0.01
