"""MCNP itself samples the SDEF card from get_mcnp_sdef, and the source particles are checked
against the calculated source, for every beam direction and the different forms the card
takes.

MCNP writes every source particle to a binary PTRAC file (full precision), read with
mcnptools. The SDEF card cannot hold the full correlations (see the notes in the card): the
direction does not depend on position, and the energy depends only on the direction. So
the particles are checked against what the card represents: the axial profile, the beam
radius in each position bin, the angular distribution, the energy spectrum in each angle
bin, and (confirming the approximations) position and angle independent and the energy
not depending on position.

Skipped unless MCNP (the mcnp6 executable on the PATH, or MCNP_EXE) and mcnptools are
available.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import MAX_Z, assert_counts_match, assert_mean_matches, make_source

from beamtarget import Material
from beamtarget.units import MEV

ENDF_DIR = Path(__file__).parent / "data" / "endf" / "ENDF-B-VIII.1"

mcnptools = pytest.importorskip("mcnptools")
MCNP = os.environ.get("MCNP_EXE") or shutil.which("mcnp6")
if MCNP is None:
    pytest.skip("MCNP (mcnp6) is not available", allow_module_level=True)

NUM_SAMPLES = 100_000

# Beam direction -> unit vector (written out here, not taken from SDEF_DIRECTIONS, so that
# the card's own vectors are checked)
AXES = {
    "-x": (-1, 0, 0), "+x": (1, 0, 0),
    "-y": (0, -1, 0), "+y": (0, 1, 0),
    "-z": (0, 0, -1), "+z": (0, 0, 1),
}  # fmt: skip


def _gas(formula):
    return Material(formula, "gas", molecules={formula: 1})


# Cases: (make_source overrides, direction, placement keyword, point)
CASES = {
    # The 50/50 D/T case (diverging beam, 10 degree bins: radius and energy depend on
    # position and angle), along every axis, placed by its backstop or its entry point
    "DT50_-z": ({}, "-z", "backstop_position", (0.0, 0.0, 0.0)),
    "DT50_+z": ({}, "+z", "entry_position", (1.0, 2.0, 3.0)),
    "DT50_-x": ({}, "-x", "backstop_position", (5.0, -3.0, 10.0)),
    "DT50_+x": ({}, "+x", "entry_position", (-100.0, 0.0, 0.0)),
    "DT50_-y": ({}, "-y", "entry_position", (0.0, 7.0, 0.0)),
    "DT50_+y": ({}, "+y", "backstop_position", (0.0, 0.0, 0.0)),
    # Three regions of 1, 2, and 3 output bins (si1 with no, one "i", and "2i" points)
    "regions": (
        dict(
            length=[10.0, 20.0, 30.0],
            pressure=[5.0, 5.0, 5.0],
            temperature=[25.0, 25.0, 25.0],
            position_bins=[1, 2, 3],
            position_substeps=[2, 2, 2],
            divergence=[30.0, 30.0, 30.0],
        ),
        "-z", "backstop_position", (0.0, 0.0, 0.0),
    ),
    # One angle bin: an isotropic source (no d2, one energy distribution)
    "isotropic": (dict(angle_bin_width=180.0), "-z", "backstop_position", (0.0, 0.0, 0.0)),
    # No divergence: one radius (rad = d3, no fext)
    "parallel": (dict(divergence=[0.0], wall_radius=1.0), "+x", "entry_position", (0.0, 0.0, 0.0)),
    # 2.5 MeV protons on 7Li: the n0 and n1 lines overlap, so their energies share segments
    "lithium": (
        dict(
            beam="proton",
            material=Material("7Li", "solid", atoms={"7Li": 1}, density=0.534),
            beam_energy_entry=2500.0,
            pressure=[None],
            temperature=[None],
            length=[1e-3],
            divergence=[0.0],
            position_bins=[5],
            position_substeps=[4],
        ),
        "+y", "backstop_position", (0.0, 0.0, 0.0),
    ),
    # H(t,n)3He near threshold: a forward cone, so most angle bins are empty (their energy
    # distributions are never sampled)
    "triton_cone": (
        dict(beam="triton", material=_gas("1H2"), beam_energy_entry=3300.0, length=[1.0]),
        "-z", "backstop_position", (0.0, 0.0, 0.0),
    ),
    # 18 MeV protons stopping in beryllium, from evaluated data (ENDF/B-VIII.1): continuum
    # spectra from 0 to 18 MeV, plus two-body lines
    "p_Be_endf": (
        dict(
            beam="proton",
            material=Material("Be", "solid", atoms={"Be": 1}, density=1.848),
            beam_energy_entry=18000.0,
            pressure=[None],
            temperature=[None],
            length=[0.5],
            divergence=[0.0],
            position_bins=[10],
            position_substeps=[10],
            energy_bin_width=0.1,  # MeV
            nuclear_data=["drosg", str(ENDF_DIR)],
        ),
        "+x", "entry_position", (0.0, 0.0, 0.0),
    ),
}  # fmt: skip


def run_mcnp(directory, sdef, n=NUM_SAMPLES):
    """Return (positions, directions, energies [MeV]) of n source particles sampled by MCNP
    from an SDEF card, in a void sphere (nothing is transported)."""
    directory.mkdir(parents=True, exist_ok=True)
    deck = (
        "SDEF sampling test\n"
        "1 0 -1 imp:n=1\n"
        "2 0 1 imp:n=0\n"
        "\n"
        "1 so 1000\n"
        "\n"
        "mode n\n"
        f"{sdef}"
        f"nps {n}\n"
        f"ptrac file=bin event=src write=all max={n}\n"
    )
    (directory / "deck.i").write_text(deck, encoding="ascii", newline="\n")
    result = subprocess.run(
        [MCNP, "i=deck.i", "n=run."], cwd=directory, capture_output=True, text=True
    )
    ptrac = directory / "run.p"
    if result.returncode != 0 or not ptrac.exists():
        output = directory / "run.o"
        tail = output.read_text(errors="replace")[-3000:] if output.exists() else result.stdout
        pytest.fail(f"MCNP failed:\n{tail}")
    P = mcnptools.Ptrac
    histories = P(str(ptrac), P.BIN_PTRAC).ReadHistories(n)
    events = [h.GetEvent(0) for h in histories]
    assert len(events) == n and all(e.Type() == P.SRC for e in events)
    r = np.array([[e.Get(P.X), e.Get(P.Y), e.Get(P.Z)] for e in events])
    u = np.array([[e.Get(P.U), e.Get(P.V), e.Get(P.W)] for e in events])
    E = np.array([e.Get(P.ENERGY) for e in events])
    return r, u, E


@pytest.fixture(scope="module", params=list(CASES))
def sampled(request, tmp_path_factory):
    """The source of one case and the particles MCNP sampled from its SDEF card, in beam
    coordinates (see the OpenMC sampling tests)."""
    overrides, direction, placement, point = CASES[request.param]
    s = make_source(**overrides)
    sdef = s.get_mcnp_sdef(direction, **{placement: list(point)})
    entry, _ = s.sdef_positions(direction, **{placement: list(point)})
    if placement == "entry_position":
        assert np.allclose(entry, point)
    r, u, E = run_mcnp(tmp_path_factory.mktemp(request.param), sdef)

    axis = np.array(AXES[direction], dtype=float)
    d = r - entry
    along = d @ axis
    across = d - along[:, None] * axis
    rho = np.linalg.norm(across, axis=1)
    mu = u @ axis
    u_across = u - mu[:, None] * axis
    angle = np.degrees(np.arccos(np.clip(mu, -1, 1)))
    return {
        "name": request.param,
        "source": s,
        "along": along,
        "rho": rho,
        "mu": mu,
        "E": E,
        "i": np.searchsorted(s.x_grid_out, along, side="right") - 1,
        "j": np.minimum((angle / s.angle_bin_width).astype(int), s.num_angle_bins_out - 1),
        "across_unit": across / np.maximum(rho, 1e-300)[:, None],
        "u_across_unit": u_across / np.linalg.norm(u_across, axis=1)[:, None],
    }


def _total(s, axes):
    """The source summed over the given axes of (position, angle, energy) and the reactions
    (energy-resolved sums use the common mesh of neutrons_by_angle_and_energy)."""
    return sum(np.sum(s.neutrons[rxn], axes) for rxn in s.reactions)


def test_positions_along_the_beam(sampled):
    """Every particle lies on the segment from the entry point to the backstop, in the right
    direction, and the numbers per position bin follow the axial profile."""
    s, along = sampled["source"], sampled["along"]
    outside = np.sum((along < 0) | (along > s.x_grid_out[-1]))
    assert outside == 0, f"{sampled['name']}: {outside} particles off the beam segment"
    counts, _ = np.histogram(along, s.x_grid_out)
    assert_counts_match(counts, _total(s, (1, 2)), f"{sampled['name']}: axial profile")


def test_radius_and_azimuth(sampled):
    """Within each position bin, particles fill a disc of the beam edge radius uniformly in
    area (the radius as written in the card, to SDEF_RADIUS_DECIMALS), at every azimuth."""
    s = sampled["source"]
    from beamtarget.source import SDEF_RADIUS_DECIMALS

    for k in range(s.position_bins_total):
        in_bin = sampled["i"] == k
        if not np.any(in_bin):  # No neutrons there (past where the beam stops, for example)
            continue
        R = round(s.radius_mids_out[k], SDEF_RADIUS_DECIMALS)
        rho = sampled["rho"][in_bin]
        assert np.all(rho <= R * (1 + 1e-9)), f"{sampled['name']}: outside the beam radius"
        assert_mean_matches((rho / R) ** 2, 0.5, 1 / 12, f"{sampled['name']}: rho^2, bin {k}")
    n = len(sampled["rho"])
    for key in ("across_unit", "u_across_unit"):
        mean = np.mean(sampled[key], axis=0)
        assert np.all(np.abs(mean) < MAX_Z * np.sqrt(0.5 / n)), f"{sampled['name']}: {key}"


def test_angles(sampled):
    """The emission angle follows the angular distribution (uniform in cosine for an
    isotropic source), independently of position, as the card is built."""
    s = sampled["source"]
    name = sampled["name"]
    if s.num_angle_bins_out == 1:  # Isotropic: check uniformity in cosine on finer bins
        counts, _ = np.histogram(sampled["mu"], np.linspace(-1, 1, 21))
        assert_counts_match(counts, np.ones(20), f"{name}: isotropic angles")
    angular = _total(s, (0, 2))
    counts = np.bincount(sampled["j"], minlength=s.num_angle_bins_out)
    assert_counts_match(counts, angular, f"{name}: angles")
    joint = np.zeros((s.position_bins_total, s.num_angle_bins_out))
    np.add.at(joint, (sampled["i"], sampled["j"]), 1)
    independent = np.outer(_total(s, (1, 2)), angular)
    assert_counts_match(joint, independent, f"{name}: position x angle (independent)")


def test_energy_in_each_angle_bin(sampled):
    """In each angle bin, the energies follow the spectrum at that angle bin by bin, with no
    particles in empty bins (the segments, the zero bins between them, and the trimmed
    ends are read as intended)."""
    s = sampled["source"]
    grid, by_angle = s.neutrons_by_angle_and_energy()
    grid = grid / MEV  # MCNP's energies are in MeV
    spectra = sum(by_angle[rxn] for rxn in s.reactions)  # (angle, energy)
    for j in range(s.num_angle_bins_out):
        in_bin = sampled["j"] == j
        if not np.any(spectra[j]):
            assert not np.any(in_bin), f"{sampled['name']}: particles in empty angle bin {j}"
            continue
        counts, _ = np.histogram(sampled["E"][in_bin], grid)
        outside = np.sum((sampled["E"][in_bin] < grid[0]) | (sampled["E"][in_bin] > grid[-1]))
        assert outside == 0, f"{sampled['name']}: energies off the mesh, angle bin {j}"
        assert_counts_match(counts, spectra[j], f"{sampled['name']}: energy, angle bin {j}")


def test_energy_does_not_depend_on_position(sampled):
    """As the card notes, the energy does not depend on position: the mean energy in every
    position bin is that of the whole source."""
    s = sampled["source"]
    grid, by_angle = s.neutrons_by_angle_and_energy()
    spectrum = sum(np.sum(by_angle[rxn], 0) for rxn in s.reactions)
    mids = (grid[:-1] + grid[1:]) / 2 / MEV  # MCNP's energies are in MeV
    width = s.energy_bin_width / MEV
    mean = np.sum(spectrum * mids) / spectrum.sum()
    variance = np.sum(spectrum * (mids**2 + width**2 / 12)) / spectrum.sum()
    variance -= mean**2
    for k in range(s.position_bins_total):
        energies = sampled["E"][sampled["i"] == k]
        if len(energies) == 0:  # No neutrons there (the axial check covers that)
            continue
        assert_mean_matches(energies, mean, variance, f"{sampled['name']}: energy, bin {k}")
