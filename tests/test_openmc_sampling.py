"""OpenMC itself samples the sources from get_openmc_sources, and the particles are checked
against the calculated source: position along the beam, distance from the axis, emission
angle, energy, and their correlations, for the case's placement and for moved and turned
ones.

Uses openmc.lib (the compiled OpenMC library) to sample source particles exactly as a
transport run would, without transporting them. Skipped if it is not available.
"""

import os

import numpy as np
import pytest
from conftest import MAX_Z, assert_counts_match, assert_mean_matches

openmc = pytest.importorskip("openmc")
try:
    import openmc.lib
except (ImportError, OSError):  # No compiled library
    pytest.skip("openmc.lib is not available", allow_module_level=True)

from beamtarget.openmc_source import get_openmc_sources  # noqa: E402

NUM_SAMPLES = 200_000

# Placements: get_openmc_sources arguments, and the expected beam axis and entry point
# ("backstop" means the backstop is at that point, so entry = backstop - L * axis)
PLACEMENTS = {
    "case": ({}, (0, 0, -1), ("backstop", (0, 0, 0))),
    "moved": (dict(backstop_position=[5.0, -3.0, 10.0]), (0, 0, -1), ("backstop", (5, -3, 10))),
    "turned": (dict(direction="+x", backstop_position=[0.0, 0.0, 0.0]), (1, 0, 0),
               ("backstop", (0, 0, 0))),
    "tilted": (dict(direction=[1.0, 2.0, 2.0], entry_position=[1.0, 2.0, 3.0]), (1, 2, 2),
               ("entry", (1, 2, 3))),
}  # fmt: skip


def sample(directory, sources, n=NUM_SAMPLES):
    """Return (positions, directions, energies [MeV]) of n source particles sampled by OpenMC
    from sources, in a void sphere with no materials."""
    directory.mkdir(parents=True, exist_ok=True)
    # OpenMC needs a cross-section library file even with no materials; it only opens the
    # data files of nuclides that are used, so a placeholder entry will do
    library = directory / "cross_sections.xml"
    library.write_text(
        '<?xml version="1.0"?>\n<cross_sections>\n'
        '  <library materials="H1" path="unused.h5" type="neutron"/>\n</cross_sections>\n'
    )
    sphere = openmc.Sphere(r=1e4, boundary_type="vacuum")
    model = openmc.Model(
        geometry=openmc.Geometry([openmc.Cell(region=-sphere)]), materials=openmc.Materials()
    )
    model.settings = openmc.Settings(run_mode="fixed source", particles=100, batches=1)
    model.settings.source = sources
    model.export_to_xml(directory)

    cwd, old_library = os.getcwd(), os.environ.get("OPENMC_CROSS_SECTIONS")
    os.environ["OPENMC_CROSS_SECTIONS"] = str(library)
    os.chdir(directory)
    try:
        openmc.lib.init(output=False)
        try:
            sites = openmc.lib.sample_external_source(n, prn_seed=12345)
        finally:
            openmc.lib.finalize()
    finally:
        os.chdir(cwd)
        if old_library is None:
            del os.environ["OPENMC_CROSS_SECTIONS"]
        else:
            os.environ["OPENMC_CROSS_SECTIONS"] = old_library
    r = np.array([site.r for site in sites])
    u = np.array([site.u for site in sites])
    E = np.array([site.E for site in sites]) / 1e6
    return r, u, E


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory, mixed_source):
    path = tmp_path_factory.mktemp("sampling") / "source.h5"
    mixed_source.write_hdf5(path, direction="-z", backstop_position=[0.0, 0.0, 0.0])
    return path


@pytest.fixture(scope="module", params=list(PLACEMENTS))
def sampled(request, h5_path, mixed_source, tmp_path_factory):
    """Particles sampled by OpenMC for one placement, in beam coordinates: distance along the
    beam from the entry point, distance from the axis, cosine of the direction with the beam,
    energy, and the unit vectors of position and direction across the axis."""
    kwargs, axis, (end, point) = PLACEMENTS[request.param]
    axis = np.array(axis, dtype=float) / np.linalg.norm(axis)
    L = mixed_source.x_grid_out[-1]
    entry = np.array(point, dtype=float) - (L * axis if end == "backstop" else 0)
    sources = get_openmc_sources(h5_path, **kwargs)
    r, u, E = sample(tmp_path_factory.mktemp(request.param), sources)
    d = r - entry
    along = d @ axis
    across = d - along[:, None] * axis
    rho = np.linalg.norm(across, axis=1)
    mu = u @ axis
    u_across = u - mu[:, None] * axis
    return {
        "name": request.param,
        "along": along,
        "rho": rho,
        "mu": mu,
        "E": E,
        "across_unit": across / rho[:, None],
        "u_across_unit": u_across / np.linalg.norm(u_across, axis=1)[:, None],
        "axis": axis,
    }


def _bins(s, sampled):
    """Output-mesh bin of each particle: position i, angle j."""
    i = np.searchsorted(s.x_grid_out, sampled["along"], side="right") - 1
    angle = np.degrees(np.arccos(np.clip(sampled["mu"], -1, 1)))
    j = np.minimum((angle / s.angle_step_out).astype(int), s.num_angle_bins_out - 1)
    return i, j


def test_positions_along_the_beam(sampled, mixed_source):
    """Every particle lies on the segment from the entry point to the backstop, and the
    numbers per position bin follow the axial profile."""
    s = mixed_source
    along = sampled["along"]
    outside = np.sum((along < 0) | (along > s.x_grid_out[-1]))
    assert outside == 0, f"{sampled['name']}: {outside} particles off the beam segment"
    counts, _ = np.histogram(along, s.x_grid_out)
    axial = sum(np.sum(s.neutrons[rxn], (1, 2)) for rxn in s.reactions)
    assert_counts_match(counts, axial, f"{sampled['name']}: axial profile")


def test_radius_and_azimuth(sampled, mixed_source):
    """Within each position bin, particles fill a disc of the beam edge radius uniformly in
    area (rho^2 / R^2 uniform on [0, 1]), at every azimuth around the axis."""
    s = mixed_source
    i, _ = _bins(s, sampled)
    for k in range(s.num_x_bins_total_out):
        in_bin = i == k
        R = s.radius_mids_out[k]
        rho = sampled["rho"][in_bin]
        assert np.all(rho <= R * (1 + 1e-9)), f"{sampled['name']}: outside the beam radius"
        assert_mean_matches((rho / R) ** 2, 0.5, 1 / 12, f"{sampled['name']}: rho^2, bin {k}")
    n = len(sampled["rho"])
    for key in ("across_unit", "u_across_unit"):  # Uniform azimuth: unit vectors average 0
        mean = np.mean(sampled[key], axis=0)
        assert np.all(np.abs(mean) < MAX_Z * np.sqrt(0.5 / n)), f"{sampled['name']}: {key}"


def test_angles_and_their_correlation_with_position(sampled, mixed_source):
    """The emission angle from the beam direction follows the angular distribution, and the
    joint (position, angle) distribution matches the source bin by bin."""
    s = mixed_source
    i, j = _bins(s, sampled)
    joint = np.zeros((s.num_x_bins_total_out, s.num_angle_bins_out))
    np.add.at(joint, (i, j), 1)
    expected = sum(np.sum(s.neutrons[rxn], 2) for rxn in s.reactions)
    assert_counts_match(joint.sum(0), expected.sum(0), f"{sampled['name']}: angles")
    assert_counts_match(joint, expected, f"{sampled['name']}: position x angle")


def test_energies_and_their_correlation(sampled, mixed_source):
    """The energy spectrum matches bin by bin (no particles outside the filled bins, so the
    histogram is not smeared or shifted), and the mean energy in each angle bin and in each
    position bin matches the source (the energy keeps its correlation with angle and
    position)."""
    s = mixed_source
    grid, by_angle = s.neutrons_by_angle_and_energy()
    counts, _ = np.histogram(sampled["E"], grid)
    spectrum = sum(np.sum(by_angle[rxn], 0) for rxn in s.reactions)
    assert_counts_match(counts, spectrum, f"{sampled['name']}: energy spectrum")

    # Expected mean and variance of the energy, uniform within each energy bin
    width = s.energy_bin_width
    mids = {rxn: s.neutron_energy_mids[rxn] for rxn in s.reactions}
    i, j = _bins(s, sampled)
    groups = [("angle", j, s.num_angle_bins_out), ("position", i, s.num_x_bins_total_out)]
    for axis_name, index, num in groups:
        for k in range(num):
            weights = {
                rxn: (s.neutrons[rxn][:, k, :] if axis_name == "angle" else s.neutrons[rxn][k])
                for rxn in s.reactions
            }
            w = {rxn: np.sum(weights[rxn], 0) for rxn in s.reactions}  # Per energy bin
            total = sum(np.sum(w[rxn]) for rxn in s.reactions)
            mean = sum(np.sum(w[rxn] * mids[rxn]) for rxn in s.reactions) / total
            second = sum(np.sum(w[rxn] * (mids[rxn] ** 2 + width**2 / 12)) for rxn in s.reactions)
            variance = second / total - mean**2
            assert_mean_matches(
                sampled["E"][index == k], mean, variance,
                f"{sampled['name']}: mean energy, {axis_name} bin {k}",
            )  # fmt: skip


def test_checks_detect_a_wrong_axis(sampled, mixed_source):
    """The checks have the power to catch a mistake: read along the opposite axis, the same
    particles fail the angular distribution."""
    s = mixed_source
    flipped = {**sampled, "mu": -sampled["mu"]}
    _, j = _bins(s, flipped)
    counts = np.bincount(j, minlength=s.num_angle_bins_out)
    expected = sum(np.sum(s.neutrons[rxn], (0, 2)) for rxn in s.reactions)
    with pytest.raises(AssertionError):
        assert_counts_match(counts, expected, "flipped")
