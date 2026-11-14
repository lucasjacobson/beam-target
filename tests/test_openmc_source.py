"""OpenMC sources from source.h5 (openmc_source.py); skipped if openmc is not installed."""

import h5py
import numpy as np
import pytest
from conftest import make_source

openmc = pytest.importorskip("openmc")

from beamtarget.openmc_source import (  # noqa: E402
    _merged_spectrum,
    get_openmc_sources,
    read_placement,
)


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory, mixed_source):
    """The 50/50 D/T case along -z with its backstop at (0, 0, 0)."""
    path = tmp_path_factory.mktemp("openmc") / "source.h5"
    mixed_source.write_hdf5(path, direction="-z", backstop_position=[0.0, 0.0, 0.0])
    return path


def test_one_source_per_filled_cell(h5_path, mixed_source):
    """One source per (position, angle) bin with neutrons, the reactions together."""
    s = mixed_source
    sources = get_openmc_sources(h5_path)
    filled = np.count_nonzero(sum(np.sum(s.neutrons[rxn], 2) for rxn in ("DD", "DT")))
    assert len(sources) == filled
    assert sum(src.strength for src in sources) == pytest.approx(s.yield_total, rel=1e-12)
    assert all(str(src.particle) == "neutron" for src in sources)


def test_case_placement(h5_path, mixed_source):
    """By default the source sits where the case put it: along -z, the backstop at 0."""
    L = mixed_source.x_grid_out[-1]
    direction, entry, backstop = read_placement(h5_path)
    assert direction == "-z"
    np.testing.assert_allclose(entry, [0, 0, L])
    np.testing.assert_allclose(backstop, [0, 0, 0])
    space = get_openmc_sources(h5_path)[0].space
    np.testing.assert_allclose(space.origin, [0, 0, L])
    np.testing.assert_allclose(space.z_dir, [0, 0, -1])
    assert np.dot(space.r_dir, space.z_dir) == pytest.approx(0)
    np.testing.assert_allclose(
        [space.z.a, space.z.b], mixed_source.x_grid_out[:2]
    )  # Along the beam


@pytest.mark.parametrize(
    "kwargs, origin, axis",
    [
        (dict(backstop_position=[1.0, 2.0, 3.0]), "backstop - L u", [0, 0, -1]),
        (dict(entry_position=[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], [0, 0, -1]),
        (dict(direction="+x", backstop_position=[0.0, 0.0, 0.0]), "backstop - L u", [1, 0, 0]),
        (dict(direction=[0.0, 3.0, 4.0], entry_position=[1.0, 2.0, 3.0]), [1, 2, 3], [0, 0.6, 0.8]),
        # The case's own direction, however written, needs no position: nothing moves
        (dict(direction="-z"), "backstop - L u", [0, 0, -1]),
        (dict(direction=[0.0, 0.0, -2.0]), "backstop - L u", [0, 0, -1]),
    ],
)
def test_moved_or_turned(h5_path, mixed_source, kwargs, origin, axis):
    L = mixed_source.x_grid_out[-1]
    u = np.array(axis, dtype=float)
    backstop = np.array(kwargs.get("backstop_position", [0.0, 0.0, 0.0]))
    expected = backstop - L * u if isinstance(origin, str) else origin
    src = get_openmc_sources(h5_path, **kwargs)[0]
    np.testing.assert_allclose(src.space.origin, expected, atol=1e-12)
    np.testing.assert_allclose(src.space.z_dir, u, atol=1e-12)
    np.testing.assert_allclose(src.angle.reference_uvw, u, atol=1e-12)
    for v in (src.space.r_dir, src.angle.reference_vwu):
        assert np.dot(v, u) == pytest.approx(0, abs=1e-12)


def cdf(edges, p, x):
    """Cumulative probability of a histogram (bin edges, probability per bin) at x."""
    cum = np.concatenate(([0.0], np.cumsum(p)))
    return np.interp(x, edges, cum, left=0.0, right=cum[-1])


def test_cells_match_the_output_mesh(h5_path, mixed_source):
    """Each source covers one position bin and one angle bin, and its energy histogram is
    the sum of the reactions' spectra there: the same cumulative distribution at every
    edge of their output energy bins."""
    s = mixed_source
    sources = iter(get_openmc_sources(h5_path))
    for i in range(s.position_bins_total):
        for j in range(s.num_angle_bins_out):
            spectra = {rxn: s.neutrons[rxn][i, j, :] for rxn in ("DD", "DT")}
            if not any(np.any(sp) for sp in spectra.values()):
                continue
            src = next(sources)
            assert (src.space.z.a, src.space.z.b) == tuple(s.x_grid_out[i : i + 2])
            assert src.space.r.b == s.radius_mids_out[i] and src.space.r.n == 1
            assert src.angle.mu.a == pytest.approx(s.cosine_grid_out[j + 1], abs=1e-15)
            assert src.angle.mu.b == pytest.approx(s.cosine_grid_out[j], abs=1e-15)
            assert src.energy.interpolation == "histogram"
            total = sum(sp.sum() for sp in spectra.values())
            assert src.strength == pytest.approx(total, rel=1e-12)
            edges = np.concatenate([s.neutron_energy_grid[r] for r in spectra])
            expected = sum(cdf(s.neutron_energy_grid[r], sp, edges) for r, sp in spectra.items())
            got = cdf(src.energy.x, src.energy.p[:-1], edges)
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12 * total)
    assert next(sources, None) is None


def test_merging_separate_spectra():
    """Two spectra far apart: one histogram, the empty gap between them a single bin."""
    edges, p = _merged_spectrum(
        [np.arange(7.0), np.arange(7.0)],
        [np.array([0, 1, 2, 0, 0, 0.0]), np.array([0, 0, 0, 0, 0, 3.0])],
    )
    np.testing.assert_array_equal(edges, [1, 2, 3, 5, 6])
    np.testing.assert_array_equal(p, [1, 2, 0, 3])


def test_merging_offset_grids():
    """Grids offset by half a bin: the sum keeps every cumulative probability exactly."""
    a_edges, a = np.arange(0.0, 5.0), np.array([1.0, 2.0, 3.0, 4.0])
    b_edges, b = np.arange(0.5, 5.0), np.array([2.0, 1.0, 5.0, 1.0])
    edges, p = _merged_spectrum([a_edges, b_edges], [a, b])
    x = np.linspace(-1, 6, 141)
    np.testing.assert_allclose(cdf(edges, p, x), cdf(a_edges, a, x) + cdf(b_edges, b, x))


def test_single_spectrum_unchanged_and_empty_runs_merged():
    edges, p = _merged_spectrum([np.arange(8.0)], [np.array([0, 2, 0, 0, 0, 1, 0.0])])
    np.testing.assert_array_equal(edges, [1, 2, 5, 6])
    np.testing.assert_array_equal(p, [2, 0, 1])
    assert _merged_spectrum([np.arange(3.0)], [np.zeros(2)]) is None


def test_sampled_energies_follow_the_histogram(h5_path):
    """Sampling a source's energy reproduces its histogram (no shift by half a bin): checked
    on the DT part of the strongest source (above the empty gap after its DD part)."""
    src = max(get_openmc_sources(h5_path), key=lambda s: s.strength)
    x, p = src.energy.x, src.energy.p[:-1]
    samples = src.energy.sample(2_000_000, seed=1)
    samples = samples[0] if isinstance(samples, tuple) else samples  # (values, weights) in 0.16
    assert np.all((samples >= x[0]) & (samples <= x[-1]))
    gap = np.argmax(np.diff(x) * (p == 0))  # The widest empty bin
    assert p[gap] == 0 and gap > 0
    part = slice(gap + 1, None)
    mids, w = (x[:-1] + x[1:])[part] / 2, p[part]
    mean = np.sum(w * mids) / np.sum(w)
    spread = np.sqrt(np.sum(w * (mids - mean) ** 2) / np.sum(w))
    above = samples[samples > x[gap + 1]]
    tolerance = 5 * spread / np.sqrt(len(above))
    assert tolerance < 0.3 * (x[gap + 2] - x[gap + 1])  # Fine enough to see half a bin
    assert np.mean(above) == pytest.approx(mean, abs=tolerance)


def test_exports_to_settings_xml(h5_path, tmp_path):
    """The sources write to settings.xml, with every tabular distribution holding as many
    probabilities as values (OpenMC reads them as one list of two equal halves)."""
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = get_openmc_sources(h5_path)
    settings.export_to_xml(tmp_path)
    text = (tmp_path / "settings.xml").read_text()
    assert text.count("<source ") == len(settings.source)
    import xml.etree.ElementTree as ET

    for element in ET.parse(tmp_path / "settings.xml").iter():
        if element.get("type") == "tabular":
            assert len(element.find("parameters").text.split()) % 2 == 0


def test_old_file_rejected(tmp_path):
    path = tmp_path / "old.h5"
    make_source(position_bins=[2], position_substeps=[2]).write_hdf5(path)
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = 5
    with pytest.raises(ValueError, match="format_version 5"):
        get_openmc_sources(path)


def test_bad_placement_rejected(h5_path):
    with pytest.raises(ValueError, match="not both"):
        get_openmc_sources(h5_path, entry_position=[0, 0, 0], backstop_position=[0, 0, 0])
    with pytest.raises(ValueError, match="non-zero vector"):
        get_openmc_sources(h5_path, direction=[0, 0, 0])
    with pytest.raises(ValueError, match="direction must be one of"):
        get_openmc_sources(h5_path, direction="z")
    # Turning the source needs a position, to say which end goes where
    with pytest.raises(ValueError, match=r"differs from the case's \(-z\): give entry_position"):
        get_openmc_sources(h5_path, direction="+x")
