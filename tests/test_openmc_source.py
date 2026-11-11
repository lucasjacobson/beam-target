"""OpenMC sources from source.h5 (openmc_source.py); skipped if openmc is not installed."""

import h5py
import numpy as np
import pytest
from conftest import make_source

openmc = pytest.importorskip("openmc")

from shine_neutron_source.openmc_source import get_openmc_sources, read_placement  # noqa: E402


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory, mixed_source):
    """The 50/50 D/T case along -z with its backstop at (0, 0, 0)."""
    path = tmp_path_factory.mktemp("openmc") / "source.h5"
    mixed_source.write_hdf5(path, direction="-z", backstop_position=[0.0, 0.0, 0.0])
    return path


def test_one_source_per_filled_cell(h5_path, mixed_source):
    s = mixed_source
    sources = get_openmc_sources(h5_path)
    filled = sum(np.count_nonzero(np.sum(s.neutrons[rxn], 2)) for rxn in ("DD", "DT"))
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
    np.testing.assert_allclose(space.z.x, mixed_source.x_grid_out[:2])  # Along the beam


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


def test_cells_match_the_output_mesh(h5_path, mixed_source):
    """Each source covers one position bin and one angle bin, with that cell's spectrum as a
    histogram on the output energy bins."""
    s = mixed_source
    sources = iter(get_openmc_sources(h5_path))
    for i in range(s.num_x_bins_total_out):
        for j in range(s.num_angle_bins_out):
            for rxn in ("DD", "DT"):
                spectrum = s.neutrons[rxn][i, j, :]
                if not np.any(spectrum):
                    continue
                src = next(sources)
                np.testing.assert_allclose(src.space.z.x, s.x_grid_out[i : i + 2])
                assert src.space.r.b == s.radius_mids_out[i] and src.space.r.n == 1
                np.testing.assert_allclose(
                    src.angle.mu.x, np.sort(s.cosine_grid_out[j : j + 2]), atol=1e-15
                )
                assert src.energy.interpolation == "histogram"
                lo, hi = np.flatnonzero(spectrum)[[0, -1]]
                np.testing.assert_allclose(
                    src.energy.x, s.neutron_energy_grid[rxn][lo : hi + 2] * 1e6
                )
                np.testing.assert_allclose(src.energy.p[:-1], spectrum[lo : hi + 1])
                assert src.strength == pytest.approx(spectrum.sum())
    assert next(sources, None) is None


def test_sampled_energies_follow_the_histogram(h5_path):
    """Sampling a source's energy reproduces its histogram mean (no shift by half a bin)."""
    src = max(get_openmc_sources(h5_path), key=lambda s: s.strength)
    x, p = src.energy.x, src.energy.p[:-1]
    mean = np.sum(p * (x[:-1] + x[1:]) / 2) / np.sum(p)
    samples = src.energy.sample(200_000, seed=1)
    samples = samples[0] if isinstance(samples, tuple) else samples  # (values, weights) in 0.16
    assert np.all((samples >= x[0]) & (samples <= x[-1]))
    assert np.mean(samples) == pytest.approx(mean, abs=0.02 * (x[1] - x[0]))


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
    make_source(num_x_bins=[4], num_x_bins_out=[2]).write_hdf5(path)
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
