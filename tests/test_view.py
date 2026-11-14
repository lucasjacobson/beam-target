"""The surface viewer (python -m beamtarget.view)."""

import matplotlib
import pytest
from conftest import make_source

from beamtarget import view


@pytest.fixture(scope="module")
def source_h5(tmp_path_factory):
    path = tmp_path_factory.mktemp("view") / "source.h5"
    make_source(T_frac=0.5).write_hdf5(path)
    return path


def test_read_surfaces(source_h5):
    surfaces = view.read_surfaces(source_h5)
    assert [s[0].split(" neutron")[0] for s in surfaces] == ["D(d,n)$^{3}$He", "T(d,n)$^{4}$He"]
    title, angles, energies, density = view.read_surfaces(source_h5, ["DT"], (14.0, 15.0))[0]
    assert title.endswith("(test)")
    assert density.shape == (len(angles), len(energies))
    assert 14.0 <= energies.min() and energies.max() <= 15.0


def test_unknown_reaction_or_empty_range(source_h5):
    with pytest.raises(SystemExit, match="no reaction"):
        view.read_surfaces(source_h5, ["XX"])
    with pytest.raises(SystemExit, match="no neutrons between"):
        view.read_surfaces(source_h5, ["DT"], (0.0, 1.0))


def test_main_opens_a_window_per_reaction(source_h5, monkeypatch):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(len(plt.get_fignums())))
    plt.close("all")
    view.main([str(source_h5)])
    assert shown == [2]
    plt.close("all")
