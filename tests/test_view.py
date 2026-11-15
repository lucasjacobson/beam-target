"""The surface viewer (python -m beamtarget.view)."""

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
    assert [s[0].split(" neutron")[0] for s in surfaces] == [
        "D(d,n)$^{3}$He",
        "T(d,n)$^{4}$He",
        "All reactions",
    ]
    title, angles, energies, density, unit = view.read_surfaces(source_h5, ["DT"], (14.0, 15.0))[0]
    assert title.endswith("(test)") and unit == "MeV"
    assert density.shape == (len(angles), len(energies))
    assert 14.0 <= energies.min() and energies.max() <= 15.0


def test_unknown_reaction_or_empty_range(source_h5):
    with pytest.raises(SystemExit, match="no reaction"):
        view.read_surfaces(source_h5, ["XX"])
    with pytest.raises(SystemExit, match="no neutrons between"):
        view.read_surfaces(source_h5, ["DT"], (0.0, 1.0))


def test_web_page(source_h5, tmp_path):
    """--html writes a self-contained page holding every reaction's surface and the colormaps."""
    import json
    import re

    from beamtarget.webview import COLORMAPS, DEFAULT_COLORMAP

    out = tmp_path / "page" / "surface.html"
    view.main([str(source_h5), "--html", str(out), "--no-open"])
    text = out.read_text(encoding="utf-8")
    data = json.loads(re.search(r'id="surface-data"[^>]*>(.*?)</script>', text, re.S).group(1))
    assert data["name"] == "test" and data["default_colormap"] == DEFAULT_COLORMAP
    assert [r["key"] for r in data["reactions"]] == ["D(d,n)3He", "T(d,n)4He", "all"]
    assert data["reactions"][1]["label_html"] == "T(d,n)<sup>4</sup>He"
    assert set(data["colormaps"]) == set(COLORMAPS)
    for r in data["reactions"]:
        assert len(r["heights"]) == len(r["angles"]) * len(r["energies"])
        assert max(r["heights"]) == 1.0 and r["unit"] == "MeV"


def test_web_page_by_default(source_h5, monkeypatch):
    """With no options the web page is written next to the source.h5 and opened."""
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    out = source_h5.with_name("surface.html")
    out.unlink(missing_ok=True)
    view.main([str(source_h5), "DT"])
    assert out.stat().st_size > 1000 and opened == [out.resolve().as_uri()]


def test_all_reactions_is_their_sum(source_h5):
    """The "all" surface holds every reaction's neutrons, on one energy mesh."""
    import h5py
    import numpy as np

    ((_, angles, energies, density, _),) = view.read_surfaces(source_h5, ["all"])
    with h5py.File(source_h5) as f:
        total = sum(f[f"neutrons_{r}"][()].sum() for r in ("DD", "DT"))
        cos = np.cos(np.radians(f["angle_grid"][()]))
    solid = 2 * np.pi * (cos[:-1] - cos[1:])
    width = energies[1] - energies[0]
    assert np.sum(density * solid[:, None]) * width == pytest.approx(total, rel=1e-9)
    with pytest.raises(SystemExit, match="it has DD, DT and all"):
        view.read_surfaces(source_h5, ["XX"])
