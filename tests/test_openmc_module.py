"""The copy of openmc_source.py written with each case: self-contained, and the same as the
package's reader."""

import ast
import json
import subprocess
import sys

import numpy as np
import pytest

from beamtarget import openmc_source
from beamtarget.cli import OPENMC_MODULE, write_openmc_module
from beamtarget.source import CODE_URL, SDEF_DIRECTIONS, get_code_version
from beamtarget.units import UNITS, to_internal


def test_imports_nothing_from_beamtarget():
    """Only numpy, h5py, and (inside get_openmc_sources) openmc."""
    with open(OPENMC_MODULE, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            imported.add(node.module.split(".")[0])
    assert imported == {"h5py", "numpy", "openmc"}


def test_directions_match_sdef_directions():
    assert list(openmc_source.DIRECTIONS) == list(SDEF_DIRECTIONS)
    for direction, (_, vector, _) in SDEF_DIRECTIONS.items():
        assert openmc_source.DIRECTIONS[direction] == tuple(map(float, vector.split()))


def test_unit_factors_match_units():
    """Every length and energy unit a source.h5 can be written in, with the same factor."""
    expected = {
        name for name, (quantity, _, _) in UNITS.items() if quantity in ("length", "energy")
    }
    assert set(openmc_source.TO_CM_OR_EV) == expected
    for unit, factor in openmc_source.TO_CM_OR_EV.items():
        assert factor == to_internal(1.0, unit)


def test_written_copy(tmp_path):
    write_openmc_module(tmp_path)
    text = (tmp_path / "openmc_source.py").read_text(encoding="utf-8")
    header, _, body = text.partition("\n\n")
    assert f"beam-target {get_code_version()} ({CODE_URL})" in header
    with open(OPENMC_MODULE, encoding="utf-8") as f:
        assert body == f.read()
    assert "Copyright (c) 2026 SHINE Technologies and beam-target contributors" in body


def test_copy_works_without_beamtarget(tmp_path, mixed_source):
    """Loaded from the case directory with beamtarget blocked, the copy gives the same sources
    as the package's module."""
    pytest.importorskip("openmc")
    path = tmp_path / "source.h5"
    mixed_source.write_hdf5(path, direction="+x", entry_position=[1.0, 2.0, 3.0])
    write_openmc_module(tmp_path)
    script = (
        "import json, sys\n"
        "sys.modules['beamtarget'] = None  # Any import of beamtarget fails\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from openmc_source import get_openmc_sources\n"
        f"sources = get_openmc_sources({str(path)!r})\n"
        "print(json.dumps([[s.strength, *s.space.origin, *s.space.z_dir, s.space.z.a, s.space.z.b,"
        " s.angle.mu.a, s.angle.mu.b, *s.energy.x, *s.energy.p] for s in sources]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    copied = json.loads(result.stdout)
    reference = [
        [s.strength, *s.space.origin, *s.space.z_dir, s.space.z.a, s.space.z.b,
         s.angle.mu.a, s.angle.mu.b, *s.energy.x, *s.energy.p]
        for s in openmc_source.get_openmc_sources(path)
    ]  # fmt: skip
    assert len(copied) == len(reference) > 0
    for a, b in zip(copied, reference):
        np.testing.assert_allclose(a, b, rtol=1e-15)
