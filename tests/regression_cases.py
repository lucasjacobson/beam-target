"""Regression cases: thick-target sources from evaluated data whose results are pinned, so that
a code change that moves them is noticed (see test_regression.py).

Each case uses the evaluation best suited to it, bundled in tests/data/endf (the FENDL-3.2c
files xz-compressed, unpacked by unpacked_library). Run this file to write the reference
results again, only when a change is meant to change them:

    python tests/regression_cases.py
"""

import lzma
import shutil
from pathlib import Path

import numpy as np

from beamtarget import BeamTarget, Material, Region

DATA = Path(__file__).resolve().parent / "data"
LIBRARIES = DATA / "endf"
REFERENCE_DIR = DATA / "regression"

BE = Material("Be", "solid", atoms={"Be": 1}, density=1.848)
TA = Material("Ta", "solid", atoms={"181Ta": 1}, density=16.654)
U238 = Material("U-238", "solid", atoms={"238U": 1}, density=18.95)
LI7 = Material("7Li", "solid", atoms={"7Li": 1}, density=0.534)

# name: (beam, material, beam energy [MeV], library directory in tests/data/endf)
CASES = {
    "p_Be_18_endfb81": ("proton", BE, 18.0, "ENDF-B-VIII.1"),
    "p_Ta_30_fendl32c": ("proton", TA, 30.0, "FENDL-3.2c"),
    "p_U238_30_fendl32c": ("proton", U238, 30.0, "FENDL-3.2c"),
    "d_Li7_20_fendl32c": ("deuteron", LI7, 20.0, "FENDL-3.2c"),
    "a_Be_30_tendl2025": ("alpha", BE, 30.0, "TENDL-2025"),  # ENDF/B-VIII.1 ends at 15 MeV
}


def unpacked_library(name, directory):
    """Return a directory holding the library name from tests/data/endf, with its .xz files
    unpacked under directory (the others are used where they are)."""
    source = LIBRARIES / name
    packed = list(source.rglob("*.xz"))
    if not packed:
        return source
    target = Path(directory) / name
    for path in source.rglob("*"):
        if path.is_file():
            out = target / path.relative_to(source)
            out.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".xz":
                out = out.with_suffix("")
                if not out.exists():
                    out.write_bytes(lzma.decompress(path.read_bytes()))
            else:
                shutil.copyfile(path, out)
    return target


def run_case(name, directory):
    """Run a regression case (unpacking its library under directory) and return its results:
    the neutrons per beam particle and their mean energy [MeV], the target length [cm], and
    per beam particle the neutrons in each (angle, energy) bin and each position bin."""
    beam, material, energy, library = CASES[name]
    s = BeamTarget(
        name=name,
        regions=[Region(material, wall_radius=10.0, position_bins=10)],
        beam_energy_entry=energy * 1000,
        beam_energy_backstop=10.0,  # The target stops the beam: its length is solved
        beam_current=1.0,
        beam_radius_entry=0.1,
        angle_bin_width=10.0,
        energy_bin_width=0.5,
        beam=beam,
        nuclear_data=[str(unpacked_library(library, directory))],
    )
    (rxn,) = s.reactions
    neutrons = s.neutrons[rxn] / s.particle_rate
    return {
        "yield": np.array(s.yield_total / s.particle_rate),
        "mean_energy": np.array(s.mean_neutron_energy()),
        "length": np.array(s.length[0]),
        "angle_energy": neutrons.sum(axis=0),
        "energy_grid": s.neutron_energy_grid[rxn],
        "position": neutrons.sum(axis=(1, 2)),
    }


if __name__ == "__main__":
    import tempfile

    REFERENCE_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        for name in CASES:
            results = run_case(name, directory)
            np.savez(REFERENCE_DIR / f"{name}.npz", **results)
            print(f"{name}: {results['yield']:.6e} n per particle, mean "
                  f"{results['mean_energy']:.4f} MeV")  # fmt: skip
