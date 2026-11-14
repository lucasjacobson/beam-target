"""Regression cases: neutron sources whose results are pinned, so that a code change that moves
them is noticed (see test_regression.py).

The cases cover every beam particle with trustworthy data, gas and solid targets, thin and
thick (stopping) targets, the built-in DROSG-2000 reactions (inverse ones too) and evaluated
data of every kind the reader handles: Kalbach-Mann continua, two-body levels, lab-frame
tables with linear or histogram spectra, fission, and deuteron breakup. Each uses the
library best suited to it, from tests/data/endf (the larger files xz-compressed, unpacked by
unpacked_library). Run this file to write the reference results again, only when a change is
meant to change them:

    python tests/regression_cases.py
"""

import lzma
import shutil
from pathlib import Path

import numpy as np

from beamtarget import BeamTarget, Material, Region
from beamtarget.source import dt_gas

DATA = Path(__file__).resolve().parent / "data"
LIBRARIES = DATA / "endf"
REFERENCE_DIR = DATA / "regression"

BE = Material("Be", "solid", atoms={"Be": 1}, density=1.848)
TA = Material("Ta", "solid", atoms={"181Ta": 1}, density=16.654)
U238 = Material("U-238", "solid", atoms={"238U": 1}, density=18.95)
W184 = Material("W-184", "solid", atoms={"184W": 1}, density=19.3)
C12 = Material("C-12", "solid", atoms={"12C": 1}, density=1.8)
LI7 = Material("7Li", "solid", atoms={"7Li": 1}, density=0.534)
LI7D = Material("7LiD", "solid", atoms={"7Li": 1, "D": 1}, density=0.82)
TID = Material("TiD1.5", "solid", atoms={"Ti": 1, "D": 1.5}, density=3.9)
D2 = Material("D2", "gas", molecules={"D2": 1})
T2 = Material("T2", "gas", molecules={"T2": 1})
H2 = Material("1H2", "gas", molecules={"1H2": 1})

ENDFB = "ENDF-B-VIII.1"
FENDL = "FENDL-3.2c"


def stopping(material):
    """A solid thick enough to stop the beam: its length is solved for 10 keV at the
    backstop."""
    return dict(regions=[Region(material, wall_radius=10.0, position_bins=10)], backstop=10.0)


def gas(material, length, pressure, wall_radius=10.0, divergence=0.0, position_bins=10):
    """A room-temperature gas region the beam passes through."""
    region = Region(
        material, length, pressure=pressure, temperature=25.0, wall_radius=wall_radius,
        divergence=divergence, position_bins=position_bins,
    )  # fmt: skip
    return dict(regions=[region], backstop=None)


# name: (beam, beam energy [MeV], target (see stopping and gas), nuclear_data (library
# directory names in tests/data/endf, or "drosg"), energy bin width [MeV])
CASES = {
    # Built-in reactions (DROSG-2000)
    "d_DT50gas_0.3_drosg": (
        "deuteron", 0.3,
        gas(dt_gas(0.5), None, 38.0, wall_radius=1.0, divergence=30.0, position_bins=20)
        | {"backstop": 10.0},  # Stops in the gas; the beam spreads and is scraped
        ["drosg"], 0.01,
    ),
    "d_TiD_0.3_drosg": ("deuteron", 0.3, stopping(TID), ["drosg"], 0.01),
    "p_Li7film_2.5_drosg": (
        "proton", 2.5, dict(regions=[Region(LI7, 1e-3, wall_radius=1.0, position_bins=5)],
                            backstop=None),
        ["drosg"], 0.005,
    ),
    "p_T2gas_2_drosg": ("proton", 2.0, gas(T2, 50.0, 5.0), ["drosg"], 0.005),
    "t_D2gas_0.6_drosg": ("triton", 0.6, gas(D2, 10.0, 10.0), ["drosg"], 0.01),  # Inverse
    "t_H2gas_3.3_drosg": ("triton", 3.3, gas(H2, 1.0, 38.0), ["drosg"], 0.005),  # Forward cone
    # Evaluated data
    "p_Be_18_endfb81": ("proton", 18.0, stopping(BE), [ENDFB], 0.5),
    "p_Be_70_endfb81": ("proton", 70.0, stopping(BE), [ENDFB], 0.5),
    "p_Ta_30_fendl32c": ("proton", 30.0, stopping(TA), [FENDL], 0.5),
    "p_U238_30_fendl32c": ("proton", 30.0, stopping(U238), [FENDL], 0.5),  # Fission
    "p_W184_30_endfb81": ("proton", 30.0, stopping(W184), [ENDFB], 0.5),
    "p_C12_30_fendl32c": ("proton", 30.0, stopping(C12), [FENDL], 0.5),  # Histogram tables
    "p_Li7_10_fendl32c": ("proton", 10.0, stopping(LI7), [FENDL], 0.5),
    "d_Li7_20_fendl32c": ("deuteron", 20.0, stopping(LI7), [FENDL], 0.5),
    "d_Be_18_fendl32c": ("deuteron", 18.0, stopping(BE), [FENDL], 0.5),
    "a_Be_10_endfb81": ("alpha", 10.0, stopping(BE), [ENDFB], 0.5),
    # Built-in D(d,n) and evaluated d + 7Li in one target
    "d_Li7D_10_drosg_fendl32c": ("deuteron", 10.0, stopping(LI7D), ["drosg", FENDL], 0.25),
}  # fmt: skip


def unpacked_library(name, directory):
    """Return a directory holding the library name from tests/data/endf, with its .xz files
    unpacked under directory (a library without them is used where it is)."""
    source = LIBRARIES / name
    if not list(source.rglob("*.xz")):
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
            elif not out.exists():
                shutil.copyfile(path, out)
    return target


def run_case(name, directory):
    """Run a regression case (unpacking its libraries under directory) and return its results:
    the target length [cm], the beam energy at its end [keV], the total neutrons per beam
    particle, and for each reaction the neutrons per beam particle, their mean energy [MeV],
    and per beam particle the neutrons in each (angle, energy) bin and each position bin."""
    beam, energy, target, nuclear_data, energy_bin_width = CASES[name]
    libraries = [d if d == "drosg" else str(unpacked_library(d, directory)) for d in nuclear_data]
    s = BeamTarget(
        name=name,
        regions=target["regions"],
        beam_energy_entry=energy * 1000,
        beam_energy_backstop=target["backstop"],
        beam_current=1.0,
        beam_radius_entry=0.1,
        angle_bin_width=10.0,
        energy_bin_width=energy_bin_width,
        beam=beam,
        nuclear_data=libraries,
    )
    results = {
        "length": np.array(s.length),
        "beam_energy_end": np.array(s.beam_energy_grid[-1]),
        "yield": np.array(s.yield_total / s.particle_rate),
    }
    for rxn in s.reactions_present():
        neutrons = s.neutrons[rxn] / s.particle_rate
        results |= {
            f"{rxn}/yield": np.array(neutrons.sum()),
            f"{rxn}/mean_energy": np.array(s.mean_neutron_energy(rxn)),
            f"{rxn}/angle_energy": neutrons.sum(axis=0),
            f"{rxn}/energy_grid": s.neutron_energy_grid[rxn],
            f"{rxn}/position": neutrons.sum(axis=(1, 2)),
        }
    return results


if __name__ == "__main__":
    import tempfile
    import time

    REFERENCE_DIR.mkdir(exist_ok=True)
    for old in REFERENCE_DIR.glob("*.npz"):
        if old.stem not in CASES:
            old.unlink()  # A case no longer in CASES
    with tempfile.TemporaryDirectory() as directory:
        for name in CASES:
            t = time.time()
            results = run_case(name, directory)
            np.savez(REFERENCE_DIR / f"{name}.npz", **results)
            reactions = sorted({k.split("/")[0] for k in results if "/" in k})
            print(f"{name:28s} {results['yield']:.6e} n per particle "
                  f"({', '.join(reactions)}), {time.time() - t:.1f} s")  # fmt: skip
