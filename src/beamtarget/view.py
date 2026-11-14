"""Open the surface plots of a source.h5 in windows, to turn them around with the mouse.

    python -m beamtarget.view output/p18_Be/source.h5              # every reaction
    python -m beamtarget.view output/Apollo/source.h5 DT           # one reaction
    python -m beamtarget.view output/p18_Be/source.h5 --energy 0 3 # only 0-3 MeV (if in MeV)

Each window shows the same plot as plots/neutron_energy_angle_surface_<rxn>.png (see
source.draw_energy_surface), starting from the same view: drag to turn it, right-drag or
scroll to zoom.
"""

import argparse

import h5py
import numpy as np

from .source import _plot_label, draw_energy_surface, surface_density


def read_surfaces(path, reactions=None, energy=None):
    """Return [(title, angles, energies, density, unit)] for the reactions in a source.h5 (all,
    or those named), for draw_energy_surface, with the neutron energies in the file's unit;
    energy = (low, high), in that unit, keeps only that range."""
    with h5py.File(path, "r") as f:
        name = f.attrs.get("name", path)
        found = [k.removeprefix("neutrons_") for k in f if k.startswith("neutrons_")]
        unknown = sorted(set(reactions or []) - set(found))
        if unknown:
            raise SystemExit(
                f"{path}: no reaction(s) {', '.join(unknown)}; it has {', '.join(found)}"
            )
        surfaces = []
        for rxn in reactions or found:
            data = f[f"neutrons_{rxn}"]
            grid = f[f"neutron_energy_grid_{rxn}"]
            unit = str(grid.attrs.get("units", "MeV"))
            angles, energies, density = surface_density(f["angle_grid"][()], grid[()], data[()])
            if energy is not None:
                keep = (energies >= energy[0]) & (energies <= energy[1])
                if np.count_nonzero(keep) < 2:
                    raise SystemExit(
                        f"{path}: {rxn} has no neutrons between {energy[0]} and {energy[1]} {unit}"
                    )
                energies, density = energies[keep], density[:, keep]
            label = _plot_label(str(data.attrs.get("reaction", rxn)))
            surfaces.append(
                (
                    f"{label} neutron energy against emission angle ({name})",
                    angles,
                    energies,
                    density,
                    unit,
                )
            )
    return surfaces


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m beamtarget.view",
        description="Open the energy-angle surface plots of a source.h5 to turn them around.",
    )
    parser.add_argument("path", help="a source.h5 written by beamtarget")
    parser.add_argument("reactions", nargs="*", help="reaction keys, such as DT (default: all)")
    parser.add_argument(
        "--energy",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="neutron energies, in the file's unit (that of the case's energy_bin_width)",
    )
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt  # The default (interactive) backend

    from .source import PLOT_STYLE

    surfaces = read_surfaces(args.path, args.reactions, args.energy)
    with plt.rc_context(PLOT_STYLE):
        for title, angles, energies, density, unit in surfaces:
            draw_energy_surface(plt.figure(figsize=(9, 7)), angles, energies, density, title, unit)
        plt.show()


if __name__ == "__main__":
    main()
