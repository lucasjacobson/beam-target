"""Open the surface plots of a source.h5 in the web browser, to turn them around with the
mouse.

    python -m beamtarget.view output/p18_Be/source.h5              # every reaction
    python -m beamtarget.view output/DT50/source.h5 DT             # one reaction
    python -m beamtarget.view output/p18_Be/source.h5 --energy 0 3 # only 0-3 MeV (if in MeV)

A web page (surface.html next to source.h5, or the file given with --html) is written and
opened: a WebGL view with its own controls (see webview.py).
"""

import argparse

import h5py
import numpy as np

from .source import ALL_REACTIONS, _plot_label, sum_on_common_grid, surface_density


def read_surfaces(path, reactions=None, energy=None):
    """Return [(title, angles, energies, density, unit)] for the reactions in a source.h5 (all,
    or those named), for the web page (see webview.py), with the neutron energies in the file's unit;
    energy = (low, high), in that unit, keeps only that range. "all" names the sum over the
    reactions, which comes last by default when there are several."""
    with h5py.File(path, "r") as f:
        name = f.attrs.get("name", path)
        found = [k.removeprefix("neutrons_") for k in f if k.startswith("neutrons_")]
        present = [rxn for rxn in found if np.any(f[f"neutrons_{rxn}"][()] > 0)]
        key, all_label = ALL_REACTIONS
        unknown = sorted(set(reactions or []) - set(found) - {key})
        if unknown:
            raise SystemExit(
                f"{path}: no reaction(s) {', '.join(unknown)}; it has {', '.join(found)}"
                + (f" and {key}" if len(present) > 1 else "")
            )
        if not reactions:
            reactions = found + ([key] if len(present) > 1 else [])
        surfaces = []
        for rxn in reactions:
            if rxn == key:
                grids = [f[f"neutron_energy_grid_{r}"] for r in present]
                unit = str(grids[0].attrs.get("units", "MeV"))
                by_angle = [np.sum(f[f"neutrons_{r}"][()], 0) for r in present]
                grid, total = sum_on_common_grid([g[()] for g in grids], by_angle)
                angles, energies, density = surface_density(f["angle_grid"][()], grid, total[None])
                label = all_label
            else:
                data = f[f"neutrons_{rxn}"]
                grid = f[f"neutron_energy_grid_{rxn}"]
                unit = str(grid.attrs.get("units", "MeV"))
                angles, energies, density = surface_density(f["angle_grid"][()], grid[()], data[()])
                label = _plot_label(str(data.attrs.get("reaction", rxn)))
            if energy is not None:
                keep = (energies >= energy[0]) & (energies <= energy[1])
                if np.count_nonzero(keep) < 2:
                    raise SystemExit(
                        f"{path}: {rxn} has no neutrons between {energy[0]} and {energy[1]} {unit}"
                    )
                energies, density = energies[keep], density[:, keep]
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
        description="Open the energy-angle surface plots of a source.h5 in the web browser, to "
        "turn them around.",
    )
    parser.add_argument("path", help="a source.h5 written by beamtarget")
    parser.add_argument(
        "reactions",
        nargs="*",
        help='reaction keys, such as DT, or "all" for their sum (default: each, and their sum)',
    )
    parser.add_argument(
        "--energy",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="neutron energies, in the file's unit (that of the case's energy_bin_width)",
    )
    parser.add_argument(
        "--html",
        metavar="FILE",
        help="where to write the web page (default: surface.html next to the source.h5)",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="write the web page without opening it"
    )
    args = parser.parse_args(argv)

    import webbrowser

    from .webview import write_surface_page

    out = write_surface_page(args.path, args.html, args.reactions, args.energy)
    print(f"Wrote {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
