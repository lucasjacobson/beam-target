"""OpenMC sources from a source.h5 file written by ShineSource.write_hdf5.

    from shine_neutron_source.openmc_source import get_openmc_sources

    settings.source = get_openmc_sources("output/DT90/source.h5")

Needs the openmc Python package, version 0.16 or later (for cylinders along any axis), which
is imported only here: the rest of shine_neutron_source runs without it.

Unlike the MCNP SDEF card, the OpenMC source keeps the full correlation between position,
emission angle and neutron energy: there is one source for each (position bin, angle bin,
reaction) of the output mesh, each with its own energy spectrum.
"""

import h5py
import numpy as np

from .source import SDEF_DIRECTIONS

# Oldest source.h5 layout this module reads (it needs the placement attributes)
MIN_FORMAT_VERSION = 6


def _perpendicular(u):
    """Return a unit vector perpendicular to the unit vector u."""
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = a - np.dot(a, u) * u
    return v / np.linalg.norm(v)


def read_placement(path):
    """Return (direction, entry, backstop) stored in a source.h5 file: the beam direction (one
    of SDEF_DIRECTIONS) and the points [cm] where the beam enters the first region and where
    it reaches the backstop."""
    with h5py.File(path, "r") as hf:
        _check_format(path, hf)
        return (
            str(hf.attrs["direction"]),
            np.array(hf.attrs["entry_position"], dtype=float),
            np.array(hf.attrs["backstop_position"], dtype=float),
        )


def _check_format(path, hf):
    if hf.attrs.get("format") != "shine-neutron-source":
        raise ValueError(f"{path} is not a shine-neutron-source HDF5 file")
    version = int(hf.attrs.get("format_version", 0))
    if version < MIN_FORMAT_VERSION:
        raise ValueError(
            f"{path} has format_version {version}; get_openmc_sources needs "
            f"{MIN_FORMAT_VERSION} or later (rerun the case to write it again)"
        )


def get_openmc_sources(path):
    """Return a list of openmc.IndependentSource for the neutron source in a source.h5 file.

    There is one source for each (position bin, emission-angle bin, reaction) of the output
    mesh that has neutrons:

    - space: a cylinder along the beam axis, from the start to the end of the position bin,
      uniform in area out to the beam edge radius over the bin (radius_mids, as in the SDEF
      card).
    - angle: uniform in the cosine of the emission angle from the beam direction over the
      angle bin, and uniform in azimuth.
    - energy: the neutron spectrum of that position, angle and reaction, as a histogram on
      the output energy bins.
    - strength: the neutrons it emits [n/s]. OpenMC treats strengths as relative weights and
      normalises tallies per source particle, so multiply tallies by the total of the
      strengths (yield_total in the file's results) for absolute rates.

    The source is placed as it was for the case: the direction, entry_position and
    backstop_position the case was run with, stored in the file. Everything else (lengths,
    radii, the yield) comes from the calculation; to change it, rerun the case.
    """
    import openmc

    with h5py.File(path, "r") as hf:
        _check_format(path, hf)
        x_grid = hf["x_grid"][()]  # cm, from the entry point along the beam
        radius = hf["radius_mids"][()]  # cm
        cosines = np.cos(np.radians(hf["angle_grid"][()]))  # Decreasing
        reactions = sorted(
            key.removeprefix("neutrons_") for key in hf if key.startswith("neutrons_")
        )
        energy_grids = {rxn: hf[f"neutron_energy_grid_{rxn}"][()] for rxn in reactions}  # MeV
        neutrons = {rxn: hf[f"neutrons_{rxn}"][()] for rxn in reactions}  # n/s
        case_direction = str(hf.attrs["direction"])
        case_entry = np.array(hf.attrs["entry_position"], dtype=float)

    u = np.array(SDEF_DIRECTIONS[case_direction][1].split(), dtype=float)
    entry = case_entry
    v = _perpendicular(u)

    def histogram(x):
        """A uniform distribution between x[0] and x[1] (increasing)."""
        return openmc.stats.Tabular(x, [1.0, 0.0], interpolation="histogram")

    phi = openmc.stats.Uniform(0.0, 2 * np.pi)
    sources = []
    for i in range(len(x_grid) - 1):
        space = openmc.stats.CylindricalIndependent(
            r=openmc.stats.PowerLaw(a=0.0, b=radius[i], n=1),  # Uniform in area
            phi=phi,
            z=histogram(x_grid[i : i + 2]),
            origin=tuple(entry),
            r_dir=tuple(v),
            z_dir=tuple(u),
        )
        for j in range(len(cosines) - 1):
            angle = openmc.stats.PolarAzimuthal(
                mu=histogram([cosines[j + 1], cosines[j]]),
                phi=phi,
                reference_uvw=tuple(u),
                reference_vwu=tuple(v),
            )
            for rxn in reactions:
                spectrum = neutrons[rxn][i, j, :]
                filled = np.flatnonzero(spectrum)
                if len(filled) == 0:
                    continue
                lo, hi = filled[0], filled[-1] + 1
                # Histogram: one probability per bin, plus a final 0 that is not used (the
                # XML holds energies and probabilities as one list of equal halves)
                energy = openmc.stats.Tabular(
                    energy_grids[rxn][lo : hi + 1] * 1e6,  # MeV -> eV
                    np.append(spectrum[lo:hi], 0.0),
                    interpolation="histogram",
                )
                sources.append(
                    openmc.IndependentSource(
                        space=space,
                        angle=angle,
                        energy=energy,
                        strength=float(np.sum(spectrum)),
                        particle="neutron",
                    )
                )
    return sources
