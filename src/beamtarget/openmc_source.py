"""OpenMC sources from a source.h5 file written by beam-target (BeamTarget.write_hdf5).

    from beamtarget.openmc_source import get_openmc_sources

    settings.source = get_openmc_sources("output/DT90/source.h5")

Needs only numpy, h5py, and the openmc Python package, version 0.16 or later (for cylinders
along any axis); the rest of beamtarget runs without openmc. This module is self-contained:
beam-target writes a copy of it next to each case's source.h5, which can be used without
beam-target installed, by importing it from that folder (or loading the file by its path):

    import sys
    sys.path.insert(0, "output/DT90")
    from openmc_source import get_openmc_sources

Part of beam-target, https://github.com/lucasjacobson/beam-target, under the MIT License:

    Copyright (c) 2026 SHINE Technologies and beam-target contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

Unlike the MCNP SDEF card, the OpenMC source keeps the full correlation between position,
emission angle, and neutron energy: there is one source for each (position bin, angle bin)
of the output mesh, each with its own energy spectrum (all the reactions together).
"""

import h5py
import numpy as np

# This module imports nothing from beamtarget, so that a copy of it works on its own. The two
# tables below repeat what it needs from source.py and units.py; tests check they agree.

# Oldest source.h5 layout this module reads (it needs the placement attributes)
MIN_FORMAT_VERSION = 6

# The beam directions a case can be run with (source.SDEF_DIRECTIONS), as unit vectors
DIRECTIONS = {
    "-x": (-1.0, 0.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-z": (0.0, 0.0, -1.0),
    "+z": (0.0, 0.0, 1.0),
}

# Factors to cm and eV of the units the lengths, radii, and neutron energies in a source.h5
# can be in (as in units.UNITS)
TO_CM_OR_EV = {
    "m": 100.0, "cm": 1.0, "mm": 0.1, "um": 1e-4, "nm": 1e-7, "in": 2.54, "inch": 2.54,
    "ft": 30.48, "eV": 1.0, "keV": 1e3, "MeV": 1e6, "GeV": 1e9,
}  # fmt: skip


def _unit_vector(direction):
    """Return the beam direction as a unit vector: from one of DIRECTIONS ("-z", ...) or any
    three numbers (x, y, z)."""
    if isinstance(direction, str):
        if direction not in DIRECTIONS:
            raise ValueError(
                f"direction must be one of {', '.join(DIRECTIONS)} or a vector (x, y, z), "
                f"got {direction!r}"
            )
        return np.array(DIRECTIONS[direction])
    u = np.array(direction, dtype=float)
    if u.shape != (3,) or not np.all(np.isfinite(u)) or not np.any(u):
        raise ValueError(f"direction must be a non-zero vector (x, y, z), got {direction!r}")
    return u / np.linalg.norm(u)


def _perpendicular(u):
    """Return a unit vector perpendicular to the unit vector u."""
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = a - np.dot(a, u) * u
    return v / np.linalg.norm(v)


def _point(name, value):
    point = np.array(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be three numbers (x, y, z), got {value!r}")
    return point


def read_placement(path):
    """Return (direction, entry, backstop) stored in a source.h5 file: the beam direction (one
    of DIRECTIONS) and the points [cm] where the beam enters the first region and where
    it reaches the backstop."""
    with h5py.File(path, "r") as hf:
        _check_format(path, hf)
        return (
            str(hf.attrs["direction"]),
            np.array(hf.attrs["entry_position"], dtype=float),
            np.array(hf.attrs["backstop_position"], dtype=float),
        )


def _check_format(path, hf):
    if hf.attrs.get("format") != "beam-target":
        raise ValueError(f"{path} is not a beam-target HDF5 file")
    version = int(hf.attrs.get("format_version", 0))
    if version < MIN_FORMAT_VERSION:
        raise ValueError(
            f"{path} has format_version {version}; get_openmc_sources needs "
            f"{MIN_FORMAT_VERSION} or later (rerun the case to write it again)"
        )


def _read(hf, key):
    """Return a dataset of a source.h5 in the units OpenMC uses (cm, eV), from the units it
    was written in (its units attribute)."""
    unit = str(hf[key].attrs["units"])  # Always written with "u" for micro
    if unit not in TO_CM_OR_EV:
        raise ValueError(f"{key}: unknown unit {unit!r} (expected one of {', '.join(TO_CM_OR_EV)})")
    return hf[key][()] * TO_CM_OR_EV[unit]


def _merged_spectrum(grids, spectra):
    """Return (edges, probabilities) of the sum of histograms, each given by its bin edges
    and the neutrons in each bin, or None if they are all empty. The sum is a histogram on
    the union of their edges (the density of each added over each piece), the same
    distribution as choosing a histogram by its total and then sampling it. Empty bins at
    either end are dropped, and each run of empty bins inside becomes one bin, which samples
    the same and keeps the source small."""
    pieces = []
    for edges, spectrum in zip(grids, spectra):
        filled = np.flatnonzero(spectrum)
        if len(filled):
            lo, hi = filled[0], filled[-1] + 1
            pieces.append((edges[lo : hi + 1], spectrum[lo:hi]))
    if not pieces:
        return None
    if len(pieces) == 1:  # Nothing to add: the histogram as it is
        edges, p = pieces[0]
    else:
        edges = np.unique(np.concatenate([e for e, _ in pieces]))
        # The same edge from two grids can differ in the last bits: keep one
        edges = edges[np.r_[True, np.diff(edges) > 1e-12 * np.max(np.abs(edges))]]
        widths = np.diff(edges)
        mids = edges[:-1] + widths / 2
        p = np.zeros(len(widths))
        for e, spectrum in pieces:
            k = np.searchsorted(e, mids, side="right") - 1
            inside = (k >= 0) & (k < len(spectrum))
            p[inside] += spectrum[k[inside]] / np.diff(e)[k[inside]] * widths[inside]
    # One bin for each run of empty bins
    keep = np.r_[True, ~((p[1:] == 0) & (p[:-1] == 0))]
    return np.r_[edges[:-1][keep], edges[-1]], p[keep]


def get_openmc_sources(path, direction=None, entry_position=None, backstop_position=None):
    """Return a list of openmc.IndependentSource for the neutron source in a source.h5 file.

    There is one source for each (position bin, emission-angle bin) of the output mesh that
    has neutrons:

    - space: a cylinder along the beam axis, from the start to the end of the position bin,
      uniform in area out to the beam edge radius over the bin (radius_mids, as in the SDEF
      card).
    - angle: uniform in the cosine of the emission angle from the beam direction over the
      angle bin, and uniform in azimuth.
    - energy: the neutron spectrum of that position and angle, all reactions together, as a
      histogram on the output energy bins (each run of empty bins merged into one).
    - strength: the neutrons it emits [n/s]. OpenMC samples the sources in proportion to
      their strengths and scales tallies by the total of the strengths (yield_total in the
      file's results), so tallies come out per second: absolute rates, with no further
      normalization.

    Adding the reactions' spectra gives the same distribution as a source for each reaction,
    with half the sources or fewer, so a smaller settings.xml.

    The source is placed as it was for the case (the direction, entry_position, and
    backstop_position the case was run with, stored in the file), unless moved here:

    Args:
        path: The source.h5 file.
        direction: Beam direction: one of DIRECTIONS ("-x", ..., "+z") or any vector
            (x, y, z). None keeps the case's.
        entry_position: Point where the beam enters the first region, (x, y, z) [cm].
        backstop_position: Point where the beam reaches the backstop, (x, y, z) [cm]. Give
            this or entry_position, not both. With neither, the source stays where it was for
            the case. A direction other than the case's needs one of them, since turning the
            source moves at least one of its ends.

    Everything else (lengths, radii, the yield) comes from the calculation; to change it,
    rerun the case.
    """
    import openmc

    if entry_position is not None and backstop_position is not None:
        raise ValueError("give entry_position or backstop_position, not both")

    with h5py.File(path, "r") as hf:
        _check_format(path, hf)
        x_grid = _read(hf, "x_grid")  # cm, from the entry point along the beam
        radius = _read(hf, "radius_mids")  # cm
        cosines = np.cos(np.radians(hf["angle_grid"][()]))  # Decreasing
        reactions = sorted(
            key.removeprefix("neutrons_") for key in hf if key.startswith("neutrons_")
        )
        energy_grids = {rxn: _read(hf, f"neutron_energy_grid_{rxn}") for rxn in reactions}  # eV
        neutrons = {rxn: hf[f"neutrons_{rxn}"][()] for rxn in reactions}  # n/s
        case_direction = str(hf.attrs["direction"])
        case_entry = np.array(hf.attrs["entry_position"], dtype=float)

    case_u = _unit_vector(case_direction)
    u = case_u if direction is None else _unit_vector(direction)
    length = x_grid[-1]
    if entry_position is not None:
        entry = _point("entry_position", entry_position)
    elif backstop_position is not None:
        entry = _point("backstop_position", backstop_position) - length * u
    elif np.allclose(u, case_u, rtol=0, atol=1e-12):
        entry = case_entry
    else:
        raise ValueError(
            f"direction {direction!r} differs from the case's ({case_direction}): give "
            "entry_position or backstop_position too, to say where the source goes"
        )
    v = _perpendicular(u)

    phi = openmc.stats.Uniform(0.0, 2 * np.pi)
    grids = [energy_grids[rxn] for rxn in reactions]
    sources = []
    for i in range(len(x_grid) - 1):
        space = openmc.stats.CylindricalIndependent(
            r=openmc.stats.PowerLaw(a=0.0, b=radius[i], n=1),  # Uniform in area
            phi=phi,
            z=openmc.stats.Uniform(x_grid[i], x_grid[i + 1]),
            origin=tuple(entry),
            r_dir=tuple(v),
            z_dir=tuple(u),
        )
        for j in range(len(cosines) - 1):
            merged = _merged_spectrum(grids, [neutrons[rxn][i, j, :] for rxn in reactions])
            if merged is None:
                continue
            edges, p = merged
            angle = openmc.stats.PolarAzimuthal(
                mu=openmc.stats.Uniform(cosines[j + 1], cosines[j]),
                phi=phi,
                reference_uvw=tuple(u),
                reference_vwu=tuple(v),
            )
            # Histogram: one probability per bin, plus a final 0 that is not used (the XML
            # holds energies and probabilities as one list of equal halves)
            energy = openmc.stats.Tabular(edges, np.append(p, 0.0), interpolation="histogram")
            sources.append(
                openmc.IndependentSource(
                    space=space,
                    angle=angle,
                    energy=energy,
                    strength=float(np.sum(p)),
                    particle="neutron",
                )
            )
    return sources
