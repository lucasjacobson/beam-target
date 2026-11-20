# beam-target

Neutron source terms for beam-target neutron generators, for MCNP and OpenMC.

An accelerator-driven neutron generator makes neutrons as a light-ion beam slows down in a gas
or solid target, so each neutron's starting position, direction, and energy are all
correlated. This package calculates that source term deterministically, in seconds to minutes,
with no statistical noise, and writes it as an MCNP SDEF card, OpenMC sources, and an HDF5
file. It handles any light-ion beam on any target that nuclear data cover. Its development
is funded by SHINE Technologies.

## What it calculates

- **Beams:** protons, deuterons, tritons, ³He, and alphas.
- **Targets:** one or more regions in a row, each a gas (any mixture of molecules, by pressure
  and temperature or by density) or a solid (any atoms, by density): for example, a D/T gas
  cell with pumping stages, a titanium deuteride film, or a lithium or beryllium block.
- **The beam:** slowing down with SRIM (or ICRU 49) stopping powers, spreading with its
  divergence, scraping on each region's wall, and losses to nuclear reactions, with the power
  it deposits along the way.
- **Neutrons:** from the built-in DROSG-2000 reactions D(d,n)³He, T(d,n)⁴He, T(p,n)³He, and
  ⁷Li(p,n)⁷Be, or from any ENDF-6 evaluation (ENDF/B, FENDL, TENDL, etc.), including
  multi-body breakup, converted to the lab frame with relativistic kinematics, on a (position,
  angle, energy) mesh. Two-body lines (the DROSG reactions and the discrete levels of an
  evaluation) are split exactly at the bin edges, so they need no internal angle mesh.
- **Solving for a setting:** give the beam energy wanted at the backstop, and the code finds
  a region's length, pressure, or density; give the neutron yield wanted, and it finds the
  beam current.

## Installation

Python 3.11 or later. From a clone of this repository:

    pip install -e .

This installs the package, its dependencies (numpy, h5py, matplotlib, and numba), and the
`beam-target` command. The first case that uses evaluated data compiles a few routines with
numba, which takes a few seconds once; the compiled code is cached for later runs. The nuclear data it needs are included, except evaluated
libraries, which are optional (see [Nuclear data](#nuclear-data)). Writing OpenMC sources
also needs the `openmc` Python package (0.16 or later). For development,
`pip install -e ".[dev]"` adds pytest (with pytest-xdist) and ruff.

## Quick start

    beam-target examples/simple.toml

runs the case in `examples/simple.toml` and writes its results to `output/DT90/`. The same
works as `python -m beamtarget`. Options: `-c NAME` runs one case, `-o DIR` sets the
output directory, `-j N` sets the number of CPU cores to use (default: all; cases run in
parallel and share them, and a long case takes over the cores of those that finish), `--list`
lists the cases, `--no-plots` skips the plots (about 2 s per case), and `-v` shows more
detail.

Each case's energy-angle surfaces are written as a web page, `surface.html` next to its
`source.h5`: open it in your browser to turn them around with the mouse. It is a WebGL view lit
from a direction you choose, with a choice of colormaps, optional shadows, and the value under
the pointer (it loads three.js from cdn.jsdelivr.net).

    python -m beamtarget.view output/DT90/source.h5

writes the page again and opens it; give reaction keys (or `all`) to show only those, and
`--energy LOW HIGH` to show only those neutron energies, in the case's unit for them.

From Python:

```python
from beamtarget import load_input, run_case

for name, settings in load_input("examples/simple.toml").items():
    source = run_case(name, settings, output_dir="output")
    print(name, source.yield_total, "n/s")
```

A script runs one thread per case unless told otherwise: `beamtarget.set_parallel(8)` lets
each case use up to 8 threads for its evaluated continua (results are the same either way).

`BeamTarget` can also be built directly from `Material` and `Region` objects; see its
docstring. It takes and holds its values in eV (beam and neutron energies), A, Pa, K, rad, W,
and cm (`beamtarget.units` has the factors), and reports its results in the units of its
`units` argument.

## Input files

Cases are described in TOML input files, and the examples are the reference for every
setting. Values with units are written with them, such as `beam_energy_entry = "280 keV"` or
`pressure = "5 torr"`:

| Quantity | Units | Plain number |
|---|---|---|
| Energies (beam energies, `energy_bin_width`) | eV, keV, MeV, GeV | not accepted |
| Beam current | A, mA, uA, nA | not accepted |
| Pressure | Pa, kPa, torr (or Torr), mtorr (or mTorr), mbar, bar, atm, psi | not accepted |
| Temperature | K, degC (or C), degF (or F) | not accepted |
| Divergence | rad, mrad, urad, deg | not accepted |
| Lengths and radii | m, cm, mm, um, nm, in (or inch), ft | cm |
| Density | g/cm3, kg/m3, g/L | g/cm3 |

Unit names are case-sensitive, and `µ` may be written for `u`. Emission angles
(`angle_bin_width`) are plain numbers in degrees, and `yield_total` in n/s. A case's results
(summary, text files, `source.h5`, and plots) are in the units it uses, the first one where it
uses several for one kind of value (beam energies, neutron energies, lengths along the beam,
radii, and so on); the beam power is in kW. The MCNP SDEF card is always in cm and MeV.

- [`examples/simple.toml`](examples/simple.toml): one case, using every setting that can be
  given together, each explained.
- [`examples/generic.toml`](examples/generic.toml): one case per feature (gas mixtures,
  solid targets, other beams, calculated settings, placement in the MCNP geometry, etc.),
  sharing settings through a `[defaults]` table and named `[materials.<name>]` and
  `[regions.<name>]` tables.

## Outputs

Each case writes `source.h5` (the full source), `source.sdef` (an MCNP SDEF card),
`summary.toml`, text profiles, plots, and `surface.html` to its own directory. `get_openmc_sources` turns
`source.h5` into OpenMC sources that keep the full correlation between position, angle, and
energy, which an SDEF card cannot. [`docs/outputs.md`](docs/outputs.md) describes every file.

## Nuclear data

The cross sections of the built-in reactions (DROSG-2000), stopping powers (SRIM and
ICRU 49), atomic masses (AME2020), and natural isotopic compositions (NIST) are included in
[`src/beamtarget/data/`](src/beamtarget/data/). For other reactions, point
the code at evaluated libraries you download (ENDF/B, FENDL, TENDL, etc.); a per-user
configuration file names them once for every input file. The
[data README](src/beamtarget/data/README.md) gives the source of every data file,
how evaluated libraries are set up and used, how the results compare with MCNP and with
measurements, and how to choose between libraries.

## Validation

The test suite (`python -m pytest -m ""`, about 680 tests) checks the physics and the outputs,
including:

- **MCNP and OpenMC sampling:** MCNP and OpenMC sample the generated sources, and the sampled
  positions, angles, and energies are checked against the calculation. These tests are
  skipped without MCNP and mcnptools, or without the `openmc` package.
- **Thick-target yields:** with evaluated data, the neutron yields of beams stopping in a
  target agree with MCNP's own calculation from the same data within 1-5%: protons,
  deuterons, and alphas on beryllium (in `tests/test_endf.py`), protons and deuterons on
  aluminum, tantalum, and uranium, and protons on thorium (see the
  [data README](src/beamtarget/data/README.md)). This tests the calculation, not the data:
  for deuterons on uranium and protons on thorium, the data lack or may lack fission
  neutrons.
- **Reference results:** yields for D/T gas targets are pinned, so that any change in them is
  deliberate.

## Assumptions and limitations

- **Straggling** of the beam's energy and direction is neglected. In gas targets its effect is
  well below the output bins (range straggling of about 3%, and a few degrees of direction
  spread late in the range); in solid targets it smears the correlation between angle and
  energy more, but does not change the yields.
- **Divergence** only grows the beam radius: the beam particles are taken to move along the
  axis (a 36 mrad divergence tilts them by about 2°, the same order as straggling).
- **Beam profile:** the beam is uniform across its radius and expands self-similarly, so the
  scraped fraction is that of a flat profile. This is not necessarily conservative: it scrapes
  more than a centrally peaked beam, but less than a hollow one.
- **Compounds and mixtures** stop the beam as the sum of their atoms (Bragg additivity). SRIM's
  compound corrections (for water vapor, CO₂, air, etc.) are not applied, a few-percent
  effect for such additives; there are none for D/T gas or metal hydrides.
- **ICRU 49 stopping** (`stopping_power = "icru49"`) is available for hydrogen-isotope gases
  only.
- **Open neutron channels** that are not modeled, such as the competing channels of the
  built-in reactions or reactions on other isotopes in the target, are reported as warnings
  (in the log, the summary, and the HDF5 file); the case still runs. When a built-in reaction
  misses channels that an evaluation in your libraries gives (such as FENDL-3.2c's p + ⁷Li
  above 3.7 MeV), a further warning names it and says to list it before `"drosg"`. A beam
  energy beyond the end of a built-in reaction's data is refused.
- **Position mesh:** the beam is followed in `position_substeps` steps (default 10) across
  each position bin, and its energy is taken at the middle of each step. A warning names any
  region where the steps are too long: where the two-body neutron lines of neighboring steps
  are more than half an energy bin apart (the energy spectrum then shows stripes, most in the
  first and last angle bins), or where the yield is off by more than 0.1% (estimated with
  Simpson's rule); it gives the `position_substeps` to use.
- **Evaluated data** come with their own limitations, including fission (often missing from
  the libraries) and neutrons from secondary charged particles; see the
  [data README](src/beamtarget/data/README.md).
- **Target (n,2n) multiplication** and any other transport of the neutrons in the target are
  left to the transport code, which models the target.

## Development

Run the tests with `python -m pytest`: in parallel, and without the slow ones (MCNP and OpenMC
sampling, convergence, and regression), about 20 s on 16 cores. `python -m pytest -m slow` runs
those alone (about a minute), and `python -m pytest -m ""` runs everything. Format and lint
with `ruff format` and `ruff check` (settings in `pyproject.toml`). The naming conventions used in the code are at the top of
[`source.py`](src/beamtarget/source.py).
[`scripts/make_srim_tables.py`](scripts/make_srim_tables.py) regenerates the SRIM stopping
tables, and [`docs/figures/`](docs/figures/) holds the figures in the data README and the
scripts that draw them.

## Citing

If you use this package in published work, please cite it; [`CITATION.cff`](CITATION.cff)
gives the details (GitHub shows them under "Cite this repository"). Please also cite the
nuclear data you use (see the [data README](src/beamtarget/data/README.md)).

## License

The code and documentation are released under the MIT License (see [`LICENSE`](LICENSE)),
copyright SHINE Technologies and beam-target contributors. The data files are redistributed
under their own sources' terms; see the [data README](src/beamtarget/data/README.md#licensing).
