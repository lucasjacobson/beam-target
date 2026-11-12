# Output files

Each case is written to its own directory, `<output-dir>/<case>/` (`output/<case>/` by
default; set it with `-o`):

| File | Contents | Read it with |
|---|---|---|
| `source.h5` | The full source: neutrons on the (position, angle, energy) mesh for each reaction, the beam along its path, the inputs, the results, and the materials | `h5py`; `get_openmc_sources` for OpenMC |
| `source.sdef` | An MCNP SDEF source definition, headed by the case summary and notes on using it | MCNP |
| `summary.toml` | The key inputs and results | `tomllib` |
| `axial_profile.txt`, `energy_spectrum.txt`, `angular_distribution.txt` | The source integrated down to one variable | `numpy.loadtxt` |
| `plots/*.png` | Plots of the beam and the source | an image viewer |

All the text files are ASCII with Unix (LF) line endings. Beam energies are in keV, and
neutron energies in MeV; positions are in cm, measured along the beam from where it enters the
first region; emission angles are in degrees from the beam direction (0° is forward). The
calculation itself runs on a finer internal mesh; every file holds the coarser output mesh,
set by `num_x_bins_out` (per region), `angle_step_out`, and `energy_bin_width`.

Each reaction has a key, used in dataset and field names: `DD`, `DT`, `pT`, `tD`, `p7Li_n0`,
and so on for the built-in DROSG reactions, and `<beam><target>_xn` (such as `p9Be_xn`) for
one taken from an evaluated library. Only reactions whose target isotope is in the target
appear.

## `source.h5`

The complete result, from which the other files can all be remade. Its file attributes are:

| Attribute | Meaning |
|---|---|
| `format`, `format_version` | `"shine-neutron-source"`, and the layout version, increased whenever the layout changes |
| `name`, `created`, `code_version` | Case name, UTC creation time, and the code version: the git commit (with `-dirty` if there were uncommitted changes) when run from a checkout, otherwise the package version |
| `input_file` | The input file, when run from one |
| `direction`, `entry_position`, `backstop_position` | Where the source was placed for the case: the beam direction, and the points where the beam enters and reaches the backstop [cm] |
| `warnings` | Open neutron channels that were not modeled (empty if none) |

Every dataset has a `units` attribute, and those at the top level a `description` too. Arrays
named `*_grid` hold values at mesh boundaries (one more entry than there are bins); `*_mids`
hold one value per bin.

| Dataset | Contents |
|---|---|
| `x_grid` | Position mesh along the beam [cm] |
| `beam_energy_grid`, `radius_grid`, `scrape_grid`, `current_grid` | Beam energy, edge radius, fraction scraped off, and current at each position grid point |
| `beam_energy_mids`, `radius_mids`, `scrape_mids`, `current_mids` | The same per position bin, as used in the SDEF card and `axial_profile.txt` |
| `angle_grid` | Emission-angle mesh [deg] |
| `neutron_energy_grid_<rxn>` | Neutron energy mesh of each reaction [MeV] |
| `neutrons_<rxn>` | Neutrons emitted [n/s] in each (position, angle, energy) bin, shape `(len(x_grid) - 1, len(angle_grid) - 1, len(neutron_energy_grid_<rxn>) - 1)`; attributes `dimensions` (the axes), `reaction` (such as `D(d,n)3He`), and `data` (the nuclear data used) |

| Group | Contents |
|---|---|
| `inputs` | Every input, one dataset each; values per region are arrays in beam order. A value the calculation solved for (a length, pressure, or density, or the current) says so in its `units` |
| `results` | The key results, the same as `[results]` in `summary.toml` |
| `materials/<name>` | Each material used, as attributes: `phase`, `density` (solids), its composition (`composition_kind`, which is `"atoms"` or `"molecules"`, `composition_formulas`, and `composition_amounts`), and its atom and isotope fractions |

For example, the neutrons per emission angle, summed over position and energy:

```python
import h5py

with h5py.File("output/DT90/source.h5") as f:
    angles = f["angle_grid"][()]
    per_angle = f["neutrons_DT"][()].sum(axis=(0, 2))  # n/s in each angle bin
```

## `source.sdef`

An MCNP source definition. The comment lines at the top hold the case summary (the same as
`summary.toml`) and notes on using the source; the cards follow. Copy them into the data block
of an MCNP input.

The source is a cylinder along the beam axis, from where the beam enters the first region to
the backstop:

| Distribution | Variable | Depends on |
|---|---|---|
| `d1` | Position along the axis (`EXT`) | |
| `d2` | Emission direction: cosine of the angle to `VEC`, the beam direction (left out for an isotropic source) | |
| `d3` | Radius (`RAD`) | Position, through distributions 101, 102, etc., when the beam diverges |
| `d4` | Neutron energy (`ERG`) | Direction, through distributions 501, 502, etc. (one per angle bin) |

- **Normalization.** MCNP tallies are per source neutron: multiply them by `yield_total` (in
  the summary) for absolute rates.
- **Placement.** The `direction` and the `entry_position` or `backstop_position` settings
  place the source (see `examples/generic.toml`). For `-z` the card runs along +z with `POS`
  at the backstop end; for every other direction `POS` is where the beam enters.
- **What MCNP cannot hold.** An SDEF distribution can depend on only one other variable, so
  the neutron energy depends on emission angle but not on position along the beam, and the
  angular distribution is averaged over position. The OpenMC source and `source.h5` keep the
  full correlation between position, angle, and energy.

## OpenMC

`get_openmc_sources` turns a `source.h5` file into OpenMC sources, one for each (position bin,
angle bin), each with its own energy spectrum (all reactions together), so the full
correlation between position, angle, and energy is kept:

```python
from shine_neutron_source.openmc_source import get_openmc_sources

settings.source = get_openmc_sources("output/DT90/source.h5")
```

It needs the `openmc` Python package (0.16 or later); the rest of the package runs without
it. It can also move or turn the source; see its docstring.

## `summary.toml`

The key inputs and results as TOML, with units in comments, in these tables:

- `[run]`: case name, code version, creation time, input file, the source placement, and any
  warnings.
- `[inputs]`: every input; values per region are lists in beam order.
- `[results]`: the yield of each reaction and in total, the share of each reaction, the mean
  neutron energies, the atom density and the beam energy at the end of each region, the beam
  energy, current, and radius at the backstop, the beam power (entering, deposited in the
  target, carried onto the wall by scraping, lost to nuclear reactions, and reaching the
  backstop), and the fractions of the beam and of the neutrons lost.
- `[materials.<name>]`: each material's phase, density (solids), and composition.
- `[reactions.<rxn>]`: each reaction present, and the nuclear data it came from.

## Profiles: `axial_profile.txt`, `energy_spectrum.txt`, `angular_distribution.txt`

The source integrated over all but one variable, one row per output bin, with a column for
each reaction and one for the total. Each file's header (lines starting with `#`) lists its
columns and units.

- `axial_profile.txt`: neutrons per position bin, with the beam energy, radius, and scraped
  fraction, and the beam power deposited in the target and carried onto the wall in each bin.
- `energy_spectrum.txt`: neutrons per energy bin.
- `angular_distribution.txt`: neutrons per emission-angle bin, and per steradian.

## Plots

| File | Shows |
|---|---|
| `beam.png` | Beam energy, radius (with the wall), current, and deposited power along the beam |
| `neutron_position.png`, `neutron_position_log.png` | Neutrons per unit length along the beam |
| `neutron_angle.png` | Neutrons per steradian against emission angle |
| `neutron_energy_angle_<rxn>.png` | Neutron energy against emission angle, colored by intensity, for each reaction |
| `neutron_energy.png`, `neutron_energy_log.png` | The neutron energy spectrum |

Position plots mark the boundaries between regions with dashed lines.
