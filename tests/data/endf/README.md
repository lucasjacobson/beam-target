# Evaluated nuclear data for the tests

Evaluations copied unmodified from their libraries, so that the tests of the ENDF reader and
of evaluated reactions run without a local nuclear-data library. They are laid out like the
libraries themselves, so each directory can be given as a `nuclear_data` entry. The larger
FENDL-3.2c files are compressed with xz (`.endf.xz`); the regression tests unpack them into a
temporary directory (see `tests/regression_cases.py`).

| File | Library | Contents | Used to test |
|---|---|---|---|
| `ENDF-B-VIII.1/protons/p-004_Be_009.endf` | ENDF/B-VIII.1 (LANL, P. G. Young and E. D. Arthur, 1988) | p + Be-9 to 113 MeV | Kalbach-Mann continuum (LAW=1, linear outgoing energies) and two-body level channels (LAW=2); the proton thick-target benchmark; regression case p + Be-9 at 18 MeV |
| `ENDF-B-VIII.1/alphas/a-004_Be_009.endf` | ENDF/B-VIII.1 (NNL, Holmes et al., 2018) | alpha + Be-9 to 15 MeV | Skipping an evaluation that ends below the beam energy |
| `TENDL-2025/a/a-Be009.tendl` | TENDL-2025 (TALYS) | alpha + Be-9 to 200 MeV | Kalbach-Mann with the slope from the systematics (histogram outgoing energies); regression case alpha + Be-9 at 30 MeV |
| `FENDL-3.2c/proton/p_7328_73-Ta-181.endf.xz` | FENDL-3.2c | p + Ta-181 | Lab-frame tables (LAW=7); regression case p + Ta-181 at 30 MeV |
| `FENDL-3.2c/proton/p_9237_92-U-238.endf.xz` | FENDL-3.2c | p + U-238, with fission (MT=18) | Lab-frame tables with fission neutrons; regression case p + U-238 at 30 MeV |
| `FENDL-3.2c/deuteron/d_0328_3-Li-7.endf.xz` | FENDL-3.2c (JENDL/DEU-2020) | d + Li-7 | Deuteron breakup in lab-frame tables; regression case d + Li-7 at 20 MeV |

ENDF/B-VIII.1 is from the National Nuclear Data Center (<https://www.nndc.bnl.gov/endf/>),
distributed freely; cite D. A. Brown et al., Nuclear Data Sheets 148 (2018) 1 and the
ENDF/B-VIII.1 release. TENDL-2025 is from <https://tendl.web.psi.ch/> under the CC-BY-4.0
license; cite A. J. Koning et al., Nuclear Data Sheets 155 (2019) 1. FENDL-3.2c is from the
IAEA Nuclear Data Section (<https://www-nds.iaea.org/fendl/>), distributed freely; cite the
FENDL-3.2 release.

The benchmark values in `test_endf.py` are MCNP results with the same data: neutrons created
by charged-particle reactions ("tabular sampling" in MCNP's neutron creation table) per beam
particle in a beryllium target thick enough to stop the beam. The regression cases pin this
code's own results (`tests/data/regression/*.npz`), so that a code change that moves them is
noticed.
