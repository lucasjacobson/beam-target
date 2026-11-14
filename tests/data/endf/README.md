# Evaluated nuclear data for the tests

Evaluations copied unmodified from their libraries, so that the tests of the ENDF reader and
of evaluated reactions run without a local nuclear-data library. They are laid out like the
libraries themselves, so each directory can be given as a `nuclear_data` entry. The larger
files are compressed with xz (`.endf.xz`); the regression tests unpack them into a
temporary directory (see `tests/regression_cases.py`).

| File | Library | Contents | Used to test |
|---|---|---|---|
| `ENDF-B-VIII.1/protons/p-004_Be_009.endf` | ENDF/B-VIII.1 (LANL, P. G. Young and E. D. Arthur, 1988) | p + Be-9 to 113 MeV | Kalbach-Mann continuum (LAW=1, linear outgoing energies) and two-body level channels (LAW=2); the proton thick-target benchmark; regression cases at 18 and 70 MeV |
| `ENDF-B-VIII.1/protons/p-074_W_184.endf.xz` | ENDF/B-VIII.1 | p + W-184 | A heavy target from ENDF/B; regression case at 30 MeV |
| `ENDF-B-VIII.1/alphas/a-004_Be_009.endf` | ENDF/B-VIII.1 (NNL, Holmes et al., 2018) | alpha + Be-9 to 15 MeV | Skipping an evaluation that ends below the beam energy; regression case at 10 MeV |
| `TENDL-2025/a/a-Be009.tendl` | TENDL-2025 (TALYS) | alpha + Be-9 to 200 MeV | Kalbach-Mann with the slope from the systematics (histogram outgoing energies) |
| `FENDL-3.2c/proton/p_7328_73-Ta-181.endf.xz` | FENDL-3.2c | p + Ta-181 | Lab-frame tables (LAW=7); regression case at 30 MeV |
| `FENDL-3.2c/proton/p_9237_92-U-238.endf.xz` | FENDL-3.2c | p + U-238, with fission (MT=18) | Lab-frame tables with fission neutrons; regression case at 30 MeV |
| `FENDL-3.2c/proton/p_0625_6-C-12.endf.xz` | FENDL-3.2c | p + C-12 | Lab-frame tables with histogram spectra; regression case at 30 MeV |
| `FENDL-3.2c/proton/p_0328_3-Li-7.endf.xz` | FENDL-3.2c (JENDL-5) | p + Li-7 | Neutrons listed twice in one section; regression case at 10 MeV |
| `FENDL-3.2c/deuteron/d_0328_3-Li-7.endf.xz` | FENDL-3.2c (JENDL/DEU-2020) | d + Li-7 | Deuteron breakup in lab-frame tables; regression cases at 20 MeV, and at 10 MeV in 7LiD with the built-in D(d,n) |
| `FENDL-3.2c/deuteron/d_0425_4-Be-9.endf.xz` | FENDL-3.2c (JENDL/DEU-2020) | d + Be-9 | Deuteron breakup on beryllium; regression case at 18 MeV |

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
