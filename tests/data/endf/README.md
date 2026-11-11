# Evaluated nuclear data for the tests

Three small evaluations, unmodified, so that the tests of the ENDF reader and of evaluated
reactions run without a local nuclear-data library. They are laid out like the libraries
themselves, so each directory can be given as a `nuclear_data` entry.

| File | Library | Contents | Used to test |
|---|---|---|---|
| `ENDF-B-VIII.1/protons/p-004_Be_009.endf` | ENDF/B-VIII.1 (LANL, P. G. Young and E. D. Arthur, 1988) | p + Be-9 to 113 MeV | Kalbach-Mann continuum (LAW=1, linear outgoing energies) and two-body level channels (LAW=2); the proton thick-target benchmark |
| `ENDF-B-VIII.1/alphas/a-004_Be_009.endf` | ENDF/B-VIII.1 (NNL, Holmes et al., 2018) | alpha + Be-9 to 15 MeV | Skipping an evaluation that ends below the beam energy |
| `TENDL-2021/a-Be009.tendl` | TENDL-2021 (TALYS) | alpha + Be-9 to 200 MeV | Kalbach-Mann with the slope from the systematics (histogram outgoing energies); the alpha thick-target benchmark |

ENDF/B-VIII.1 is from the National Nuclear Data Center (<https://www.nndc.bnl.gov/endf/>),
distributed freely; cite D. A. Brown et al., Nuclear Data Sheets 148 (2018) 1 and the
ENDF/B-VIII.1 release. TENDL-2021 is from <https://tendl.web.psi.ch/> under the CC-BY-4.0
licence; cite A. J. Koning et al., Nuclear Data Sheets 155 (2019) 1.

The benchmark values in the tests are MCNP results with the same data: neutrons created by
charged-particle reactions ("tabular sampling" in MCNP's neutron creation table) per beam
particle in a beryllium target thick enough to stop the beam.
