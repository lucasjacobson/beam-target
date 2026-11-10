# Nuclear data

The files in this directory are read by `ShineSource` (see `read_drosg_koe`,
`read_cross_sections` and `read_stopping_power` in `source.py`). They are kept exactly as
distributed, including their Windows (CRLF) line endings (see `.gitattributes`).

| File | Contents | Source |
|---|---|---|
| `d2hags.koe` | D(d,n)³He cross sections | DROSG-2000 v12.01 |
| `d3hags.koe` | T(d,n)⁴He cross sections | DROSG-2000 v12.01 |
| `SP_DT.txt` | Stopping power of deuterons in tritium gas | SRIM-2012.01 |

## Cross sections: `d2hags.koe`, `d3hags.koe`

Unmodified data files from DROSG-2000 version 12.01 (March 2017), taken from
`DROSG2000-v12.01.zip` (`pentiumpro/2017/`). DROSG-2000 is a collection of codes and evaluated
data for accelerator-based neutron source reactions by M. Drosg, distributed by the IAEA
Nuclear Data Section at <https://www-nds.iaea.org/public/libraries/drosg2000/>. The
`d2hags.koe` file notes that up to 40 MeV it is identical to the ENDF/B-VII evaluation by Hale
and Drosg. Each file ends with the evaluator's comments on sources and changes (lines starting
with `!`).

### Format

The code reads the files the same way DROSG-2000 does (`FORMAT(10f8.0)` with `BLANK='NULL'`
in `LEGINT.f95`):

- Every line is ten 8-character fields. Spaces inside a field are ignored, blank fields are
  zero, and an exponent may be written without `e` (`8.619-4` means 8.619 × 10⁻⁴). Adjacent
  fields can therefore run together, as in `0.05115-0.04275`.
- The first line holds EMIN, EMAX, EMINL and EMAXL [MeV]. EMAXL (40 MeV in both files) is the
  highest energy of the Legendre table.
- Each record is then two lines (20 fields):

  | Field | Meaning | Units |
  |---|---|---|
  | 1 | Incident deuteron energy (lab frame) | MeV |
  | 2 | Centre-of-mass differential cross section at 0° | mb/sr |
  | 3–20 | Legendre coefficients C₀ … C₁₇ of the angular distribution, normalised to sum to 1 | — |

The centre-of-mass differential cross section at deuteron energy E and cosine μ of the
centre-of-mass emission angle is

    dσ/dΩ(E, μ) = σ₀(E) × Σ C_l(E) P_l(μ),   l = 0, 1, …, 17

where σ₀ is field 2. While reading, the code checks that the energies increase and that each
record's coefficients sum to 1 within 0.001 (the tolerance DROSG-2000 uses), so a damaged or
misaligned file is rejected rather than silently misread.

This code interpolates linearly between tabulated energies; the DROSG-2000 codes themselves use
spline interpolation, so values between table points can differ slightly.

**Valid range.** The Legendre data cover 1.8 keV to 40 MeV for DD and 10 keV to 40 MeV for DT.
Below the first energy the code takes the cross section to be zero. The records above EMAXL
(40.0001 MeV upwards) are single-angle data rather than Legendre coefficients (the DROSG
documentation states there is no angular-distribution information outside the Legendre
table), so the code stops reading at EMAXL and limits the beam voltage to 40,000 kV
(`MAX_VOLTAGE`).

## Stopping power: `SP_DT.txt`

Unmodified output of SRIM (<http://www.srim.org/>), version SRIM-2012.01 with the SRIM-2006
calculation method, generated 6 June 2024: stopping power of deuterons (mass 2.013553 amu) in a
pure tritium gas target (2.5526 × 10⁻⁶ g/cm³, Bragg correction 0%), from 1.1 eV to 100 MeV. The
code uses the electronic and nuclear stopping columns, in MeV/(mg/cm²).

The table is used for any D/T mixture: electronic stopping per atom is the same for D and T,
so the code multiplies the per-mass stopping power by the density the gas would have if every
atom were tritium. This neglects the small isotope dependence of nuclear stopping. See the
note on `rho_T_equiv_list` in `source.py`.

## References

- M. Drosg, DROSG-2000, Codes and database for 60 neutron source reactions, documented in the
  IAEA report IAEA-NDS-87 Rev. 10 (January 2016), downloaded from the homepage of IAEA Nuclear
  Data Section. (Citation as recommended by the DROSG-2000 documentation; the version used here,
  12.01, is documented in IAEA-NDS-87 Rev. 10, March 2017:
  <https://www-nds.iaea.org/public/libraries/drosg2000/iaeands-v12.01.pdf>.)
- J. F. Ziegler, M. D. Ziegler and J. P. Biersack, "SRIM – The stopping and range of ions in
  matter (2010)", Nucl. Instrum. Methods Phys. Res. B 268 (2010) 1818–1823.
