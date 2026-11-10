"""Reading the nuclear data files: DROSG-2000, SRIM and PSTAR."""

import re
import shutil

import numpy as np
import pytest
from conftest import data_path

from shine_neutron_source.source import (
    DROSG_FILES,
    HYDROGEN_ATOM_MASSES,
    NA,
    PSTAR_FILE,
    SRIM_FILES,
    _fortran_float,
    read_drosg_koe,
    read_pstar,
    read_srim_stopping,
)

# --- DROSG-2000 cross sections ---------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("  0.0100", 0.01),
        ("8.619-4", 8.619e-4),  # Fortran exponent without "e"
        ("10.14e-3", 0.01014),
        ("0 0.1209", 0.1209),  # Spaces inside a field are ignored (BLANK='NULL')
        ("-.02515", -0.02515),
        ("  1.0e2", 100.0),
        ("     1.", 1.0),
        (" .83542", 0.83542),
        ("        ", 0.0),  # Blank field is zero
    ],
)
def test_fortran_float(field, value):
    assert _fortran_float(field) == pytest.approx(value, rel=1e-15)


@pytest.mark.parametrize("rxn, num_records, E_first", [("DD", 102, 0.0018), ("DT", 171, 0.01)])
def test_drosg_tables(rxn, num_records, E_first):
    E, total, coeffs = read_drosg_koe(data_path(DROSG_FILES[rxn]))
    assert len(E) == num_records
    assert E[0] == E_first
    assert E[-1] == 40.0  # Reading stops at EMAXL; the single-angle records above are skipped
    assert coeffs.shape == (num_records, 18)
    assert np.all(np.diff(E) > 0)
    assert np.all(np.abs(coeffs.sum(axis=1) - 1) <= 1e-3)
    assert np.all(total > 0)


def test_drosg_awkward_fields_read_correctly():
    """Spot values copied by hand from the DROSG files, at the awkward entries."""
    E, total, c = read_drosg_koe(data_path(DROSG_FILES["DT"]))
    i = np.flatnonzero(E == 0.75)[0]  # "0.05115-0.04275": two run-together fields
    assert total[i] == 27.253
    assert list(c[i, :4]) == [0.9863, 0.05115, -0.04275, 0.00265]

    E, total, c = read_drosg_koe(data_path(DROSG_FILES["DD"]))
    assert total[np.flatnonzero(E == 0.01)[0]] == 8.619e-4  # "8.619-4"
    i = np.flatnonzero(E == 0.03)[0]  # Line shifted by one character: "  0.03000 0.1209"
    assert total[i] == 0.1209
    assert c[i, 0] == 0.77329


def _damaged_copy(tmp_path, damage):
    """Write a copy of d3hags.koe with damage(lines) applied and return its path."""
    lines = open(data_path(DROSG_FILES["DT"]), newline="").read().split("\r\n")
    damage(lines)
    path = tmp_path / "damaged.koe"
    path.write_text("\r\n".join(lines), newline="")
    return path


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda lines: lines.__setitem__(1, lines[1][:24] + "  0.5000" + lines[1][32:]), "sum to"),
        (
            lambda lines: lines.__delitem__(10),
            "sum to",
        ),  # A missing line misaligns every record after it
        (
            lambda lines: (lines.__setitem__(3, lines[5]), lines.__setitem__(4, lines[6])),
            "does not increase",
        ),
        (lambda lines: lines.__delitem__(slice(100, None)), "no record at EMAXL"),
    ],
    ids=["changed coefficient", "deleted line", "energies out of order", "truncated"],
)
def test_drosg_damaged_file_rejected(tmp_path, damage, message):
    with pytest.raises(ValueError, match=message):
        read_drosg_koe(_damaged_copy(tmp_path, damage))


# --- SRIM stopping powers --------------------------------------------------------------------


def _srim_footer_per_atom(path):
    """Total stopping per atom [keV cm^2] using only the file's own footer conversion factor."""
    text = open(path).read()
    factor = float(re.search(r"^\s*(\S+)\s+eV/\(1E15 atoms/cm2\)", text, re.M).group(1))
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("-"):
            inside = not inside
        elif inside:
            k = line.split()
            rows.append(float(k[2]) + float(k[3]))
    return np.array(rows) * factor * 1e-18  # eV/(1e15 atoms/cm^2) -> keV cm^2


@pytest.mark.parametrize("isotope", ["D", "T"])
def test_srim_isotope_and_per_atom_conversion(isotope):
    path = data_path(SRIM_FILES[isotope])
    E, S_elec, S_nuc, found = read_srim_stopping(path)
    assert found == isotope
    assert E[0] == pytest.approx(1.1e-3) and E[-1] == pytest.approx(1e5)  # 1.1 eV - 100 MeV
    # Per-mass -> per-atom conversion agrees with SRIM's own factor (5 significant figures)
    np.testing.assert_allclose(S_elec + S_nuc, _srim_footer_per_atom(path), rtol=5e-5)


def test_srim_tables_consistent():
    E_D, Se_D, Sn_D, _ = read_srim_stopping(data_path(SRIM_FILES["D"]))
    E_T, Se_T, Sn_T, _ = read_srim_stopping(data_path(SRIM_FILES["T"]))
    np.testing.assert_array_equal(E_D, E_T)
    # Electronic stopping per atom does not depend on the isotope (4-figure rounding only)
    np.testing.assert_allclose(Se_D, Se_T, rtol=1e-3)
    # Nuclear stopping per atom is higher in deuterium (about 1.45x at 10-300 keV)
    sel = (E_D >= 10) & (E_D <= 300)
    assert np.all((Sn_D[sel] / Sn_T[sel] > 1.4) & (Sn_D[sel] / Sn_T[sel] < 1.5))


def test_srim_non_hydrogen_target_rejected(tmp_path):
    text = open(data_path(SRIM_FILES["T"]), newline="").read()
    path = tmp_path / "heavy.txt"
    path.write_text(text.replace("2.5526E-06 g/cm3", "1.6855E-05 g/cm3"), newline="")
    with pytest.raises(ValueError, match="not a hydrogen isotope"):
        read_srim_stopping(path)


def test_hydrogen_atom_masses():
    # Masses of neutral atoms (including the electron), close to nuclear mass + electron mass
    assert HYDROGEN_ATOM_MASSES["D"] == pytest.approx(2.013553212544 + 5.485799e-4, rel=1e-8)
    assert HYDROGEN_ATOM_MASSES["T"] == pytest.approx(3.01550071597 + 5.485799e-4, rel=1e-8)
    assert NA == 6.02214076e23


# --- PSTAR (ICRU 49) ---------------------------------------------------------------------------


def test_pstar_table():
    E, S_elec, S_nuc = read_pstar(data_path(PSTAR_FILE))
    assert len(E) == 132
    assert E[0] == 1e-3 and E[-1] == 1e4  # 1 keV - 10 GeV protons
    assert np.all(np.diff(E) > 0)
    assert np.all(S_elec > 0) and np.all(S_nuc > 0)
    # Peak electronic stopping of protons in hydrogen gas: about 3800 MeV cm^2/g near 80 keV,
    # i.e. 6.4 eV/(1e15 atoms/cm^2) per atom (hydrogen has the most electrons per gram)
    i = np.argmax(S_elec)
    assert 0.05 < E[i] < 0.12
    assert 3700 < S_elec[i] < 3950


def test_data_files_unchanged_line_endings():
    """The DROSG and SRIM files are kept byte-for-byte as distributed (CRLF)."""
    for fname in list(DROSG_FILES.values()) + list(SRIM_FILES.values()):
        raw = open(data_path(fname), "rb").read()
        assert raw.count(b"\r\n") == raw.count(b"\n"), fname


def test_copy_of_data_dir_is_usable(tmp_path):
    """ShineSource can read a data directory other than the default (data_dir argument)."""
    from conftest import make_source

    data_dir = tmp_path / "data"
    shutil.copytree(data_path(""), data_dir)
    assert make_source(data_dir=str(data_dir)).yield_total == make_source().yield_total
