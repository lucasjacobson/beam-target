"""Reading the nuclear data files: DROSG-2000, SRIM and PSTAR."""

import shutil

import h5py
import numpy as np
import pytest
from conftest import data_path

from shine_neutron_source.source import (
    DROSG_FILES,
    PSTAR_FILE,
    SRIM_STOPPING_FILE,
    _fortran_float,
    read_drosg_koe,
    read_pstar,
    read_srim_h5,
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
    """The DROSG files are kept byte-for-byte as distributed (CRLF)."""
    for fname in DROSG_FILES.values():
        raw = open(data_path(fname), "rb").read()
        assert raw.count(b"\r\n") == raw.count(b"\n"), fname


def test_copy_of_data_dir_is_usable(tmp_path):
    """ShineSource can read a data directory other than the default (data_dir argument)."""
    from conftest import make_source

    data_dir = tmp_path / "data"
    shutil.copytree(data_path(""), data_dir)
    assert make_source(data_dir=str(data_dir)).yield_total == make_source().yield_total


# --- SRIM tables for every element (srim_stopping.h5) ---------------------------------------


@pytest.fixture(scope="module")
def srim_h5():
    with h5py.File(data_path(SRIM_STOPPING_FILE), "r") as f:
        yield f


def test_srim_h5_layout(srim_h5):
    f = srim_h5
    assert f.attrs["srim_version"] == "SRIM-2012.01"
    targets = [t.decode() for t in f["targets"][()]]
    assert len(targets) == 95 and targets[:2] == ["H", "He"] and targets[-3:] == ["D", "T", "He3"]
    assert list(f["target_Z"][:92]) == list(range(1, 93))
    E = f["energy"][()]
    assert len(E) == 208 and E[0] == pytest.approx(0.0011) and E[-1] == pytest.approx(1e5)
    for particle, z in [("p", 1), ("d", 1), ("t", 1), ("h", 2), ("a", 2)]:
        assert f[particle].attrs["Z"] == z
        for phase in ("solid", "gas"):
            for key in ("electronic", "nuclear"):
                data = f[particle][phase][key][()]
                assert data.shape == (95, 208)
                assert np.all(np.isfinite(data)) and np.all(data >= 0)
                assert np.all(data[:, :-1].max(axis=1) > 0)


def test_read_srim_h5(srim_h5):
    """read_srim_h5 returns stopping per atom in keV cm^2 and the targets' masses."""
    E, masses, stopping = read_srim_h5(data_path(SRIM_STOPPING_FILE))
    np.testing.assert_array_equal(E, srim_h5["energy"][()])
    i = [t.decode() for t in srim_h5["targets"][()]].index("Ti")
    elec, nuc = stopping["solid"]["Ti"]
    np.testing.assert_array_equal(elec, srim_h5["d/solid/electronic"][i] * 1e-18)
    np.testing.assert_array_equal(nuc, srim_h5["d/solid/nuclear"][i] * 1e-18)
    assert masses["D"] == pytest.approx(2.01410177812) and masses["T"] == pytest.approx(3.0160493)
    assert masses["Ti"] == pytest.approx(47.9, rel=1e-3)


def test_srim_hydrogen_isotopes(srim_h5):
    """Deuterons in D and T gas: the same electronic stopping per atom (one electron each, up to
    SRIM's 4-figure rounding), and nuclear stopping about 1.45 times higher in deuterium."""
    E, _, stopping = read_srim_h5(data_path(SRIM_STOPPING_FILE))
    (Se_D, Sn_D), (Se_T, Sn_T) = stopping["gas"]["D"], stopping["gas"]["T"]
    np.testing.assert_allclose(Se_D, Se_T, rtol=1e-3)
    sel = (E >= 10) & (E <= 300)
    assert np.all((Sn_D[sel] / Sn_T[sel] > 1.4) & (Sn_D[sel] / Sn_T[sel] < 1.5))


def test_srim_h5_phases_differ(srim_h5):
    """SRIM treats gas and solid targets differently, so both phases are kept."""
    for particle in ("p", "d", "a"):
        solid = srim_h5[particle]["solid"]["electronic"][()]
        gas = srim_h5[particle]["gas"]["electronic"][()]
        assert np.max(np.abs(gas / solid - 1)) > 0.05
