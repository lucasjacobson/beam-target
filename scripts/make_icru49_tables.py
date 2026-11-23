"""Generate the ICRU 49 stopping-power data file, data/icru49_stopping.h5 (maintainers only).

The package reads the HDF5 file this script writes; run the script only to regenerate it. It
needs internet access, and downloads:

- NIST PSTAR and ASTAR (ICRU Report 49 stopping powers of protons and alpha particles), the
  electronic stopping power on NIST's default energy grid for every element they tabulate
  (26, with carbon as graphite), from https://physics.nist.gov/PhysRefData/Star/Text/. NIST
  Standard Reference Database 124: data of the US government, not subject to copyright in
  the United States.
- From Geant4 release GEANT4_TAG (https://github.com/Geant4/geant4), the coefficients of the
  ICRU 49 / Ziegler parameterizations of electronic stopping of protons (G4BraggModel) and of
  He ions (G4BraggIonModel) in every element Z = 1-92, the ICRU 37 mean excitation energies
  of the elements (G4NistMaterialBuilder), and the Ashley-Ritchie-Brandt Barkas correction
  function (G4EmCorrections). Geant4 Software License; see data/NOTICE_GEANT4.txt.

Every number is stored exactly as printed by its source. See stopping.py for how they are
used, and the data README for the file layout.

Usage:
    python scripts/make_icru49_tables.py src/beamtarget/data/icru49_stopping.h5 --raw-dir icru49_raw
"""

import argparse
import datetime
import re
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import numpy as np

NIST_URL = "https://physics.nist.gov/cgi-bin/Star/apdata.pl"
# NIST material numbers of the elements: Z -> matno (carbon: graphite, 906)
NIST_ELEMENTS = {1: "001", 2: "002", 4: "004", 6: "906", 7: "007", 8: "008", 10: "010",
                 13: "013", 14: "014", 18: "018", 22: "022", 26: "026", 29: "029", 32: "032",
                 36: "036", 42: "042", 47: "047", 50: "050", 54: "054", 64: "064", 74: "074",
                 78: "078", 79: "079", 82: "082", 92: "092"}  # fmt: skip
GEANT4_TAG = "v11.3.2"
GEANT4_URL = "https://raw.githubusercontent.com/Geant4/geant4/{tag}/{path}"
GEANT4_FILES = {
    "bragg": "source/processes/electromagnetic/standard/src/G4BraggModel.cc",
    "bragg_ion": "source/processes/electromagnetic/standard/src/G4BraggIonModel.cc",
    "corrections": "source/processes/electromagnetic/utils/src/G4EmCorrections.cc",
    "materials": "source/materials/src/G4NistMaterialBuilder.cc",
}
FORMAT_VERSION = 1


def fetch(url, cache, data=None):
    """Return the text at url (POSTing data, a dict, if given), cached in the file cache."""
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data).encode("ascii")
    # NIST refuses Python's default User-Agent; any other name will do
    request = urllib.request.Request(url, data=body, headers={"User-Agent": "beam-target-data/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def nist_table(program, matno, raw_dir):
    """Return (energies [MeV], electronic stopping [MeV cm2/g], header) from NIST PSTAR or
    ASTAR, on its default energy grid, exactly as printed."""
    text = fetch(
        NIST_URL,
        raw_dir / f"{program.lower()}_{matno}.txt",
        {"prog": program, "matno": matno, "ShowDefault": "on", "NumofEnergies": "0",
         "character": "space", "electronic": "on", "nuclear": "on", "total": "on"},
    )  # fmt: skip
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 4 and re.fullmatch(r"\d\.\d{3}E[-+]\d{2}", fields[0]):
            rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        raise ValueError(f"{program} {matno}: no table in the response")
    name = text.splitlines()[2].strip() if len(text.splitlines()) > 2 else ""
    energies, electronic = np.array(rows).T
    return energies, electronic, name


def _active_numbers(block):
    """The numbers in a C++ initializer, with // comments removed."""
    block = re.sub(r"//[^\n]*", "", block)
    return [
        float(x.rstrip("fF")) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?[fF]?", block)
    ]


def geant4_coefficients(function, text):
    """The a[92][5] coefficient table of a Geant4 stopping function."""
    body = text[text.index(function) :]
    body = body[body.index("a[92][5]") :]
    numbers = _active_numbers(body[body.index("{") : body.index("};")])
    if len(numbers) != 92 * 5:
        raise ValueError(f"{function}: {len(numbers)} coefficients, expected 460")
    return np.array(numbers).reshape(92, 5)


def geant4_mean_excitation(text):
    """Mean excitation energies [eV] of the elements Z = 1-92 (G4_<symbol> materials)."""
    values = {}
    pattern = r'AddMaterial\("G4_[A-Za-z]+"\s*,\s*[-+.\deE]+\s*,\s*(\d+)\s*,\s*([-+.\deE]+)'
    for match in re.finditer(pattern, text):
        values.setdefault(int(match.group(1)), float(match.group(2)))
    return np.array([values[Z] for Z in range(1, 93)])


def geant4_barkas(text):
    """The Barkas correction function: (W, value) pairs."""
    end = text.index("sBarkasCorr = new G4PhysicsFreeVector(47")
    block = text[text.rindex("fTable", 0, end) : end]
    pairs = re.findall(r"\{\s*([-+.\deE]+)\s*,\s*([-+.\deE]+)\s*\}", block)
    if len(pairs) != 47:
        raise ValueError(f"Barkas table: {len(pairs)} points, expected 47")
    return np.array(pairs, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="HDF5 file to write")
    parser.add_argument("--raw-dir", type=Path, default=Path("icru49_raw"),
                        help="cache of the downloaded files (reused if present)")  # fmt: skip
    args = parser.parse_args()

    geant4 = {
        key: fetch(GEANT4_URL.format(tag=GEANT4_TAG, path=path), args.raw_dir / Path(path).name)
        for key, path in GEANT4_FILES.items()
    }
    today = datetime.date.today().isoformat()
    with h5py.File(args.output, "w") as f:
        f.attrs["format"] = "beam-target ICRU 49 stopping data"
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["created"] = today
        f.attrs["description"] = (
            "Electronic stopping powers of ICRU Report 49 (NIST PSTAR and ASTAR, per gram, as "
            "printed), and the Geant4 coefficients of the ICRU 49 / Ziegler parameterizations "
            "for every element (see scripts/make_icru49_tables.py and the data README)"
        )
        for program, group_name in [("PSTAR", "pstar"), ("ASTAR", "astar")]:
            group = f.create_group(group_name)
            group.attrs["source"] = (
                f"NIST {program}, {NIST_URL}, default energies, retrieved {today}"
            )
            group.attrs["units"] = "energy [MeV], electronic stopping power [MeV cm2/g]"
            for Z, matno in NIST_ELEMENTS.items():
                energies, electronic, name = nist_table(program, matno, args.raw_dir)
                data = group.create_dataset(str(Z), data=np.column_stack([energies, electronic]))
                data.attrs["material"] = name
                data.attrs["nist_material_number"] = matno
                print(f"{program} Z={Z:2d} {name}: {len(energies)} energies")
        g4 = f.create_group("geant4")
        g4.attrs["source"] = f"Geant4 {GEANT4_TAG}, https://github.com/Geant4/geant4"
        g4.attrs["license"] = "Geant4 Software License, see NOTICE_GEANT4.txt"
        for name, key, function in [
            ("proton_fit", "bragg", "G4BraggModel::ElectronicStoppingPower"),
            ("helium_fit", "bragg_ion", "G4BraggIonModel::HeElectronicStoppingPower"),
        ]:
            fit = g4.create_dataset(name, data=geant4_coefficients(function, geant4[key]))
            fit.attrs["source"] = f"{GEANT4_FILES[key]}, {function}"
        g4.create_dataset(
            "mean_excitation_energy", data=geant4_mean_excitation(geant4["materials"])
        ).attrs["source"] = f"{GEANT4_FILES['materials']}, NistSimpleMaterials [eV], Z = 1-92"
        g4.create_dataset("barkas", data=geant4_barkas(geant4["corrections"])).attrs["source"] = (
            f"{GEANT4_FILES['corrections']}, Initialise (W, value)"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
