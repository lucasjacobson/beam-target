"""Generate the SRIM stopping-power data file, data/srim_stopping.h5 (maintainers only).

The package does not need SRIM: it reads the HDF5 file this script writes. Run the script
only to regenerate or extend that file. It needs a Windows SRIM installation (tested with
SRIM-2013 Pro, whose SR Module reports SRIM-2012.01) and runs SRIM's batch program,
"SR Module\\SRModule.exe", once per table.

Tables are made for every beam particle (p, d, t, 3He, alpha), every target (elements
Z = 1-92 at their natural atomic mass, plus the isotopes D, T and 3He) and both target phases
(solid and gas): 950 in all. Stopping is requested per atom, in eV/(1e15 atoms/cm2), so it
does not depend on the target density. Densities are SRIM's own (from SCOEF03.dat); for the
gas phase of elements SRIM gives no gas density for, an ideal monatomic gas at 0 degC and
1 atm is used, which affects only SRIM's range and straggling columns (not stored).

SRIM's own output files are kept in a cache directory (--raw-dir); tables already there are
not recalculated, so the HDF5 file can be rebuilt from the cache without running SRIM again.

SR Module needs the Visual Basic 5 runtime (MSVBVM50.DLL), either next to SRModule.exe or in
Windows. If neither has it, the script unpacks it from SRIM's own setup files into its private
copy of SR Module, so nothing is installed on the system.

Usage:
    python scripts/make_srim_tables.py "C:\\path\\to\\SRIM-2013-Pro" \\
        src/shine_neutron_source/data/srim_stopping.h5 --raw-dir srim_raw
"""

import argparse
import datetime
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

# Beam particles: name -> (description, Z, nuclear mass [u], CODATA 2022)
PARTICLES = {
    "p": ("proton", 1, 1.007276466579),
    "d": ("deuteron", 1, 2.013553212544),
    "t": ("triton", 1, 3.015500715970),
    "h": ("helion (3He nucleus)", 2, 3.014932246932),
    "a": ("alpha (4He nucleus)", 2, 4.001506179129),
}

# Isotopic targets in addition to the natural elements: name -> (Z, atomic mass [u], AME2020)
ISOTOPE_TARGETS = {"D": (1, 2.01410177812), "T": (1, 3.0160492779), "He3": (2, 3.01602932197)}

PHASES = {"solid": 0, "gas": 1}
STOPPING_UNITS = 7  # eV/(1E15 atoms/cm2)
# 1.1 eV to 5 GeV: the widest range SR Module gives. It starts its tables at 1.1 eV whatever lower
# E-Min it is given, and above about 5 GeV it stops with "Run-time error '6': Overflow" (though
# it accepts up to 10 GeV/u), for protons in hydrogen as for alphas in uranium
E_MIN_KEV, E_MAX_KEV = 0.0011, 5.0e6
STP_ATOM_DENSITY = 2.6867811e19  # Loschmidt constant [1/cm3]
AVOGADRO = 6.02214076e23
FORMAT_VERSION = 1


def read_elements(sr_module):
    """Return [(symbol, Z, name, natural mass [u], solid density, gas density)] from
    SCOEF03.dat, in order of Z (densities in g/cm3; gas density 0 if SRIM gives none)."""
    elements = []
    with open(sr_module / "SCOEF03.dat", encoding="latin-1") as f:
        for line in f:
            fields = shlex.split(line)
            elements.append((fields[0].strip(), int(fields[2]), fields[1].strip(),
                             float(fields[5]), float(fields[6]), float(fields[10])))  # fmt: skip
    return elements


def targets(elements):
    """Return [(target, Z, element name, mass [u], solid density, gas density)]: the natural
    elements, then the isotopic targets (at the natural element's atom density)."""
    result = list(elements)
    by_z = {z: (name, mass, rho_solid) for _, z, name, mass, rho_solid, _ in elements}
    for symbol, (z, mass) in ISOTOPE_TARGETS.items():
        name, natural_mass, rho_solid = by_z[z]
        result.append((symbol, z, name, mass, rho_solid * mass / natural_mass, 0.0))
    return result


def gas_density(mass, rho_gas):
    """SRIM's gas density, or an ideal monatomic gas at 0 degC and 1 atm [g/cm3]."""
    return rho_gas or STP_ATOM_DENSITY * mass / AVOGADRO


def sr_in(out_name, ion_z, ion_mass, phase, density, target_z, target_name, target_mass):
    """Return the text of an SR.IN control file for one table."""
    lines = [
        "---Stopping/Range Input Data (Number-format: Period = Decimal Point)",
        "---Output File Name",
        f'"{out_name}"',
        "---Ion(Z), Ion Mass(u)",
        f"{ion_z}   {ion_mass}",
        "---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.",
        f"{phase}    {density:.6E}    1",
        "---Number of Target Elements",
        " 1 ",
        "---Target Elements: (Z), Target name, Stoich, Target Mass(u)",
        f'{target_z}   "{target_name}"   1   {target_mass}',
        "---Output Stopping Units (1-8)",
        f" {STOPPING_UNITS}",
        "---Ion Energy : E-Min(keV), E-Max(keV)",
        f" {E_MIN_KEV}    {E_MAX_KEV}",
    ]
    return "\r\n".join(lines) + "\r\n"


def prepare_sr_module(srim_dir, work):
    """Copy SR Module into work and make sure it can find the VB5 runtime."""
    sr_module = work / "SR Module"
    shutil.copytree(srim_dir / "SR Module", sr_module)
    # The runtime may be next to SRModule.exe (as a fixed SRIM install puts it) or in Windows
    windows = Path(r"C:\Windows")
    places = [sr_module, windows / "SysWOW64", windows / "System32"]
    if not any((place / "MSVBVM50.DLL").exists() for place in places):
        packed = srim_dir / "SRIM-Setup" / "SRIM-AutoSetup" / "MSVBVM50.dl_"
        # Windows' expand.exe by full path (Git Bash has an unrelated "expand" on its PATH)
        expand = windows / "System32" / "expand.exe"
        subprocess.run([str(expand), str(packed), str(sr_module / "MSVBVM50.DLL")], check=True,
                       capture_output=True)  # fmt: skip
    return sr_module


def raw_name(particle, target, phase):
    return f"{particle}_in_{target}_{phase}.txt"


def run_srim(sr_module, raw_dir, all_targets):
    """Run SR Module for every table not already in raw_dir. Returns the number run."""
    count = 0
    for particle, (_, ion_z, ion_mass) in PARTICLES.items():
        for target, z, name, mass, rho_solid, rho_gas in all_targets:
            for phase, flag in PHASES.items():
                out_name = raw_name(particle, target, phase)
                if (raw_dir / out_name).exists():
                    continue
                density = rho_solid if phase == "solid" else gas_density(mass, rho_gas)
                (sr_module / "SR.IN").write_bytes(
                    sr_in(out_name, ion_z, ion_mass, flag, density, z, name, mass).encode()
                )
                (sr_module / out_name).unlink(missing_ok=True)
                subprocess.run([str(sr_module / "SRModule.exe")], cwd=sr_module, check=True)
                if not (sr_module / out_name).exists():
                    raise RuntimeError(f"SR Module wrote no {out_name}")
                shutil.move(sr_module / out_name, raw_dir / out_name)
                count += 1
        print(f"{particle}: done", flush=True)
    return count


UNIT_TO_KEV = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3, "GeV": 1e6}


def read_raw(path):
    """Return (SRIM version, E [keV], electronic, nuclear [eV/(1e15 atoms/cm2)]) from one
    SRIM output file."""
    text = path.read_text(encoding="latin-1")
    if "eV/(1E15 atoms/cm2)" not in text:
        raise ValueError(f"{path}: not in eV/(1E15 atoms/cm2)")
    version = re.search(r"SRIM version ---> (\S+)", text).group(1)
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("-"):
            inside = not inside
            continue
        if inside:
            t = line.split()
            rows.append([float(t[0]) * UNIT_TO_KEV[t[1]], float(t[2]), float(t[3])])
    data = np.array(rows)
    return version, data[:, 0], data[:, 1], data[:, 2]


def write_hdf5(path, raw_dir, all_targets):
    """Pack the raw tables into one HDF5 file (see the data README for the layout)."""
    names = [t[0] for t in all_targets]
    energy, versions = None, set()
    stopping = {}
    for particle in PARTICLES:
        for phase in PHASES:
            rows_e, rows_n = [], []
            for target in names:
                version, E, e, n = read_raw(raw_dir / raw_name(particle, target, phase))
                versions.add(version)
                if energy is None:
                    energy = E
                elif not np.array_equal(E, energy):
                    raise ValueError(f"{particle} in {target} ({phase}): different energy grid")
                rows_e.append(e)
                rows_n.append(n)
            elec = np.array(rows_e)
            stopping[particle, phase] = (elec, np.array(rows_n))
    if len(versions) != 1:
        raise ValueError(f"tables come from several SRIM versions: {sorted(versions)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    gz = dict(compression="gzip", compression_opts=9, shuffle=True)
    units = "eV/(1e15 atoms/cm2)"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "shine-neutron-source SRIM stopping powers"
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["srim_version"] = versions.pop()
        f.attrs["created"] = datetime.date.today().isoformat()
        f.attrs["generated_by"] = "scripts/make_srim_tables.py (SRIM SR Module)"
        f.attrs["description"] = (
            "Electronic and nuclear stopping per target atom for light ions in single-element "
            "targets, from SRIM. Values are SRIM's output, to its 4 significant figures."
        )
        f.create_dataset("energy", data=energy, **gz)
        f["energy"].attrs["units"] = "keV"
        f["energy"].attrs["description"] = "beam particle kinetic energy (lab)"
        f.create_dataset("targets", data=np.array(names, dtype="S"))
        f["targets"].attrs["description"] = (
            "target names: element symbols for Z = 1-92 at natural isotopic mass, then the "
            "isotopes D, T and He3"
        )
        f.create_dataset("target_Z", data=np.array([t[1] for t in all_targets], dtype="i4"))
        f.create_dataset("target_mass", data=np.array([t[3] for t in all_targets]))
        f["target_mass"].attrs["units"] = "u (atomic mass, as given to SRIM)"
        for particle, (description, z, mass) in PARTICLES.items():
            g = f.create_group(particle)
            g.attrs["description"] = description
            g.attrs["Z"] = z
            g.attrs["mass"] = mass
            g.attrs["mass_units"] = "u (nuclear mass, as given to SRIM)"
            for phase in PHASES:
                elec, nuc = stopping[particle, phase]
                p = g.create_group(phase)
                p.attrs["description"] = f"{phase} targets (SRIM's {phase}-phase stopping)"
                for key, data in [("electronic", elec), ("nuclear", nuc)]:
                    p.create_dataset(key, data=data, **gz)
                    p[key].attrs["units"] = units
                    p[key].attrs["dimensions"] = ["target", "energy"]


def check_hdf5(path, raw_dir):
    """Raise if the HDF5 file does not hold exactly the raw tables' values."""
    with h5py.File(path, "r") as f:
        names = [n.decode() for n in f["targets"][()]]
        for particle in PARTICLES:
            for phase in PHASES:
                elec = f[particle][phase]["electronic"][()]
                nuc = f[particle][phase]["nuclear"][()]
                for i, target in enumerate(names):
                    _, _, e, n = read_raw(raw_dir / raw_name(particle, target, phase))
                    if not (np.array_equal(elec[i], e) and np.array_equal(nuc[i], n)):
                        raise ValueError(f"{particle} in {target} ({phase}) does not match")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("srim_dir", type=Path, help="SRIM installation directory")
    parser.add_argument("output", type=Path, help="HDF5 file to write")
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="cache of SRIM output files (kept; existing tables are reused)")  # fmt: skip
    args = parser.parse_args(argv)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    all_targets = targets(read_elements(args.srim_dir / "SR Module"))
    expected = len(PARTICLES) * len(all_targets) * len(PHASES)
    missing = sum(
        not (args.raw_dir / raw_name(p, t[0], ph)).exists()
        for p in PARTICLES
        for t in all_targets
        for ph in PHASES
    )
    ran = 0
    if missing:
        if sys.platform != "win32":
            parser.exit(2, f"error: {missing} tables missing and SR Module runs only on Windows\n")
        with tempfile.TemporaryDirectory() as tmp:
            sr_module = prepare_sr_module(args.srim_dir, Path(tmp))
            ran = run_srim(sr_module, args.raw_dir, all_targets)
    print(f"{expected} tables ({ran} calculated now, {expected - ran} from {args.raw_dir})")

    write_hdf5(args.output, args.raw_dir, all_targets)
    check_hdf5(args.output, args.raw_dir)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB); values verified")


if __name__ == "__main__":
    main()
