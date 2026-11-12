"""Plot the DD and DT total (angle-integrated) cross sections from the DROSG-2000 data.

Writes total_cross_sections.png next to this script. Needs the package installed
(pip install -e .).

The total cross section is sigma = 4 pi * sigma_0(0 deg, CM) * C_0, since only the l = 0
Legendre term survives integration over solid angle.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from beamtarget.nuclear import REACTIONS
from beamtarget.source import DEFAULT_DATA_DIR, read_drosg_koe

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.default"] = "regular"

labels = {"DD": r"D(d,n)$^3$He", "DT": r"T(d,n)$^4$He"}

fig, ax = plt.subplots(figsize=(8, 6))
for rxn in ["DD", "DT"]:
    E, sigma0, coeffs = read_drosg_koe(os.path.join(DEFAULT_DATA_DIR, REACTIONS[rxn].data_file))
    sigma_tot = 4 * np.pi * sigma0 * coeffs[:, 0] / 1000  # mb -> b
    ax.semilogy(E * 1000, sigma_tot, label=labels[rxn])

ax.set_xlabel("Deuteron energy (lab) [keV]")
ax.set_ylabel("Total cross section [b]")
ax.set_title("DD and DT total cross sections (DROSG-2000)")
ax.set_xlim(0, 500)
ax.set_ylim(1e-6, 10)
ax.grid(which="major")
ax.grid(which="minor", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "total_cross_sections.png"), dpi=150
)
