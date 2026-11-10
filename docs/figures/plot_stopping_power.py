"""Plot the stopping power of deuterons in tritium gas: SRIM vs ICRU Report 49 (NIST PSTAR).

Writes stopping_power.png next to this script. Needs the package installed (pip install -e .).

Both curves are the stopping per tritium atom exactly as ShineSource uses it (T_frac = 1):
"srim" is the SRIM electronic + nuclear stopping; "icru49" is the ICRU 49 electronic stopping
for protons at the same velocity plus the SRIM nuclear stopping.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from shine_neutron_source import ShineSource

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.default"] = "regular"


def stopping_per_atom(stopping_power):
    """Return (E [keV], stopping [eV/(1e15 atoms/cm^2)]) for deuterons in tritium."""
    # A minimal pure-tritium case; only its stopping-power table is used
    source = ShineSource(
        "plot", 1.0, 280.0, 1.0, 10.0, 20.0, 1.0, 0.2, 4.0, 0.0, 2, 1, 90.0, 90.0, 0.5,
        stopping_power=stopping_power,
    )  # fmt: skip
    E, eps = source.stopping_per_region[0]
    return E, eps * 1e18  # keV cm^2 -> eV/(1e15 atoms/cm^2)


E_srim, S_srim = stopping_per_atom("srim")
E_icru, S_icru = stopping_per_atom("icru49")

fig, (ax, ax_ratio) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.plot(E_srim, S_srim, label="SRIM-2012")
ax.plot(E_icru, S_icru, label="ICRU 49 (NIST PSTAR)")
ax.set_ylabel(r"Stopping power [eV/(10$^{15}$ atoms/cm$^2$)]")
ax.set_title("Stopping power of deuterons in tritium gas")
ax.set_ylim(0, None)
ax.grid()
ax.legend()

E = np.linspace(max(E_srim[0], E_icru[0]), 500, 1000)
ax_ratio.plot(E, np.interp(E, E_srim, S_srim) / np.interp(E, E_icru, S_icru), color="k")
ax_ratio.axhline(1, color="gray", linewidth=0.8)
ax_ratio.set_xlabel("Deuteron energy (lab) [keV]")
ax_ratio.set_ylabel("SRIM / ICRU 49")
ax_ratio.set_xlim(0, 500)
ax_ratio.grid()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopping_power.png"), dpi=150)
