"""The energy-angle surfaces of a source.h5 as a web page, to turn around in a browser.

    python -m beamtarget.view output/p18_Be/source.h5 --html

writes surface.html next to source.h5 and opens it. The page holds the data and draws it
with three.js (WebGL, loaded from cdn.jsdelivr.net): the surface colored by height, lit from
a direction that can be changed, with highlights and cast shadows, a value readout under the
pointer, and a choice of colormaps.
"""

import html
import json
import re
from pathlib import Path

import numpy as np

from .source import ALL_REACTIONS
from .view import read_surfaces

TEMPLATE = Path(__file__).with_name("web") / "surface.html"

# The colormaps offered, as (matplotlib name, part used): those with a dark end start above
# it, so the low floor of a surface (most of its area) is not black
COLORMAPS = {
    "Spectral (reversed)": ("Spectral_r", 0.0, 1.0),
    "Turbo": ("turbo", 0.0, 1.0),
    "Viridis": ("viridis", 0.2, 1.0),
    "Cividis": ("cividis", 0.15, 1.0),
    "Plasma": ("plasma", 0.15, 0.97),
    "Cool to warm": ("coolwarm", 0.0, 1.0),
    "Red, yellow, blue (reversed)": ("RdYlBu_r", 0.0, 1.0),
}
DEFAULT_COLORMAP = "Spectral (reversed)"
COLORMAP_SAMPLES = 256
HEIGHT_DECIMALS = 5  # Heights are stored as fractions of the peak, rounded to this


def _label_html(label):
    """A matplotlib label such as "$^{9}$Be(p,xn)" in HTML (superscripts as <sup>)."""
    return re.sub(r"\$\^\{([^}]*)\}\$", r"<sup>\1</sup>", html.escape(label, quote=False))


def _label_text(label):
    """The same in plain text ("9Be(p,xn)")."""
    return re.sub(r"\$\^\{([^}]*)\}\$", r"\1", label)


def surface_page(path, reactions=None, energy=None):
    """Return the web page (HTML) for the reactions of a source.h5 (all, or those named),
    over the neutron energies energy = (low, high) in the file's unit (all if None)."""
    from matplotlib import colormaps

    entries = []
    name = None
    for title, angles, energies, density, unit in read_surfaces(path, reactions, energy):
        label, _, rest = title.partition(" neutron energy against emission angle (")
        name = rest.rstrip(")") or name
        top = float(np.max(density))
        heights = np.round(density / top, HEIGHT_DECIMALS) if top > 0 else np.zeros_like(density)
        entries.append(
            {
                "key": "all" if label == ALL_REACTIONS[1] else _label_text(label),
                "label_html": _label_html(label),
                "label_text": _label_text(label),
                "angles": [round(float(a), 6) for a in angles],
                "energies": [float(f"{e:.8g}") for e in energies],
                "unit": unit,
                "max": top,
                "heights": heights.ravel().tolist(),  # (angle, energy), row-major
            }
        )
    maps = {}
    for title, (cmap, lo, hi) in COLORMAPS.items():
        rgb = colormaps[cmap](np.linspace(lo, hi, COLORMAP_SAMPLES))[:, :3]
        maps[title] = np.round(255 * rgb).astype(int).tolist()
    data = {
        "name": name or str(path),
        "reactions": entries,
        "colormaps": maps,
        "default_colormap": DEFAULT_COLORMAP,
    }
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", text)


def write_surface_page(path, out=None, reactions=None, energy=None):
    """Write the web page of a source.h5 (see surface_page) to out (default: surface.html
    next to it) and return its path."""
    out = Path(path).with_name("surface.html") if out is None else Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(surface_page(path, reactions, energy), encoding="utf-8", newline="\n")
    return out
