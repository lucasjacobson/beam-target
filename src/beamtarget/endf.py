"""Neutron production from ENDF-6 evaluations of light-ion reactions (TENDL, ENDF/B, FENDL, ...).

An evaluation file holds, for one beam particle (the projectile) on one target nuclide, the
cross section of each reaction channel (MF=3) and, for each channel, the yield and
energy-angle distribution of every particle it emits (MF=6). This module reads the parts
that describe neutrons: every channel with a neutron product, summed ("merged") into one
neutron production, and the non-elastic cross section (for the loss of beam particles to
nuclear reactions).

Supported neutron distributions (ENDF-102 section 6.2):

- LAW=1, continuum energy-angle: LANG=2 (Kalbach-Mann, with the slope from the systematics,
  NA=1, or given, NA=2), LANG=3 with NA=2 (as TENDL uses it: Kalbach-Mann with the slope
  given), LANG=1 (Legendre) or NA=0 (isotropic); outgoing energies as a histogram (LEP=1) or
  linear (LEP=2); center-of-mass (LCT=2, 3, 4) or lab (LCT=1) frame.
- LAW=2, discrete two-body reactions (such as (p,n) to one level of the residual nucleus):
  Legendre (LANG=0) or tabulated (LANG=12) center-of-mass angular distributions; the
  neutron energy follows from the kinematics, with the channel's Q value (QI in MF=3).
- LAW=7, lab-frame angle-energy tables: an outgoing-energy distribution at each of a set of
  lab cosines.

Between tabulated incident energies, distributions are mixed linearly (INT=2) or by unit base
(INT=22: each stretched to the outgoing-energy range at the wanted energy). Continuum (LAW=1)
distributions are always mixed by unit base, even where the evaluation gives INT=2, as MCNP
does: mixed literally, every feature of a tabulated spectrum (a peak, its upper edge) stays at
the same outgoing energy between tabulated incident energies 1-2 MeV apart, which in a thick
target makes a stripe for each one, and the top of the higher energy's spectrum reaches above
what the reaction can give at the lower beam energies. For 18 MeV protons on thick 9Be
(ENDF/B-VII.0), spectra mixed by unit base match MCNP 6.2's (1e10 histories) in their fine
structure, and literally mixed ones do not; the yields are the same either way. Anything else
is refused with a message naming it.

Reference: ENDF-6 Formats Manual, BNL-203218-2023-INRE (ENDF-102).
"""

import functools
import os
import re

import numpy as np

from .nuclear import BEAMS, atom_name, bin_two_body, cm_to_lab, nuclear_mass

# ENDF sub-library number (NSUB) of each beam particle's incident data
NSUB_BEAMS = {10010: "p", 10020: "d", 10030: "t", 20030: "h", 20040: "a"}

# TENDL's variants of one evaluation, {beam}-{symbol}{A:03d}{suffix}, such as p-Be009.tendl:
# the same calculation with its channels given separately up to 30 MeV and lumped into MT=5
# above (the standard file), up to 20 or 60 MeV (-s20, -s60), everywhere (-mt), or nowhere
# (-s0). They are not equally complete: in TENDL-2025 the separate channels of -mt (above
# about 30 MeV for p, t, 3He and alpha beams) and of -s60 (from 30 to 60 MeV) cover as little
# as 10-60% of the non-elastic cross section, so the neutrons of the rest are missing, while
# the standard file's separate channels miss up to 22% of it for d + 9Be, where -mt is
# complete. find_evaluation therefore takes the most complete of the standard, -mt and -s60
# files present, and among equally complete ones the first in this order. The lumped
# spectra are used only where no such file is present (-s20, then -s0), because they can be
# wrong for light targets: for p + 9Be at 10 MeV, -s0 gives every neutron the (p,n) shape,
# leaving out the low-energy neutrons of the breakup (p,np) that the separate channels give.
# (Being lumped, -s0 is always complete, so completeness alone would always pick it.)
FILE_SUFFIXES = (".tendl", "-mt.tendl", "-s60.tendl", "-s20.tendl", "-s0.tendl")
NUM_SPLIT = 3  # The first NUM_SPLIT of FILE_SUFFIXES are compared by completeness
# Variants whose uncovered share of the non-elastic cross section (integrated over incident
# energy) differs by less than this count as equally complete
VARIANT_TIE = 0.01

M_NEUTRON = nuclear_mass(0, 1)  # MeV
# CM outgoing-energy pieces of a continuum are narrow enough that their neutrons' lab angle
# changes by at most this many angle bins across a piece (see _cm_energy_step)
CONTINUUM_ANGLE_SHIFT = 0.2
# Samples per tabulated interval for placing the narrow pieces (see _energy_points)
STEP_SAMPLES = 256

# Kalbach systematics (ENDF-102 section 6.2.3.2): the factor M of the incident particle.
# Kalbach gives 1 for p and d and 0 for alpha, and never specified t or 3He; they are taken
# as 1 here, like the other hydrogen and light ions (the term only matters at high energy).
KALBACH_M = {"p": 1.0, "d": 1.0, "t": 1.0, "h": 1.0, "a": 0.0}
KALBACH_M_NEUTRON = 0.5  # m for an emitted neutron
# Energy to break each particle into nucleons [MeV] (ENDF-102 Appendix H)
BREAKUP_ENERGY = {(0, 1): 0.0, (1, 1): 0.0, (1, 2): 2.224566, (1, 3): 8.481798,
                  (2, 3): 7.718043, (2, 4): 28.29566}  # fmt: skip

# Reaction sums (MT: the range of channels it is the sum of), left out of the non-elastic
# total when their parts are given
SUMS = {4: range(50, 92), 103: range(600, 650), 104: range(650, 700), 105: range(700, 750),
        106: range(750, 800), 107: range(800, 850)}  # fmt: skip


def _number(field):
    """Read one 11-character ENDF number, such as " 1.234567+3", "-2.5-12" or " 6.01200+ 3"
    (Fortran output can leave a blank between the exponent's sign and its digits)."""
    s = field.replace(" ", "")
    if not s:
        return 0.0
    for i in range(1, len(s)):
        if s[i] in "+-" and s[i - 1] not in "eE":
            return float(s[:i] + "e" + s[i:])
    return float(s)


class _Records:
    """Reads the records of one (MF, MT) section in order."""

    def __init__(self, lines):
        self.lines = lines
        self.i = 0

    def cont(self):
        line = self.lines[self.i]
        self.i += 1
        return [_number(line[k : k + 11]) for k in range(0, 66, 11)]

    def values(self, n):
        out = []
        while len(out) < n:
            line = self.lines[self.i]
            self.i += 1
            out += [_number(line[k : k + 11]) for k in range(0, 66, 11)]
        return out[:n]

    def tab1(self):
        """Return (the CONT values, Tabulated)."""
        c = self.cont()
        nr, n = int(c[4]), int(c[5])
        interp = self.values(2 * nr)
        xy = self.values(2 * n)
        breaks = [int(v) for v in interp[0::2]]
        codes = [int(v) for v in interp[1::2]]
        return c, Tabulated(xy[0::2], xy[1::2], breaks, codes)

    def tab2(self):
        """Return (the CONT values, breakpoints, interpolation codes)."""
        c = self.cont()
        interp = self.values(2 * int(c[4]))
        return c, [int(v) for v in interp[0::2]], [int(v) for v in interp[1::2]]

    def mf3(self):
        """Read an MF=3 section: return (QM [MeV], QI [MeV], cross section)."""
        self.cont()
        c, xs = self.tab1()
        return c[0] / 1e6, c[1] / 1e6, xs


class Tabulated:
    """A TAB1 function y(x) with ENDF interpolation codes (1 histogram, 2 linear-linear,
    3 linear-log, 4 log-linear, 5 log-log); zero outside the tabulated range."""

    def __init__(self, x, y, breaks, codes):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.breaks = list(breaks)
        self.codes = list(codes)

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x)
        if len(self.x) < 2:
            return out
        inside = (x >= self.x[0]) & (x <= self.x[-1])
        i = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, len(self.x) - 2)
        x0, x1, y0, y1 = self.x[i], self.x[i + 1], self.y[i], self.y[i + 1]
        # Interval i runs from point i + 1 to i + 2 (1-based); its region is the first whose
        # last point (breakpoint) is at least i + 2
        region = np.searchsorted(self.breaks, i + 2, side="left").clip(0, len(self.codes) - 1)
        code = np.array(self.codes)[region]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
            lin = y0 + t * (y1 - y0)
            res = np.where(code == 1, y0, lin)
            logx = np.where((x0 > 0) & (x1 > 0) & (x > 0), np.log(x / x0) / np.log(x1 / x0), t)
            res = np.where(code == 3, y0 + logx * (y1 - y0), res)
            res = np.where((code == 4) & (y0 > 0) & (y1 > 0), y0 * (y1 / y0) ** t, res)
            res = np.where((code == 5) & (y0 > 0) & (y1 > 0), y0 * (y1 / y0) ** logx, res)
        res = np.where(x == self.x[-1], self.y[-1], res)
        out[inside] = res[inside]
        return out


def _sections(path):
    """Return {(MF, MT): [lines]} of an ENDF-6 file."""
    sections = {}
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            if len(line) < 75:
                continue
            try:
                mf, mt = int(line[70:72]), int(line[72:75])
            except ValueError:
                continue
            if mf > 0 and mt > 0:
                sections.setdefault((mf, mt), []).append(line)
    return sections


def separation_energy(Z_c, A_c, Z_x, A_x, particle):
    """Return the separation energy [MeV] of a particle (Z, A) from a compound nucleus
    (Z_c, A_c), leaving (Z_x, A_x), by the mass formula of the Kalbach systematics, without
    pairing or shell terms (ENDF-102 section 6.2.3.2)."""
    N_c, N_x = A_c - Z_c, A_x - Z_x
    return (
        15.68 * (A_c - A_x)
        - 28.07 * ((N_c - Z_c) ** 2 / A_c - (N_x - Z_x) ** 2 / A_x)
        - 18.56 * (A_c ** (2 / 3) - A_x ** (2 / 3))
        + 33.22 * ((N_c - Z_c) ** 2 / A_c ** (4 / 3) - (N_x - Z_x) ** 2 / A_x ** (4 / 3))
        - 0.717 * (Z_c**2 / A_c ** (1 / 3) - Z_x**2 / A_x ** (1 / 3))
        + 1.211 * (Z_c**2 / A_c - Z_x**2 / A_x)
        - BREAKUP_ENERGY[particle]
    )


def _energy_points(energies, f0, lep, max_step, step=None):
    """Split an outgoing-energy distribution (energies [MeV], density f0 [1/MeV], histogram
    for lep 1 or linear for lep 2) into pieces at most max_step [MeV] wide. Return (energy
    of each piece [MeV], its probability, index of the tabulated point below), normalized
    to a total of 1.

    step, if given, is a function of the outgoing energy (array) giving the largest piece
    width wanted there [MeV], where it is narrower than max_step. Each tabulated interval is
    then cut into pieces of equal "count" N, where dN/dE = 1 / step(E), so pieces are
    narrow only where step asks for it. Without step (or where step is at least max_step),
    each interval is cut into equal pieces at most max_step wide.
    """
    e, f = energies, f0
    widths = np.diff(e)
    probs = f[:-1] * widths if lep == 1 else 0.5 * (f[:-1] + f[1:]) * widths
    keep = probs > 0
    n_sub = np.maximum(1, np.ceil(widths / max_step).astype(int))
    # Piece edges as fractions of each interval: equal pieces, unless step asks for more
    fractions = {}
    if step is not None:
        samples = np.linspace(0.0, 1.0, STEP_SAMPLES + 1)
        for i in np.flatnonzero(keep):
            x = e[i] + samples * widths[i]
            density = 1.0 / np.minimum(step(x), max_step)
            count = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]))))
            count *= widths[i] / STEP_SAMPLES
            n = int(np.ceil(count[-1]))
            if n > n_sub[i]:
                n_sub[i] = n
                fractions[i] = np.interp(np.linspace(0, count[-1], n + 1), count, samples)
    n_sub[~keep] = 0
    index = np.repeat(np.arange(len(widths)), n_sub)
    k = np.arange(len(index)) - np.repeat(np.cumsum(n_sub) - n_sub, n_sub)
    lo = k / n_sub[index]
    hi = (k + 1) / n_sub[index]
    for i, edges in fractions.items():  # Unequal pieces
        sel = index == i
        lo[sel], hi[sel] = edges[:-1], edges[1:]
    frac = (lo + hi) / 2
    E = e[index] + frac * widths[index]
    if lep == 1:
        p = probs[index] * (hi - lo)
    else:
        dens = f[index] + frac * (f[index + 1] - f[index])
        p = dens * widths[index] * (hi - lo)
    total = probs.sum()
    return E, (p / total if total > 0 else p), index


def _cm_energy_step(beta, angle_bin, energy_bin):
    """Return a function giving the widest CM outgoing-energy piece [MeV] that keeps the lab
    angle of its neutrons within CONTINUUM_ANGLE_SHIFT angle bins, at CM energies eps [MeV].

    With the CM moving at speed V (kinetic energy E_c = m_n V^2 / 2 for a neutron) and a
    neutron leaving it at speed u (energy eps) at CM angle theta, the lab angle phi has
    tan phi = u sin theta / (V + u cos theta), and d phi / d eps = sqrt(E_c / eps) sin theta /
    (2 E_lab), with E_lab the neutron's lab energy. It is largest for slow lab neutrons, so the
    piece width is 2 a angle_bin E_lab sqrt(eps / E_c), with E_lab the lowest the piece can give,
    (sqrt(eps) - sqrt(E_c))^2, but at least half an energy bin (the angles of the slowest
    neutrons, all in the lowest bin, need not be finer), and a = CONTINUUM_ANGLE_SHIFT.
    Nonrelativistic: the CM moves slowly, so this is only used to choose the piece widths.
    """
    E_c = 0.5 * M_NEUTRON * beta**2
    a = CONTINUUM_ANGLE_SHIFT * angle_bin
    floor = (a / 2) ** 2 * E_c  # Where the formula gives zero: the CM energy is 0

    def step(eps):
        eps = np.maximum(eps, 0.0)
        if E_c <= 0:
            return np.full(np.shape(eps), np.inf)
        E_lab = np.maximum((np.sqrt(eps) - np.sqrt(E_c)) ** 2, energy_bin / 2)
        return np.maximum(2 * a * E_lab * np.sqrt(eps / E_c), floor)

    return step


class _Continuum:
    """LAW=1 at one incident energy: outgoing energies [MeV], f0 [1/MeV] and the angular
    parameters (Kalbach r and maybe a, or Legendre coefficients)."""

    def __init__(self, energies, f0, lep, kind, params):
        self.energies = energies
        self.f0 = f0
        self.lep = lep
        self.kind = kind  # "kalbach", "legendre" or "isotropic"
        self.params = params

    @property
    def top(self):
        return self.energies[-1]


class _TwoBody:
    """LAW=2 at one incident energy: the CM angular distribution of the neutron, as Legendre
    coefficients a_1..a_NL (LANG=0) or a table of (cosine, density) (LANG=12)."""

    def __init__(self, lang, values):
        self.lang = lang
        self.values = np.asarray(values, dtype=float)

    def cumulative(self, x):
        """Probability of a CM cosine between -1 and x (any array shape), normalized so that
        the whole of [-1, 1] has probability 1."""
        x = np.asarray(x, dtype=float)
        if self.lang == 0:
            coeffs = np.concatenate(([1.0], self.values))
            series = np.polynomial.legendre.Legendre(coeffs * (2 * np.arange(len(coeffs)) + 1) / 2)
            return series.integ(lbnd=-1)(x)
        # Tabulated, linear-linear: integrate the piecewise-linear density exactly
        mu, p = self.values[0::2], self.values[1::2]
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(mu))))
        if cum[-1] <= 0:  # No shape given: isotropic
            return (np.clip(x, -1, 1) + 1) / 2
        x = np.clip(x, mu[0], mu[-1])
        i = np.clip(np.searchsorted(mu, x, side="right") - 1, 0, len(mu) - 2)
        t = x - mu[i]
        slope = (p[i + 1] - p[i]) / np.where(mu[i + 1] > mu[i], mu[i + 1] - mu[i], 1)
        return (cum[i] + p[i] * t + 0.5 * slope * t**2) / cum[-1]


class _LabTable:
    """LAW=7 at one incident energy: at each lab cosine (increasing), an outgoing-energy
    distribution (Tabulated, density per unit cosine and MeV)."""

    def __init__(self, cosines, spectra):
        self.cosines = np.asarray(cosines, dtype=float)
        self.spectra = spectra

    @property
    def top(self):
        return max(s.x[-1] for s in self.spectra)


class _Channel:
    """One reaction channel's neutron production: cross section [b] and neutron yield vs
    incident energy [eV], and the distributions at the tabulated incident energies [MeV]."""

    def __init__(self, mt, law, xs, yield_, frame, incident, distributions, unit_base, q=0.0):
        self.mt = mt
        self.law = law
        self.xs = xs
        self.yield_ = yield_
        self.frame = frame  # "cm" or "lab"
        self.incident = incident
        self.distributions = distributions
        self.unit_base = unit_base
        self.q = q  # QI [MeV], for two-body channels

    def production(self, E):
        """Neutron production cross section [b] at incident energy E [MeV]."""
        return float(self.xs(E * 1e6)[0] * self.yield_(E * 1e6)[0])

    def mixture(self, E):
        """Return [(weight, distribution, energy scale)]: the distribution at E [MeV] as a
        mixture of the two tabulated ones around it (INT=2), each with the factor its
        outgoing energies are stretched by (1, or for unit base, INT=22, the ratio of the
        interpolated to its own top outgoing energy)."""
        inc = self.incident
        if E <= inc[0]:
            return [(1.0, self.distributions[0], 1.0)]
        if E >= inc[-1]:
            return [(1.0, self.distributions[-1], 1.0)]
        k = np.searchsorted(inc, E, side="right") - 1
        w = (E - inc[k]) / (inc[k + 1] - inc[k])
        lo, hi = self.distributions[k], self.distributions[k + 1]
        if not self.unit_base or self.law == 2:
            return [(1 - w, lo, 1.0), (w, hi, 1.0)]
        top = (1 - w) * lo.top + w * hi.top
        return [(1 - w, lo, top / lo.top if lo.top > 0 else 1.0),
                (w, hi, top / hi.top if hi.top > 0 else 1.0)]  # fmt: skip


class NeutronProduction:
    """The neutron production of one beam particle on one target nuclide, from an ENDF-6
    evaluation, with every neutron-producing channel merged.

    Attributes: beam (BEAMS key), target (Z, A), awr (target mass / neutron mass), awi
    (projectile mass / neutron mass), emax [MeV] (top of the evaluation), channels (list of
    _Channel), nonelastic (Tabulated [b] vs incident energy [eV]), channel_sum (the same,
    summed over the explicit channels; less than nonelastic if channels are missing), path.
    """

    def __init__(self, path):
        self.path = str(path)
        sections = _sections(path)
        if (1, 451) not in sections:
            raise ValueError(f"{path}: not an ENDF-6 file (no MF=1, MT=451 header)")
        r = _Records(sections[(1, 451)])
        c1 = r.cont()
        r.cont()
        c3 = r.cont()
        za = int(c1[0])
        self.target = (za // 1000, za % 1000)
        self.awr = c1[1]
        self.awi = c3[0]
        self.emax = c3[1] / 1e6
        nsub = int(c3[4])
        if nsub not in NSUB_BEAMS:
            raise ValueError(f"{path}: incident particle (NSUB {nsub}) is not one of the beams")
        self.beam = NSUB_BEAMS[nsub]

        mf3 = {mt: _Records(lines).mf3() for (mf, mt), lines in sections.items() if mf == 3}
        present = set(mf3)
        redundant = {mt for mt, parts in SUMS.items() if present & set(parts)}
        mf6 = sorted(mt for mf, mt in sections if mf == 6)
        has_levels = any(50 <= mt <= 91 for mt in mf6)

        self.channels = []
        for mt in mf6:
            if mt == 2 or (mt == 4 and has_levels):  # Elastic; or a sum of the level channels
                continue
            self.channels.extend(self._read_channels(mt, sections[(6, mt)], mf3))

        parts = [
            xs for mt, (_, _, xs) in mf3.items() if mt not in (1, 2, 3) and mt not in redundant
        ]
        grid = np.unique(np.concatenate([p.x for p in parts])) if parts else np.array([0.0, 1.0])
        self.channel_sum = Tabulated(grid, sum(p(grid) for p in parts), [len(grid)], [2])
        self.nonelastic = mf3[3][2] if 3 in mf3 else self.channel_sum

    def _read_channels(self, mt, lines, mf3):
        """Return a _Channel for each neutron product of an MF=6 section (none, one, or
        several: an evaluation can list the neutrons more than once, each over its own range
        of incident energies, as FENDL-3.2c p + 7Li does below and above 10 MeV)."""
        r = _Records(lines)
        head = r.cont()
        lct, nk = int(head[3]), int(head[4])
        channels = []
        for _ in range(nk):
            c, yield_ = r.tab1()
            zap, law = int(c[0]), int(c[3])
            if zap != 1:
                self._skip_law(r, law, mt)
                continue
            if mt not in mf3:
                raise ValueError(f"{self.path}: MT={mt} has neutrons but no cross section")
            _, qi, xs = mf3[mt]
            frame = "lab" if lct == 1 else "cm"
            if law == 1:
                incident, dists, unit_base = self._read_law1(r, mt)
            elif law == 2:
                incident, dists, unit_base = self._read_law2(r, mt)
                frame = "cm"  # LAW=2 is always in the center-of-mass frame
            elif law == 7:
                incident, dists, unit_base = self._read_law7(r, mt)
                frame = "lab"
            else:
                raise NotImplementedError(
                    f"{self.path}: MT={mt} neutrons use LAW={law}; only LAW=1, 2 and 7 are "
                    "supported"
                )
            channels.append(_Channel(mt, law, xs, yield_, frame, incident, dists, unit_base, q=qi))
        return channels

    def _incident_interpolation(self, codes, mt):
        if set(codes) <= {2}:
            return False
        if set(codes) <= {2, 22}:
            return 22 in codes
        raise NotImplementedError(
            f"{self.path}: MT={mt} interpolates between incident energies with INT={codes}; "
            "only 2 (linear-linear) and 22 (unit base) are supported"
        )

    def _read_law1(self, r, mt):
        c, _, codes = r.tab2()
        lang, lep, ne = int(c[2]), int(c[3]), int(c[5])
        # Always by unit base, even where the evaluation says INT=2, as MCNP does (see the
        # module docstring)
        self._incident_interpolation(codes, mt)
        unit_base = True
        if lep not in (1, 2):
            raise NotImplementedError(f"{self.path}: MT={mt} uses LEP={lep}")
        incident, dists = [], []
        for _ in range(ne):
            cc = r.cont()
            E, nd, na, nw, nep = cc[1], int(cc[2]), int(cc[3]), int(cc[4]), int(cc[5])
            values = np.array(r.values(nw)).reshape(nep, na + 2)
            if nd:
                raise NotImplementedError(
                    f"{self.path}: MT={mt} has discrete outgoing energies (ND={nd})"
                )
            if na and lang not in (1, 2, 3):
                raise NotImplementedError(f"{self.path}: MT={mt} uses LANG={lang}")
            if lang == 3 and na != 2:
                raise NotImplementedError(f"{self.path}: MT={mt} uses LANG=3 with NA={na}")
            kind = "isotropic" if na == 0 else {1: "legendre", 2: "kalbach", 3: "kalbach"}[lang]
            dists.append(
                _Continuum(values[:, 0] / 1e6, values[:, 1] * 1e6, lep, kind, values[:, 2:])
            )
            incident.append(E / 1e6)
        return np.array(incident), dists, unit_base

    def _read_law2(self, r, mt):
        c, _, codes = r.tab2()
        self._incident_interpolation(codes, mt)
        incident, dists = [], []
        for _ in range(int(c[5])):
            cc = r.cont()
            lang, nw = int(cc[2]), int(cc[4])
            values = r.values(nw)
            if lang not in (0, 12):
                raise NotImplementedError(
                    f"{self.path}: MT={mt} two-body angular distribution LANG={lang}; only 0 "
                    "(Legendre) and 12 (linear table) are supported"
                )
            dists.append(_TwoBody(lang, values))
            incident.append(cc[1] / 1e6)
        return np.array(incident), dists, False

    def _read_law7(self, r, mt):
        c, _, codes = r.tab2()
        unit_base = self._incident_interpolation(codes, mt)
        incident, dists = [], []
        for _ in range(int(c[5])):
            cm, _, mu_codes = r.tab2()
            if set(mu_codes) - {2}:
                raise NotImplementedError(f"{self.path}: MT={mt} LAW=7 cosine INT={mu_codes}")
            cosines, spectra = [], []
            for _ in range(int(cm[5])):
                cc, f = r.tab1()
                if set(f.codes) not in ({1}, {2}):  # All histogram or all linear-linear
                    raise NotImplementedError(f"{self.path}: MT={mt} LAW=7 energy INT={f.codes}")
                cosines.append(cc[1])
                spectra.append(Tabulated(f.x / 1e6, f.y * 1e6, f.breaks, f.codes))
            dists.append(_LabTable(cosines, spectra))
            incident.append(cm[1] / 1e6)
        return np.array(incident), dists, unit_base

    @staticmethod
    def _skip_law(r, law, mt):
        """Step over the law-dependent data of a product that is not a neutron."""
        if law in (0, 3, 4) or law < 0:
            return
        if law in (1, 2, 5):
            c, _, _ = r.tab2()
            for _ in range(int(c[5])):
                cc = r.cont()
                r.values(int(cc[4]))
        elif law == 6:
            r.cont()
        elif law == 7:
            c, _, _ = r.tab2()
            for _ in range(int(c[5])):
                c2, _, _ = r.tab2()
                for _ in range(int(c2[5])):
                    r.tab1()
        else:
            raise NotImplementedError(f"MT={mt}: a product uses LAW={law}")

    # --- Cross sections --------------------------------------------------------------------

    def production_cross_section(self, E):
        """Neutron production cross section [b] (neutrons per reaction times cross section,
        summed over the channels) at incident energy E [MeV]."""
        return sum(ch.production(E) for ch in self.channels)

    def nonelastic_cross_section(self, E):
        """Cross section [b] for removing a beam particle by a nuclear reaction at E [MeV]."""
        return self.nonelastic(np.asarray(E) * 1e6)

    def missing_fraction(self, E):
        """Fraction of the non-elastic cross section at E [MeV] not covered by the explicit
        channels (0 for a complete file; their neutrons would be missing)."""
        total = self.nonelastic(E * 1e6)[0]
        if total <= 0 or self.nonelastic is self.channel_sum:
            return 0.0
        return max(0.0, 1 - self.channel_sum(E * 1e6)[0] / total)

    def missing_share(self, E_max=None):
        """Share of the non-elastic cross section not covered by the explicit channels,
        integrated over incident energy up to E_max [MeV] (default: the top of the
        evaluation); 0 for a complete file."""
        if self.nonelastic is self.channel_sum:
            return 0.0
        top = (self.emax if E_max is None else min(E_max, self.emax)) * 1e6
        grid = np.union1d(self.nonelastic.x, self.channel_sum.x)
        grid = np.append(grid[grid < top], top)
        total = self.nonelastic(grid)
        gap = np.maximum(0.0, total - self.channel_sum(grid))
        dx = np.diff(grid)
        whole = np.sum(0.5 * (total[1:] + total[:-1]) * dx)
        return float(np.sum(0.5 * (gap[1:] + gap[:-1]) * dx) / whole) if whole > 0 else 0.0

    def kalbach_slope(self, E_a, E_b):
        """The Kalbach-Mann slope a for neutrons emitted with CM energy E_b [MeV] (array)
        by the beam at lab energy E_a [MeV] (ENDF-102 section 6.2.3.2)."""
        beam = BEAMS[self.beam]
        Z_a, A_a = beam.Z, beam.A
        Z_t, A_t = self.target
        Z_c, A_c = Z_t + Z_a, A_t + A_a
        eps_a = E_a * self.awr / (self.awr + self.awi)
        awr_B = self.awr + self.awi - 1.0
        eps_b = np.asarray(E_b) * (awr_B + 1.0) / awr_B
        S_a = separation_energy(Z_c, A_c, Z_t, A_t, (Z_a, A_a))
        S_b = separation_energy(Z_c, A_c, Z_c, A_c - 1, (0, 1))
        e_a = eps_a + S_a
        e_b = eps_b + S_b
        x1 = min(e_a, 130.0) * e_b / e_a
        x3 = min(e_a, 41.0) * e_b / e_a
        M, m = KALBACH_M[self.beam], KALBACH_M_NEUTRON
        return 0.04 * x1 + 1.8e-6 * x1**3 + 6.7e-7 * M * m * x3**4

    # --- Lab-frame spectra -----------------------------------------------------------------

    def _boost(self, E):
        """Return (beta, gamma) of the beam + target center of mass at beam energy E [MeV]."""
        m1, m2 = BEAMS[self.beam].mass, nuclear_mass(*self.target)
        p1 = np.sqrt(E * (E + 2 * m1))
        beta = p1 / (E + m1 + m2)
        return beta, 1 / np.sqrt(1 - beta**2)

    @property
    def has_two_body(self):
        """Whether any channel is two-body (LAW=2: discrete levels)."""
        return any(c.law == 2 for c in self.channels)

    def lab_spectrum(
        self, E, angle_edges, energy_edges, num_cm_angles=720, max_step=None, two_body=True
    ):
        """Return the neutron production cross section [b] in each (lab angle, lab energy)
        bin at beam energy E [MeV], and the total [b].

        Args:
            E: Beam (lab) energy [MeV].
            angle_edges: Lab emission-angle bin edges [deg], from 0 to 180.
            energy_edges: Lab neutron-energy bin edges [MeV], evenly spaced.
            num_cm_angles: Number of angle cells (even in angle) over which each continuum
                distribution is integrated exactly and placed at its middle angle.
            max_step: Largest outgoing-energy step [MeV] at which continuum distributions
                are split (default: half an output energy bin).
            two_body: Include the two-body (LAW=2) channels; leave them out to calculate
                them separately with two_body_spectrum.

        Each piece of each distribution (outgoing energy, angle cell) is converted to the
        lab (boosted from the center of mass of beam + target with relativistic kinematics,
        unless given in the lab) and added to the lab bin it lands in. The total is exact:
        the bins add up to the production cross section; the placement is good to about the
        size of the pieces. The two-body channels are binned exactly (see two_body_spectrum).
        """
        angle_edges = np.asarray(angle_edges, dtype=float)
        energy_edges = np.asarray(energy_edges, dtype=float)
        width = energy_edges[1] - energy_edges[0]
        shape = (len(angle_edges) - 1, len(energy_edges) - 1)
        if max_step is None:
            max_step = width / 2
        out = np.zeros(shape)
        beta, gamma = self._boost(E)
        theta_edges = np.linspace(0, np.pi, num_cm_angles + 1)
        mu_edges = np.cos(theta_edges)  # Decreasing
        mu_mid = np.cos(0.5 * (theta_edges[:-1] + theta_edges[1:]))
        grid = (angle_edges, energy_edges, width, shape)

        total = 0.0
        for channel in self.channels:
            if channel.law == 2:
                continue  # Below, with two_body_spectrum
            sigma = channel.production(E)
            if sigma <= 0:
                continue
            for weight, dist, scale in channel.mixture(E):
                if weight <= 0:
                    continue
                w = sigma * weight
                if channel.law == 7:
                    out += self._bin_lab_table(dist, scale, w, max_step, grid)
                else:
                    out += self._bin_continuum(
                        dist,
                        scale,
                        w,
                        E,
                        channel.frame,
                        beta,
                        gamma,
                        mu_edges,
                        mu_mid,
                        max_step,
                        grid,
                    )
                total += w
        if two_body and self.has_two_body:
            lines, sigma = self.two_body_spectrum([E], [1.0], [0], 1, angle_edges, energy_edges)
            out += lines[0]
            total += sigma[0]
        return out, total

    def two_body_spectrum(self, E, weights, rows, num_rows, angle_edges, energy_edges):
        """Bin the neutrons of the two-body (LAW=2) channels at many beam energies at once,
        each with its own kinematics: no interpolation between beam energies, so the sharp
        line each level makes at one beam energy is smeared exactly as the beam slows down.
        Each line is split where it crosses the lab bin edges and binned exactly (see
        nuclear.bin_two_body), with the angular distribution interpolated linearly in the
        incident energy (INT=2) as in _Channel.mixture.

        Args:
            E: Beam (lab) energies [MeV], one per internal bin.
            weights: Weight of each, multiplying the cross section [b] (n_target dx times
                the beam fraction, for neutrons per beam particle).
            rows: Output position row of each.
            num_rows: Number of output position rows.
            angle_edges, energy_edges: As for lab_spectrum.

        Returns (out, sigma): out with shape (num_rows, angle bins, energy bins), the
        weighted neutrons, and sigma, the two-body production cross section [b] at each beam
        energy (zero where a level is below its kinematic threshold or the weight is zero).
        """
        E = np.atleast_1d(np.asarray(E, dtype=float))
        weights = np.broadcast_to(np.asarray(weights, dtype=float), E.shape)
        shape = (num_rows, len(angle_edges) - 1, len(energy_edges) - 1)
        out = np.zeros(shape)
        sigma = np.zeros(len(E))
        m1, m2 = BEAMS[self.beam].mass, nuclear_mass(*self.target)
        for channel in self.channels:
            if channel.law != 2:
                continue
            sig = channel.xs(E * 1e6) * channel.yield_(E * 1e6)
            inc = channel.incident
            if len(inc) > 1:
                k = np.clip(np.searchsorted(inc, E, side="right") - 1, 0, len(inc) - 2)
                t = np.clip((E - inc[k]) / (inc[k + 1] - inc[k]), 0, 1)
            else:
                k, t = np.zeros(len(E), dtype=int), np.zeros(len(E))
            k1 = np.minimum(k + 1, len(inc) - 1)

            def cumulative(index, x, sig=sig, k=k, k1=k1, t=t, channel=channel):
                F = np.zeros_like(x)
                lo, hi, w = k[index], k1[index], t[index]
                for d in np.unique(np.concatenate((lo, hi))):
                    mix = np.where(lo == d, 1 - w, 0.0) + np.where(hi == d, w, 0.0)
                    use = mix > 0
                    if np.any(use):
                        F[use] += mix[use, None] * channel.distributions[d].cumulative(x[use])
                return F * sig[index, None]

            m4 = m1 + m2 - M_NEUTRON - channel.q  # Residual, in its level
            lines, s = bin_two_body(
                E, np.where(sig > 0, weights, 0.0), rows, num_rows, angle_edges, energy_edges,
                (m1, m2, M_NEUTRON, m4), cumulative,
            )  # fmt: skip
            out += lines
            sigma += s
        return out, sigma

    @staticmethod
    def _bin(E_lab, cos_lab, prob, grid):
        """Add pieces (lab energy [MeV], lab cosine, weight [b]) into the lab bins."""
        angle_edges, energy_edges, width, shape = grid
        angle = np.degrees(np.arccos(np.clip(cos_lab, -1, 1)))
        steps = np.diff(angle_edges)
        if np.allclose(steps, steps[0], rtol=1e-12, atol=0):  # Even bins: no search needed
            j = np.clip((angle / steps[0]).astype(int), 0, shape[0] - 1)
        else:
            j = np.clip(np.searchsorted(angle_edges, angle, side="right") - 1, 0, shape[0] - 1)
        k = np.clip(np.floor((E_lab - energy_edges[0]) / width).astype(int), 0, shape[1] - 1)
        flat = np.bincount((j * shape[1] + k).ravel(), prob.ravel(), minlength=shape[0] * shape[1])
        return flat.reshape(shape)

    def _bin_continuum(
        self, dist, scale, w, E, frame, beta, gamma, mu_edges, mu_mid, max_step, grid
    ):
        """Bin a LAW=1 distribution (weight w [b]) into the lab bins.

        Each outgoing-energy piece is placed at its middle energy. In the center-of-mass (CM)
        frame that is too coarse where the neutrons are slow in the lab: there the lab angle
        changes quickly with the CM energy, so pieces max_step wide land in separate angle
        bins (jagged angular distributions at low lab energies). The pieces are made narrower
        there, as needed (see _cm_energy_step).
        """
        step = None
        if frame != "lab":
            angle_edges, energy_edges, width, _ = grid
            step = _cm_energy_step(beta, np.radians(np.min(np.diff(angle_edges))), width)
        Eb, p, index = _energy_points(
            dist.energies * scale, dist.f0 / scale, dist.lep, max_step, step
        )
        if len(Eb) == 0:
            return 0.0
        cells = self._angular_cells(dist, index, Eb, E, mu_edges)  # (pieces, cells)
        prob = (w * p)[:, None] * cells
        if frame == "lab":
            E_lab = np.repeat(Eb[:, None], len(mu_mid), axis=1)
            cos_lab = np.repeat(mu_mid[None, :], len(Eb), axis=0)
        else:
            pb = np.sqrt(Eb * (Eb + 2 * M_NEUTRON))[:, None]
            Etot = (Eb + M_NEUTRON)[:, None]
            p_par = gamma * (pb * mu_mid[None, :] + beta * Etot)
            p_perp = pb * np.sqrt(np.clip(1 - mu_mid**2, 0, None))[None, :]
            E_lab = gamma * (Etot + beta * pb * mu_mid[None, :]) - M_NEUTRON
            cos_lab = p_par / np.sqrt(p_par**2 + p_perp**2)
        return self._bin(E_lab, cos_lab, prob, grid)

    def _bin_lab_table(self, dist, scale, w, max_step, grid):
        """Bin a LAW=7 distribution (weight w [b]) into the lab bins. Between tabulated
        cosines the density is linear in the cosine (INT=2): a cosine cell between mu_k and
        mu_k+1 holds (1 - t) of the spectrum at mu_k and t of the one at mu_k+1, integrated
        over the cell. The cells are the intervals between tabulated cosines split at the
        output angle-bin edges, so each lies in one output bin and the integral over each bin
        is exact. (Cells of a fixed width, placed by their midpoints, gave some bins one cell
        more than their neighbors: a regular stripe of up to 25% in the angular distribution.)"""
        angle_edges, energy_edges, width, shape = grid
        edge_angles = np.radians(angle_edges)
        spectra = []
        for s in dist.spectra:  # Each spectrum in the lab energy bins (unnormalized)
            e = s.x * scale
            f = s.y / scale
            lep = 1 if set(s.codes) == {1} else 2  # Histogram (INT=1) or linear (INT=2)
            Eb, p, _ = _energy_points(e, f, lep, max_step)
            if lep == 1:
                norm = np.sum(f[:-1] * np.diff(e))
            else:
                norm = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(e))
            k = np.clip(np.floor((Eb - energy_edges[0]) / width).astype(int), 0, shape[1] - 1)
            spectra.append(np.bincount(k, p * norm, minlength=shape[1]))
        out = np.zeros(shape)
        mu = dist.cosines
        for i in range(len(mu) - 1):
            a, b = mu[i], mu[i + 1]
            if b <= a:
                continue
            theta_a, theta_b = np.arccos(a), np.arccos(b)  # theta_a > theta_b
            inside = edge_angles[(edge_angles > theta_b) & (edge_angles < theta_a)]
            thetas = np.concatenate(([theta_a], np.sort(inside)[::-1], [theta_b]))
            cell_mu = np.cos(thetas)  # Increasing
            lo, hi = cell_mu[:-1], cell_mu[1:]
            # Integrals over the cell of (1 - t) and t, with t = (mu - a) / (b - a)
            t_int = ((hi - a) ** 2 - (lo - a) ** 2) / (2 * (b - a))
            one_minus = (hi - lo) - t_int
            mid = np.degrees(np.arccos(0.5 * (lo + hi)))
            j = np.clip(np.searchsorted(angle_edges, mid, side="right") - 1, 0, shape[0] - 1)
            for jj, wa, wb in zip(j, one_minus, t_int):
                out[jj] += wa * spectra[i] + wb * spectra[i + 1]
        total = out.sum()
        return out * (w / total) if total > 0 else out

    def _angular_cells(self, dist, index, Eb, E, mu_edges):
        """Return the probability of each CM angle cell (between mu_edges, decreasing) for
        each piece of a LAW=1 distribution: exact integrals of the angular distribution."""
        n = len(Eb)
        dmu = mu_edges[:-1] - mu_edges[1:]  # Positive
        if dist.kind == "isotropic":
            return np.repeat((dmu / 2)[None, :], n, axis=0)
        if dist.kind == "legendre":
            # f(mu) = sum_l (2l+1)/2 f_l/f0 P_l(mu); integrate each P_l over the cell
            f0 = dist.f0[index]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(f0[:, None] > 0, dist.params[index] / f0[:, None], 0.0)
            cells = np.repeat((dmu / 2)[None, :], n, axis=0)
            for order in range(1, ratio.shape[1] + 1):
                c = np.zeros(order + 1)
                c[order] = 1.0
                F = np.polynomial.legendre.legval(mu_edges, np.polynomial.legendre.legint(c))
                cells += (
                    (2 * order + 1) / 2 * ratio[:, order - 1 : order] * (F[None, :-1] - F[None, 1:])
                )
            return cells
        # Kalbach-Mann: f = a / (2 sinh a) [cosh(a mu) + r sinh(a mu)]
        r = dist.params[index, 0]
        a = dist.params[index, 1] if dist.params.shape[1] >= 2 else self.kalbach_slope(E, Eb)
        a = np.maximum(a, 1e-8)[:, None]
        r = r[:, None]
        # The integral of f up to each edge, from one exponential per edge (cells share edges):
        # (sinh(a mu) + r cosh(a mu)) / (2 sinh a), with sinh and cosh from exp(+-a mu)
        up = np.exp(a * mu_edges[None, :])
        down = 1.0 / up
        F = ((1 + r) * up - (1 - r) * down) / (4 * np.sinh(a))
        return F[:, :-1] - F[:, 1:]

    def max_lab_energy(self, E):
        """Highest lab neutron energy [MeV] at beam energy E [MeV]."""
        top = 0.0
        beta, gamma = self._boost(E)
        m1, m2 = BEAMS[self.beam].mass, nuclear_mass(*self.target)
        for channel in self.channels:
            if channel.production(E) <= 0:
                continue
            if channel.law == 2:
                T3, _ = cm_to_lab(E, [1.0], m1, m2, M_NEUTRON, m1 + m2 - M_NEUTRON - channel.q)
                if not np.isnan(T3[0]):
                    top = max(top, T3[0])
                continue
            for _, dist, scale in channel.mixture(E):
                Eb = dist.top * scale
                if channel.frame == "lab":
                    top = max(top, Eb)
                else:
                    pb = np.sqrt(Eb * (Eb + 2 * M_NEUTRON))
                    top = max(top, gamma * (Eb + M_NEUTRON + beta * pb) - M_NEUTRON)
        return top

    @property
    def label(self):
        """Such as "9Be(p,xn)"."""
        name = "H" if self.target == (1, 1) else atom_name(*self.target)
        return f"{name}({self.beam},xn)"


# File names of the libraries, parsed into (beam, symbol, A, rank): TENDL
# (p-Be009.tendl; rank by FILE_SUFFIXES, metastable targets such as p-Ac222m are not
# used), ENDF/B (p-004_Be_009.endf) and FENDL (p_0409_4-Be-9.endf)
_TENDL_NAME = re.compile(r"^([pdtha])-([A-Z][a-z]?)(\d{3})(-mt|-s60|-s20|-s0|)\.tendl$")
_ENDFB_NAME = re.compile(r"^([pdtha])-(\d{3})_([A-Z][a-z]?)_(\d{3})\.endf$")
_FENDL_NAME = re.compile(r"^([pdtha])_\d+_(\d+)-([A-Z][a-z]?)-(\d+)\.endf$")


def _parse_name(name):
    """Return (beam, symbol, A, rank) for an evaluation file name, or None."""
    m = _TENDL_NAME.match(name)
    if m:
        rank = FILE_SUFFIXES.index(m.group(4) + ".tendl")
        return m.group(1), m.group(2), int(m.group(3)), rank
    m = _ENDFB_NAME.match(name)
    if m:
        return m.group(1), m.group(3), int(m.group(4)), 0
    m = _FENDL_NAME.match(name)
    if m:
        return m.group(1), m.group(3), int(m.group(4)), 0
    return None


@functools.lru_cache(maxsize=16)
def library_index(directory):
    """Return {(beam, symbol, A): [path, ...]} for the evaluation files in a directory and
    its subdirectories (indexed once per run), recognizing TENDL, ENDF/B, and FENDL names.
    Several paths are TENDL variants of one evaluation, in FILE_SUFFIXES order."""
    found = {}
    for root, _, files in os.walk(directory):
        for name in files:
            parsed = _parse_name(name)
            if parsed is None:
                continue
            beam, symbol, A, rank = parsed
            found.setdefault((beam, symbol, A), []).append((rank, os.path.join(root, name)))
    return {key: [path for _, path in sorted(paths)] for key, paths in found.items()}


def choose_variant(paths, E_max=None):
    """Return the variant to use of several TENDL variants of one evaluation (paths in
    FILE_SUFFIXES order). Of the standard, -mt and -s60 files (or, without any, of the
    others), the most complete up to E_max [MeV] (default: the top of each): the one whose
    explicit channels leave the least of the non-elastic cross section uncovered, integrated
    over incident energy, and the first among those within VARIANT_TIE of it. Variants that
    cannot be read are passed over (the error is raised if none can be)."""
    split = [p for p in paths if _parse_name(os.path.basename(p))[3] < NUM_SPLIT]
    paths = split or paths
    if len(paths) == 1:
        return paths[0]
    shares, error = {}, None
    for path in paths:
        try:
            shares[path] = load_evaluation(path).missing_share(E_max)
        except (ValueError, NotImplementedError) as e:
            error = error or e
    if not shares:
        raise error
    best = min(shares.values())
    return next(p for p in paths if p in shares and shares[p] <= best + VARIANT_TIE)


def find_evaluation(directory, beam, target, E_max=None):
    """Return the path of the evaluation for a beam on a target (Z, A) in a library
    directory, or None. Of several TENDL variants, the most complete up to the beam energy
    E_max [MeV] is used (choose_variant)."""
    if directory is None or not os.path.isdir(directory):
        return None
    Z, A = target
    paths = library_index(os.path.abspath(directory)).get((beam, atom_name(Z), A))
    return choose_variant(paths, E_max) if paths else None


def describe_evaluation(path):
    """Return a one-line description of an evaluation: the file's label (TPID) and its
    origin line (target, laboratory, evaluation date, and authors), from the header."""
    with open(path, encoding="ascii", errors="replace") as f:
        lines = [next(f, "") for _ in range(6)]
    tpid = lines[0][:66].strip()
    text = lines[5][:66].strip() if len(lines) > 5 else ""
    return " | ".join(part for part in (tpid, text) if part)


@functools.lru_cache(maxsize=32)
def _load(path, mtime):
    return NeutronProduction(path)


def load_evaluation(path):
    """Read an evaluation (cached while the file is unchanged)."""
    return _load(str(path), os.path.getmtime(path))


class EndfReaction:
    """The neutron production of a beam on one target isotope from an evaluated file, as a
    reaction of BeamTarget (alongside the DROSG two-body reactions of nuclear.REACTIONS).

    Attributes: key (such as "p9Be_xn"), label (such as "9Be(p,xn)"), beam, target (Z, A),
    path, library (the nuclear_data entry it was found in), evaluation (see
    describe_evaluation) and production (NeutronProduction).
    """

    def __init__(self, path, library=""):
        self.path = str(path)
        self.library = str(library)
        self.production = load_evaluation(self.path)
        self.beam = self.production.beam
        self.target = self.production.target
        name = "H" if self.target == (1, 1) else atom_name(*self.target)
        self.key = f"{self.beam}{name}_xn"
        self.label = self.production.label
        self.evaluation = describe_evaluation(self.path)

    def __repr__(self):
        return f"EndfReaction({self.label}, {self.path!r})"
