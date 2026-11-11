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
  linear (LEP=2); centre-of-mass (LCT=2, 3, 4) or lab (LCT=1) frame.
- LAW=2, discrete two-body reactions (such as (p,n) to one level of the residual nucleus):
  Legendre (LANG=0) or tabulated (LANG=12) centre-of-mass angular distributions; the
  neutron energy follows from the kinematics, with the channel's Q value (QI in MF=3).
- LAW=7, lab-frame angle-energy tables: an outgoing-energy distribution at each of a set of
  lab cosines.

Between tabulated incident energies, distributions are interpolated linearly (INT=2) or by
unit base (INT=22: each stretched to the outgoing-energy range at the wanted energy).
Anything else is refused with a message naming it.

Reference: ENDF-6 Formats Manual, BNL-203218-2023-INRE (ENDF-102).
"""

import functools
import os
import re

import numpy as np

from .nuclear import BEAMS, atom_name, cm_to_lab, nuclear_mass

# ENDF sub-library number (NSUB) of each beam particle's incident data
NSUB_BEAMS = {10010: "p", 10020: "d", 10030: "t", 20030: "h", 20040: "a"}

# Evaluation file names tried for a beam on a target, in order: TENDL's file with every
# channel explicit ("-mt"), then its other variants. {beam}-{symbol}{A:03d}, such as
# p-Be009-mt.tendl.
FILE_SUFFIXES = ("-mt.tendl", ".tendl", "-s60.tendl", "-s20.tendl", "-s0.tendl")

M_NEUTRON = nuclear_mass(0, 1)  # MeV

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
    """Read one 11-character ENDF number, such as " 1.234567+3" or "-2.5-12"."""
    s = field.strip()
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


def _energy_points(energies, f0, lep, max_step):
    """Split an outgoing-energy distribution (energies [MeV], density f0 [1/MeV], histogram
    for lep 1 or linear for lep 2) into pieces at most max_step [MeV] wide. Return (energy
    of each piece [MeV], its probability, index of the tabulated point below), normalised
    to a total of 1."""
    e, f = energies, f0
    widths = np.diff(e)
    probs = f[:-1] * widths if lep == 1 else 0.5 * (f[:-1] + f[1:]) * widths
    keep = probs > 0
    n_sub = np.maximum(1, np.ceil(widths / max_step).astype(int))
    n_sub[~keep] = 0
    index = np.repeat(np.arange(len(widths)), n_sub)
    k = np.arange(len(index)) - np.repeat(np.cumsum(n_sub) - n_sub, n_sub)
    frac = (k + 0.5) / n_sub[index]
    E = e[index] + frac * widths[index]
    if lep == 1:
        p = probs[index] / n_sub[index]
    else:
        dens = f[index] + frac * (f[index + 1] - f[index])
        p = dens * widths[index] / n_sub[index]
        sums = np.bincount(index, p, minlength=len(widths))
        p = p * np.where(
            sums[index] > 0, probs[index] / np.where(sums[index] > 0, sums[index], 1), 0
        )
    total = probs.sum()
    return E, (p / total if total > 0 else p), index


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

    def cell_probabilities(self, mu_edges):
        """Probability of each cosine cell between mu_edges (decreasing), normalised to 1."""
        if self.lang == 0:
            coeffs = np.concatenate(([1.0], self.values))
            series = np.polynomial.legendre.Legendre(coeffs * (2 * np.arange(len(coeffs)) + 1) / 2)
            F = series.integ()(mu_edges)
            probs = F[:-1] - F[1:]
        else:  # Tabulated, linear-linear: integrate the piecewise-linear density exactly
            mu, p = self.values[0::2], self.values[1::2]
            cum = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(mu))))

            def F(x):
                x = np.clip(x, mu[0], mu[-1])
                i = np.clip(np.searchsorted(mu, x, side="right") - 1, 0, len(mu) - 2)
                t = x - mu[i]
                slope = (p[i + 1] - p[i]) / np.where(mu[i + 1] > mu[i], mu[i + 1] - mu[i], 1)
                return cum[i] + p[i] * t + 0.5 * slope * t**2

            probs = F(mu_edges[:-1]) - F(mu_edges[1:])
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        return probs / total if total > 0 else np.full(len(probs), 1 / len(probs))


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
            channel = self._read_channel(mt, sections[(6, mt)], mf3)
            if channel is not None:
                self.channels.append(channel)

        parts = [
            xs for mt, (_, _, xs) in mf3.items() if mt not in (1, 2, 3) and mt not in redundant
        ]
        grid = np.unique(np.concatenate([p.x for p in parts])) if parts else np.array([0.0, 1.0])
        self.channel_sum = Tabulated(grid, sum(p(grid) for p in parts), [len(grid)], [2])
        self.nonelastic = mf3[3][2] if 3 in mf3 else self.channel_sum

    def _read_channel(self, mt, lines, mf3):
        """Return the _Channel of an MF=6 section, or None if it has no neutrons."""
        r = _Records(lines)
        head = r.cont()
        lct, nk = int(head[3]), int(head[4])
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
                frame = "cm"  # LAW=2 is always in the centre-of-mass frame
            elif law == 7:
                incident, dists, unit_base = self._read_law7(r, mt)
                frame = "lab"
            else:
                raise NotImplementedError(
                    f"{self.path}: MT={mt} neutrons use LAW={law}; only LAW=1, 2 and 7 are "
                    "supported"
                )
            return _Channel(mt, law, xs, yield_, frame, incident, dists, unit_base, q=qi)
        return None

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
        unit_base = self._incident_interpolation(codes, mt)
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
                if set(f.codes) - {2}:
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
        """Return (beta, gamma) of the beam + target centre of mass at beam energy E [MeV]."""
        m1, m2 = BEAMS[self.beam].mass, nuclear_mass(*self.target)
        p1 = np.sqrt(E * (E + 2 * m1))
        beta = p1 / (E + m1 + m2)
        return beta, 1 / np.sqrt(1 - beta**2)

    def lab_spectrum(self, E, angle_edges, energy_edges, num_cm_angles=720, max_step=None):
        """Return the neutron production cross section [b] in each (lab angle, lab energy)
        bin at beam energy E [MeV], and the total [b].

        Args:
            E: Beam (lab) energy [MeV].
            angle_edges: Lab emission-angle bin edges [deg], from 0 to 180.
            energy_edges: Lab neutron-energy bin edges [MeV], evenly spaced.
            num_cm_angles: Number of angle cells (even in angle) over which each
                distribution is integrated exactly and placed at its middle angle.
            max_step: Largest outgoing-energy step [MeV] at which continuum distributions
                are split (default: half an output energy bin).

        Each piece of each distribution (outgoing energy, angle cell) is converted to the
        lab (boosted from the centre of mass of beam + target with relativistic kinematics,
        unless given in the lab) and added to the lab bin it lands in. The total is exact:
        the bins add up to the production cross section; the placement is good to about the
        size of the pieces.
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
            sigma = channel.production(E)
            if sigma <= 0:
                continue
            for weight, dist, scale in channel.mixture(E):
                if weight <= 0:
                    continue
                w = sigma * weight
                if channel.law == 2:
                    m1, m2 = BEAMS[self.beam].mass, nuclear_mass(*self.target)
                    m4 = m1 + m2 - M_NEUTRON - channel.q  # Residual, in its level
                    T3, cos_lab = cm_to_lab(E, mu_mid, m1, m2, M_NEUTRON, m4)
                    if np.isnan(T3[0]):
                        continue  # Below the channel's threshold
                    prob = w * dist.cell_probabilities(mu_edges)
                    out += self._bin(T3[None, :], cos_lab[None, :], prob[None, :], grid)
                elif channel.law == 7:
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
        return out, total

    @staticmethod
    def _bin(E_lab, cos_lab, prob, grid):
        """Add pieces (lab energy [MeV], lab cosine, weight [b]) into the lab bins."""
        angle_edges, energy_edges, width, shape = grid
        angle = np.degrees(np.arccos(np.clip(cos_lab, -1, 1)))
        j = np.clip(np.searchsorted(angle_edges, angle, side="right") - 1, 0, shape[0] - 1)
        k = np.clip(np.floor((E_lab - energy_edges[0]) / width).astype(int), 0, shape[1] - 1)
        flat = np.bincount((j * shape[1] + k).ravel(), prob.ravel(), minlength=shape[0] * shape[1])
        return flat.reshape(shape)

    def _bin_continuum(
        self, dist, scale, w, E, frame, beta, gamma, mu_edges, mu_mid, max_step, grid
    ):
        """Bin a LAW=1 distribution (weight w [b]) into the lab bins."""
        Eb, p, index = _energy_points(dist.energies * scale, dist.f0 / scale, dist.lep, max_step)
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
        over the cell. Cells are at most a quarter of an output angle bin wide."""
        angle_edges, energy_edges, width, shape = grid
        step = np.radians(np.min(np.diff(angle_edges))) / 4
        spectra = []
        for s in dist.spectra:  # Each spectrum in the lab energy bins (unnormalised)
            e = s.x * scale
            f = s.y / scale
            Eb, p, _ = _energy_points(e, f, 2, max_step)
            norm = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(e))
            k = np.clip(np.floor((Eb - energy_edges[0]) / width).astype(int), 0, shape[1] - 1)
            spectra.append(np.bincount(k, p * norm, minlength=shape[1]))
        out = np.zeros(shape)
        mu = dist.cosines
        for i in range(len(mu) - 1):
            a, b = mu[i], mu[i + 1]
            if b <= a:
                continue
            theta_a, theta_b = np.arccos(a), np.arccos(b)
            n = max(1, int(np.ceil((theta_a - theta_b) / step)))
            cell_mu = np.cos(np.linspace(theta_a, theta_b, n + 1))  # Increasing
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
        hi, lo = mu_edges[None, :-1], mu_edges[None, 1:]
        return (np.sinh(a * hi) - np.sinh(a * lo) + r * (np.cosh(a * hi) - np.cosh(a * lo))) / (
            2 * np.sinh(a)
        )

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


# File names of the libraries, parsed into (beam, Z, symbol, A, rank): TENDL
# (p-Be009-mt.tendl; rank by FILE_SUFFIXES, metastable targets such as p-Ac222m are not
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
    """Return {(beam, symbol, A): path} for the evaluation files in a directory and its
    subdirectories (indexed once per run), recognising TENDL, ENDF/B and FENDL names. Where
    a directory holds several TENDL variants of one evaluation, the one earliest in
    FILE_SUFFIXES (every channel explicit first) is used."""
    found = {}
    for root, _, files in os.walk(directory):
        for name in files:
            parsed = _parse_name(name)
            if parsed is None:
                continue
            beam, symbol, A, rank = parsed
            key = (beam, symbol, A)
            if key not in found or rank < found[key][0]:
                found[key] = (rank, os.path.join(root, name))
    return {key: path for key, (_, path) in found.items()}


def find_evaluation(directory, beam, target):
    """Return the path of the evaluation for a beam on a target (Z, A) in a library
    directory, or None."""
    if directory is None or not os.path.isdir(directory):
        return None
    Z, A = target
    return library_index(os.path.abspath(directory)).get((beam, atom_name(Z), A))


def describe_evaluation(path):
    """Return a one-line description of an evaluation: the file's label (TPID) and its
    origin line (target, laboratory, evaluation date and authors), from the header."""
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
    reaction of ShineSource (alongside the DROSG two-body reactions of nuclear.REACTIONS).

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
