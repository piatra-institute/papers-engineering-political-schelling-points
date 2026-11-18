"""A laboratory design and its model-derived predictions and power.

Groups of N participants play R rounds. Each round draws theta ~ N(mu, 1/alpha),
private signals x_i = theta + eps_i (precision beta) and a bulletin z = theta +
eta (precision tau). Two between-group factors:

  incentive I: 'coord'  - participation pays 1 - c iff the group's participation
                          share is at least theta, -c otherwise;
               'hurdle' - participation pays 1 - c iff theta <= h (h announced),
                          -c otherwise; others' choices are irrelevant.
  visibility V: 'public' - every member sees the bulletin on a shared screen with
                           mutual acknowledgement (reach q = 1, common knowledge);
                'partial' - a random share q of members sees it and is told q;
                            the others know a bulletin was shown to a share q.

Model cutoffs for receivers are exact equilibria: model.py for public bulletins,
aware.py for partial reach, a closed form for the hurdle task. Behaviour is
simulated with attenuation omega of the coordination component, logistic
decision noise and participant- and group-level cutoff heterogeneity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from aware import nonreceiver_cutoff, receiver_cutoff
from model import Env, Group, Phiinv, cutoff, no_message, solve


@dataclass(frozen=True)
class Design:
    env: Env
    tau: float
    q: float
    N: int
    R: int
    h: float  # hurdle; set to theta0 so the hurdle cutoff equals the evidential cutoff
    tau_true: float | None = None  # precision used to draw bulletins (default: tau, the perceived precision)


def hurdle_cutoff(env: Env, tau: float, h: float, z):
    a, b = env.alpha, env.beta
    P = a + tau + b
    return (P * h - a * env.mu - tau * z - math.sqrt(P) * float(Phiinv(env.c))) / b


class CutoffTable:
    """Exact receiver cutoffs on a z-grid.

    public-coord: reach 1 with common knowledge (Lemma 1).
    partial-coord: a share q receives; non-receivers are informed (aware.py).
    private-coord-<p>: every member receives, but receipt is only commonly believed to degree p;
        receivers act as in the reach-p equilibrium (model.cutoff with v = p).
    *-hurdle: closed form; others' choices are irrelevant.
    """

    def __init__(self, design: Design, zgrid: np.ndarray, beliefs=()):
        env, tau, q = design.env, design.tau, design.q
        self.z = zgrid
        self.coord = {"public-coord": np.array([cutoff(env, tau, 1.0, float(z)) for z in zgrid])}
        self.kN = nonreceiver_cutoff(env, tau, q)
        self.coord["partial-coord"] = np.array([receiver_cutoff(env, tau, q, float(z), self.kN) for z in zgrid])
        for p in beliefs:
            self.coord[f"private-coord-{p:g}"] = np.array([cutoff(env, tau, p, float(z)) for z in zgrid])
        self.hurdle = hurdle_cutoff(env, tau, design.h, zgrid)

    def k(self, cell: str, z, omega: float = 1.0):
        kh = np.interp(z, self.z, self.hurdle)
        if cell.endswith("-hurdle"):
            return kh
        kc = np.interp(z, self.z, self.coord[cell])
        return kh + omega * (kc - kh)


CELLS = ("public-coord", "partial-coord", "public-hurdle", "partial-hurdle")


def simulate_group(design: Design, table: CutoffTable, cell: str, omega: float, noise: float, sd_subject: float,
                   sd_group: float, rng: np.random.Generator) -> tuple[float, float, int]:
    """Simulate one group; return OLS slopes (on z, on x) from receivers' choices."""
    env = design.env
    N, R = design.N, design.R
    q = design.q if cell.startswith("partial") else 1.0
    nrec = max(1, int(round(q * N)))
    tau_true = design.tau if design.tau_true is None else design.tau_true
    theta = rng.normal(env.mu, 1 / math.sqrt(env.alpha), R)
    z = theta + rng.normal(0, 1 / math.sqrt(tau_true), R)
    u_subj = rng.normal(0, sd_subject, N)
    u_grp = rng.normal(0, sd_group)
    members = np.argsort(rng.random((R, N)), axis=1)[:, :nrec]
    x = theta[:, None] + rng.normal(0, 1 / math.sqrt(env.beta), (R, nrec))
    k = table.k(cell, z, omega)[:, None] + u_subj[members] + u_grp
    p = 1.0 / (1.0 + np.exp(-(k - x) / noise))
    a = (rng.random((R, nrec)) < p).astype(float)
    X = np.column_stack([np.ones(R * nrec), x.ravel(), np.repeat(z, nrec)])
    coef, *_ = np.linalg.lstsq(X, a.ravel(), rcond=None)
    return float(coef[2]), float(coef[1]), nrec * R


def expected_slopes(design: Design, table: CutoffTable, omega: float, noise: float, sd_subject: float,
                    sd_group: float, groups: int, seed: int) -> dict:
    """Large-sample group-level slope on z by cell (mean over many simulated groups)."""
    rng = np.random.default_rng(seed)
    out = {}
    for cell in CELLS:
        bz = [simulate_group(design, table, cell, omega, noise, sd_subject, sd_group, rng)[0] for _ in range(groups)]
        out[cell] = {"mean": float(np.mean(bz)), "sd": float(np.std(bz, ddof=1))}
    out["interaction"] = (out["public-coord"]["mean"] - out["partial-coord"]["mean"]) - (
        out["public-hurdle"]["mean"] - out["partial-hurdle"]["mean"])
    return out


def interaction_test(design: Design, table: CutoffTable, omega: float, noise: float, sd_subject: float,
                     sd_group: float, G: int, rng: np.random.Generator, alpha_level: float = 0.05) -> bool:
    """One simulated experiment with G groups per cell; pooled-variance t-test of the
    visibility-by-incentive interaction on group-level slopes."""
    est = {}
    for cell in CELLS:
        est[cell] = np.array([simulate_group(design, table, cell, omega, noise, sd_subject, sd_group, rng)[0]
                              for _ in range(G)])
    means = {c: est[c].mean() for c in CELLS}
    pooled = sum(((est[c] - means[c]) ** 2).sum() for c in CELLS) / (4 * G - 4)
    diff = (means["public-coord"] - means["partial-coord"]) - (means["public-hurdle"] - means["partial-hurdle"])
    se = math.sqrt(pooled * 4 / G)
    tcrit = stats.t.ppf(1 - alpha_level / 2, 4 * G - 4)
    return abs(diff / se) > tcrit


def power_curve(design: Design, table: CutoffTable, omega: float, noise: float, sd_subject: float,
                sd_group: float, Gs, reps: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for G in Gs:
        rej = sum(interaction_test(design, table, omega, noise, sd_subject, sd_group, G, rng) for _ in range(reps))
        out.append({"G": int(G), "power": rej / reps})
    return out


def two_cell_test(design: Design, tabs: dict, cells: tuple, omega: float, noise: float, sd_subject: float,
                  sd_group: float, G: int, rng: np.random.Generator, alpha_level: float = 0.05):
    """Difference in group-level slopes between two coordination conditions that may
    use different bulletin precisions (tabs maps condition -> (design, table, cell))."""
    est = []
    for cond in cells:
        d, tab, cell = tabs[cond]
        est.append(np.array([simulate_group(d, tab, cell, omega, noise, sd_subject, sd_group, rng)[0]
                             for _ in range(G)]))
    diff = est[0].mean() - est[1].mean()
    pooled = (((est[0] - est[0].mean()) ** 2).sum() + ((est[1] - est[1].mean()) ** 2).sum()) / (2 * G - 2)
    se = math.sqrt(pooled * 2 / G)
    return abs(diff / se) > stats.t.ppf(1 - alpha_level / 2, 2 * G - 2), diff


def interaction_cells(design: Design, table: CutoffTable, cells: tuple, omega: float, noise: float,
                      sd_subject: float, sd_group: float, G: int, rng: np.random.Generator,
                      alpha_level: float = 0.05) -> bool:
    """Pooled-variance t-test of (c0 - c1) - (c2 - c3) on group-level slopes, G groups per cell."""
    est = [np.array([simulate_group(design, table, c, omega, noise, sd_subject, sd_group, rng)[0]
                     for _ in range(G)]) for c in cells]
    diff = (est[0].mean() - est[1].mean()) - (est[2].mean() - est[3].mean())
    pooled = sum(((e - e.mean()) ** 2).sum() for e in est) / (4 * G - 4)
    return abs(diff / math.sqrt(pooled * 4 / G)) > stats.t.ppf(1 - alpha_level / 2, 4 * G - 4)


def simulated_power(design: Design, table: CutoffTable, cells: tuple, omega: float, noise: float,
                    sd_subject: float, sd_group: float, G: int, reps: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    return sum(interaction_cells(design, table, cells, omega, noise, sd_subject, sd_group, G, rng)
               for _ in range(reps)) / reps


def groups_for_power(design: Design, table: CutoffTable, cells: tuple, omega: float, noise: float,
                     sd_subject: float, sd_group: float, start: int, target: float, reps: int,
                     seed: int) -> dict:
    """Smallest G (searched upward from an analytic start) whose simulated power reaches target."""
    G = max(2, start)
    tried = []
    while True:
        pw = simulated_power(design, table, cells, omega, noise, sd_subject, sd_group, G, reps, seed)
        tried.append([G, pw])
        if pw >= target or G > 2000:
            return {"groups_per_cell": G, "simulated_power": pw, "tried": tried}
        G += max(1, int(math.ceil(0.04 * G)))
