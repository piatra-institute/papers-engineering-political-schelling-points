"""Repeated play with erosion of the status quo by past participation.

Each period is the static game of model.py with a public prior N(mu_t, 1/alpha).
The status quo's expected strength next period falls with participation now:

  mu_{t+1} = theta_bar - gamma * A_t.

The deterministic skeleton sets the realised fundamental equal to its public
expectation (theta_t = mu_t), so A_t = Phi(sqrt(beta)(k_t - mu_t)) for a single
cutoff k_t. Without a message the period equilibrium is indexed by
u = Phi^{-1}(theta*), with

  mu(u)  = Phi(u) - (sqrt(beta)/alpha)(u + s),  s = sqrt(1 + alpha/beta) Phi^{-1}(c)
  psi(u) = Phi((1 + beta/alpha) u + beta s/alpha)   (participation at theta = mu).

Steady states are the roots of L(u) = mu(u) + gamma psi(u) - theta_bar; a root is
stable iff L'(u) < 0.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from model import SQRT2PI, Env, Group, Phi, Phiinv, XTOL, g, no_message, phi, solve


def _s(env: Env) -> float:
    return math.sqrt(1.0 + env.alpha / env.beta) * float(Phiinv(env.c))


def mu_of_u(env: Env, u):
    return Phi(u) - (math.sqrt(env.beta) / env.alpha) * (u + _s(env))


def psi(env: Env, u):
    a, b = env.alpha, env.beta
    return Phi((1 + b / a) * u + b * _s(env) / a)


def dpsi(env: Env, u):
    a, b = env.alpha, env.beta
    return (1 + b / a) * phi((1 + b / a) * u + b * _s(env) / a)


def L(env: Env, gamma: float, theta_bar: float, u):
    return mu_of_u(env, u) + gamma * psi(env, u) - theta_bar


def dL(env: Env, gamma: float, u):
    return phi(u) - math.sqrt(env.beta) / env.alpha + gamma * dpsi(env, u)


def gamma_critical_closed_form(env: Env) -> float:
    """For c = 1/2: bistability is possible iff gamma > (sqrt(2 pi beta) - alpha)/(alpha + beta)."""
    a, b = env.alpha, env.beta
    return (math.sqrt(2 * math.pi * b) - a) / (a + b)


def max_dL(env: Env, gamma: float) -> float:
    res = minimize_scalar(lambda u: -float(dL(env, gamma, u)), bounds=(-6.0, 6.0), method="bounded",
                          options={"xatol": 1e-12})
    grid = np.linspace(-6, 6, 24001)
    return max(-res.fun, float(np.max(dL(env, gamma, grid))))


def gamma_critical_numeric(env: Env) -> float:
    return brentq(lambda gm: max_dL(env, gm), 1e-9, 50.0, xtol=1e-13)


def steady_states(env: Env, gamma: float, theta_bar: float, grid: int = 60001) -> list[dict]:
    u = np.linspace(-8.0, 8.0, grid)
    Lu = L(env, gamma, theta_bar, u)
    roots = []
    for i in range(grid - 1):
        if Lu[i] == 0.0:
            roots.append(float(u[i]))
        elif Lu[i] * Lu[i + 1] < 0:
            roots.append(brentq(lambda x: float(L(env, gamma, theta_bar, x)), u[i], u[i + 1], xtol=XTOL))
    return [{"u": r, "mu": float(mu_of_u(env, r)), "A": float(psi(env, r)), "stable": bool(dL(env, gamma, r) < 0)}
            for r in roots]


def bistable_interval(env: Env, gamma: float) -> tuple[float, float] | None:
    """Open interval of theta_bar with three steady states (None if gamma <= gamma_c)."""
    u = np.linspace(-8.0, 8.0, 160001)
    d = dL(env, gamma, u)
    idx = np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0]
    if len(idx) < 2:
        return None
    crit = [brentq(lambda x: float(dL(env, gamma, x)), u[i], u[i + 1], xtol=XTOL) for i in idx[:2]]
    h = [float(mu_of_u(env, c) + gamma * psi(env, c)) for c in crit]
    return min(h), max(h)


# ------------------------------------------------------------------ one period
def period_participation(env: Env, mu: float, theta: float | None = None, message=None) -> float:
    """Participation in one period at realised fundamental theta (default: mu).

    message: None, or (z, tau, v) for a message with perceived precision tau and reach v.
    """
    e = replace(env, mu=mu)
    theta = mu if theta is None else theta
    rb = math.sqrt(env.beta)
    t0, k0 = no_message(e)
    if message is None:
        return float(Phi(rb * (k0 - theta)))
    z, tau, v = message
    eq = solve(e, [Group(v, tau)], z)
    if not eq.unique:
        raise ValueError("multiple period equilibria")
    return float((1 - v) * Phi(rb * (k0 - theta)) + v * Phi(rb * (eq.k[0] - theta)))


def simulate(env: Env, gamma: float, theta_bar: float, mu0: float, schedule, periods: int) -> dict:
    """Deterministic path. schedule(t) returns None or (z, tau, v) for period t (0-based)."""
    mus, As = [mu0], []
    mu = mu0
    for t in range(periods):
        A = period_participation(env, mu, message=schedule(t))
        As.append(A)
        mu = theta_bar - gamma * A
        mus.append(mu)
    return {"mu": mus, "A": As}


def persistent(env: Env, gamma: float, theta_bar: float, schedule, forced: int, tail: int = 400) -> tuple[bool, dict]:
    ss = steady_states(env, gamma, theta_bar)
    low, mid, high = ss[0], ss[1], ss[2]
    path = simulate(env, gamma, theta_bar, low["mu"], schedule, forced + tail)
    final = path["A"][-1]
    return abs(final - high["A"]) < abs(final - low["A"]), path


def critical_message(env: Env, gamma: float, theta_bar: float, tau: float, v: float, forced: int) -> float:
    """Largest message content z (least mobilizing) for which a campaign of `forced`
    periods leaves the high state. Persistence iff mu after the campaign is below
    the unstable steady state, a monotone function of z."""
    ss = steady_states(env, gamma, theta_bar)
    low, mid = ss[0], ss[1]

    def mu_after(z):
        mu = low["mu"]
        for _ in range(forced):
            mu = theta_bar - gamma * period_participation(env, mu, message=(z, tau, v))
        return mu - mid["mu"]

    return brentq(mu_after, -3.0, low["mu"] + 2.0, xtol=1e-12)


# ------------------------------------------------------------------ shocks
def residence_times(env: Env, gamma: float, theta_bar: float, sigma: float, periods: int, seed: int,
                    grid: int = 4001) -> dict:
    """Stochastic version: theta_t = mu_t + xi_t, xi_t ~ N(0, sigma^2).

    Uses the exact period cutoff k*(mu) tabulated on a fine grid of mu.
    Basins are separated by the unstable steady state of the skeleton.
    """
    ss = steady_states(env, gamma, theta_bar)
    mid = ss[1]
    lo_mu, hi_mu = theta_bar - gamma - 0.5, theta_bar + 0.5
    mus = np.linspace(lo_mu, hi_mu, grid)
    ks = np.array([no_message(replace(env, mu=float(m)))[1] for m in mus])
    rng = np.random.default_rng(seed)
    xi = rng.normal(0.0, sigma, periods)
    rb = math.sqrt(env.beta)
    mu = ss[-1]["mu"]  # start in the high state
    in_high = []
    for t in range(periods):
        k = float(np.interp(mu, mus, ks))
        A = float(Phi(rb * (k - (mu + xi[t]))))
        mu = theta_bar - gamma * A
        in_high.append(mu < mid["mu"])
    in_high = np.array(in_high)
    switches_out_high = int(np.sum(in_high[:-1] & ~in_high[1:]))
    switches_out_low = int(np.sum(~in_high[:-1] & in_high[1:]))
    return {"share_high": float(in_high.mean()),
            "mean_residence_high": float(in_high.sum() / max(switches_out_high, 1)),
            "mean_residence_low": float((~in_high).sum() / max(switches_out_low, 1)),
            "exits_high": switches_out_high, "exits_low": switches_out_low, "periods": periods}
