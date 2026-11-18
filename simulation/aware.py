"""Partial reach with aware non-receivers.

A message z with precision tau is shown to a random mass q of citizens. Everyone
knows q and tau; receivers see z, non-receivers do not. This is a common-prior
Bayesian game. Given the non-receivers' cutoff kN, the success threshold for
each z solves the same one-dimensional equation as in model.py with k0 replaced
by kN. Non-receivers integrate over the unseen z:

  P(success | x) = E[ P(z <= zbar(theta) | theta) | x ],

where zbar(theta) is the message at which theta is exactly the success
threshold. It has a closed form, so the non-receivers' condition is a single
integral and kN solves one scalar equation.
"""
from __future__ import annotations

import math

from scipy import integrate
from scipy.optimize import brentq

from model import Env, Group, Phi, Phiinv, XTOL, no_message, solve


def success_prob_given_theta(env: Env, tau: float, q: float, kN: float, theta: float) -> float:
    rb = math.sqrt(env.beta)
    p = (theta - (1.0 - q) * float(Phi(rb * (kN - theta)))) / q
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    a, b = env.alpha, env.beta
    zbar = theta - (rb * float(Phiinv(p)) + math.sqrt(a + tau + b) * float(Phiinv(env.c)) - a * (theta - env.mu)) / tau
    return float(Phi(math.sqrt(tau) * (zbar - theta)))


def nonreceiver_success_prob(env: Env, tau: float, q: float, kN: float, x: float) -> float:
    """Success is certain where non-receivers alone reach theta (p <= 0), impossible where
    even full receiver participation falls short (p >= 1), and depends on z in between.
    The middle band has width of order q, so it is integrated separately."""
    a, b = env.alpha, env.beta
    rb = math.sqrt(b)
    m = (a * env.mu + b * x) / (a + b)
    sd = 1.0 / math.sqrt(a + b)
    base = lambda t: t - (1.0 - q) * float(Phi(rb * (kN - t)))  # increasing in t
    t_lo = brentq(base, -5.0, 5.0, xtol=XTOL)  # p = 0
    t_hi = brentq(lambda t: base(t) - q, -5.0, 5.0, xtol=XTOL)  # p = 1
    dens = lambda t: math.exp(-0.5 * ((t - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    certain = float(Phi((t_lo - m) / sd))
    middle, _ = integrate.quad(lambda t: success_prob_given_theta(env, tau, q, kN, t) * dens(t), t_lo, t_hi,
                               epsabs=1e-14, epsrel=1e-11, limit=400)
    return certain + middle


def nonreceiver_cutoff(env: Env, tau: float, q: float) -> float:
    """Symmetric cutoff kN of aware non-receivers (each indifferent at x = kN)."""
    _, k0 = no_message(env)
    f = lambda k: nonreceiver_success_prob(env, tau, q, k, k) - env.c
    return brentq(f, k0 - 1.0, k0 + 1.0, xtol=1e-12)


def receiver_cutoff(env: Env, tau: float, q: float, z: float, kN: float) -> float:
    eq = solve(env, [Group(q, tau)], z, kN=kN)
    if not eq.unique:
        raise ValueError("multiple equilibria")
    return eq.k[0]
