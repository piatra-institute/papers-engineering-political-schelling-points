"""Symmetric cutoff equilibrium of the public-bulletin game in a group of N players.

Success requires that the number of participants be at least N * theta. Given a
common cutoff k, each other member participates independently given theta with
probability Phi(sqrt(beta)(k - theta)). A participant succeeds iff
1 + Binomial(N - 1, p) >= N theta. The success probability is piecewise smooth
in theta with breaks at multiples of 1/N, so it is integrated piece by piece.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import binom

from model import Env, Phi


def success_prob(env: Env, tau: float, N: int, z: float, k: float, x: float) -> float:
    a, b = env.alpha, env.beta
    P = a + tau + b
    m = (a * env.mu + tau * z + b * x) / P
    sd = 1.0 / math.sqrt(P)
    dens = lambda t: math.exp(-0.5 * ((t - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    rb = math.sqrt(b)
    # theta <= 1/N: a participant succeeds alone
    total = float(Phi((1.0 / N - m) / sd))
    for j in range(1, N):
        lo, hi = j / N, (j + 1) / N
        need = j  # others needed: smallest integer >= N*theta - 1 for theta in (j/N, (j+1)/N]

        def f(t, need=need):
            p = float(Phi(rb * (k - t)))
            return float(binom.sf(need - 1, N - 1, p)) * dens(t)

        total += integrate.quad(f, lo, hi, epsabs=1e-13, epsrel=1e-10, limit=200)[0]
    return total


def public_cutoff(env: Env, tau: float, N: int, z: float, k_guess: float) -> float:
    g = lambda k: success_prob(env, tau, N, z, k, k) - env.c
    return brentq(g, k_guess - 0.6, k_guess + 0.6, xtol=1e-11)


def roots_on_grid(env: Env, tau: float, N: int, z: float, k_guess: float, width: float = 0.6, n: int = 61) -> int:
    ks = np.linspace(k_guess - width, k_guess + width, n)
    vals = np.array([success_prob(env, tau, N, z, k, k) - env.c for k in ks])
    return int(np.sum(np.sign(vals[1:]) != np.sign(vals[:-1])))
