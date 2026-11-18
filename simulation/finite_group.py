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


# ------------------------------------------------------------------ partial cell: nR receivers, informed non-receivers
_TH = np.linspace(-1.5, 2.5, 4001)


def _success_given_theta(N: int, nR_others: int, nN_others: int, pR, pN):
    """P(1 + X_R + X_N >= N theta) on the theta grid, X_R ~ Bin(nR_others, pR), X_N ~ Bin(nN_others, pN)."""
    kR = np.arange(nR_others + 1)
    kN = np.arange(nN_others + 1)
    from scipy.stats import binom as _b
    fR = _b.pmf(kR[None, :], nR_others, pR[:, None])
    fN = _b.pmf(kN[None, :], nN_others, pN[:, None])
    tot = np.zeros((len(pR), nR_others + nN_others + 1))
    for i in range(nR_others + 1):
        tot[:, i:i + nN_others + 1] += fR[:, i:i + 1] * fN
    need = np.ceil(N * _TH - 1.0 - 1e-12).clip(0, None).astype(int)
    cum = np.concatenate([np.cumsum(tot[:, ::-1], axis=1)[:, ::-1], np.zeros((len(pR), 1))], axis=1)
    idx = np.minimum(need, nR_others + nN_others + 1)
    return cum[np.arange(len(pR)), idx]


def _post(env: Env, x: float, z, tau: float):
    a, b = env.alpha, env.beta
    if z is None:
        P, m = a + b, (a * env.mu + b * x) / (a + b)
    else:
        P, m = a + b + tau, (a * env.mu + b * x + tau * z) / (a + b + tau)
    w = np.exp(-0.5 * P * (_TH - m) ** 2)
    return w / np.trapz(w, _TH)


def partial_equilibrium(env: Env, tau: float, N: int, nR: int, zgrid: np.ndarray, kN_guess: float) -> dict:
    """Symmetric equilibrium (receiver cutoff function kR(z), non-receiver cutoff kN) of the
    finite partial-visibility game in which nR of N members see the bulletin and everyone knows nR."""
    rb = math.sqrt(env.beta)

    def receiver_cut(z, kN, guess):
        pN = Phi(rb * (kN - _TH))

        def f(k):
            pR = Phi(rb * (k - _TH))
            s = _success_given_theta(N, nR - 1, N - nR, pR, pN)
            return np.trapz(s * _post(env, k, z, tau), _TH) - env.c
        return brentq(f, guess - 0.8, guess + 0.8, xtol=1e-10)

    def solve_kR(kN):
        out, g = [], kN
        for z in zgrid:
            g = receiver_cut(float(z), kN, g)
            out.append(g)
        return np.array(out)

    def nonreceiver_gap(kN):
        kR = solve_kR(kN)
        pN = Phi(rb * (kN - _TH))
        post = _post(env, kN, None, tau)
        # integrate over z | theta ~ N(theta, 1/tau) on the z grid
        dens = np.exp(-0.5 * tau * (zgrid[None, :] - _TH[:, None]) ** 2) * math.sqrt(tau / (2 * math.pi))
        acc = np.zeros_like(_TH)
        for j, z in enumerate(zgrid):
            pR = Phi(rb * (kR[j] - _TH))
            acc += _success_given_theta(N, nR, N - nR - 1, pR, pN) * dens[:, j]
        acc *= (zgrid[1] - zgrid[0])
        return np.trapz(acc * post, _TH) - env.c

    kN = brentq(nonreceiver_gap, kN_guess - 0.15, kN_guess + 0.15, xtol=1e-7)
    return {"kN": kN, "kR": solve_kR(kN)}
