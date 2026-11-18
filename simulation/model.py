"""Static model: a regime-change global game with an amplified message.

A continuum of citizens chooses whether to participate. Collective action
succeeds iff the participating mass A is at least the fundamental theta.
Participation pays 1 - c on success and -c on failure; abstention pays 0.

Information
  prior          theta ~ N(mu, 1/alpha)
  private signal x_i = theta + eps_i, eps_i ~ N(0, 1/beta)
  message        z = theta + eta, perceived precision tau_j by receiver group j

Reach. Receiver group j has mass w_j; total reach v = sum_j w_j. The masses
and perceived precisions of all groups are common knowledge among receivers.
Non-receivers do not condition on the message and use the no-message cutoff
k0 (the "unaware" benchmark; aware.py relaxes this).

In a monotone equilibrium every group uses a cutoff and the success region is
theta <= theta*. Each group's cutoff is pinned down by theta* through its own
indifference condition, so theta* solves one equation:

  S(theta) = (1 - v) Phi(sqrt(beta)(k0 - theta)) + sum_j w_j Phi(g_j(theta)) - theta = 0,
  g_j(theta) = [alpha(theta - mu) + tau_j(theta - z)]/sqrt(beta)
               - sqrt((alpha + tau_j + beta)/beta) Phi^{-1}(c),
  k_j = theta* + g_j(theta*)/sqrt(beta).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

SQRT2PI = math.sqrt(2.0 * math.pi)
XTOL = 1e-14


def Phi(x):
    return ndtr(x)


def Phiinv(p):
    return ndtri(p)


def phi(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQRT2PI


@dataclass(frozen=True)
class Env:
    alpha: float  # prior precision
    mu: float  # prior mean
    beta: float  # private-signal precision
    c: float  # participation cost (share of the success payoff)


@dataclass(frozen=True)
class Group:
    w: float  # mass of the receiver group
    tau: float  # perceived precision of the message in this group


@dataclass
class Equilibrium:
    thetas: list  # all roots of S on [0, 1]
    k: list = field(default_factory=list)  # cutoffs of receiver groups at the first root
    k0: float = float("nan")
    theta0: float = float("nan")

    @property
    def unique(self) -> bool:
        return len(self.thetas) == 1

    @property
    def theta(self) -> float:
        return self.thetas[0]


# ---------------------------------------------------------------- no message
def no_message(env: Env) -> tuple[float, float]:
    """Success threshold theta0 and cutoff k0 of the game without a message."""
    a, b = env.alpha, env.beta
    s = math.sqrt(1.0 + a / b) * float(Phiinv(env.c))

    def f(t):
        return float(Phiinv(t)) - (a / math.sqrt(b)) * (t - env.mu) + s

    t0 = brentq(f, 1e-15, 1.0 - 1e-15, xtol=XTOL)
    return t0, t0 + float(Phiinv(t0)) / math.sqrt(b)


def no_message_roots(env: Env, grid: int = 20001) -> int:
    a, b = env.alpha, env.beta
    s = math.sqrt(1.0 + a / b) * float(Phiinv(env.c))
    t = np.linspace(1e-9, 1 - 1e-9, grid)
    f = Phiinv(t) - (a / math.sqrt(b)) * (t - env.mu) + s
    return int(np.sum(np.sign(f[1:]) != np.sign(f[:-1])))


# ---------------------------------------------------------------- with message
def g(theta, env: Env, tau: float, z: float):
    a, b = env.alpha, env.beta
    return (a * (theta - env.mu) + tau * (theta - z)) / math.sqrt(b) - math.sqrt((a + tau + b) / b) * float(
        Phiinv(env.c))


def S(theta, env: Env, groups, z: float, kN: float):
    v = sum(gr.w for gr in groups)
    out = (1.0 - v) * Phi(math.sqrt(env.beta) * (kN - theta)) - theta
    for gr in groups:
        out = out + gr.w * Phi(g(theta, env, gr.tau, z))
    return out


def solve(env: Env, groups, z: float, kN: float | None = None, grid: int = 4001) -> Equilibrium:
    """All monotone equilibria (roots of S on [0, 1]); kN defaults to k0."""
    t0, k0 = no_message(env)
    if kN is None:
        kN = k0
    th = np.linspace(0.0, 1.0, grid)
    s = S(th, env, groups, z, kN)
    roots = []
    for i in range(grid - 1):
        if s[i] == 0.0:
            roots.append(float(th[i]))
        elif s[i] * s[i + 1] < 0:
            roots.append(brentq(lambda t: float(S(t, env, groups, z, kN)), th[i], th[i + 1], xtol=XTOL))
    if s[-1] == 0.0:
        roots.append(1.0)
    eq = Equilibrium(thetas=roots, k0=k0, theta0=t0)
    if roots:
        t = roots[0]
        eq.k = [t + float(g(t, env, gr.tau, z)) / math.sqrt(env.beta) for gr in groups]
    return eq


def cutoff(env: Env, tau: float, v: float, z: float, kN: float | None = None) -> float:
    """Cutoff of a single receiver group with perceived precision tau and reach v."""
    if v <= 0.0:
        t0, k0 = no_message(env)
        return t0 + float(g(t0, env, tau, z)) / math.sqrt(env.beta)
    eq = solve(env, [Group(v, tau)], z, kN)
    if not eq.unique:
        raise ValueError(f"multiple equilibria: {eq.thetas}")
    return eq.k[0]


def slopes(env: Env, groups, z: float, kN: float | None = None) -> dict:
    """Exact derivatives of theta* and of every group's cutoff with respect to z."""
    eq = solve(env, groups, z, kN)
    if not eq.unique:
        raise ValueError("multiple equilibria")
    t = eq.theta
    kN = eq.k0 if kN is None else kN
    v = sum(gr.w for gr in groups)
    b = env.beta
    phiN = float(phi(math.sqrt(b) * (kN - t)))
    phis = [float(phi(g(t, env, gr.tau, z))) for gr in groups]
    dS_dtheta = -(1 - v) * math.sqrt(b) * phiN - 1.0 + sum(
        gr.w * p * (env.alpha + gr.tau) / math.sqrt(b) for gr, p in zip(groups, phis))
    dS_dz = sum(gr.w * p * (-gr.tau / math.sqrt(b)) for gr, p in zip(groups, phis))
    dtheta = -dS_dz / dS_dtheta
    dk = [dtheta * (1 + (env.alpha + gr.tau) / b) - gr.tau / b for gr in groups]
    return {"theta": t, "k": eq.k, "dtheta_dz": dtheta, "dk_dz": dk, "dS_dtheta": dS_dtheta, "phiN": phiN,
            "phiR": phis}


def multiplier(env: Env, tau: float, v: float, z: float) -> float:
    """Exact marginal coordination multiplier: dk/dz divided by its Bayesian value -tau/beta."""
    if v <= 0:
        return 1.0
    sl = slopes(env, [Group(v, tau)], z)
    return sl["dk_dz"][0] / (-tau / env.beta)


# ---------------------------------------------------------------- closed forms
def neutral_message(env: Env, tau: float) -> float:
    """z0(tau): the message content that leaves every cutoff at k0, for any reach."""
    t0, _ = no_message(env)
    a, b = env.alpha, env.beta
    return t0 - float(Phiinv(env.c)) * (math.sqrt(a + tau + b) - math.sqrt(a + b)) / tau


def evidential(env: Env, tau: float, z: float) -> float:
    """Closed-form evidential channel E = k(tau, v=0) - k0 = tau (z0 - z)/beta."""
    return tau * (neutral_message(env, tau) - z) / env.beta


def multiplier_at_neutral(env: Env, tau: float, v: float) -> float:
    """Closed-form multiplier at z = z0(tau), where phi_R = phi_N = phi(Phi^{-1}(theta0))."""
    t0, _ = no_message(env)
    p0 = float(phi(Phiinv(t0)))
    a, b = env.alpha, env.beta
    return 1.0 + v * p0 * (a + tau + b) / (math.sqrt(b) + (1 - v) * b * p0 - v * p0 * (a + tau))


def perceived_precision(n: int, rho: float, r: float) -> float:
    """Precision of the average of n reports of precision rho with pairwise error correlation r."""
    return n * rho / (1.0 + (n - 1) * r)


# ---------------------------------------------------------------- certificates
def posterior_success_prob(env: Env, x: float, z: float | None, tau: float, theta_star: float) -> float:
    """P(theta <= theta_star | x, z) by direct numerical integration of prior x likelihoods.

    The posterior is never written in closed form here; the integrand is the
    product of the prior and the likelihoods, scaled by its value at the mode.
    """
    a, b = env.alpha, env.beta
    use_z = z is not None
    prec = a + b + (tau if use_z else 0.0)
    mode = (a * env.mu + b * x + (tau * z if use_z else 0.0)) / prec
    sd = 1.0 / math.sqrt(prec)

    def logp(t):
        lp = -0.5 * a * (t - env.mu) ** 2 - 0.5 * b * (x - t) ** 2
        if use_z:
            lp -= 0.5 * tau * (z - t) ** 2
        return lp

    ref = logp(mode)
    dens = lambda t: math.exp(logp(t) - ref)
    lo, hi = mode - 40 * sd, mode + 40 * sd
    cut = min(max(theta_star, lo), hi)
    pts = [mode] if lo < mode < cut else None
    num = integrate.quad(dens, lo, cut, epsabs=0, epsrel=1e-12, limit=400, points=pts)[0] if cut > lo else 0.0
    den = integrate.quad(dens, lo, hi, epsabs=0, epsrel=1e-12, limit=400, points=[mode])[0]
    return num / den


def success_threshold_from_cutoffs(env: Env, groups, ks, kN: float) -> float:
    """Root of A(theta) = theta for fixed cutoffs (A decreasing, so a single crossing)."""
    v = sum(gr.w for gr in groups)
    rb = math.sqrt(env.beta)

    def A(t):
        out = (1 - v) * float(Phi(rb * (kN - t)))
        for gr, k in zip(groups, ks):
            out += gr.w * float(Phi(rb * (k - t)))
        return out - t

    return brentq(A, -1.0, 2.0, xtol=XTOL)


def best_response_by_quadrature(env: Env, tau: float, z: float, theta_hat: float) -> float:
    """Cutoff x at which P(success | x, z) = c, using numerical integration only."""
    f = lambda x: posterior_success_prob(env, x, z, tau, theta_hat) - env.c
    return brentq(f, theta_hat - 10.0, theta_hat + 10.0, xtol=1e-12)


def certify(env: Env, groups, z: float, iters: int = 400, tol: float = 1e-10) -> dict:
    """Independent checks of an equilibrium.

    1. Indifference at each computed cutoff, by quadrature of the posterior.
    2. Iterated best responses (quadrature-based) from 'nobody participates' and
       'everybody participates' among receivers. In this supermodular game the two
       sequences bracket every equilibrium; if both converge to the computed
       cutoffs, the equilibrium is the unique one among receivers' strategies.
    """
    eq = solve(env, groups, z)
    t0, k0 = eq.theta0, eq.k0
    indiff = [posterior_success_prob(env, k, z, gr.tau, eq.theta) - env.c for gr, k in zip(groups, eq.k)]
    indiff0 = posterior_success_prob(env, k0, None, 0.0, t0) - env.c
    out = {"theta_star": eq.theta, "unique_roots": eq.unique, "max_abs_indifference_error": float(
        max(abs(v) for v in indiff + [indiff0]))}
    paths = {}
    for start, k_init in (("low", -5.0), ("high", 6.0)):
        ks = [k_init] * len(groups)
        for it in range(iters):
            th = success_threshold_from_cutoffs(env, groups, ks, k0)
            new = [best_response_by_quadrature(env, gr.tau, z, th) for gr in groups]
            delta = max(abs(a - b) for a, b in zip(new, ks))
            ks = new
            if delta < tol:
                break
        paths[start] = (ks, it + 1)
    out["iterated_low"] = paths["low"][0]
    out["iterated_high"] = paths["high"][0]
    out["iterations"] = [paths["low"][1], paths["high"][1]]
    out["max_abs_gap_iterated_vs_analytic"] = float(max(
        max(abs(a - b) for a, b in zip(paths["low"][0], eq.k)),
        max(abs(a - b) for a, b in zip(paths["high"][0], eq.k))))
    return out


def kappa(env: Env, tau: float) -> float:
    """M0(tau, v) = 1/(1 - kappa v): kappa = phi0(alpha+tau+beta)/(sqrt(beta) + beta phi0)."""
    t0, _ = no_message(env)
    p0 = float(phi(Phiinv(t0)))
    return p0 * (env.alpha + tau + env.beta) / (math.sqrt(env.beta) + env.beta * p0)


# ---------------------------------------------------------------- receipt versus common belief
def realized(env: Env, tau: float, v_belief: float, v_actual: float, z: float) -> dict:
    """Receivers act on a commonly believed reach v_belief; a share v_actual actually receives.

    Receivers' cutoff is the equilibrium cutoff at reach v_belief (their beliefs about who else
    responds); realized participation and success use the actual share v_actual. When the two
    coincide this is the equilibrium of Lemma 1.
    """
    t0, k0 = no_message(env)
    k = cutoff(env, tau, v_belief, z)
    rb = math.sqrt(env.beta)
    A = lambda t: (1 - v_actual) * float(Phi(rb * (k0 - t))) + v_actual * float(Phi(rb * (k - t))) - t
    theta_a = brentq(A, -1.0, 2.0, xtol=XTOL)
    return {"k": k, "k0": k0, "theta_actual": theta_a,
            "receiver_pp": 100 * (float(Phi(rb * (k - t0))) - t0),
            "aggregate_pp": 100 * v_actual * (float(Phi(rb * (k - t0))) - t0)}


def multiplicity_share(env: Env, tau: float, v: float, zs) -> float:
    """Share of message contents zs at which the reach-v game has more than one equilibrium."""
    return float(np.mean([len(solve(env, [Group(v, tau)], float(z)).thetas) > 1 for z in zs]))
