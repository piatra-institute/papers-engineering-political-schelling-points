"""Recorded run: tests, exact equilibrium computations, verification grids,
dynamics, identification and the experimental design with power analysis.

Writes output/results.json and ../paper/figures/{channels,dynamics,design}.pdf.
Every random process is seeded from CONFIG.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import unittest
from dataclasses import replace

import matplotlib
import numpy as np
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import aware  # noqa: E402
import dynamics as dyn  # noqa: E402
import experiment as ex  # noqa: E402
import finite_group as fg  # noqa: E402
from model import (Env, Group, Phi, Phiinv, certify, cutoff, evidential, multiplier,  # noqa: E402
                   multiplier_at_neutral, neutral_message, no_message, no_message_roots, perceived_precision,
                   slopes, solve)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIG = os.path.normpath(os.path.join(HERE, "..", "paper", "figures"))

CONFIG = {
    "baseline": {"alpha": 4.0, "mu": 0.6, "beta": 25.0, "c": 0.5},
    "rho": 1.0,  # precision of one report
    "n_small": 3, "n_large": 5,  # accounts in the campaign
    "surprise": 0.3,  # baseline mobilizing message: z = theta0 - surprise
    "v_private": 0.25,  # reach that is common knowledge before public disclosure
    "dynamics": {"gamma": 1.0, "theta_bar_position": 0.3, "campaign_lengths": [1, 2, 4, 8],
                 "shock_sds": [0.1, 0.2, 0.3, 0.5], "shock_gammas": [0.6, 1.0, 1.5], "shock_periods": 50000,
                 "shock_seed": 20261118, "disclosure_campaign": 6},
    "design": {"N": 20, "R": 30, "q": 0.25, "tau": 1.0, "noise_sd_units": 0.5, "subject_sd_units": 0.5,
               "group_sd_units": 0.25, "omegas": [1.0, 0.5, 0.25], "calibration_groups": 4000,
               "calibration_seed": 11, "power_reps": 1000, "power_seed": 12, "size_seed": 13,
               "paradox_seed": 14, "paradox_reps": 1000, "power_targets": [0.8, 0.9]},
}

BASE = Env(**CONFIG["baseline"])
RB = math.sqrt(BASE.beta)


def r(x, d=6):
    return float(round(float(x), d))


def participation(env: Env, k: float, theta: float) -> float:
    return float(Phi(math.sqrt(env.beta) * (k - theta)))


# ====================================================================== tests
def run_tests() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromName("test_model")
    res = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    if not res.wasSuccessful():
        sys.exit("unit tests failed")
    return {"ran": res.testsRun, "failures": len(res.failures), "errors": len(res.errors)}


# ====================================================================== static
def static_baseline() -> dict:
    t0, k0 = no_message(BASE)
    return {"theta0": t0, "k0": k0, "prior_success_prob": float(Phi(math.sqrt(BASE.alpha) * (t0 - BASE.mu))),
            "uniqueness_ratio": BASE.alpha / math.sqrt(BASE.beta), "uniqueness_bound": math.sqrt(2 * math.pi),
            "no_message_roots": no_message_roots(BASE),
            "max_public_precision_unique": math.sqrt(2 * math.pi * BASE.beta) - BASE.alpha}


def certificates() -> dict:
    t0, _ = no_message(BASE)
    configs = []
    for v in (0.25, 0.5, 1.0):
        for tau in (1.0, 3.0):
            for dz in (-0.3, 0.0, 0.3):
                configs.append(([Group(v, tau)], t0 + dz))
    configs.append(([Group(0.3, 3.0), Group(0.4, 1.0)], t0 - 0.2))
    configs.append(([Group(0.5, 3.0), Group(0.25, 0.0)], t0 - 0.3))
    errs, gaps, uniq, its = [], [], 0, []
    for groups, z in configs:
        cert = certify(BASE, groups, z)
        errs.append(cert["max_abs_indifference_error"])
        gaps.append(cert["max_abs_gap_iterated_vs_analytic"])
        uniq += int(cert["unique_roots"])
        its.extend(cert["iterations"])
    return {"configurations": len(configs), "all_unique": uniq == len(configs),
            "max_abs_indifference_error": float(max(errs)), "max_abs_gap_iterated_vs_analytic": float(max(gaps)),
            "max_iterations": int(max(its))}


def verify_propositions() -> dict:
    """Grid check of Proposition 1, the multiplier formula and the precision lemma."""
    n = {"points": 0, "nonunique_skipped": 0, "sign_fail": 0, "evidential_fail": 0, "coord_sign_fail": 0,
         "coord_monotone_fail": 0, "multiplier_fd_fail": 0, "multiplier_neutral_fail": 0,
         "precision_lemma_paths": 0, "precision_lemma_fail": 0}
    max_ev, max_fd, max_neu = 0.0, 0.0, 0.0
    vs = (0.1, 0.3, 0.6, 1.0)
    for alpha in (1.0, 4.0):
        for beta in (9.0, 25.0, 100.0):
            for c in (0.3, 0.5, 0.7):
                for mu in (0.4, 0.6):
                    env = Env(alpha, mu, beta, c)
                    if no_message_roots(env) != 1:
                        continue
                    t0, k0 = no_message(env)
                    for tau in (0.5, 1.0, 3.0):
                        z0 = neutral_message(env, tau)
                        # multiplier closed form at the neutral message
                        for v in vs:
                            eq = solve(env, [Group(v, tau)], z0)
                            if eq.unique:
                                err = abs(multiplier(env, tau, v, z0) - multiplier_at_neutral(env, tau, v))
                                max_neu = max(max_neu, err)
                                n["multiplier_neutral_fail"] += int(err > 1e-8)
                        for dz in (-0.5, -0.2, -0.05, 0.05, 0.2, 0.5):
                            z = z0 + dz
                            E = evidential(env, tau, z)
                            e_num = cutoff(env, tau, 0.0, z) - k0
                            max_ev = max(max_ev, abs(E - e_num))
                            n["evidential_fail"] += int(abs(E - e_num) > 1e-12)
                            prevC = 0.0
                            for v in vs:
                                eq = solve(env, [Group(v, tau)], z)
                                n["points"] += 1
                                if not eq.unique:
                                    n["nonunique_skipped"] += 1
                                    continue
                                k = eq.k[0]
                                n["sign_fail"] += int(np.sign(k - k0) != np.sign(-dz))
                                C = (k - k0) - E
                                n["coord_sign_fail"] += int(np.sign(C) != np.sign(E))
                                n["coord_monotone_fail"] += int(abs(C) < abs(prevC) - 1e-13)
                                prevC = C
                                h = 1e-6
                                fd = (cutoff(env, tau, v, z + h) - cutoff(env, tau, v, z - h)) / (2 * h)
                                rel = abs(fd / (-tau / beta) - multiplier(env, tau, v, z)) / multiplier(env, tau, v, z)
                                max_fd = max(max_fd, rel)
                                n["multiplier_fd_fail"] += int(rel > 1e-5)
                    # precision lemma: along tau, the cutoff changes direction at most once,
                    # never (c = 1/2), from rising to falling (c < 1/2) or falling to rising (c > 1/2)
                    taus = np.linspace(0.05, 4.0, 80)
                    for v in (0.3, 1.0):
                        for dz in (-0.4, -0.1, 0.0, 0.1, 0.4):
                            z = t0 + dz
                            ks = []
                            ok = True
                            for tau in taus:
                                eq = solve(env, [Group(v, tau)], z)
                                if not eq.unique:
                                    ok = False
                                    break
                                ks.append(eq.k[0])
                            if not ok:
                                continue
                            n["precision_lemma_paths"] += 1
                            d = np.sign(np.diff(ks))
                            d = d[np.abs(np.diff(ks)) > 1e-12]
                            changes = np.where(d[1:] != d[:-1])[0]
                            bad = len(changes) > 1
                            if len(changes) == 1:
                                first = d[0]
                                bad = bad or (c == 0.5) or (c < 0.5 and first < 0) or (c > 0.5 and first > 0)
                            n["precision_lemma_fail"] += int(bad)
    n["max_abs_evidential_error"] = max_ev
    n["max_rel_multiplier_fd_error"] = max_fd
    n["max_abs_multiplier_neutral_error"] = max_neu
    return n


def decomposition() -> dict:
    t0, k0 = no_message(BASE)
    s = CONFIG["surprise"]
    z = t0 - s
    tau = CONFIG["rho"]
    out = {"z": z, "tau": tau, "z0": neutral_message(BASE, tau), "by_reach": {}}
    E = evidential(BASE, tau, z)
    for v in (0.0, 0.25, 0.5, 1.0):
        k = cutoff(BASE, tau, v, z)
        out["by_reach"][f"{v:.2f}"] = {
            "k_minus_k0": k - k0, "evidential": E, "coordination": (k - k0) - E,
            "coordination_share": ((k - k0) - E) / (k - k0),
            "participation_pp": 100 * (participation(BASE, k, t0) - t0),
            "evidential_pp": 100 * (participation(BASE, k0 + E, t0) - t0),
            "multiplier_exact": multiplier(BASE, tau, v, z) if v > 0 else 1.0,
            "multiplier_neutral": multiplier_at_neutral(BASE, tau, v)}
    # zero-precision limit: coordination share stays bounded away from zero
    small = 1e-4
    zz = neutral_message(BASE, small) - s
    Es = evidential(BASE, small, zz)
    ks = cutoff(BASE, small, 1.0, zz)
    out["small_precision"] = {"tau": small, "C_over_E": ((ks - k0) - Es) / Es,
                              "closed_form_limit": multiplier_at_neutral(BASE, 0.0, 1.0) - 1.0,
                              "k_minus_k0_at_zero": cutoff(BASE, 0.0, 1.0, z) - k0}
    # multipliers by private precision (closed form at the neutral message, tau = 1, v = 1)
    out["multiplier_by_beta"] = {f"{b:g}": multiplier_at_neutral(replace(BASE, beta=b), 1.0, 1.0)
                                 for b in (9.0, 25.0, 100.0, 400.0)}
    return out


def isolated_disclosure() -> dict:
    """A receiver learns the source is common while believing the others have not."""
    t0, k0 = no_message(BASE)
    z = t0 - CONFIG["surprise"]
    n, rho = CONFIG["n_small"], CONFIG["rho"]
    tau_h = perceived_precision(n, rho, 0.0)
    out = {}
    for v in (CONFIG["v_private"], 1.0):
        res = {}
        for tau_l, name in ((rho, "common_source"), (0.0, "uninformative")):
            sl = slopes(BASE, [Group(v, tau_h), Group(1e-12, tau_l)], z)
            k_h, k_l = sl["k"]
            # closed form: same theta*, different own precision
            cf = ((tau_l - tau_h) * (sl["theta"] - z) - float(Phiinv(BASE.c)) * (
                math.sqrt(BASE.alpha + tau_l + BASE.beta) - math.sqrt(BASE.alpha + tau_h + BASE.beta))) / BASE.beta
            pp_h = 100 * (participation(BASE, k_h, t0) - t0)
            pp_l = 100 * (participation(BASE, k_l, t0) - t0)
            res[name] = {"tau_l": tau_l, "share_of_effect_removed": (pp_h - pp_l) / pp_h,
                         "share_of_cutoff_shift_removed": (k_h - k_l) / (k_h - k0),
                         "closed_form_error": abs((k_l - k_h) - cf),
                         "response_ratio_to_undisclosed": sl["dk_dz"][1] / sl["dk_dz"][0],
                         "participation_pp_undisclosed": 100 * (participation(BASE, k_h, t0) - t0),
                         "participation_pp_disclosed": 100 * (participation(BASE, k_l, t0) - t0)}
        # common knowledge of the disclosure at the same reach
        kc = cutoff(BASE, rho, v, z)
        res["common_disclosure"] = {"participation_pp": 100 * (participation(BASE, kc, t0) - t0),
                                    "uninformative_participation_pp": 100 * (
                                        participation(BASE, cutoff(BASE, 0.0, v, z), t0) - t0)}
        out[f"{v:.2f}"] = res
    out["tau_h"] = tau_h
    out["max_closed_form_error"] = max(out[v][name]["closed_form_error"] for v in ("0.25", "1.00")
                                       for name in ("common_source", "uninformative"))
    full = out["1.00"]
    out["common_share_removed_full_reach"] = 1 - full["common_disclosure"]["participation_pp"] / full["common_source"][
        "participation_pp_undisclosed"]
    return out


def paradox() -> dict:
    t0, k0 = no_message(BASE)
    s = CONFIG["surprise"]
    z = t0 - s
    rho = CONFIG["rho"]
    v0 = CONFIG["v_private"]
    out = {"examples": {}}
    for n in (CONFIG["n_small"], CONFIG["n_large"]):
        tau_h = perceived_precision(n, rho, 0.0)
        kb = cutoff(BASE, tau_h, v0, z)
        ka = cutoff(BASE, rho, 1.0, z)
        kp = cutoff(BASE, rho, v0, z)  # disclosure without a reach change
        Pi = rho * multiplier_at_neutral(BASE, rho, 1.0) / (tau_h * multiplier_at_neutral(BASE, tau_h, v0))
        out["examples"][str(n)] = {
            "tau_h": tau_h, "Pi": Pi, "before_pp": 100 * (participation(BASE, kb, t0) - t0),
            "after_public_pp": 100 * (participation(BASE, ka, t0) - t0),
            "after_private_pp": 100 * (participation(BASE, kp, t0) - t0),
            "public_raises": bool(ka > kb), "private_raises": bool(kp > kb)}
    # neglect threshold for the larger campaign: Pi(chi) = 1 with tau_h = n rho/(1 + (n - 1)(1 - chi))
    n = CONFIG["n_large"]

    def Pi_chi(chi):
        th = perceived_precision(n, rho, 1.0 - chi)
        return rho * multiplier_at_neutral(BASE, rho, 1.0) / (th * multiplier_at_neutral(BASE, th, v0)) - 1.0

    chi_star = brentq(Pi_chi, 0.0, 1.0, xtol=1e-12)
    out["neglect_threshold"] = {"n": n, "chi_star": chi_star,
                                "tau_h_at_chi_star": perceived_precision(n, rho, 1.0 - chi_star)}
    # validation of the first-order criterion against exact equilibria (c = 1/2)
    agree = {}
    total = {}
    skipped = 0
    for alpha in (1.0, 4.0):
        for beta in (9.0, 25.0, 100.0):
            env = Env(alpha, BASE.mu, beta, 0.5)
            tt0, _ = no_message(env)
            for tau_l in (0.5, 1.0):
                for R in (1.25, 1.5, 2.0, 3.0, 5.0, 8.0):
                    tau_h = R * tau_l
                    for v0_ in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                        Pi = tau_l * multiplier_at_neutral(env, tau_l, 1.0) / (
                            tau_h * multiplier_at_neutral(env, tau_h, v0_))
                        for sz in (0.01, 0.05, 0.15, 0.3):
                            for sign in (1, -1):
                                zz = tt0 - sign * sz
                                try:
                                    kb = cutoff(env, tau_h, v0_, zz)
                                    ka = cutoff(env, tau_l, 1.0, zz)
                                except ValueError:
                                    skipped += 1
                                    continue
                                amplifies = (ka - kb) * sign > 0  # pushes further in the message's direction
                                key = f"{sz:g}"
                                total[key] = total.get(key, 0) + 1
                                agree[key] = agree.get(key, 0) + int(amplifies == (Pi > 1))
    out["local_criterion"] = {k: {"points": total[k], "agree": agree[k], "share": agree[k] / total[k]} for k in total}
    out["local_criterion_nonunique_skipped"] = skipped
    # map for the figure: exact sign over (R, v0) at the baseline
    Rs = np.linspace(1.0, 8.0, 57)
    v0s = np.linspace(0.0, 0.95, 39)
    sign_map = np.zeros((len(v0s), len(Rs)))
    for i, vv in enumerate(v0s):
        for j, R in enumerate(Rs):
            try:
                sign_map[i, j] = np.sign(cutoff(BASE, rho, 1.0, z) - cutoff(BASE, R * rho, vv, z))
            except ValueError:
                sign_map[i, j] = np.nan
    boundary = []
    for vv in v0s:
        f = lambda R: rho * multiplier_at_neutral(BASE, rho, 1.0) / (R * rho * multiplier_at_neutral(BASE, R * rho, vv)) - 1
        boundary.append(brentq(f, 1.0001, 50.0) if f(1.0001) > 0 > f(50.0) else np.nan)
    out["_map"] = {"R": Rs.tolist(), "v0": v0s.tolist(), "sign": sign_map.tolist(), "boundary": boundary}
    return out


def aware_robustness() -> dict:
    t0, k0 = no_message(BASE)
    out = {}
    worst = 0.0
    for tau in (1.0, 3.0):
        for q in (0.1, 0.25, 0.5, 0.9):
            kN = aware.nonreceiver_cutoff(BASE, tau, q)
            zs = np.linspace(t0 - 0.6, t0 + 0.6, 13)
            diffs, effects = [], []
            for zz in zs:
                ku = cutoff(BASE, tau, q, float(zz))
                ka = aware.receiver_cutoff(BASE, tau, q, float(zz), kN)
                diffs.append(100 * abs(participation(BASE, ka, t0) - participation(BASE, ku, t0)))
                effects.append(100 * abs(participation(BASE, ku, t0) - t0))
            share = max(diffs) / max(effects)
            worst = max(worst, share)
            out[f"tau{tau:g}_q{q:g}"] = {"kN_minus_k0_pp": 100 * (participation(BASE, kN, t0) - t0),
                                         "max_abs_diff_pp": max(diffs), "max_effect_pp": max(effects),
                                         "diff_share_of_max_effect": share}
    # Monte Carlo check of the non-receivers' indifference at the baseline design
    rng = np.random.default_rng(CONFIG["design"]["calibration_seed"])
    tau, q = CONFIG["design"]["tau"], CONFIG["design"]["q"]
    kN = aware.nonreceiver_cutoff(BASE, tau, q)
    m = (BASE.alpha * BASE.mu + BASE.beta * kN) / (BASE.alpha + BASE.beta)
    th = rng.normal(m, 1 / math.sqrt(BASE.alpha + BASE.beta), 400000)
    zz = th + rng.normal(0, 1 / math.sqrt(tau), th.size)
    zgrid = np.linspace(zz.min(), zz.max(), 1601)
    tstar = np.array([solve(BASE, [Group(q, tau)], float(x), kN=kN).theta for x in zgrid])
    p = float(np.mean(th <= np.interp(zz, zgrid, tstar)))
    out["max_diff_share"] = worst
    out["max_kN_shift_pp"] = max(v["kN_minus_k0_pp"] for k, v in out.items() if k.startswith("tau"))
    out["max_receiver_diff_pp"] = max(v["max_abs_diff_pp"] for k, v in out.items() if k.startswith("tau"))
    out["mc_indifference"] = {"draws": int(th.size), "success_prob": p, "target": BASE.c,
                              "mc_se": math.sqrt(p * (1 - p) / th.size)}
    return out


# ====================================================================== dynamics
def dynamics_results() -> dict:
    D = CONFIG["dynamics"]
    out = {}
    gc = dyn.gamma_critical_closed_form(BASE)
    out["gamma_c_closed_form"] = gc
    out["gamma_c_numeric"] = dyn.gamma_critical_numeric(BASE)
    out["gamma_c_other_costs"] = {f"{c:g}": dyn.gamma_critical_numeric(replace(BASE, c=c)) for c in (0.3, 0.7)}
    out["gamma_c_by_precision"] = {}
    for a, b in ((1.0, 9.0), (4.0, 100.0), (4.0, 400.0)):
        e = replace(BASE, alpha=a, beta=b)
        out["gamma_c_by_precision"][f"a{a:g}_b{b:g}"] = {"closed": dyn.gamma_critical_closed_form(e),
                                                         "numeric": dyn.gamma_critical_numeric(e)}
    # grid check of the bistability characterisation
    mism, pts = 0, 0
    for gm in np.linspace(0.05, 2.0, 40):
        iv = dyn.bistable_interval(BASE, gm)
        for tb in np.linspace(0.2, 2.2, 41):
            nss = len(dyn.steady_states(BASE, gm, tb))
            pred = 3 if (iv is not None and iv[0] < tb < iv[1]) else 1
            pts += 1
            mism += int(nss != pred)
    out["bistability_grid"] = {"points": pts, "mismatches": mism}
    gm = D["gamma"]
    lo, hi = dyn.bistable_interval(BASE, gm)
    tb = lo + D["theta_bar_position"] * (hi - lo)
    ss = dyn.steady_states(BASE, gm, tb)
    out["baseline"] = {"gamma": gm, "interval": [lo, hi], "theta_bar": tb,
                       "steady_states": ss}
    # critical campaigns
    rho = CONFIG["rho"]
    tau_h = perceived_precision(CONFIG["n_small"], rho, 0.0)
    crit = {}
    for T in D["campaign_lengths"]:
        crit[str(T)] = {"undisclosed": dyn.critical_message(BASE, gm, tb, tau_h, 1.0, T),
                        "disclosed": dyn.critical_message(BASE, gm, tb, rho, 1.0, T)}
    out["critical_message"] = crit
    # sharpness at T = 4, undisclosed
    T = 4
    zc = crit[str(T)]["undisclosed"]
    sharp = {}
    for dz in (-1e-6, 1e-6):
        sched = lambda t, zz=zc + dz: (zz, tau_h, 1.0) if t < T else None
        p, path = dyn.persistent(BASE, gm, tb, sched, T)
        sharp[f"{dz:+g}"] = {"persistent": p, "final_A": path["A"][-1]}
    out["sharpness"] = sharp
    # trajectories for the figure
    traj = {}
    for name, zz in (("below", zc - 0.05), ("above", zc + 0.05)):
        sched = lambda t, zz=zz: (zz, tau_h, 1.0) if t < T else None
        traj[name] = dyn.simulate(BASE, gm, tb, ss[0]["mu"], sched, 30)["A"]
    gm_u = 0.8 * gc
    tb_u = tb
    ss_u = dyn.steady_states(BASE, gm_u, tb_u)
    sched = lambda t, zz=zc - 0.05: (zz, tau_h, 1.0) if t < T else None
    traj["unique"] = dyn.simulate(BASE, gm_u, tb_u, ss_u[0]["mu"], sched, 30)["A"]
    out["_trajectories"] = traj
    out["unique_regime"] = {"gamma": gm_u, "steady_states": len(ss_u), "final_A": traj["unique"][-1],
                            "initial_A": ss_u[0]["A"]}
    # disclosure timing: a campaign that persists undisclosed but not when disclosed from the start
    Tc = D["disclosure_campaign"]
    z3 = dyn.critical_message(BASE, gm, tb, tau_h, 1.0, 3)
    zd = 0.5 * (z3 + crit["4"]["undisclosed"])  # undisclosed campaign crosses in period 4
    timing = []
    for td in range(Tc + 1):
        sched = lambda t, td=td: ((zd, tau_h if t < td else rho, 1.0) if t < Tc else None)
        p, path = dyn.persistent(BASE, gm, tb, sched, Tc)
        timing.append({"disclosure_period": td, "persistent": p})
    first_persistent = next((x["disclosure_period"] for x in timing if x["persistent"]), None)
    # period at which the undisclosed campaign crosses the unstable state
    sched = lambda t: (zd, tau_h, 1.0) if t < Tc else None
    path = dyn.simulate(BASE, gm, tb, ss[0]["mu"], sched, Tc)
    cross = next((t + 1 for t, mu in enumerate(path["mu"][1:]) if mu < ss[1]["mu"]), None)
    out["disclosure_timing"] = {"campaign_length": Tc, "z": zd, "z_c_3": z3, "timing": timing,
                                "disclosed_from_start_z_c": dyn.critical_message(BASE, gm, tb, rho, 1.0, Tc),
                                "latest_preventing_disclosure": (first_persistent - 1) if first_persistent else None,
                                "undisclosed_crossing_period": cross}
    # shocks
    rows = {}
    for g_ in D["shock_gammas"]:
        lo_, hi_ = dyn.bistable_interval(BASE, g_)
        tb_ = lo_ + D["theta_bar_position"] * (hi_ - lo_)
        for sd in D["shock_sds"]:
            rr = dyn.residence_times(BASE, g_, tb_, sd, D["shock_periods"], D["shock_seed"])
            rows[f"g{g_:g}_s{sd:g}"] = rr
    out["shocks"] = rows
    out["consistent_shock_sd"] = 1 / math.sqrt(BASE.alpha)
    return out


# ====================================================================== identification
def identification() -> dict:
    t0, k0 = no_message(BASE)
    z = t0 - CONFIG["surprise"]
    tau, v = CONFIG["rho"], 1.0
    k = cutoff(BASE, tau, v, z)
    dk = k - k0
    # (i) private evidence alone: tau_E with E(tau_E) = dk
    tau_E = brentq(lambda t: evidential(BASE, t, z) - dk, 1e-6, 200.0, xtol=1e-12)
    # (ii) private valuation persuasion: cost c' for exposed agents, others at k0
    s0 = math.sqrt(BASE.alpha + BASE.beta)

    def kp(cprime):
        return t0 + BASE.alpha * (t0 - BASE.mu) / BASE.beta - s0 * float(Phiinv(cprime)) / BASE.beta

    c_p = brentq(lambda cc: kp(cc) - k, 1e-9, BASE.c, xtol=1e-14)
    # (iii) majority illusion: neglected bias b in the private signal
    b = dk
    # (iv) reach needed at a weaker message
    out = {"z": z, "target_k_minus_k0": dk, "target_pp": 100 * (participation(BASE, k, t0) - t0),
           "private_evidence_tau": tau_E, "persuasion_cost": c_p, "persuasion_cost_cut": BASE.c - c_p,
           "illusion_bias": b}
    # signatures in the hurdle task (payoffs independent of others): relative weight on
    # the bulletin's content (tau/beta) and level shift at theta0. The illusion's level
    # shift disappears when the peer sample is representative.
    sig = {}
    for name, tt, lvl in (("coordination", tau, evidential(BASE, tau, z)),
                          ("private_evidence", tau_E, evidential(BASE, tau_E, z)),
                          ("persuasion", 0.0, k - k0),
                          ("illusion", 0.0, b)):
        sig[name] = {"hurdle_weight": tt / BASE.beta, "hurdle_level_pp": 100 * (participation(BASE, k0 + lvl, t0) - t0)}
    out["signatures"] = sig
    out["coordination_partial_weight"] = tau * multiplier(BASE, tau, CONFIG["design"]["q"], z) / BASE.beta
    out["coordination_public_weight"] = tau * multiplier(BASE, tau, 1.0, z) / BASE.beta
    # iso-response curve: (tau, v) with the same response to content at the neutral message
    target = tau * multiplier_at_neutral(BASE, tau, 1.0)
    p0 = float(np.exp(-0.5 * float(Phiinv(t0)) ** 2) / math.sqrt(2 * math.pi))
    curve = []
    for vv in np.linspace(0.0, 1.0, 21):
        f = lambda t: t * multiplier_at_neutral(BASE, t, vv) - target
        # stay inside the region where the multiplier's denominator is positive
        tmax = 60.0 if vv == 0 else min(60.0, 0.999 * ((RB + (1 - vv) * BASE.beta * p0) / (vv * p0) - BASE.alpha))
        tt = brentq(f, 1e-6, tmax, xtol=1e-12)
        curve.append([float(vv), float(tt)])
    out["iso_response"] = curve
    out["iso_tau_at_v0"] = curve[0][1]
    out["iso_tau_at_v_half"] = curve[10][1]
    return out


# ====================================================================== design
def design_results() -> dict:
    Dc = CONFIG["design"]
    t0, k0 = no_message(BASE)
    d = ex.Design(env=BASE, tau=Dc["tau"], q=Dc["q"], N=Dc["N"], R=Dc["R"], h=t0)
    zg = np.linspace(-4.0, 5.0, 451)
    tab = ex.CutoffTable(d, zg)
    unit = 1 / math.sqrt(BASE.beta)
    noise, sds, sdg = Dc["noise_sd_units"] * unit, Dc["subject_sd_units"] * unit, Dc["group_sd_units"] * unit
    zsd = math.sqrt(1 / BASE.alpha + 1 / Dc["tau"])
    out = {"design": {"N": Dc["N"], "R": Dc["R"], "q": Dc["q"], "tau": Dc["tau"], "hurdle": t0,
                      "bulletin_sd": zsd, "kN_minus_k0": tab.kN - k0},
           "theory": {"hurdle_relative_weight": Dc["tau"] / BASE.beta,
                      "public_multiplier_neutral": multiplier_at_neutral(BASE, Dc["tau"], 1.0),
                      "partial_multiplier_neutral_unaware": multiplier_at_neutral(BASE, Dc["tau"], Dc["q"])},
           "by_omega": {}}
    for om in Dc["omegas"] + [0.0]:
        es = ex.expected_slopes(d, tab, om, noise, sds, sdg, Dc["calibration_groups"], Dc["calibration_seed"])
        cells = {c: es[c] for c in ex.CELLS}
        inter = es["interaction"]
        sd_sum = sum(cells[c]["sd"] ** 2 for c in ex.CELLS)
        row = {"cells": cells, "interaction_slope": inter, "interaction_response_pp_per_sd": -100 * inter * zsd,
               "cell_response_pp_per_sd": {c: -100 * cells[c]["mean"] * zsd for c in ex.CELLS}}
        if om > 0:
            req = {}
            for target in Dc["power_targets"]:
                G = 2
                while True:
                    df = 4 * G - 4
                    ncp = abs(inter) / math.sqrt(sd_sum / G)
                    tc = stats.t.ppf(0.975, df)
                    pw = 1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)
                    if pw >= target:
                        break
                    G += 1
                req[f"{target:g}"] = {"groups_per_cell": G, "participants": 4 * G * Dc["N"]}
            row["required"] = req
            G80 = req["0.8"]["groups_per_cell"]
            pc = ex.power_curve(d, tab, om, noise, sds, sdg, [G80], Dc["power_reps"], Dc["power_seed"])
            row["simulated_power_at_G80"] = pc[0]["power"]
        out["by_omega"][f"{om:g}"] = row
    # size of the test when the coordination component is absent
    G_half = out["by_omega"]["0.5"]["required"]["0.8"]["groups_per_cell"]
    pc0 = ex.power_curve(d, tab, 0.0, noise, sds, sdg, [G_half], Dc["power_reps"], Dc["size_seed"])
    out["size_at_omega0"] = {"G": G_half, "rejection_rate": pc0[0]["power"]}
    # power curves for the figure (analytic, calibrated by the simulated cell s.d.)
    curves = {}
    for om in Dc["omegas"]:
        row = out["by_omega"][f"{om:g}"]
        sd_sum = sum(row["cells"][c]["sd"] ** 2 for c in ex.CELLS)
        pts = []
        for G in range(2, 161):
            df = 4 * G - 4
            ncp = abs(row["interaction_slope"]) / math.sqrt(sd_sum / G)
            tc = stats.t.ppf(0.975, df)
            pts.append([G, float(1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))])
        curves[f"{om:g}"] = pts
    out["_power_curves"] = curves
    # provenance arm: undisclosed common source seen with partial reach versus the same
    # source disclosed and public, as a function of correlation neglect chi
    rho, n = CONFIG["rho"], CONFIG["n_large"]
    par = {}
    rng = np.random.default_rng(Dc["paradox_seed"])
    for chi in (0.0, 0.5, 1.0):
        th = perceived_precision(n, rho, 1.0 - chi)
        d_und = ex.Design(env=BASE, tau=th, q=Dc["q"], N=Dc["N"], R=Dc["R"], h=t0, tau_true=rho)
        d_dis = ex.Design(env=BASE, tau=rho, q=Dc["q"], N=Dc["N"], R=Dc["R"], h=t0, tau_true=rho)
        t_und = ex.CutoffTable(d_und, zg)
        t_dis = ex.CutoffTable(d_dis, zg)
        tabs = {"disclosed_public": (d_dis, t_dis, "public-coord"), "undisclosed_partial": (d_und, t_und, "partial-coord")}
        sims = [ex.simulate_group(d_dis, t_dis, "public-coord", 1.0, noise, sds, sdg, rng)[0] for _ in range(2000)]
        simu = [ex.simulate_group(d_und, t_und, "partial-coord", 1.0, noise, sds, sdg, rng)[0] for _ in range(2000)]
        diff = float(np.mean(sims) - np.mean(simu))
        sd2 = float(np.var(sims, ddof=1) + np.var(simu, ddof=1))
        G = 2
        while True:
            df = 2 * G - 2
            ncp = abs(diff) / math.sqrt(sd2 / G)
            tc = stats.t.ppf(0.975, df)
            if 1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp) >= 0.8 or G > 2000:
                break
            G += 1
        Pi = rho * multiplier_at_neutral(BASE, rho, 1.0) / (th * multiplier_at_neutral(BASE, th, Dc["q"]))
        par[f"{chi:g}"] = {"tau_undisclosed": th, "Pi": Pi, "slope_diff": diff, "response_gain_pp_per_sd": -100 * diff * zsd,
                           "groups_per_condition_80": G}
    out["provenance_arm"] = par
    # finite group: symmetric equilibrium of the public-bulletin game with N players
    z0 = neutral_message(BASE, Dc["tau"])
    fin = {}
    for N in (Dc["N"], 50):
        kf = {dz: fg.public_cutoff(BASE, Dc["tau"], N, z0 + dz, cutoff(BASE, Dc["tau"], 1.0, z0 + dz))
              for dz in (-0.3, -0.01, 0.0, 0.01)}
        kc = {dz: cutoff(BASE, Dc["tau"], 1.0, z0 + dz) for dz in kf}
        fin[str(N)] = {"response_ratio": ((kf[0.01] - kf[-0.01]) / (kc[0.01] - kc[-0.01])),
                       "effect_ratio_at_minus_0.3": (kf[-0.3] - kf[0.0]) / (kc[-0.3] - kc[0.0]),
                       "level_shift_pp": 100 * (participation(BASE, kf[0.0], t0) - participation(BASE, kc[0.0], t0)),
                       "roots_on_grid": fg.roots_on_grid(BASE, Dc["tau"], N, z0, kc[0.0])}
    out["finite_group"] = fin
    return out


# ====================================================================== figures
def style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


COLORS = ["#1b4f72", "#b9770e", "#7d3c98", "#1e8449", "#922b21"]


def fig_channels(res: dict):
    t0, k0 = no_message(BASE)
    tau = CONFIG["rho"]
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.6))
    a = ax[0, 0]
    zs = np.linspace(t0 - 0.8, t0 + 0.8, 161)
    for v, col in zip((0.0, 0.25, 0.5, 1.0), COLORS):
        ys = [100 * (participation(BASE, cutoff(BASE, tau, v, float(zz)), t0) - t0) for zz in zs]
        a.plot(zs - t0, ys, color=col, label=f"reach {v:g}" + (" (evidential only)" if v == 0 else ""))
    a.axhline(0, color="0.6", lw=0.6)
    a.axvline(0, color="0.6", lw=0.6)
    a.set_xlabel(r"message content relative to $\theta_0$, $z-\theta_0$")
    a.set_ylabel("change in participation at $\\theta_0$ (pp)")
    a.set_title("(a) Effect of the message by reach")
    a.legend(frameon=False, loc="upper right")
    a = ax[0, 1]
    vs = np.linspace(0, 1, 101)
    for t_, col in zip((0.5, 1.0, 3.0), COLORS):
        a.plot(vs, [multiplier_at_neutral(BASE, t_, v) for v in vs], color=col, label=f"$\\tau$ = {t_:g}")
    a.set_xlabel("reach $v$")
    a.set_ylabel("coordination multiplier $M$")
    a.set_title("(b) Multiplier on the evidential response")
    a.legend(frameon=False, loc="upper left")
    a = ax[1, 0]
    mp = res["paradox"]["_map"]
    S = np.array(mp["sign"])
    a.contourf(mp["R"], mp["v0"], S, levels=[-1.5, 0, 1.5], colors=["#f2f3f4", "#d6eaf8"])
    a.plot(mp["boundary"], mp["v0"], color=COLORS[0], lw=1.4, label=r"first-order boundary $\Pi=1$")
    a.set_xlabel(r"deflation of perceived precision $R=\tau_h/\tau_l$")
    a.set_ylabel(r"reach before disclosure $v_0$")
    a.set_title("(c) Public disclosure raises participation (shaded)")
    a.legend(frameon=False, loc="upper right")
    a = ax[1, 1]
    z = t0 - CONFIG["surprise"]
    tau_h = res["isolated"]["tau_h"]
    tls = np.linspace(0.0, tau_h, 61)
    iso, com = [], []
    for v, col, ls in ((CONFIG["v_private"], COLORS[1], "-"), (1.0, COLORS[0], "-")):
        iso = [-slopes(BASE, [Group(v, tau_h), Group(1e-12, float(tl))], z)["dk_dz"][1] * BASE.beta for tl in tls]
        com = [(-slopes(BASE, [Group(v, float(tl))], z)["dk_dz"][0] * BASE.beta) if tl > 0 else 0.0 for tl in tls]
        a.plot(tls, iso, color=col, ls="-", label=f"isolated, reach {v:g}")
        a.plot(tls, com, color=col, ls="--", label=f"common knowledge, reach {v:g}")
    a.set_xlabel(r"disclosed receiver's perceived precision $\tau_l$")
    a.set_ylabel(r"response to content, $-\beta\,dk/dz$")
    a.set_title("(d) Isolated versus common disclosure")
    a.set_ylim(-1.0, 31.0)
    a.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "channels.pdf"))
    plt.close(fig)


def fig_dynamics(res: dict):
    D = res["dynamics"]
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.6))
    a = ax[0, 0]
    gms = np.linspace(0.05, 2.0, 200)
    lo, hi = [], []
    for gm in gms:
        iv = dyn.bistable_interval(BASE, gm)
        lo.append(iv[0] if iv else np.nan)
        hi.append(iv[1] if iv else np.nan)
    a.fill_between(gms, lo, hi, color="#d6eaf8", label="two stable states")
    a.axvline(D["gamma_c_closed_form"], color=COLORS[0], lw=1.2, label=r"$\gamma_c$ (closed form)")
    a.plot([D["baseline"]["gamma"]], [D["baseline"]["theta_bar"]], "o", color=COLORS[4], ms=4, label="baseline")
    a.set_xlabel(r"erosion $\gamma$")
    a.set_ylabel(r"baseline strength $\bar\theta$")
    a.set_title("(a) Bistability region")
    a.legend(frameon=False, loc="upper left")
    a = ax[0, 1]
    tr = D["_trajectories"]
    for name, col, lab in (("below", COLORS[0], "content just past threshold"),
                           ("above", COLORS[1], "content just short of threshold"),
                           ("unique", COLORS[2], r"same campaign, $\gamma<\gamma_c$")):
        a.plot(range(1, len(tr[name]) + 1), tr[name], color=col, label=lab, marker="o", ms=2, lw=1)
    a.axvspan(0.5, 4.5, color="0.92")
    a.set_xlabel("period")
    a.set_ylabel("participation $A_t$")
    a.set_title("(b) A four-period campaign")
    a.legend(frameon=False, loc="center right")
    a = ax[1, 0]
    Ts = CONFIG["dynamics"]["campaign_lengths"]
    a.plot(Ts, [D["critical_message"][str(T)]["undisclosed"] for T in Ts], "o-", color=COLORS[0],
           label=r"undisclosed ($\tau$ = %g)" % res["isolated"]["tau_h"])
    a.plot(Ts, [D["critical_message"][str(T)]["disclosed"] for T in Ts], "s-", color=COLORS[1],
           label=r"disclosed ($\tau$ = %g)" % CONFIG["rho"])
    a.set_xscale("log", base=2)
    a.set_xticks(Ts)
    a.set_xticklabels([str(T) for T in Ts])
    a.set_xlabel("campaign length (periods)")
    a.set_ylabel(r"critical content $z_c$")
    a.set_title("(c) Least mobilizing content that persists")
    a.legend(frameon=False, loc="lower right")
    a = ax[1, 1]
    for g_, col in zip(CONFIG["dynamics"]["shock_gammas"], COLORS):
        ys = []
        for sd in CONFIG["dynamics"]["shock_sds"]:
            rr = D["shocks"][f"g{g_:g}_s{sd:g}"]
            ys.append(rr["mean_residence_high"] if rr["exits_high"] > 0 else rr["periods"])
        a.plot(CONFIG["dynamics"]["shock_sds"], ys, "o-", color=col, label=f"$\\gamma$ = {g_:g}")
    a.set_yscale("log")
    a.axvline(D["consistent_shock_sd"], color="0.5", lw=0.8, ls=":")
    a.set_xlabel(r"shock s.d. $\sigma$ (dotted: prior s.d.)")
    a.set_ylabel("mean periods in high state")
    a.set_title("(d) Persistence with shocks")
    a.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dynamics.pdf"))
    plt.close(fig)


def fig_design(res: dict):
    Dr = res["design"]
    Id = res["identification"]
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.6))
    a = ax[0, 0]
    labels = ["public\ncoord", "partial\ncoord", "public\nhurdle", "partial\nhurdle"]
    width = 0.2
    for i, om in enumerate(CONFIG["design"]["omegas"] + [0.0]):
        row = Dr["by_omega"][f"{om:g}"]["cells"]
        vals = [-100 * row[c]["mean"] * Dr["design"]["bulletin_sd"] for c in ex.CELLS]
        a.bar(np.arange(4) + (i - 1.5) * width, vals, width, color=COLORS[i], label=f"$\\omega$ = {om:g}")
    a.set_xticks(np.arange(4))
    a.set_xticklabels(labels)
    a.set_ylabel("participation per s.d. of bulletin (pp)")
    a.set_title("(a) Predicted response to the bulletin")
    a.legend(frameon=False, loc="upper right")
    a = ax[0, 1]
    for om, col in zip(CONFIG["design"]["omegas"], COLORS):
        pts = np.array(Dr["_power_curves"][f"{om:g}"])
        a.plot(pts[:, 0], pts[:, 1], color=col, label=f"$\\omega$ = {om:g}")
    a.axhline(0.8, color="0.6", lw=0.6, ls=":")
    a.set_xscale("log")
    a.set_xlabel("groups per cell")
    a.set_ylabel("power (two-sided 5%)")
    a.set_title("(b) Power for the interaction")
    a.legend(frameon=False, loc="lower right")
    a = ax[1, 0]
    cv = np.array(Id["iso_response"])
    a.plot(cv[:, 0], cv[:, 1], color=COLORS[0])
    a.plot([1.0], [CONFIG["rho"]], "o", color=COLORS[4], ms=4)
    a.set_xlabel("reach $v$")
    a.set_ylabel(r"perceived precision $\tau$")
    a.set_title("(c) Equal responses to content")
    a = ax[1, 1]
    chis = [0.0, 0.5, 1.0]
    pa = Dr["provenance_arm"]
    a.bar(np.arange(3), [pa[f"{c:g}"]["response_gain_pp_per_sd"] for c in chis], 0.5, color=COLORS[0])
    a.axhline(0, color="0.5", lw=0.6)
    a.set_xticks(np.arange(3))
    a.set_xticklabels([f"neglect $\\chi$ = {c:g}" for c in chis])
    a.set_ylabel("response gain, disclosed-public\nover undisclosed-partial (pp per s.d.)")
    a.set_title("(d) Provenance arm, five accounts")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "design.pdf"))
    plt.close(fig)


# ====================================================================== main
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    started = time.time()
    res = {"config": CONFIG, "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                                             "platform": platform.platform()}}
    res["tests"] = run_tests()
    res["static"] = static_baseline()
    res["certificates"] = certificates()
    res["propositions"] = verify_propositions()
    res["decomposition"] = decomposition()
    res["isolated"] = isolated_disclosure()
    res["paradox"] = paradox()
    res["aware"] = aware_robustness()
    res["dynamics"] = dynamics_results()
    res["identification"] = identification()
    res["design"] = design_results()
    style()
    fig_channels(res)
    fig_dynamics(res)
    fig_design(res)
    res["runtime_seconds"] = time.time() - started
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("tests", "certificates", "runtime_seconds")}, indent=1))


if __name__ == "__main__":
    main()
