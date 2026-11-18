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
from model import (Env, Group, Phi, Phiinv, certify, cutoff, evidential, kappa, multiplicity_share,  # noqa: E402
                   multiplier, multiplier_at_neutral, neutral_message, no_message, no_message_roots,
                   perceived_precision, realized, slopes, solve)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIG = os.path.normpath(os.path.join(HERE, "..", "paper", "figures"))

CONFIG = {
    "baseline": {"alpha": 4.0, "mu": 0.6, "beta": 25.0, "c": 0.5},
    "rho": 1.0,  # precision of one report
    "n_small": 3, "n_large": 5,  # accounts in the campaign
    "surprise": 0.3,  # baseline mobilizing message: z = theta0 - surprise
    "v_private": 0.25,  # commonly believed reach before public disclosure
    "post_neglect": [0.0, 0.25, 0.5],  # neglect of a disclosed (known) correlation
    "arm_post_neglect": [0.0, 0.5],  # provenance arm of the experiment
    "label_p": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],  # probability that another receiver noticed a label
    "large_campaign_precisions": [9.0, 20.0, 40.0, 80.0],
    "dynamics": {"gamma": 0.6, "theta_bar_position": 0.5, "campaign_lengths": [1, 2, 4, 8],
                 "shock_sds": [0.1, 0.2, 0.3, 0.5], "shock_gammas": [0.6, 1.0], "shock_periods": 50000,
                 "shock_seed": 20261118, "disclosure_campaign": 6,
                 "precise": {"alpha": 16.0, "beta": 100.0, "campaign_tau": 8.0},
                 "campaign_length": 4, "campaign_offset": 0.05, "horizons": [1, 5, 10, 50, 200],
                 "paths": 4000, "path_seed": 20261119},
    "design": {"N": 20, "R": 30, "q": 0.25, "tau": 1.0, "noise_sd_units": 0.5, "subject_sd_units": 0.5,
               "group_sd_units": 0.25, "omegas": [1.0, 0.5, 0.25], "calibration_groups": 4000,
               "calibration_seed": 11, "power_reps": 2000, "power_seed": 12, "size_seed": 13,
               "paradox_seed": 14, "power_target": 0.8, "beliefs": [0.25, 0.5, 0.75],
               "belief_omegas": [1.0, 0.5], "finite_partial_zgrid": 201},
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
    out = {"z": z, "tau": tau, "z0": neutral_message(BASE, tau), "kappa": kappa(BASE, tau), "by_reach": {}}
    E = evidential(BASE, tau, z)
    for v in (0.0, 0.25, 0.5, 1.0):
        k = cutoff(BASE, tau, v, z)
        out["by_reach"][f"{v:.2f}"] = {
            "k_minus_k0": k - k0, "evidential": E, "coordination": (k - k0) - E,
            "coordination_share": ((k - k0) - E) / (k - k0),
            "kappa_v": kappa(BASE, tau) * v,
            "participation_pp": 100 * (participation(BASE, k, t0) - t0),
            "evidential_pp": 100 * (participation(BASE, k0 + E, t0) - t0),
            "multiplier_exact": multiplier(BASE, tau, v, z) if v > 0 else 1.0,
            "multiplier_neutral": multiplier_at_neutral(BASE, tau, v),
            "average_multiplier": (k - k0) / E}
    # identity M0 = 1/(1 - kappa v) on a grid
    err = 0.0
    for t_ in (0.5, 1.0, 3.0):
        for v in (0.1, 0.25, 0.5, 0.75, 1.0):
            err = max(err, abs(multiplier_at_neutral(BASE, t_, v) - 1 / (1 - kappa(BASE, t_) * v)))
    out["kappa_identity_max_error"] = err
    out["kappa_by_tau"] = {f"{t_:g}": kappa(BASE, t_) for t_ in (0.5, 1.0, 3.0)}
    # zero-precision limit
    small = 1e-4
    zz = neutral_message(BASE, small) - s
    Es = evidential(BASE, small, zz)
    ks = cutoff(BASE, small, 1.0, zz)
    out["small_precision"] = {"tau": small, "C_over_E": ((ks - k0) - Es) / Es,
                              "closed_form_limit": multiplier_at_neutral(BASE, 0.0, 1.0) - 1.0,
                              "k_minus_k0_at_zero": cutoff(BASE, 0.0, 1.0, z) - k0}
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
            cf = ((tau_l - tau_h) * (sl["theta"] - z) - float(Phiinv(BASE.c)) * (
                math.sqrt(BASE.alpha + tau_l + BASE.beta) - math.sqrt(BASE.alpha + tau_h + BASE.beta))) / BASE.beta
            pp_h = 100 * (participation(BASE, k_h, t0) - t0)
            pp_l = 100 * (participation(BASE, k_l, t0) - t0)
            res[name] = {"tau_l": tau_l, "share_of_effect_removed": (pp_h - pp_l) / pp_h,
                         "share_of_cutoff_shift_removed": (k_h - k_l) / (k_h - k0),
                         "closed_form_error": abs((k_l - k_h) - cf),
                         "response_ratio_to_undisclosed": sl["dk_dz"][1] / sl["dk_dz"][0],
                         "participation_pp_undisclosed": pp_h, "participation_pp_disclosed": pp_l}
        kc = cutoff(BASE, rho, v, z)
        kh = cutoff(BASE, tau_h, v, z)
        Eh, El = evidential(BASE, tau_h, z), evidential(BASE, rho, z)
        res["common_disclosure"] = {
            "participation_pp": 100 * (participation(BASE, kc, t0) - t0),
            "uninformative_participation_pp": 100 * (participation(BASE, cutoff(BASE, 0.0, v, z), t0) - t0),
            "share_of_cutoff_shift_removed": 1 - (kc - k0) / (kh - k0),
            "evidential_fall": 1 - El / Eh, "coordination_fall": 1 - ((kc - k0) - El) / ((kh - k0) - Eh),
            "C_over_E_before": ((kh - k0) - Eh) / Eh, "C_over_E_after": ((kc - k0) - El) / El}
        out[f"{v:.2f}"] = res
    out["tau_h"] = tau_h
    out["max_closed_form_error"] = max(out[v][name]["closed_form_error"] for v in ("0.25", "1.00")
                                       for name in ("common_source", "uninformative"))
    full = out["1.00"]
    out["common_share_removed_full_reach"] = 1 - full["common_disclosure"]["participation_pp"] / full["common_source"][
        "participation_pp_undisclosed"]
    return out


def label_disclosure() -> dict:
    """A label each receiver notices independently with probability p, commonly known.
    Noticers (mass p v) assign tau_l, the rest tau_h. p -> 0 is the isolated case, p = 1 common."""
    t0, k0 = no_message(BASE)
    z = t0 - CONFIG["surprise"]
    tau_h = perceived_precision(CONFIG["n_small"], CONFIG["rho"], 0.0)
    tau_l = CONFIG["rho"]
    v = 1.0
    pp_h = 100 * (participation(BASE, cutoff(BASE, tau_h, v, z), t0) - t0)
    rows = []
    for p in CONFIG["label_p"]:
        pe = min(max(p, 1e-12), 1 - 1e-12)
        eq = solve(BASE, [Group(v * pe, tau_l), Group(v * (1 - pe), tau_h)], z)
        k_l, k_h = eq.k
        pp_l = 100 * (participation(BASE, k_l, t0) - t0)
        pp_u = 100 * (participation(BASE, k_h, t0) - t0)
        avg = pe * pp_l + (1 - pe) * pp_u
        rows.append({"p": p, "noticer_pp": pp_l, "non_noticer_pp": pp_u, "average_pp": avg,
                     "noticer_share_removed": (pp_h - pp_l) / pp_h, "average_share_removed": (pp_h - avg) / pp_h,
                     "unique": eq.unique})
    return {"undisclosed_pp": pp_h, "tau_h": tau_h, "tau_l": tau_l, "rows": rows}


def paradox() -> dict:
    """Before disclosure receivers believe the n accounts independent (tau_h = n rho). Disclosure
    reveals that they copy one report; a neglect chi' of that known correlation leaves
    tau_l = n rho/(1 + (n - 1)(1 - chi'))."""
    t0, k0 = no_message(BASE)
    s = CONFIG["surprise"]
    z = t0 - s
    rho = CONFIG["rho"]
    v0 = CONFIG["v_private"]
    out = {"examples": {}}
    for n in (CONFIG["n_small"], CONFIG["n_large"]):
        tau_h = perceived_precision(n, rho, 0.0)
        kb = cutoff(BASE, tau_h, v0, z)
        ex_ = {"tau_h": tau_h, "before_pp": 100 * (participation(BASE, kb, t0) - t0), "by_post_neglect": {}}
        for chi in CONFIG["post_neglect"]:
            tau_l = perceived_precision(n, rho, 1.0 - chi)
            ka = cutoff(BASE, tau_l, 1.0, z)
            kp = cutoff(BASE, tau_l, v0, z)
            Pi = tau_l * multiplier_at_neutral(BASE, tau_l, 1.0) / (tau_h * multiplier_at_neutral(BASE, tau_h, v0))
            ex_["by_post_neglect"][f"{chi:g}"] = {
                "tau_l": tau_l, "Pi": Pi, "after_public_pp": 100 * (participation(BASE, ka, t0) - t0),
                "after_private_pp": 100 * (participation(BASE, kp, t0) - t0), "public_raises": bool(ka > kb)}
        out["examples"][str(n)] = ex_
    n = CONFIG["n_large"]

    def Pi_chi(chi):
        tl = perceived_precision(n, rho, 1.0 - chi)
        th = perceived_precision(n, rho, 0.0)
        return tl * multiplier_at_neutral(BASE, tl, 1.0) / (th * multiplier_at_neutral(BASE, th, v0)) - 1.0

    chi_star = brentq(Pi_chi, 0.0, 1.0 - 1e-9, xtol=1e-12)
    out["post_neglect_threshold"] = {"n": n, "chi_star": chi_star,
                                     "tau_l_at_chi_star": perceived_precision(n, rho, 1.0 - chi_star)}
    agree, total, skipped = {}, {}, 0
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
                                amplifies = (ka - kb) * sign > 0
                                key = f"{sz:g}"
                                total[key] = total.get(key, 0) + 1
                                agree[key] = agree.get(key, 0) + int(amplifies == (Pi > 1))
    out["local_criterion"] = {k: {"points": total[k], "agree": agree[k], "share": agree[k] / total[k]} for k in total}
    out["local_criterion_nonunique_skipped"] = skipped
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


def receipt_vs_belief() -> dict:
    """Actual receipt v_a versus commonly believed receipt v_b, baseline message tau = 1."""
    t0, k0 = no_message(BASE)
    z = t0 - CONFIG["surprise"]
    tau = CONFIG["rho"]
    rows = {}
    for va in (0.25, 1.0):
        for vb in (0.0, 0.25, 0.5, 1.0):
            rr = realized(BASE, tau, vb, va, z)
            rows[f"a{va:g}_b{vb:g}"] = {"v_actual": va, "v_belief": vb, "receiver_pp": rr["receiver_pp"],
                                        "aggregate_pp": rr["aggregate_pp"], "theta_actual": rr["theta_actual"],
                                        "response_multiplier": multiplier(BASE, tau, vb, z) if vb > 0 else 1.0}
    return {"rows": rows, "theta0": t0}


def large_campaigns() -> dict:
    """Outside the uniqueness bound: share of contents within theta0 +/- 1 with several equilibria."""
    t0, _ = no_message(BASE)
    zs = np.linspace(t0 - 1.0, t0 + 1.0, 201)
    bound = math.sqrt(2 * math.pi * BASE.beta) - BASE.alpha
    out = {"precision_bound": bound, "max_independent_reports": int(math.floor(bound / CONFIG["rho"])),
           "shares": {f"{t:g}": multiplicity_share(BASE, t, 1.0, zs) for t in CONFIG["large_campaign_precisions"]}}
    n = 80
    # neglect chi with n rho/(1 + (n-1)(1-chi)) <= bound
    out["neglect_bound_80"] = 1.0 - (n * CONFIG["rho"] / bound - 1.0) / (n - 1)
    out["roots_80_at_baseline_message"] = len(solve(BASE, [Group(1.0, 80.0)], t0 - CONFIG["surprise"]).thetas)
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
    for a, b in ((1.0, 9.0), (4.0, 100.0), (16.0, 100.0), (4.0, 400.0)):
        e = replace(BASE, alpha=a, beta=b)
        out["gamma_c_by_precision"][f"a{a:g}_b{b:g}"] = {"closed": dyn.gamma_critical_closed_form(e),
                                                         "numeric": dyn.gamma_critical_numeric(e)}
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
    out["baseline"] = {"gamma": gm, "interval": [lo, hi], "theta_bar": tb, "steady_states": ss}
    rho = CONFIG["rho"]
    tau_h = perceived_precision(CONFIG["n_small"], rho, 0.0)
    crit = {}
    for T in D["campaign_lengths"]:
        crit[str(T)] = {"undisclosed": dyn.critical_message(BASE, gm, tb, tau_h, 1.0, T),
                        "disclosed": dyn.critical_message(BASE, gm, tb, rho, 1.0, T)}
    out["critical_message"] = crit
    T = D["campaign_length"]
    zc = crit[str(T)]["undisclosed"]
    sharp = {}
    for dz in (-1e-6, 1e-6):
        sched = lambda t, zz=zc + dz: (zz, tau_h, 1.0) if t < T else None
        p, path = dyn.persistent(BASE, gm, tb, sched, T)
        sharp[f"{dz:+g}"] = {"persistent": p, "final_A": path["A"][-1]}
    out["sharpness"] = sharp
    traj = {}
    off = D["campaign_offset"]
    for name, zz in (("below", zc - off), ("above", zc + off)):
        sched = lambda t, zz=zz: (zz, tau_h, 1.0) if t < T else None
        traj[name] = dyn.simulate(BASE, gm, tb, ss[0]["mu"], sched, 30)["A"]
    gm_u = 0.8 * gc
    ss_u = dyn.steady_states(BASE, gm_u, tb)
    sched = lambda t, zz=zc - off: (zz, tau_h, 1.0) if t < T else None
    traj["unique"] = dyn.simulate(BASE, gm_u, tb, ss_u[0]["mu"], sched, 30)["A"]
    out["_trajectories"] = traj
    out["unique_regime"] = {"gamma": gm_u, "steady_states": len(ss_u), "final_A": traj["unique"][-1],
                            "initial_A": ss_u[0]["A"]}
    Tc = D["disclosure_campaign"]
    z3 = dyn.critical_message(BASE, gm, tb, tau_h, 1.0, 3)
    zd = 0.5 * (z3 + crit["4"]["undisclosed"])
    timing = []
    for td in range(Tc + 1):
        sched = lambda t, td=td: ((zd, tau_h if t < td else rho, 1.0) if t < Tc else None)
        p, path = dyn.persistent(BASE, gm, tb, sched, Tc)
        timing.append({"disclosure_period": td, "persistent": p})
    first_persistent = next((x["disclosure_period"] for x in timing if x["persistent"]), None)
    sched = lambda t: (zd, tau_h, 1.0) if t < Tc else None
    path = dyn.simulate(BASE, gm, tb, ss[0]["mu"], sched, Tc)
    cross = next((t + 1 for t, mu in enumerate(path["mu"][1:]) if mu < ss[1]["mu"]), None)
    out["disclosure_timing"] = {"campaign_length": Tc, "z": zd, "z_c_3": z3, "timing": timing,
                                "latest_preventing_disclosure": (first_persistent - 1) if first_persistent else None,
                                "undisclosed_crossing_period": cross}
    # belief-consistent shocks: residence in each state and stationary shares
    rows = {}
    for g_ in D["shock_gammas"]:
        lo_, hi_ = dyn.bistable_interval(BASE, g_)
        tb_ = lo_ + D["theta_bar_position"] * (hi_ - lo_)
        for sd in D["shock_sds"]:
            rows[f"g{g_:g}_s{sd:g}"] = dyn.residence_times(BASE, g_, tb_, sd, D["shock_periods"], D["shock_seed"])
    out["shocks"] = rows
    out["consistent_shock_sd"] = 1 / math.sqrt(BASE.alpha)
    # temporary campaign under belief-consistent shocks, baseline and precise environments
    camp = {}
    envs = {"baseline": (BASE, tau_h), "precise": (replace(BASE, alpha=D["precise"]["alpha"], beta=D["precise"]["beta"]),
                                                   D["precise"]["campaign_tau"])}
    for name, (env, tau_c) in envs.items():
        lo_, hi_ = dyn.bistable_interval(env, gm)
        tb_ = lo_ + D["theta_bar_position"] * (hi_ - lo_)
        ss_ = dyn.steady_states(env, gm, tb_)
        zc_ = dyn.critical_message(env, gm, tb_, tau_c, 1.0, T)
        sd = 1 / math.sqrt(env.alpha)
        res = {"alpha": env.alpha, "beta": env.beta, "theta_bar": tb_, "shock_sd": sd, "campaign_tau": tau_c,
               "z_c": zc_, "steady_states": ss_, "gamma_c": dyn.gamma_critical_closed_form(env),
               "residence": dyn.residence_times(env, gm, tb_, sd, D["shock_periods"], D["shock_seed"])}
        for lab, L, zz in (("none", 0, zc_), ("past", T, zc_ - off), ("short", T, zc_ + off)):
            res[lab] = dyn.stochastic_campaign(env, gm, tb_, sd, (zz, tau_c, 1.0), L, D["horizons"], D["paths"],
                                               D["path_seed"])
        camp[name] = res
    out["campaign_shocks"] = camp
    return out


# ====================================================================== identification
def identification() -> dict:
    t0, k0 = no_message(BASE)
    z = t0 - CONFIG["surprise"]
    tau, v = CONFIG["rho"], 1.0
    k = cutoff(BASE, tau, v, z)
    dk = k - k0
    tau_E = brentq(lambda t: evidential(BASE, t, z) - dk, 1e-6, 200.0, xtol=1e-12)
    s0 = math.sqrt(BASE.alpha + BASE.beta)

    def kp(cprime):
        return t0 + BASE.alpha * (t0 - BASE.mu) / BASE.beta - s0 * float(Phiinv(cprime)) / BASE.beta

    c_p = brentq(lambda cc: kp(cc) - k, 1e-9, BASE.c, xtol=1e-14)
    b = dk

    # majority illusion about others: the citizen believes others' cutoff is k0 + b_o
    def k_illusion(bo):
        th = brentq(lambda t: float(Phi(RB * (k0 + bo - t))) - t, -1.0, 2.0, xtol=1e-14)
        return th + (BASE.alpha * (th - BASE.mu) / math.sqrt(BASE.beta)
                     - math.sqrt((BASE.alpha + BASE.beta) / BASE.beta) * float(Phiinv(BASE.c))) / RB

    b_o = brentq(lambda bo: k_illusion(bo) - k, 0.0, 2.0, xtol=1e-14)
    out = {"z": z, "target_k_minus_k0": dk, "target_pp": 100 * (participation(BASE, k, t0) - t0),
           "private_evidence_tau": tau_E, "persuasion_cost": c_p, "persuasion_cost_cut": BASE.c - c_p,
           "illusion_bias": b, "illusion_others_shift": b_o}
    sig = {}
    for name, tt, lvl in (("coordination", tau, evidential(BASE, tau, z)),
                          ("private_evidence", tau_E, evidential(BASE, tau_E, z)),
                          ("persuasion", 0.0, k - k0),
                          ("illusion", 0.0, b),
                          ("illusion_others", 0.0, 0.0)):
        sig[name] = {"hurdle_weight": tt / BASE.beta,
                     "hurdle_level_pp": 100 * (participation(BASE, k0 + lvl, t0) - t0)}
    out["signatures"] = sig
    q = CONFIG["design"]["q"]
    out["coordination_partial_weight"] = tau * multiplier(BASE, tau, q, z) / BASE.beta
    out["coordination_public_weight"] = tau * multiplier(BASE, tau, 1.0, z) / BASE.beta
    target = tau * multiplier_at_neutral(BASE, tau, 1.0)
    p0 = float(np.exp(-0.5 * float(Phiinv(t0)) ** 2) / math.sqrt(2 * math.pi))
    curve = []
    for vv in np.linspace(0.0, 1.0, 21):
        f = lambda t: t * multiplier_at_neutral(BASE, t, vv) - target
        tmax = 60.0 if vv == 0 else min(60.0, 0.999 * ((RB + (1 - vv) * BASE.beta * p0) / (vv * p0) - BASE.alpha))
        curve.append([float(vv), float(brentq(f, 1e-6, tmax, xtol=1e-12))])
    out["iso_response"] = curve
    out["iso_tau_at_v0"] = curve[0][1]
    out["iso_tau_at_v_half"] = curve[10][1]
    # global (weak) identification from curvature: Kullback-Leibler divergence per noise-free choice
    # between the reference (tau = 1, v = 1) and the iso-response alternatives, with theta drawn from the
    # prior and the observed bulletin drawn with precision 1 around theta (lab data: theta and z observed)
    zg = np.linspace(t0 - 4.0, t0 + 4.0, 321)
    tabs = {"ref": np.array([cutoff(BASE, tau, 1.0, float(x)) for x in zg]),
            "half": np.array([cutoff(BASE, curve[10][1], 0.5, float(x)) for x in zg]),
            "zero": np.array([cutoff(BASE, curve[0][1], 0.0, float(x)) for x in zg])}
    rng = np.random.default_rng(CONFIG["design"]["calibration_seed"] + 100)
    th = rng.normal(BASE.mu, 1 / math.sqrt(BASE.alpha), 200000)
    zz = th + rng.normal(0, 1, th.size)
    keep = (zz > zg[0]) & (zz < zg[-1])
    th, zz = th[keep], zz[keep]
    pr = np.clip(Phi(RB * (np.interp(zz, zg, tabs["ref"]) - th)), 1e-12, 1 - 1e-12)
    kl = {}
    for name in ("half", "zero"):
        pa = np.clip(Phi(RB * (np.interp(zz, zg, tabs[name]) - th)), 1e-12, 1 - 1e-12)
        d = float(np.mean(pr * np.log(pr / pa) + (1 - pr) * np.log((1 - pr) / (1 - pa))))
        kl[name] = {"kl_per_choice": d, "choices_for_lr20": math.log(20.0) / d,
                    "max_abs_diff_pp_central": float(100 * np.max(np.abs(
                        Phi(RB * (np.interp(np.linspace(t0 - 2.2, t0 + 2.2, 101), zg, tabs["ref"]) - t0))
                        - Phi(RB * (np.interp(np.linspace(t0 - 2.2, t0 + 2.2, 101), zg, tabs[name]) - t0)))))}
    out["weak_identification"] = kl
    return out


# ====================================================================== design
def _analytic_G(diff, sd_sum, cells=4, target=0.8):
    G = 2
    while True:
        df = cells * G - cells
        ncp = abs(diff) / math.sqrt(sd_sum / G)
        tc = stats.t.ppf(0.975, df)
        if 1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp) >= target or G > 3000:
            return G
        G += 1


def _two_cell_power(specs, omega, noise, sds, sdg, G, reps, seed):
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(reps):
        est = [np.array([ex.simulate_group(d, t, c, omega, noise, sds, sdg, rng)[0] for _ in range(G)])
               for d, t, c in specs]
        diff = est[0].mean() - est[1].mean()
        pooled = (((est[0] - est[0].mean()) ** 2).sum() + ((est[1] - est[1].mean()) ** 2).sum()) / (2 * G - 2)
        rej += abs(diff / math.sqrt(pooled * 2 / G)) > stats.t.ppf(0.975, 2 * G - 2)
    return rej / reps


def _two_cell_G(specs, omega, noise, sds, sdg, start, target, reps, seed):
    if start > 500:  # effect too small to be a practical target; report the analytic count only
        return {"groups_per_condition": start, "simulated_power": None}
    G = max(2, start)
    while True:
        pw = _two_cell_power(specs, omega, noise, sds, sdg, G, reps, seed)
        if pw >= target or G > 2000:
            return {"groups_per_condition": G, "simulated_power": pw}
        G += max(1, int(math.ceil(0.04 * G)))


def design_results() -> dict:
    Dc = CONFIG["design"]
    t0, k0 = no_message(BASE)
    d = ex.Design(env=BASE, tau=Dc["tau"], q=Dc["q"], N=Dc["N"], R=Dc["R"], h=t0)
    zg = np.linspace(-4.0, 5.0, 451)
    tab = ex.CutoffTable(d, zg, beliefs=tuple(Dc["beliefs"]))
    unit = 1 / math.sqrt(BASE.beta)
    noise, sds, sdg = Dc["noise_sd_units"] * unit, Dc["subject_sd_units"] * unit, Dc["group_sd_units"] * unit
    zsd = math.sqrt(1 / BASE.alpha + 1 / Dc["tau"])
    tgt = Dc["power_target"]
    out = {"design": {"N": Dc["N"], "R": Dc["R"], "q": Dc["q"], "tau": Dc["tau"], "hurdle": t0,
                      "bulletin_sd": zsd, "kN_minus_k0": tab.kN - k0},
           "theory": {"hurdle_relative_weight": Dc["tau"] / BASE.beta,
                      "public_multiplier_neutral": multiplier_at_neutral(BASE, Dc["tau"], 1.0),
                      "partial_multiplier_neutral_unaware": multiplier_at_neutral(BASE, Dc["tau"], Dc["q"])},
           "by_omega": {}}
    main_cells = ("public-coord", "partial-coord", "public-hurdle", "partial-hurdle")
    belief_cells = tuple(f"private-coord-{p:g}" for p in Dc["beliefs"])
    for om in Dc["omegas"] + [0.0]:
        rng = np.random.default_rng(Dc["calibration_seed"])
        cells = {}
        for c in main_cells + belief_cells:
            bz = [ex.simulate_group(d, tab, c, om, noise, sds, sdg, rng)[0] for _ in range(Dc["calibration_groups"])]
            cells[c] = {"mean": float(np.mean(bz)), "sd": float(np.std(bz, ddof=1))}
        inter = (cells["public-coord"]["mean"] - cells["partial-coord"]["mean"]) - (
            cells["public-hurdle"]["mean"] - cells["partial-hurdle"]["mean"])
        row = {"cells": cells, "interaction_slope": inter, "interaction_response_pp_per_sd": -100 * inter * zsd,
               "cell_response_pp_per_sd": {c: -100 * cells[c]["mean"] * zsd for c in cells}}
        if om > 0:
            sd_sum = sum(cells[c]["sd"] ** 2 for c in main_cells)
            G_an = _analytic_G(inter, sd_sum)
            found = ex.groups_for_power(d, tab, main_cells, om, noise, sds, sdg, G_an, tgt, Dc["power_reps"],
                                        Dc["power_seed"])
            row["analytic_groups_per_cell"] = G_an
            row["required"] = {"groups_per_cell": found["groups_per_cell"],
                               "participants": 4 * found["groups_per_cell"] * Dc["N"],
                               "simulated_power": found["simulated_power"], "tried": found["tried"]}
        out["by_omega"][f"{om:g}"] = row
    G_half = out["by_omega"]["0.5"]["required"]["groups_per_cell"]
    out["size_at_omega0"] = {"G": G_half, "rejection_rate": ex.simulated_power(
        d, tab, main_cells, 0.0, noise, sds, sdg, G_half, Dc["power_reps"], Dc["size_seed"])}
    curves = {}
    for om in Dc["omegas"]:
        row = out["by_omega"][f"{om:g}"]
        sd_sum = sum(row["cells"][c]["sd"] ** 2 for c in main_cells)
        pts = []
        for G in range(2, 161):
            df = 4 * G - 4
            ncp = abs(row["interaction_slope"]) / math.sqrt(sd_sum / G)
            tc = stats.t.ppf(0.975, df)
            pts.append([G, float(1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))])
        curves[f"{om:g}"] = pts
    out["_power_curves"] = curves
    # acknowledgment cell: everyone receives; public acknowledgment versus private screens with a truthful
    # statement that every member received the bulletin (common belief of receipt p). Interaction with the
    # incentive: (public-coord - private-coord-p) - (public-hurdle - private-hurdle); the private hurdle
    # cell has the same model cutoff as the public hurdle cell.
    ack = {}
    for om in Dc["belief_omegas"]:
        row = out["by_omega"][f"{om:g}"]
        for p, c in zip(Dc["beliefs"], belief_cells):
            diff = row["cells"]["public-coord"]["mean"] - row["cells"][c]["mean"]
            sd_sum = (row["cells"]["public-coord"]["sd"] ** 2 + row["cells"][c]["sd"] ** 2
                      + 2 * row["cells"]["public-hurdle"]["sd"] ** 2)
            G_an = _analytic_G(diff, sd_sum)
            found = ex.groups_for_power(d, tab, ("public-coord", c, "public-hurdle", "public-hurdle"), om, noise,
                                        sds, sdg, G_an, tgt, Dc["power_reps"], Dc["power_seed"] + 1)
            ack[f"w{om:g}_p{p:g}"] = {"omega": om, "p": p, "response_gap_pp_per_sd": -100 * diff * zsd,
                                     "private_cell_response_pp_per_sd": row["cell_response_pp_per_sd"][c],
                                     "groups_per_cell": found["groups_per_cell"],
                                     "simulated_power": found["simulated_power"]}
    out["acknowledgment"] = ack
    # provenance arm: five accounts believed independent before disclosure (tau_h = 5); after disclosure
    # the known correlation is neglected to degree chi' (tau_l = 5/(1 + 4(1 - chi'))). Undisclosed bulletin
    # with partial visibility versus disclosed bulletin with public visibility. Pi from the aware model's
    # slopes at the neutral message (the model used for the cells) and from the closed form.
    rho, n = CONFIG["rho"], CONFIG["n_large"]
    th = perceived_precision(n, rho, 0.0)
    d_und = ex.Design(env=BASE, tau=th, q=Dc["q"], N=Dc["N"], R=Dc["R"], h=t0, tau_true=rho)
    t_und = ex.CutoffTable(d_und, zg)
    i0 = int(np.argmin(np.abs(zg - t0)))
    slope_und = (t_und.coord["partial-coord"][i0 + 1] - t_und.coord["partial-coord"][i0 - 1]) / (zg[i0 + 1] - zg[i0 - 1])
    par = {}
    rng = np.random.default_rng(Dc["paradox_seed"])
    for chi in CONFIG["arm_post_neglect"]:
        tl = perceived_precision(n, rho, 1.0 - chi)
        d_dis = ex.Design(env=BASE, tau=tl, q=Dc["q"], N=Dc["N"], R=Dc["R"], h=t0, tau_true=rho)
        t_dis = ex.CutoffTable(d_dis, zg)
        slope_dis = (t_dis.coord["public-coord"][i0 + 1] - t_dis.coord["public-coord"][i0 - 1]) / (zg[i0 + 1] - zg[i0 - 1])
        sims = [ex.simulate_group(d_dis, t_dis, "public-coord", 1.0, noise, sds, sdg, rng)[0] for _ in range(4000)]
        simu = [ex.simulate_group(d_und, t_und, "partial-coord", 1.0, noise, sds, sdg, rng)[0] for _ in range(4000)]
        diff = float(np.mean(sims) - np.mean(simu))
        sd2 = float(np.var(sims, ddof=1) + np.var(simu, ddof=1))
        G_an = _analytic_G(diff, sd2, cells=2)
        found = _two_cell_G([(d_dis, t_dis, "public-coord"), (d_und, t_und, "partial-coord")], 1.0, noise, sds, sdg,
                            G_an, tgt, Dc["power_reps"], Dc["paradox_seed"] + 1)
        par[f"{chi:g}"] = {"tau_l": tl, "tau_h": th,
                           "Pi_closed_form": tl * multiplier_at_neutral(BASE, tl, 1.0) / (th * multiplier_at_neutral(BASE, th, Dc["q"])),
                           "Pi_aware": slope_dis / slope_und, "slope_diff": diff,
                           "response_gain_pp_per_sd": -100 * diff * zsd, "groups_per_condition": found["groups_per_condition"],
                           "simulated_power": found["simulated_power"]}
    out["provenance_arm"] = par
    # finite groups: public cell (N players) and partial cell (q N receivers, informed non-receivers)
    z0 = neutral_message(BASE, Dc["tau"])
    fin = {}
    for N in (Dc["N"], 50):
        kf = {dz: fg.public_cutoff(BASE, Dc["tau"], N, z0 + dz, cutoff(BASE, Dc["tau"], 1.0, z0 + dz))
              for dz in (-0.3, -0.01, 0.0, 0.01)}
        kc = {dz: cutoff(BASE, Dc["tau"], 1.0, z0 + dz) for dz in kf}
        fin[str(N)] = {"response_ratio": ((kf[0.01] - kf[-0.01]) / (kc[0.01] - kc[-0.01])),
                       "slope_finite": (kf[0.01] - kf[-0.01]) / 0.02, "slope_continuum": (kc[0.01] - kc[-0.01]) / 0.02,
                       "effect_ratio_at_minus_0.3": (kf[-0.3] - kf[0.0]) / (kc[-0.3] - kc[0.0]),
                       "level_shift_pp": 100 * (participation(BASE, kf[0.0], t0) - participation(BASE, kc[0.0], t0)),
                       "roots_on_grid": fg.roots_on_grid(BASE, Dc["tau"], N, z0, kc[0.0])}
    nR = int(round(Dc["q"] * Dc["N"]))
    zgp = np.linspace(z0 - 5.0, z0 + 5.0, Dc["finite_partial_zgrid"])
    part = fg.partial_equilibrium(BASE, Dc["tau"], Dc["N"], nR, zgp, tab.kN)
    j = int(np.argmin(np.abs(zgp - z0)))
    sl_pf = (part["kR"][j + 1] - part["kR"][j - 1]) / (zgp[j + 1] - zgp[j - 1])
    h = zgp[1] - zgp[0]
    sl_pc = (aware.receiver_cutoff(BASE, Dc["tau"], Dc["q"], z0 + h, tab.kN)
             - aware.receiver_cutoff(BASE, Dc["tau"], Dc["q"], z0 - h, tab.kN)) / (2 * h)
    hurdle_slope = -Dc["tau"] / BASE.beta
    pub = fin[str(Dc["N"])]
    fin["partial"] = {"receivers": nR, "kN_finite": part["kN"], "kN_continuum": tab.kN,
                      "slope_finite": sl_pf, "slope_continuum": sl_pc, "response_ratio": sl_pf / sl_pc,
                      "interaction_ratio": ((pub["slope_finite"] - sl_pf) - 0.0) / ((pub["slope_continuum"] - sl_pc) - 0.0),
                      "hurdle_slope": hurdle_slope}
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
    a.set_xlabel("commonly believed reach $v$")
    a.set_ylabel(r"multiplier $M_0 = 1/(1-\kappa v)$")
    a.set_title("(b) Multiplier on the evidential response")
    a.legend(frameon=False, loc="upper left")
    a = ax[1, 0]
    mp = res["paradox"]["_map"]
    S = np.array(mp["sign"])
    a.contourf(mp["R"], mp["v0"], S, levels=[-1.5, 0, 1.5], colors=["#f2f3f4", "#d6eaf8"])
    a.plot(mp["boundary"], mp["v0"], color=COLORS[0], lw=1.4, label=r"first-order boundary $\Pi=1$")
    a.set_xlabel(r"deflation of perceived precision $\tau_h/\tau_l$")
    a.set_ylabel(r"reach before disclosure $v_0$")
    a.set_title("(c) Public disclosure raises participation")
    a.legend(frameon=False, loc="upper right")
    a = ax[1, 1]
    rows = res["label"]["rows"]
    ps = [r_["p"] for r_ in rows]
    a.plot(ps, [100 * r_["noticer_share_removed"] for r_ in rows], "o-", color=COLORS[0],
           label="receiver who noticed the label")
    a.plot(ps, [100 * r_["average_share_removed"] for r_ in rows], "s--", color=COLORS[1],
           label="average over receivers")
    a.set_ylim(0, 100)
    a.set_xlabel("probability another receiver noticed, $p$")
    a.set_ylabel("effect removed (%)")
    a.set_title("(d) Effect removed by a label")
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
    for name, col, lab in (("below", COLORS[0], "just past threshold"),
                           ("above", COLORS[1], "just short of threshold"),
                           ("unique", COLORS[2], r"past threshold, $\gamma<\gamma_c$")):
        a.plot(range(1, len(tr[name]) + 1), tr[name], color=col, label=lab, marker="o", ms=2, lw=1)
    a.axvspan(0.5, CONFIG["dynamics"]["campaign_length"] + 0.5, color="0.92")
    a.set_xlabel("period")
    a.set_ylabel("participation $A_t$")
    a.set_title("(b) Skeleton: a campaign of 4 periods")
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
    hz = CONFIG["dynamics"]["horizons"]
    for name, col, lab in (("baseline", COLORS[0], "baseline"), ("precise", COLORS[2], "precise")):
        cs = D["campaign_shocks"][name]
        a.plot(hz, [cs["past"]["share_high"][str(h)] for h in hz], "o-", color=col, label=f"{lab}")
        a.plot(hz, [cs["none"]["share_high"][str(h)] for h in hz], "o--", color=col, label=f"{lab}, none")
    a.set_xscale("log")
    a.set_ylim(0, 1.3)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_xlabel("periods after the campaign began")
    a.set_ylabel("share of paths in the high state")
    a.set_title("(d) With belief-consistent shocks")
    a.legend(frameon=False, loc="upper right", ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dynamics.pdf"))
    plt.close(fig)


def fig_design(res: dict):
    Dr = res["design"]
    Id = res["identification"]
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.6))
    a = ax[0, 0]
    cells = ["public-coord", "partial-coord", "private-coord-0.5", "public-hurdle", "partial-hurdle"]
    labels = ["public\ncoord", "partial\ncoord", "private\ncoord\n$p$=0.5", "public\nhurdle", "partial\nhurdle"]
    width = 0.2
    for i, om in enumerate(CONFIG["design"]["omegas"] + [0.0]):
        row = Dr["by_omega"][f"{om:g}"]["cell_response_pp_per_sd"]
        a.bar(np.arange(5) + (i - 1.5) * width, [row[c] for c in cells], width, color=COLORS[i],
              label=f"$\\omega$ = {om:g}")
    a.set_xticks(np.arange(5))
    a.set_xticklabels(labels)
    a.set_ylabel("participation per s.d. of bulletin (pp)")
    a.set_title("(a) Predicted response to the bulletin")
    a.legend(frameon=False, loc="upper right")
    a = ax[0, 1]
    for om, col in zip(CONFIG["design"]["omegas"], COLORS):
        pts = np.array(Dr["_power_curves"][f"{om:g}"])
        a.plot(pts[:, 0], pts[:, 1], color=col, label=f"$\\omega$ = {om:g} (analytic)")
        req = Dr["by_omega"][f"{om:g}"]["required"]
        a.plot([req["groups_per_cell"]], [req["simulated_power"]], "o", color=col, ms=4)
    a.axhline(0.8, color="0.6", lw=0.6, ls=":")
    a.set_xscale("log")
    a.set_xlabel("groups per cell")
    a.set_ylabel("power (two-sided 5%)")
    a.set_title("(b) Power (dots: simulated)")
    a.legend(frameon=False, loc="lower right")
    a = ax[1, 0]
    cv = np.array(Id["iso_response"])
    a.plot(cv[:, 0], cv[:, 1], color=COLORS[0])
    a.plot([1.0], [CONFIG["rho"]], "o", color=COLORS[4], ms=4)
    a.set_xlabel("commonly believed reach $v$")
    a.set_ylabel(r"perceived precision $\tau$")
    a.set_title("(c) Equal responses at the neutral message")
    a = ax[1, 1]
    chis = CONFIG["arm_post_neglect"]
    pa = Dr["provenance_arm"]
    a.bar(np.arange(len(chis)), [pa[f"{c:g}"]["response_gain_pp_per_sd"] for c in chis], 0.5, color=COLORS[0])
    a.axhline(0, color="0.5", lw=0.6)
    a.set_xticks(np.arange(len(chis)))
    a.set_xticklabels([f"$\\chi'$ = {c:g}" for c in chis])
    a.set_xlabel("neglect of the disclosed correlation")
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
    res["label"] = label_disclosure()
    res["paradox"] = paradox()
    res["receipt"] = receipt_vs_belief()
    res["large"] = large_campaigns()
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
