"""Unit tests for the equilibrium code. run_all.py runs them first and aborts on failure."""
import math
import unittest

import numpy as np

from aware import nonreceiver_cutoff, receiver_cutoff
from dynamics import bistable_interval, gamma_critical_closed_form, gamma_critical_numeric, steady_states
from model import (Env, Group, Phiinv, certify, cutoff, evidential, multiplier, multiplier_at_neutral,
                   neutral_message, no_message, perceived_precision, slopes, solve)

ENV = Env(alpha=4.0, mu=0.6, beta=25.0, c=0.5)


class StaticModel(unittest.TestCase):
    def test_no_message_fixed_point(self):
        for env in (ENV, Env(1.0, 0.4, 9.0, 0.3), Env(4.0, 0.6, 100.0, 0.7)):
            t0, k0 = no_message(env)
            lhs = float(Phiinv(t0))
            rhs = (env.alpha / math.sqrt(env.beta)) * (t0 - env.mu) - math.sqrt(1 + env.alpha / env.beta) * float(
                Phiinv(env.c))
            self.assertAlmostEqual(lhs, rhs, places=11)

    def test_certificate_heterogeneous(self):
        cert = certify(ENV, [Group(0.3, 3.0), Group(0.4, 1.0)], 0.3)
        self.assertTrue(cert["unique_roots"])
        self.assertLess(cert["max_abs_indifference_error"], 1e-9)
        self.assertLess(cert["max_abs_gap_iterated_vs_analytic"], 1e-8)

    def test_evidential_closed_form(self):
        for env in (ENV, Env(1.0, 0.4, 9.0, 0.3)):
            for tau in (0.5, 2.0):
                for z in (-0.2, 0.3, 0.9):
                    _, k0 = no_message(env)
                    self.assertAlmostEqual(cutoff(env, tau, 0.0, z) - k0, evidential(env, tau, z), places=12)
                    self.assertAlmostEqual(cutoff(env, tau, 1e-9, z) - k0, evidential(env, tau, z), places=7)

    def test_neutral_message(self):
        for env in (ENV, Env(1.0, 0.4, 9.0, 0.3), Env(4.0, 0.6, 100.0, 0.7)):
            _, k0 = no_message(env)
            for tau in (0.5, 2.0):
                z0 = neutral_message(env, tau)
                for v in (0.2, 0.7, 1.0):
                    self.assertAlmostEqual(cutoff(env, tau, v, z0), k0, places=10)

    def test_multiplier_matches_finite_difference(self):
        for v in (0.3, 1.0):
            for z in (0.1, 0.45, 0.8):
                h = 1e-6
                fd = (cutoff(ENV, 1.0, v, z + h) - cutoff(ENV, 1.0, v, z - h)) / (2 * h)
                self.assertAlmostEqual(fd / (-1.0 / ENV.beta), multiplier(ENV, 1.0, v, z), places=5)

    def test_multiplier_closed_form_at_neutral(self):
        for tau in (0.5, 1.0, 3.0):
            for v in (0.25, 1.0):
                z0 = neutral_message(ENV, tau)
                self.assertAlmostEqual(multiplier(ENV, tau, v, z0), multiplier_at_neutral(ENV, tau, v), places=9)

    def test_zero_precision_has_no_effect(self):
        _, k0 = no_message(ENV)
        for v in (0.2, 1.0):
            self.assertAlmostEqual(cutoff(ENV, 0.0, v, -2.0), k0, places=12)

    def test_isolated_disclosure_keeps_response(self):
        sl = slopes(ENV, [Group(0.5, 3.0), Group(1e-9, 0.0)], 0.2)
        self.assertLess(sl["dk_dz"][1], 0.0)  # responds to content despite zero own precision

    def test_perceived_precision(self):
        self.assertAlmostEqual(perceived_precision(5, 1.0, 0.0), 5.0)
        self.assertAlmostEqual(perceived_precision(5, 1.0, 1.0), 1.0)


class Aware(unittest.TestCase):
    def test_aware_close_to_unaware_for_small_reach(self):
        _, k0 = no_message(ENV)
        kN = nonreceiver_cutoff(ENV, 1.0, 0.01)
        self.assertLess(abs(kN - k0), 1e-4)
        self.assertAlmostEqual(receiver_cutoff(ENV, 1.0, 0.01, 0.3, kN), cutoff(ENV, 1.0, 0.01, 0.3), places=4)


class Dynamics(unittest.TestCase):
    def test_gamma_critical(self):
        self.assertAlmostEqual(gamma_critical_closed_form(ENV), gamma_critical_numeric(ENV), places=9)

    def test_three_states_inside_interval(self):
        gc = gamma_critical_closed_form(ENV)
        gm = 2.5 * gc
        lo, hi = bistable_interval(ENV, gm)
        self.assertEqual(len(steady_states(ENV, gm, 0.5 * (lo + hi))), 3)
        self.assertEqual(len(steady_states(ENV, gm, hi + 0.05)), 1)
        self.assertIsNone(bistable_interval(ENV, 0.9 * gc))


if __name__ == "__main__":
    unittest.main()
