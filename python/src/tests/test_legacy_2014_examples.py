import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ex5_hypersensitive import run_example as run_hypersensitive
from experiments.ex6_nonsmoothham import run_example as run_nonsmooth
from experiments.ex7_singular import run_example as run_singular


class LegacyPaper2014ExamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plt.show = lambda *args, **kwargs: None

    def test_example_31_hypersensitive_matches_legacy_setup_and_terminal_behavior(self):
        result = run_hypersensitive()
        last = result["log"][-1]
        x_terminal = float(result["X"][-1, 0])

        self.assertEqual(result["problem"].T, 25.0)
        self.assertAlmostEqual(float(result["problem"].x0[0]), 1.0)
        self.assertGreaterEqual(last["N"], 300)
        self.assertGreaterEqual(last["M"], 20)
        self.assertLess(result["delta"], 1.0e-3)
        self.assertLess(abs(x_terminal - 1.0), 5.0e-4)
        self.assertLessEqual(last["eta_time"], last["tol_time_star"])
        self.assertLess(last["eta_PA"], 1.0e-2)
        self.assertLess(last["eta_delta"], 1.0e-2)
        self.assertTrue(last["all_indicators_within_tolerance"])

    def test_example_32_nonsmooth_tracks_exact_cost_reasonably(self):
        result = run_nonsmooth()
        last = result["log"][-1]

        mesh = np.asarray(result["t_nodes"], dtype=float)
        X = np.asarray(result["X"], dtype=float)[:, 0]
        J_mesh = float(np.sum(np.diff(mesh) * (X[:-1] ** 10)))
        J_star = float((0.5 ** 11) / 11.0)
        rel_err = abs(J_mesh - J_star) / J_star

        self.assertEqual(result["problem"].T, 1.0)
        self.assertAlmostEqual(float(result["problem"].x0[0]), 0.5)
        self.assertGreaterEqual(last["N"], 100)
        self.assertEqual(last["M"], 3)
        self.assertLess(result["delta"], 1.0e-6)
        self.assertLessEqual(last["eta_time"], last["tol_time_star"])
        self.assertAlmostEqual(last["eta_PA"], 0.0)
        self.assertLessEqual(last["eta_delta"], 1.0e-6)
        self.assertLess(rel_err, 1.0e-2)
        self.assertLess(abs(float(result["X"][-1, 0])), 1.5e-1)

    def test_example_33_singular_tracking_reaches_small_objective(self):
        result = run_singular()
        last = result["log"][-1]

        self.assertEqual(result["problem"].T, 4.0)
        self.assertAlmostEqual(float(result["problem_data"]["epsilon"]), 1.0e-10)
        self.assertAlmostEqual(float(result["problem_data"]["beta"]), 0.75)
        self.assertAlmostEqual(float(result["problem_data"]["t0"]), 5.0 / 3.0)
        self.assertGreaterEqual(last["N"], 1000)
        self.assertLess(last["objective_mesh_approx"], 1.0e-4)
        self.assertLessEqual(last["eta_time"], 1.0e-5)


if __name__ == "__main__":
    unittest.main()
