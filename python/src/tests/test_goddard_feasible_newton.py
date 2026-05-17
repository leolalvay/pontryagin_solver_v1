import unittest

import numpy as np

from core.newton import solve_tpbvp
from core.pa_bundle import PABundle
from experiments.ex8_goddard_fixedtime import build_goddard_problem, build_goddard_initial_guess


class GoddardFeasibleNewtonTests(unittest.TestCase):
    def test_problem_state_feasibility_and_projection(self):
        prob, params = build_goddard_problem(T=0.15, q_max=17.0, rho_m=1.0e4)
        infeasible = np.array([1.0, 0.06, 0.8], dtype=float)
        self.assertFalse(prob.state_feasible(infeasible, 0.05))
        projected = prob.project_state(infeasible, 0.05)
        self.assertTrue(prob.state_feasible(projected, 0.05, tol=1.0e-10))

    def test_fraction_to_boundary_caps_velocity_step(self):
        prob, _ = build_goddard_problem(T=0.15, q_max=17.0, rho_m=1.0e4)
        X = np.array([[1.0, 0.0, 1.0], [1.0, 0.04, 0.9]], dtype=float)
        dX = np.array([[0.0, 0.0, 0.0], [0.0, 0.04, 0.0]], dtype=float)
        t_nodes = np.array([0.0, 0.05], dtype=float)
        lam = prob.fraction_to_boundary_step(X, dX, t_nodes, safety=0.9)
        self.assertLess(lam, 1.0)
        X_trial = X + lam * dX
        self.assertTrue(prob.trajectory_feasible(X_trial, t_nodes, tol=1.0e-10))

    def test_newton_returns_feasible_trajectory_for_goddard_smoke(self):
        prob, params = build_goddard_problem(T=0.15, q_max=17.0, rho_m=1.0e4)
        t_nodes = np.linspace(0.0, params["T"], 8)
        X_init, P_init = build_goddard_initial_guess(t_nodes, m_f=params["m_f"])
        bundle = PABundle()
        bundle.add_control(np.array([0.5 * params["T_max"]]))
        bundle.add_control(np.array([0.0]))
        bundle.add_control(np.array([params["T_max"]]))
        X, P, info = solve_tpbvp(
            prob,
            t_nodes,
            bundle,
            delta=5.0e-2,
            X_init=X_init,
            P_init=P_init,
            max_iter=5,
            tol=1.0e-8,
        )
        self.assertEqual(X.shape[1], 3)
        self.assertTrue(prob.trajectory_feasible(X, t_nodes, tol=1.0e-8))
        self.assertIn("feasibility_rejections", info)
        self.assertIn("projection_fallbacks", info)


if __name__ == "__main__":
    unittest.main()
