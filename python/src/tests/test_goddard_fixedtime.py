import unittest

import numpy as np

from core.hamiltonian import compute_H
from experiments.ex8_goddard_fixedtime import (
    build_goddard_problem,
    goddard_candidate_controls,
    goddard_constrained_control,
    goddard_pressure_margin,
    goddard_pressure_time_derivative,
    goddard_singular_control,
    goddard_terminal_cost,
    goddard_terminal_gradient,
    run_goddard_barrier_then_qmax_continuation,
    run_goddard_solver,
)


class GoddardFixedTimeTests(unittest.TestCase):
    def test_terminal_penalty_gradient_matches_finite_difference(self):
        x = np.array([1.2, 0.3, 0.72], dtype=float)
        rho_m = 1.0e4
        m_f = 0.6
        analytic = goddard_terminal_gradient(x, rho_m=rho_m, m_f=m_f)
        eps = 1.0e-7
        numeric = np.zeros(3)
        for i in range(3):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            numeric[i] = (
                goddard_terminal_cost(x_plus, rho_m=rho_m, m_f=m_f)
                - goddard_terminal_cost(x_minus, rho_m=rho_m, m_f=m_f)
            ) / (2.0 * eps)
        self.assertTrue(np.allclose(analytic, numeric, rtol=1e-6, atol=1e-6))

    def test_constrained_control_satisfies_pressure_tangency(self):
        q_max = 10.0
        b = 6200.0
        beta = 500.0
        C_D = 0.05
        v = np.sqrt(q_max / b)
        x = np.array([1.0, v, 0.8], dtype=float)
        u_c = goddard_constrained_control(x, b=b, beta=beta, C_D=C_D)
        dot_g = goddard_pressure_time_derivative(x, np.array([u_c]), b=b, beta=beta, C_D=C_D)
        self.assertAlmostEqual(goddard_pressure_margin(x, q_max=q_max, b=b, beta=beta), 0.0, places=10)
        self.assertAlmostEqual(dot_g, 0.0, places=9)

    def test_singular_candidate_is_finite_for_regular_point(self):
        x = np.array([1.02, 0.08, 0.85], dtype=float)
        u_s = goddard_singular_control(x, b=6200.0, beta=500.0, C_D=0.05, c=0.5)
        self.assertIsNotNone(u_s)
        self.assertTrue(np.isfinite(u_s))

    def test_restricted_oracle_uses_step_feasible_local_control_on_active_boundary(self):
        prob, params = build_goddard_problem(q_max=10.0)
        v = np.sqrt(params["q_max"] / params["b"])
        x = np.array([1.0, v, 1.0], dtype=float)
        p = np.array([0.0, 0.0, 1.0], dtype=float)
        dt = 0.05

        value_free, u_free = compute_H(
            prob,
            p=p,
            x=x,
            t=0.0,
            candidate_controls=[],
            restricted=False,
            use_oracle=True,
        )
        value_restricted, u_restricted = compute_H(
            prob,
            p=p,
            x=x,
            t=0.0,
            candidate_controls=[],
            restricted=True,
            use_oracle=True,
            dt=dt,
        )

        self.assertAlmostEqual(float(u_free[0]), params["T_max"], places=7)
        self.assertTrue(prob.step_feasible_control(x, u_restricted, 0.0, dt))
        self.assertLess(float(u_restricted[0]), params["T_max"])
        self.assertGreaterEqual(value_restricted, value_free - 1e-9)

    def test_step_feasible_control_rejects_one_step_pressure_violation(self):
        prob, params = build_goddard_problem(q_max=10.0)
        v = np.sqrt(params["q_max"] / params["b"])
        x = np.array([1.0, v, 1.0], dtype=float)
        self.assertTrue(prob.step_feasible_control(x, np.array([0.0]), 0.0, 1.0e-3))
        self.assertFalse(prob.step_feasible_control(x, np.array([params["T_max"]]), 0.0, 1.0e-1))

    def test_problem_specific_feasibility_refinement_flags_thin_margin(self):
        prob, params = build_goddard_problem(q_max=17.0, feasibility_margin_fraction=0.1)
        x = np.array([1.0, 0.05, 0.6], dtype=float)
        p = np.array([-1.0, 0.0, -0.05], dtype=float)
        issue = prob.feasibility_refinement_fn(x, p, 0.03, 0.1, 1.0e-8)
        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "thin_step_margin")
        self.assertLessEqual(issue["predicted_pressure_margin"], issue["margin_threshold"])

    def test_candidate_list_includes_singular_and_boundary_controls(self):
        x = np.array([1.02, 0.08, 0.85], dtype=float)
        p = np.array([0.0, 1.0, 0.1], dtype=float)
        candidates = goddard_candidate_controls(
            x,
            p,
            T_max=3.5,
            q_max=10.0,
            b=6200.0,
            beta=500.0,
            C_D=0.05,
            c=0.5,
        )
        names = [name for name, _ in candidates]
        self.assertIn("coast", names)
        self.assertIn("full_thrust", names)
        self.assertIn("singular_arc", names)

    def test_smoke_run_returns_solution_and_log(self):
        result, prob = run_goddard_solver(
            n_init=6,
            T=0.15,
            q_max=10.0,
            rho_m=1.0e3,
            tol_time=1.0e-2,
            tol_PA=1.0e-2,
            tol_delta=1.0e-2,
            max_iters=1,
            verbose=False,
        )
        self.assertEqual(prob.n, 3)
        self.assertEqual(result["X"].shape[1], 3)
        self.assertEqual(result["P"].shape[1], 3)
        self.assertGreaterEqual(len(result["log"]), 1)
        self.assertIn("problem_data", result)

    def test_barrier_then_qmax_continuation_smoke_reuses_handoff(self):
        combined = run_goddard_barrier_then_qmax_continuation(
            mu_schedule=[1.0e-1, 3.0e-2],
            q_schedule=[20.0, 18.0],
            T=0.15,
            rho_m=1.0e3,
            tol_time=1.0e-2,
            tol_PA=1.0e-2,
            tol_delta=1.0e-2,
            delta0=5.0e-2,
            barrier_stage_max_iters=1,
            handoff_max_iters=1,
            continuation_stage_max_iters=1,
            verbose=False,
            store_iterates=False,
        )
        self.assertEqual(combined["q_schedule"], [20.0, 18.0])
        self.assertGreaterEqual(len(combined["stages"]), 1)
        self.assertEqual(combined["stages"][0]["source"], "true_from_barrier_handoff")
        self.assertIsNotNone(combined["last_successful_result"])
        self.assertIsNotNone(combined["last_successful_problem"])


if __name__ == "__main__":
    unittest.main()
