import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import issparse

from core.adaptivity import (
    choose_adaptive_action,
    compute_pa_ranking_scores,
    select_pa_enrichment_candidates,
    solve_optimal_control,
    validate_indicator_tolerances,
)
from core.hamiltonian import compute_H
from core.problem import OCPProblem
from core.smoothing import eval_H_smooth
from experiments.ex2_double_integrator import (
    build_example2_config,
    build_example2_problem,
    export_example2_artifacts,
    initialize_tau_augmented_guess,
    make_initial_bundle_for_config,
    pack_tau_augmented_unknowns,
    tau_augmented_jacobian,
    tau_augmented_residual,
    unpack_tau_augmented_unknowns,
)
from experiments.ex6_nonsmoothham import build_initial_guess_arrays, build_nonsmooth_problem


class ManuscriptConsistencyTests(unittest.TestCase):
    def test_example2_variant_configs_preserve_manuscript_and_comparison_cases(self):
        manuscript = build_example2_config("manuscript_tau_box")
        fixed_box = build_example2_config("archive_fixed_box")
        fixed_free = build_example2_config("archive_fixed_unconstrained")

        self.assertTrue(manuscript.use_tau)
        self.assertEqual(manuscript.penalty_weight, 1.0e4)
        self.assertEqual(manuscript.control_bounds[0][0], -1.0)
        self.assertEqual(manuscript.control_bounds[1][0], 1.0)
        self.assertTrue(np.allclose(manuscript.state_bounds[0], np.array([-2.0, -2.0])))
        self.assertTrue(np.allclose(manuscript.state_bounds[1], np.array([2.0, 2.0])))

        self.assertFalse(fixed_box.use_tau)
        self.assertEqual(fixed_box.T, 2.0)
        self.assertEqual(fixed_box.state_constraint_label, "x1 <= 0")

        self.assertFalse(fixed_free.use_tau)
        self.assertEqual(fixed_free.T, 2.0)
        self.assertIsNone(fixed_free.state_bounds)
        self.assertEqual(fixed_free.state_constraint_label, "none")

    def test_tau_augmented_unknown_layout_and_residual_sizes(self):
        config = build_example2_config("manuscript_tau_box")
        problem = build_example2_problem(config)
        mesh = np.linspace(0.0, 1.0, config.initial_nodes)
        bundle = make_initial_bundle_for_config(config)
        X, P, p_tau, tau = initialize_tau_augmented_guess(config, mesh)

        z = pack_tau_augmented_unknowns(X, P, p_tau, tau)
        X2, P2, p_tau2, tau2 = unpack_tau_augmented_unknowns(z, config.x0, len(mesh))
        residual = tau_augmented_residual(problem, mesh, z, bundle, config.delta0, config.target, config.penalty_weight)
        jacobian = tau_augmented_jacobian(problem, mesh, z, bundle, config.delta0, config.target, config.penalty_weight)

        expected_unknowns = 5 * (len(mesh) - 1) + 4
        self.assertEqual(z.shape, (expected_unknowns,))
        self.assertEqual(X2.shape, X.shape)
        self.assertEqual(P2.shape, P.shape)
        self.assertEqual(p_tau2.shape, p_tau.shape)
        self.assertAlmostEqual(tau2, tau)
        self.assertEqual(residual.shape, (expected_unknowns,))
        self.assertEqual(jacobian.shape, (expected_unknowns, expected_unknowns))
        self.assertTrue(issparse(jacobian))

    def test_restricted_hamiltonian_is_infeasible_when_no_viable_control_exists(self):
        def dynamics(x, u, t):
            return np.array([u[0]])

        def stage_cost(x, u, t):
            return float(u[0] ** 2)

        def terminal_cost(x):
            return float(x[0] ** 2)

        problem = OCPProblem(
            dynamics=dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            x0=np.array([0.0]),
            T=1.0,
            control_bounds=(np.array([1.0]), np.array([1.0])),
            state_bounds=(np.array([-np.inf]), np.array([0.0])),
        )

        value, control = compute_H(
            problem,
            p=np.array([0.0]),
            x=np.array([0.0]),
            t=0.0,
            candidate_controls=[],
            restricted=True,
        )

        self.assertTrue(np.isinf(value))
        self.assertIsNone(control)

    def test_final_resolve_log_keeps_action_and_indicator_arrays(self):
        def dynamics(x, u, t):
            return np.array([u[0]])

        def stage_cost(x, u, t):
            return 1.0 + 0.5 * float(u[0] ** 2)

        def terminal_cost(x):
            return 10.0 * float(x[0] ** 2)

        problem = OCPProblem(
            dynamics=dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            x0=np.array([1.0]),
            T=1.0,
            control_bounds=(np.array([-1.0]), np.array([1.0])),
        )

        result = solve_optimal_control(
            problem,
            initial_mesh=np.linspace(0.0, 1.0, 5),
            tol_time=1.0,
            tol_PA=1.0,
            tol_delta=1e-6,
            max_iters=1,
            delta0=0.2,
            verbose=False,
            log_path=None,
        )

        last = result["log"][-1]
        self.assertEqual(last["note"], "final_resolve")
        self.assertEqual(last["action"], "final_resolve")
        self.assertIn("rho", last)
        self.assertIn("rho_bar", last)
        self.assertIn("r_bar", last)
        self.assertIn("t_nodes_iter", last)
        self.assertEqual(len(last["rho"]), len(result["t_nodes"]) - 1)
        self.assertEqual(len(last["rho_bar"]), len(result["t_nodes"]) - 1)
        self.assertEqual(len(last["r_bar"]), len(result["t_nodes"]) - 1)
        self.assertEqual(len(last["t_nodes_iter"]), len(result["t_nodes"]))
        self.assertIn("settings", result)
        self.assertAlmostEqual(result["settings"]["s_time"], 0.5)
        self.assertAlmostEqual(result["settings"]["newton_tol"], 1e-10)
        self.assertEqual(result["settings"]["newton_max_iter"], 50)
        self.assertAlmostEqual(result["settings"]["time_balance_ratio"], 0.1)
        self.assertAlmostEqual(result["settings"]["pa_add_fraction"], 0.1)
        self.assertAlmostEqual(result["settings"]["pa_time_separation_factor"], 5.0)
        self.assertAlmostEqual(result["settings"]["pa_gap_floor_ratio"], 0.2)

    def test_time_refinement_is_suppressed_when_non_time_indicators_dominate(self):
        action = choose_adaptive_action(
            eta_time=1.5e-3,
            tol_time_star=1.0e-3,
            eta_PA=4.0e-2,
            tol_PA=1.0e-2,
            eta_delta=2.0e-2,
            tol_delta=1.0e-2,
            n_marked=7,
            explicit_mode=False,
            time_balance_ratio=0.1,
        )
        self.assertEqual(action, "add_plane")

        action = choose_adaptive_action(
            eta_time=1.5e-3,
            tol_time_star=1.0e-3,
            eta_PA=5.0e-3,
            tol_PA=1.0e-2,
            eta_delta=4.0e-2,
            tol_delta=1.0e-2,
            n_marked=7,
            explicit_mode=False,
            time_balance_ratio=0.1,
        )
        self.assertEqual(action, "delta*=0.5")

    def test_pa_candidate_selection_uses_gap_ranking_with_time_separation(self):
        t_nodes = np.linspace(0.0, 1.0, 11)
        pa_gaps = np.array([0.05, 0.2, 1.0, 0.95, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        node_dt, pa_scores = compute_pa_ranking_scores(t_nodes, pa_gaps)
        selected, meta = select_pa_enrichment_candidates(
            t_nodes,
            pa_scores,
            target_count=2,
            separation_factor=5.0,
            gap_floor_ratio=0.2,
        )

        self.assertTrue(np.allclose(node_dt, 0.1))
        self.assertEqual(selected, [2, 8])
        self.assertEqual(meta["target_count"], 2)
        self.assertGreaterEqual(meta["rejected_by_time_separation"], 1)
        self.assertAlmostEqual(meta["max_score"], 0.1)
        self.assertAlmostEqual(meta["score_floor"], 0.02)

        action = choose_adaptive_action(
            eta_time=9.0e-4,
            tol_time_star=5.0e-4,
            eta_PA=1.0e-2,
            tol_PA=1.0e-2,
            eta_delta=1.0e-2,
            tol_delta=1.0e-2,
            n_marked=7,
            explicit_mode=False,
            time_balance_ratio=0.1,
        )
        self.assertEqual(action, "delta*=0.5")

    def test_pa_candidate_selection_uses_time_weighted_scores_on_nonuniform_mesh(self):
        t_nodes = np.array([0.0, 0.1, 1.0, 2.0])
        pa_gaps = np.array([1.0, 0.6, 0.55, 0.2])
        node_dt, pa_scores = compute_pa_ranking_scores(t_nodes, pa_gaps)

        self.assertTrue(np.allclose(node_dt, np.array([0.1, 0.9, 1.0, 1.0])))
        self.assertTrue(np.allclose(pa_scores, np.array([0.1, 0.54, 0.55, 0.2])))

        selected, meta = select_pa_enrichment_candidates(
            t_nodes,
            pa_scores,
            target_count=1,
            separation_factor=0.0,
            gap_floor_ratio=0.2,
        )

        self.assertEqual(selected, [2])
        self.assertAlmostEqual(meta["max_score"], 0.55)

    def test_indicator_tolerances_remain_comparable_to_time_tolerance(self):
        validate_indicator_tolerances(1e-2, 1e-2, 1e-2)
        validate_indicator_tolerances(5e-3, 1e-2, 1e-2)

    def test_example32_initial_guess_modes_follow_exact_trajectory_shapes(self):
        mesh = np.linspace(0.0, 1.0, 6)

        X_default, P_default = build_initial_guess_arrays(mesh, mode="default")
        self.assertIsNone(X_default)
        self.assertIsNone(P_default)

        X_exact_state, P_exact_state = build_initial_guess_arrays(mesh, mode="exact_state")
        self.assertEqual(X_exact_state.shape, (6, 1))
        self.assertIsNone(P_exact_state)
        self.assertAlmostEqual(X_exact_state[0, 0], 0.5)
        self.assertAlmostEqual(X_exact_state[-1, 0], 0.0)

        X_exact_both, P_exact_both = build_initial_guess_arrays(mesh, mode="exact_state_costate")
        self.assertEqual(X_exact_both.shape, (6, 1))
        self.assertEqual(P_exact_both.shape, (6, 1))
        self.assertAlmostEqual(P_exact_both[0, 0], 0.5 ** 10)
        self.assertAlmostEqual(P_exact_both[-1, 0], 0.0)

        with self.assertRaises(ValueError):
            validate_indicator_tolerances(5e-3, 1.1e-2, 1e-2)

    def test_example32_uses_closed_form_smoothed_hamiltonian(self):
        problem, _ = build_nonsmooth_problem()

        class DummyBundle:
            def num_planes(self):
                return 1

        H_delta, grad_p, grad_x = eval_H_smooth(
            problem,
            DummyBundle(),
            p=np.array([0.3]),
            x=np.array([0.2]),
            t=0.0,
            delta=0.4,
        )

        expected_radial = np.sqrt(0.3 ** 2 + 0.4 ** 2)
        self.assertAlmostEqual(H_delta, 0.2 ** 10 - expected_radial)
        self.assertTrue(np.allclose(grad_p, np.array([-0.3 / expected_radial])))
        self.assertTrue(np.allclose(grad_x, np.array([10.0 * (0.2 ** 9)])))

        with self.assertRaises(ValueError):
            validate_indicator_tolerances(5e-3, 1e-2, 1.1e-2)

    def test_example2_export_writes_summary_trace_and_figures(self):
        class DummyBundle:
            def __init__(self, n_planes):
                self._n_planes = n_planes

            def num_planes(self):
                return self._n_planes

        config = build_example2_config("manuscript_tau_box")
        problem = build_example2_problem(config)
        mesh = np.linspace(0.0, 1.0, 3)
        X = np.array([[-1.0, 0.0], [-0.5, 1.0], [0.0, 0.0]])
        P = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])
        p_tau = np.array([0.0, 0.5, 1.0])
        controls = np.array([[1.0], [1.0], [-1.0]])

        result = {
            "config": config,
            "problem": problem,
            "t_nodes": mesh,
            "X": X,
            "P": P,
            "p_tau": p_tau,
            "tau": 2.0,
            "bundle": DummyBundle(5),
            "delta": 0.1,
            "controls": controls,
            "estimated_final_time": 2.0,
            "estimated_switch_time": 0.5,
            "log": [
                {
                    "iteration": 0,
                    "N": 2,
                    "M": 5,
                    "delta": 0.1,
                    "tau": 2.0,
                    "newton_iter": 3,
                    "newton_residual": 1.0e-12,
                    "eta_time": 1.0e-4,
                    "tol_time_star": 5.0e-4,
                    "eta_PA": 0.0,
                    "eta_delta": 2.0e-4,
                    "rho": np.array([0.2, 0.2]),
                    "r_bar": np.array([0.1, 0.1]),
                    "action": "STOP",
                    "note": "",
                }
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "outputs"
            figure_dir = out_dir / "figures"
            summary = export_example2_artifacts(result, out_dir=out_dir, figure_dir=figure_dir)

            self.assertEqual(summary["variant"], "manuscript_tau_box")
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "summary_detailed.json").exists())
            self.assertTrue((out_dir / "outer_trace.csv").exists())
            self.assertTrue((figure_dir / "manuscript_tau_box_state_costate.pdf").exists())
            self.assertTrue((figure_dir / "manuscript_tau_box_control_stepsize.pdf").exists())
            self.assertTrue((figure_dir / "manuscript_tau_box_rho_density.pdf").exists())
            self.assertTrue((figure_dir / "manuscript_tau_box_r_indicator.pdf").exists())
            self.assertTrue((figure_dir / "manuscript_tau_box_indicators.pdf").exists())


if __name__ == "__main__":
    unittest.main()
