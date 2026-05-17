"""
Initialization diagnostics for legacy 2014 Example 3.2.

This compares three first-guess modes:
- default
- exact_state
- exact_state_costate

For each mode, it runs:
1. the first fixed-mesh TPBVP solve on the initial mesh;
2. the full adaptive solve.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from core.newton import solve_tpbvp
from core.pa_bundle import PABundle
from experiments.ex6_nonsmoothham import (
    _compute_interval_controls,
    _objective_mesh_approx,
    build_initial_guess_arrays,
    build_nonsmooth_problem,
    exact_solution,
    run_nonsmooth_solver,
)


def _make_seed_bundle():
    bundle = PABundle()
    u_min = np.array([-1.0], dtype=float)
    u_max = np.array([1.0], dtype=float)
    bundle.add_control(0.5 * (u_min + u_max))
    bundle.add_control(u_min)
    bundle.add_control(u_max)
    return bundle


def _summarize_controls(control_values, exact_controls):
    control_values = np.asarray(control_values, dtype=float)
    exact_controls = np.asarray(exact_controls, dtype=float)
    return {
        "control_unique_values_rounded": sorted({round(float(v), 8) for v in control_values.tolist()}),
        "control_mean": float(np.mean(control_values)) if len(control_values) else 0.0,
        "control_last": float(control_values[-1]) if len(control_values) else 0.0,
        "control_error_inf": float(np.max(np.abs(control_values - exact_controls))) if len(control_values) else 0.0,
    }


def run_diagnostics():
    problem, problem_data = build_nonsmooth_problem()
    t_initial = np.linspace(0.0, problem_data["T"], 20)
    modes = ["default", "exact_state", "exact_state_costate"]
    out = {"fixed_mesh_first_solve": {}, "full_adaptive": {}}

    for mode in modes:
        X_init, P_init = build_initial_guess_arrays(t_initial, mode=mode)
        bundle = _make_seed_bundle()
        X, P, info = solve_tpbvp(
            problem,
            t_initial,
            bundle,
            0.02,
            X_init=X_init,
            P_init=P_init,
            tol=1.0e-10,
            max_iter=50,
            fallback_solver="least_squares",
        )
        U = _compute_interval_controls(problem, bundle, X, P, t_initial)
        _, _, u_exact = exact_solution(t_initial)
        out["fixed_mesh_first_solve"][mode] = {
            "iterations": int(info["iterations"]),
            "residual_norm": float(info["residual_norm"]),
            "solver_phase": info.get("solver_phase", "newton"),
            "fallback_used": bool(info.get("fallback_used", False)),
            "terminal_state": float(X[-1, 0]),
            "initial_costate": float(P[0, 0]),
            "objective_mesh_approx": float(_objective_mesh_approx(problem, t_initial, X, U)),
            **_summarize_controls(U[:, 0], u_exact),
        }

        adaptive_result, _ = run_nonsmooth_solver(
            n_init=20,
            tol_time=1.0e-6,
            tol_PA=1.0e-6,
            tol_delta=1.0e-6,
            max_iters=25,
            delta0=0.02,
            store_iterates=False,
            verbose=False,
            initial_guess_mode=mode,
        )
        t_final = np.asarray(adaptive_result["t_nodes"], dtype=float)
        U_final = _compute_interval_controls(problem, adaptive_result["bundle"], adaptive_result["X"], adaptive_result["P"], t_final)
        _, _, u_exact_final = exact_solution(t_final)
        out["full_adaptive"][mode] = {
            "final_action": adaptive_result["log"][-1].get("action"),
            "final_note": adaptive_result["log"][-1].get("note"),
            "mesh_intervals": int(len(t_final) - 1),
            "planes": int(adaptive_result["bundle"].num_planes()),
            "eta_time": float(adaptive_result["log"][-1]["eta_time"]),
            "eta_PA": float(adaptive_result["log"][-1]["eta_PA"]),
            "eta_delta": float(adaptive_result["log"][-1]["eta_delta"]),
            "terminal_state": float(adaptive_result["X"][-1, 0]),
            "initial_costate": float(adaptive_result["P"][0, 0]),
            "objective_mesh_approx": float(_objective_mesh_approx(problem, t_final, adaptive_result["X"], U_final)),
            **_summarize_controls(U_final[:, 0], u_exact_final),
        }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("archive_runs/legacy_2014/ex32_initialization_diagnostics.json"),
    )
    args = parser.parse_args()

    diagnostics = run_diagnostics()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
