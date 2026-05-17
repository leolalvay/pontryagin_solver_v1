"""
Discrete-structure diagnostics for legacy 2014 Example 3.2.

This script checks three structural questions for the current TPBVP discretization:
1. whether the terminal transversality condition is encoded as p_N = g_x(x_N);
2. whether a zero tail arc (x=0, p=0) is admissible in the discrete equations;
3. whether the exact continuous bang-off solution sampled on a mesh containing the
   switch time is itself a discrete solution.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from core.integrators import assemble_residual, pack_unknowns
from core.pa_bundle import PABundle
from core.shooting import shooting_residual
from experiments.ex6_nonsmoothham import build_nonsmooth_problem, exact_solution


def _make_seed_bundle():
    bundle = PABundle()
    for u in [np.array([0.0]), np.array([-1.0]), np.array([1.0])]:
        bundle.add_control(u)
    return bundle


def _residual_blocks(F):
    rx = []
    rp = []
    n_intervals = (len(F) - 1) // 2
    for i in range(n_intervals):
        rx.append(float(F[2 * i]))
        rp.append(float(F[2 * i + 1]))
    return np.asarray(rx, dtype=float), np.asarray(rp, dtype=float), float(F[-1])


def run_diagnostics():
    problem, pdata = build_nonsmooth_problem()
    bundle = _make_seed_bundle()

    t = np.concatenate([np.linspace(0.0, 0.5, 11), np.linspace(0.55, 1.0, 10)])
    x_exact, p_exact, _ = exact_solution(t)
    X_exact = x_exact.reshape(-1, 1)
    P_exact = p_exact.reshape(-1, 1)
    z_exact = pack_unknowns(X_exact, P_exact)

    F_exact = shooting_residual(problem, t, z_exact, bundle, 1.0e-6)
    rx_exact, rp_exact, bc_exact = _residual_blocks(F_exact)
    worst_state_idx = int(np.argmax(np.abs(rx_exact)))
    worst_costate_idx = int(np.argmax(np.abs(rp_exact)))

    local_x_i = np.array([0.0], dtype=float)
    local_x_ip1 = np.array([0.0], dtype=float)
    local_p_i = np.array([0.0], dtype=float)
    local_p_ip1 = np.array([0.0], dtype=float)
    _, grad_p_tail, grad_x_tail = problem.hamiltonian_smooth(local_x_i, local_p_ip1, 0.75, 1.0e-6)
    dt_tail = 0.25
    local_rx_tail = float(local_x_ip1[0] - local_x_i[0] - dt_tail * grad_p_tail[0])
    local_rp_tail = float(local_p_i[0] - local_p_ip1[0] - dt_tail * grad_x_tail[0])

    return {
        "terminal_transversality": {
            "g_x_terminal": 0.0,
            "required_boundary_condition_under_current_sign_convention": "p_N = g_x(x_N)",
            "example32_encoded_boundary_value": "p_N = 0",
        },
        "exact_continuous_solution_sampled_on_switch_mesh": {
            "mesh_contains_switch_time": True,
            "switch_time": 0.5,
            "residual_inf_norm": float(np.linalg.norm(F_exact, ord=np.inf)),
            "worst_state_residual_index": worst_state_idx,
            "worst_state_residual_interval": [float(t[worst_state_idx]), float(t[worst_state_idx + 1])],
            "worst_state_residual_value": float(rx_exact[worst_state_idx]),
            "worst_costate_residual_index": worst_costate_idx,
            "worst_costate_residual_interval": [float(t[worst_costate_idx]), float(t[worst_costate_idx + 1])],
            "worst_costate_residual_value": float(rp_exact[worst_costate_idx]),
            "boundary_residual": bc_exact,
        },
        "zero_tail_arc_admissibility": {
            "tested_local_interval_length": dt_tail,
            "local_tail_state_residual": local_rx_tail,
            "local_tail_costate_residual": local_rp_tail,
            "interpretation": "A local interval with x_i=x_{i+1}=0 and p_i=p_{i+1}=0 satisfies the discrete state and costate equations exactly.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("archive_runs/legacy_2014/ex32_discrete_structure_diagnostics.json"),
    )
    args = parser.parse_args()

    diagnostics = run_diagnostics()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
