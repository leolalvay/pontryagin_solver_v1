"""
Example 2 variants for the minimum-time double integrator.

The archive originally contained a single fixed-horizon prototype. This
module now exposes three named variants so the manuscript target and the
comparison runs are not conflated:

- ``manuscript_tau_box``: the Section 4.2 target specification with a
  tau-augmented normalized-time formulation and state box K=[-2,2]^2.
- ``archive_fixed_box``: the original fixed-T=2 archive prototype with
  state constraint x1 <= 0.
- ``archive_fixed_unconstrained``: the same fixed-T=2 prototype without
  state-space restrictions, preserved for comparison.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import lsqr, splu

from core.adaptivity import bootstrap_bundle_from_trajectory, solve_optimal_control
from core.hamiltonian import compute_H
from core.pa_bundle import PABundle
from core.problem import OCPProblem
from core.smoothing import eval_H_smooth


ArrayBounds = Optional[Tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Example2Config:
    name: str
    description: str
    use_tau: bool
    x0: np.ndarray
    target: np.ndarray
    T: float
    control_bounds: Tuple[np.ndarray, np.ndarray]
    state_bounds: ArrayBounds
    state_constraint_label: str
    penalty_weight: float
    running_cost_label: str
    stage_cost_alpha: float
    stage_cost_beta: float
    initial_nodes: int
    tol_time: float
    tol_PA: float
    tol_delta: float
    max_iters: int
    delta0: float
    s_time: float
    K_time: float
    newton_tol: float
    newton_max_iter: int
    initial_bundle_size: int


def build_example2_config(variant: str = "archive_fixed_box") -> Example2Config:
    u_bounds = (np.array([-1.0]), np.array([1.0]))
    common = dict(
        x0=np.array([-1.0, 0.0]),
        target=np.array([0.0, 0.0]),
        control_bounds=u_bounds,
        initial_nodes=21,
        tol_time=5e-3,
        tol_PA=1e-3,
        tol_delta=1e-3,
        max_iters=10,
        s_time=0.5,
        K_time=1e-6,
        newton_tol=1e-10,
        newton_max_iter=50,
    )

    if variant == "manuscript_tau_box":
        return Example2Config(
            name=variant,
            description="Section 4.2 target: tau-augmented normalized-time double integrator with K=[-2,2]^2.",
            use_tau=True,
            T=1.0,
            state_bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
            state_constraint_label="K = [-2,2]^2",
            penalty_weight=1.0e4,
            running_cost_label="tau + rho * ||x(1)-x_T||^2",
            stage_cost_alpha=1.0,
            stage_cost_beta=0.0,
            delta0=0.1,
            initial_bundle_size=5,
            **common,
        )

    if variant == "archive_fixed_box":
        return Example2Config(
            name=variant,
            description="Archive prototype: fixed T=2 with x1 <= 0 and running cost 1 + 1e-2 u^2.",
            use_tau=False,
            T=2.0,
            state_bounds=(np.array([-np.inf, -np.inf]), np.array([0.0, np.inf])),
            state_constraint_label="x1 <= 0",
            penalty_weight=100.0,
            running_cost_label="1 + 1e-2 u^2",
            stage_cost_alpha=1.0,
            stage_cost_beta=1.0e-2,
            delta0=0.2,
            initial_bundle_size=3,
            **common,
        )

    if variant == "archive_fixed_unconstrained":
        return Example2Config(
            name=variant,
            description="Archive comparison case: fixed T=2 without state-space restrictions.",
            use_tau=False,
            T=2.0,
            state_bounds=None,
            state_constraint_label="none",
            penalty_weight=100.0,
            running_cost_label="1 + 1e-2 u^2",
            stage_cost_alpha=1.0,
            stage_cost_beta=1.0e-2,
            delta0=0.2,
            initial_bundle_size=3,
            **common,
        )

    raise ValueError(f"Unknown Example 2 variant: {variant}")


def _build_fixed_horizon_problem(config: Example2Config) -> OCPProblem:
    def dynamics(x, u, t):
        return np.array([x[1], u[0]])

    def stage_cost(x, u, t):
        return float(config.stage_cost_alpha + config.stage_cost_beta * (u[0] ** 2))

    def terminal_cost(x):
        diff = x - config.target
        return float(config.penalty_weight * diff.dot(diff))

    return OCPProblem(
        dynamics,
        stage_cost,
        terminal_cost,
        config.x0,
        config.T,
        control_bounds=config.control_bounds,
        state_bounds=config.state_bounds,
    )


def _build_tau_augmented_problem(config: Example2Config) -> OCPProblem:
    def dynamics(x, u, t):
        return np.array([x[1], u[0]])

    def stage_cost(x, u, t):
        return 0.0

    def terminal_cost(x):
        diff = x - config.target
        return float(config.penalty_weight * diff.dot(diff))

    return OCPProblem(
        dynamics,
        stage_cost,
        terminal_cost,
        config.x0,
        config.T,
        control_bounds=config.control_bounds,
        state_bounds=config.state_bounds,
    )


def build_example2_problem(config: Example2Config) -> OCPProblem:
    if config.use_tau:
        return _build_tau_augmented_problem(config)
    return _build_fixed_horizon_problem(config)


def make_initial_bundle_for_config(config: Example2Config) -> PABundle:
    bundle = PABundle()
    u_min, u_max = config.control_bounds
    if config.initial_bundle_size <= 1:
        bundle.add_control(0.5 * (u_min + u_max))
        return bundle
    for value in np.linspace(float(u_min[0]), float(u_max[0]), config.initial_bundle_size):
        bundle.add_control(np.array([value], dtype=float))
    return bundle


def initialize_tau_augmented_guess(config: Example2Config, t_nodes: np.ndarray):
    n_nodes = len(t_nodes)
    X = np.zeros((n_nodes, 2), dtype=float)
    P = np.zeros((n_nodes, 2), dtype=float)
    p_tau = np.zeros(n_nodes, dtype=float)

    for i, s in enumerate(t_nodes):
        if s <= 0.5:
            X[i, 0] = -1.0 + 2.0 * s * s
            X[i, 1] = 2.0 * s
        else:
            X[i, 0] = -2.0 * (1.0 - s) ** 2
            X[i, 1] = 2.0 * (1.0 - s)
        P[i, 0] = -1.0
        P[i, 1] = 2.0 * s - 1.0
        p_tau[i] = s

    X[0] = config.x0
    tau = 2.0
    return X, P, p_tau, tau


def pack_tau_augmented_unknowns(X: np.ndarray, P: np.ndarray, p_tau: np.ndarray, tau: float) -> np.ndarray:
    return np.concatenate([X[1:, :].reshape(-1), P.reshape(-1), p_tau.reshape(-1), np.array([tau], dtype=float)])


def unpack_tau_augmented_unknowns(z: np.ndarray, x0: np.ndarray, n_nodes: int):
    n_intervals = n_nodes - 1
    offset = 0

    X = np.zeros((n_nodes, 2), dtype=float)
    X[0] = x0
    x_count = 2 * n_intervals
    X[1:, :] = z[offset:offset + x_count].reshape(n_intervals, 2)
    offset += x_count

    p_count = 2 * n_nodes
    P = z[offset:offset + p_count].reshape(n_nodes, 2)
    offset += p_count

    p_tau = z[offset:offset + n_nodes]
    offset += n_nodes

    tau = float(z[offset])
    return X, P, p_tau, tau


def _tau_terminal_gradient(x_terminal: np.ndarray, target: np.ndarray, penalty_weight: float) -> np.ndarray:
    return 2.0 * penalty_weight * (x_terminal - target)


def _example2_smooth_terms(bundle: PABundle, p: np.ndarray, x: np.ndarray, delta: float):
    controls = np.array([float(u[0]) for u in bundle.controls], dtype=float)
    q_vals = p[1] * controls
    q_min = float(np.min(q_vals))
    exps = np.exp(-(q_vals - q_min) / max(delta, 1e-12))
    weights = exps / np.sum(exps)
    u_bar = float(np.dot(weights, controls))
    u_sq_bar = float(np.dot(weights, controls ** 2))
    var_u = max(u_sq_bar - u_bar ** 2, 0.0)
    d_u_bar_dp2 = -var_u / max(delta, 1e-12)

    H_delta = x[1] * p[0] + q_min - delta * np.log(np.sum(exps) + 1e-300)
    grad_p = np.array([x[1], u_bar], dtype=float)
    grad_x = np.array([0.0, p[0]], dtype=float)
    return H_delta, grad_p, grad_x, d_u_bar_dp2


def tau_augmented_residual(
    problem: OCPProblem,
    t_nodes: np.ndarray,
    z: np.ndarray,
    bundle: PABundle,
    delta: float,
    target: np.ndarray,
    penalty_weight: float,
) -> np.ndarray:
    X, P, p_tau, tau = unpack_tau_augmented_unknowns(z, problem.x0, len(t_nodes))
    n_intervals = len(t_nodes) - 1
    residual = np.zeros(5 * n_intervals + 4, dtype=float)
    offset = 0

    for i in range(n_intervals):
        dt = t_nodes[i + 1] - t_nodes[i]
        H_delta, grad_p, grad_x, _ = _example2_smooth_terms(bundle, P[i + 1], X[i], delta)

        residual[offset:offset + 2] = X[i + 1] - X[i] - dt * tau * grad_p
        offset += 2

        residual[offset:offset + 2] = P[i] - P[i + 1] - dt * tau * grad_x
        offset += 2

        residual[offset] = p_tau[i] - p_tau[i + 1] - dt * H_delta
        offset += 1

    residual[offset:offset + 2] = P[-1] - _tau_terminal_gradient(X[-1], target, penalty_weight)
    offset += 2
    residual[offset] = p_tau[0]
    offset += 1
    residual[offset] = p_tau[-1] - 1.0
    return residual


def tau_augmented_jacobian(
    problem: OCPProblem,
    t_nodes: np.ndarray,
    z: np.ndarray,
    bundle: PABundle,
    delta: float,
    target: np.ndarray,
    penalty_weight: float,
    eps: float = 1e-7,
) -> csc_matrix:
    X, P, _, tau = unpack_tau_augmented_unknowns(z, problem.x0, len(t_nodes))
    n_intervals = len(t_nodes) - 1
    n_nodes = len(t_nodes)
    total_unknowns = 5 * n_intervals + 4
    rows = []
    cols = []
    data = []

    def col_x(k: int, dim: int) -> int:
        return 2 * (k - 1) + dim

    def col_p(j: int, dim: int) -> int:
        return 2 * n_intervals + 2 * j + dim

    def col_p_tau(j: int) -> int:
        return 2 * n_intervals + 2 * n_nodes + j

    col_tau = total_unknowns - 1

    def row_x(i: int, dim: int) -> int:
        return 5 * i + dim

    def row_p(i: int, dim: int) -> int:
        return 5 * i + 2 + dim

    def row_p_tau(i: int) -> int:
        return 5 * i + 4

    row_terminal = 5 * n_intervals
    row_p_tau_0 = row_terminal + 2
    row_p_tau_N = row_terminal + 3

    def add_entry(r: int, c: int, value: float):
        rows.append(r)
        cols.append(c)
        data.append(float(value))

    def local_terms(i: int, x_i: np.ndarray, p_ip1: np.ndarray):
        dt = t_nodes[i + 1] - t_nodes[i]
        H_delta, grad_p, grad_x, d_u_bar_dp2 = _example2_smooth_terms(bundle, p_ip1, x_i, delta)
        phi = -dt * tau * grad_p
        psi = -dt * tau * grad_x
        omega = -dt * H_delta
        return phi, psi, omega, grad_p, grad_x, d_u_bar_dp2

    for i in range(n_intervals):
        dt = t_nodes[i + 1] - t_nodes[i]
        x_i = X[i].copy()
        p_ip1 = P[i + 1].copy()
        phi, psi, _, grad_p, grad_x, d_u_bar_dp2 = local_terms(i, x_i, p_ip1)

        for dim in range(2):
            add_entry(row_x(i, dim), col_x(i + 1, dim), +1.0)
            if i >= 1:
                add_entry(row_x(i, dim), col_x(i, dim), -1.0)
            add_entry(row_p(i, dim), col_p(i, dim), +1.0)
            add_entry(row_p(i, dim), col_p(i + 1, dim), -1.0)

            add_entry(row_x(i, dim), col_tau, -dt * grad_p[dim])
            add_entry(row_p(i, dim), col_tau, -dt * grad_x[dim])

        add_entry(row_p_tau(i), col_p_tau(i), +1.0)
        add_entry(row_p_tau(i), col_p_tau(i + 1), -1.0)

        if i >= 1:
            add_entry(row_x(i, 0), col_x(i, 1), -dt * tau)
            add_entry(row_p_tau(i), col_x(i, 1), -dt * P[i + 1, 0])

        add_entry(row_x(i, 1), col_p(i + 1, 1), -dt * tau * d_u_bar_dp2)
        add_entry(row_p(i, 1), col_p(i + 1, 0), -dt * tau)
        add_entry(row_p_tau(i), col_p(i + 1, 0), -dt * X[i, 1])
        add_entry(row_p_tau(i), col_p(i + 1, 1), -dt * grad_p[1])

    for dim in range(2):
        add_entry(row_terminal + dim, col_p(n_intervals, dim), +1.0)
        add_entry(row_terminal + dim, col_x(n_intervals, dim), -2.0 * penalty_weight)

    add_entry(row_p_tau_0, col_p_tau(0), +1.0)
    add_entry(row_p_tau_N, col_p_tau(n_intervals), +1.0)

    return coo_matrix((data, (rows, cols)), shape=(total_unknowns, total_unknowns)).tocsc()


def solve_tau_augmented_tpbvp(
    problem: OCPProblem,
    t_nodes: np.ndarray,
    bundle: PABundle,
    delta: float,
    target: np.ndarray,
    penalty_weight: float,
    X_init: Optional[np.ndarray] = None,
    P_init: Optional[np.ndarray] = None,
    p_tau_init: Optional[np.ndarray] = None,
    tau_init: Optional[float] = None,
    tol: float = 1e-10,
    max_iter: int = 50,
):
    if X_init is None or P_init is None or p_tau_init is None or tau_init is None:
        X_init, P_init, p_tau_init, tau_seed = initialize_tau_augmented_guess(
            build_example2_config("manuscript_tau_box"),
            t_nodes,
        )
        if tau_init is None:
            tau_init = tau_seed
    z = pack_tau_augmented_unknowns(X_init, P_init, p_tau_init, tau_init)

    for it in range(max_iter):
        F = tau_augmented_residual(problem, t_nodes, z, bundle, delta, target, penalty_weight)
        normF = np.linalg.norm(F, ord=np.inf)
        if normF < tol:
            break

        J = tau_augmented_jacobian(problem, t_nodes, z, bundle, delta, target, penalty_weight)
        try:
            dz = splu(J, permc_spec="COLAMD").solve(-F)
        except Exception:
            dz = lsqr(J, -F, atol=1e-12, btol=1e-12, iter_lim=4 * J.shape[0])[0]

        lam = 1.0
        z_new = z + lam * dz
        while z_new[-1] <= 1e-8 and lam > 1e-4:
            lam *= 0.5
            z_new = z + lam * dz

        F_new = tau_augmented_residual(problem, t_nodes, z_new, bundle, delta, target, penalty_weight)
        normF_new = np.linalg.norm(F_new, ord=np.inf)
        while normF_new > (1.0 - 1e-4 * lam) * normF and lam > 1e-4:
            lam *= 0.5
            z_new = z + lam * dz
            if z_new[-1] <= 1e-8:
                continue
            F_new = tau_augmented_residual(problem, t_nodes, z_new, bundle, delta, target, penalty_weight)
            normF_new = np.linalg.norm(F_new, ord=np.inf)
        z = z_new

    X, P, p_tau, tau = unpack_tau_augmented_unknowns(z, problem.x0, len(t_nodes))
    info = {
        "iterations": it + 1,
        "residual_norm": float(
            np.linalg.norm(
                tau_augmented_residual(problem, t_nodes, z, bundle, delta, target, penalty_weight),
                ord=np.inf,
            )
        ),
    }
    return X, P, p_tau, tau, info


def _compute_tau_indicators(
    problem: OCPProblem,
    t_nodes: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    tau: float,
    bundle: PABundle,
    delta: float,
    tol_time: float,
    tol_PA: float,
    tol_delta: float,
    s_time: float,
    K_time: float,
):
    n_intervals = len(t_nodes) - 1
    eta_time_local = np.zeros(n_intervals, dtype=float)
    rho_arr = np.zeros(n_intervals, dtype=float)
    rho_bar_arr = np.zeros(n_intervals, dtype=float)

    if n_intervals > 0:
        dt = np.diff(t_nodes)
        floor = K_time * np.sqrt(float(np.max(dt)))
        for i in range(n_intervals):
            H_delta, grad_p, grad_x, _ = _example2_smooth_terms(bundle, P[i + 1], X[i], delta)
            grad_p_aug = tau * grad_p
            grad_x_aug = tau * grad_x
            rho_arr[i] = -0.5 * float(np.dot(grad_p_aug, grad_x_aug))
            rho_bar_arr[i] = max(abs(rho_arr[i]), floor)
            eta_time_local[i] = abs(rho_bar_arr[i]) * (dt[i] ** 2)
        eta_time = float(np.max(eta_time_local))
        tol_time_star = float(tol_time / n_intervals)
        mark_thr = float(s_time * tol_time / n_intervals)
    else:
        eta_time = 0.0
        tol_time_star = tol_time
        mark_thr = 0.0

    eta_PA = 0.0
    eta_delta = 0.0
    for i in range(n_intervals):
        dt_i = t_nodes[i + 1] - t_nodes[i]
        H_bar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
        H_bar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])
        H_true_i, _ = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
        H_true_ip1, _ = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True)
        H_delta_i, _, _, _ = _example2_smooth_terms(bundle, P[i], X[i], delta)
        H_delta_ip1, _, _, _ = _example2_smooth_terms(bundle, P[i + 1], X[i + 1], delta)

        eta_PA += 0.5 * tau * ((H_bar_i - H_true_i) + (H_bar_ip1 - H_true_ip1)) * dt_i
        eta_delta += 0.5 * tau * ((H_bar_i - H_delta_i) + (H_bar_ip1 - H_delta_ip1)) * dt_i

    n_mark = int(np.sum(eta_time_local > mark_thr)) if n_intervals > 0 else 0
    if (eta_time <= tol_time_star) and (eta_PA <= tol_PA) and (eta_delta <= tol_delta):
        action = "STOP"
    elif eta_time > tol_time_star:
        action = f"refine_time(marked={n_mark})"
    elif eta_PA > tol_PA:
        action = "add_plane"
    elif eta_delta > tol_delta:
        action = "delta*=0.5"
    else:
        action = "continue"

    return {
        "eta_time": float(eta_time),
        "eta_PA": float(eta_PA),
        "eta_delta": float(eta_delta),
        "rho": rho_arr,
        "rho_bar": rho_bar_arr,
        "r_bar": eta_time_local,
        "tol_time_star": float(tol_time_star),
        "mark_thr": float(mark_thr),
        "action": action,
    }


def _interpolate_tau_warm_start(old_nodes, new_nodes, X, P, p_tau):
    X_new = np.column_stack([np.interp(new_nodes, old_nodes, X[:, j]) for j in range(X.shape[1])])
    P_new = np.column_stack([np.interp(new_nodes, old_nodes, P[:, j]) for j in range(P.shape[1])])
    p_tau_new = np.interp(new_nodes, old_nodes, p_tau)
    return X_new, P_new, p_tau_new


def solve_optimal_control_tau_augmented(config: Example2Config, verbose: bool = True):
    problem = build_example2_problem(config)
    t_nodes = np.linspace(0.0, 1.0, config.initial_nodes)
    bundle = make_initial_bundle_for_config(config)
    delta = config.delta0
    log = []

    X_guess = None
    P_guess = None
    p_tau_guess = None
    tau_guess = 2.0

    def _log(message: str):
        if verbose:
            print(message)

    for k in range(config.max_iters):
        X, P, p_tau, tau, info = solve_tau_augmented_tpbvp(
            problem,
            t_nodes,
            bundle,
            delta,
            config.target,
            config.penalty_weight,
            X_init=X_guess,
            P_init=P_guess,
            p_tau_init=p_tau_guess,
            tau_init=tau_guess,
            tol=config.newton_tol,
            max_iter=config.newton_max_iter,
        )

        if k == 0:
            m_before = bundle.num_planes()
            added = bootstrap_bundle_from_trajectory(
                problem,
                t_nodes=t_nodes,
                X=X,
                P=P,
                bundle=bundle,
                restricted=True,
                num_support_nodes=12,
                grid_size=51,
                use_oracle=False,
            )
            _log(f"[bootstrap] M_before={m_before}, added={added}, M_after={bundle.num_planes()}")
            if added > 0:
                X, P, p_tau, tau, info = solve_tau_augmented_tpbvp(
                    problem,
                    t_nodes,
                    bundle,
                    delta,
                    config.target,
                    config.penalty_weight,
                    X_init=X,
                    P_init=P,
                    p_tau_init=p_tau,
                    tau_init=tau,
                    tol=config.newton_tol,
                    max_iter=config.newton_max_iter,
                )

        indicators = _compute_tau_indicators(
            problem,
            t_nodes,
            X,
            P,
            tau,
            bundle,
            delta,
            config.tol_time,
            config.tol_PA,
            config.tol_delta,
            config.s_time,
            config.K_time,
        )

        log_entry = {
            "iteration": k,
            "N": len(t_nodes) - 1,
            "M": bundle.num_planes(),
            "delta": float(delta),
            "tau": float(tau),
            "newton_iter": info["iterations"],
            "newton_residual": info["residual_norm"],
            "t_nodes_iter": t_nodes.copy(),
            "note": "",
            **indicators,
        }
        log.append(log_entry)

        dt_all = np.diff(t_nodes)
        dt_min = float(np.min(dt_all)) if dt_all.size else 0.0
        dt_max = float(np.max(dt_all)) if dt_all.size else 0.0
        _log(
            f"[adapt {k:02d}] N={len(t_nodes)-1:4d} M={bundle.num_planes():3d} "
            f"dt=[{dt_min:.2e},{dt_max:.2e}] delta={delta:.2e} tau={tau:.6f} | "
            f"Newton it={info['iterations']:2d} res={info['residual_norm']:.2e} | "
            f"eta_time={indicators['eta_time']:.2e}/{indicators['tol_time_star']:.2e} "
            f"eta_PA={indicators['eta_PA']:.2e}/{config.tol_PA:.2e} "
            f"eta_delta={indicators['eta_delta']:.2e}/{config.tol_delta:.2e} -> {indicators['action']}"
        )

        if indicators["action"] == "STOP":
            break
        if indicators["action"].startswith("refine_time"):
            new_nodes = [t_nodes[0]]
            for i in range(len(t_nodes) - 1):
                if indicators["r_bar"][i] > indicators["mark_thr"]:
                    new_nodes.append(0.5 * (t_nodes[i] + t_nodes[i + 1]))
                new_nodes.append(t_nodes[i + 1])
            new_nodes = np.array(sorted(set(float(v) for v in new_nodes)))
            X_guess, P_guess, p_tau_guess = _interpolate_tau_warm_start(t_nodes, new_nodes, X, P, p_tau)
            t_nodes = new_nodes
            tau_guess = tau
            continue
        if indicators["action"] == "add_plane":
            max_gap = -np.inf
            best_u = None
            for i in range(len(t_nodes)):
                H_bar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
                H_true_i, u_star = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
                gap = H_bar_i - H_true_i
                if gap > max_gap:
                    max_gap = gap
                    best_u = u_star
            if best_u is not None:
                bundle.add_control(best_u)
            X_guess, P_guess, p_tau_guess, tau_guess = X, P, p_tau, tau
            continue
        if indicators["action"] == "delta*=0.5":
            delta *= 0.5
            X_guess, P_guess, p_tau_guess, tau_guess = X, P, p_tau, tau
            continue

    if not log or log[-1]["action"] != "STOP":
        X, P, p_tau, tau, info = solve_tau_augmented_tpbvp(
            problem,
            t_nodes,
            bundle,
            delta,
            config.target,
            config.penalty_weight,
            X_init=X_guess if X_guess is not None else X,
            P_init=P_guess if P_guess is not None else P,
            p_tau_init=p_tau_guess if p_tau_guess is not None else p_tau,
            tau_init=tau_guess,
            tol=config.newton_tol,
            max_iter=config.newton_max_iter,
        )
        indicators = _compute_tau_indicators(
            problem,
            t_nodes,
            X,
            P,
            tau,
            bundle,
            delta,
            config.tol_time,
            config.tol_PA,
            config.tol_delta,
            config.s_time,
            config.K_time,
        )
        log.append({
            "iteration": len(log),
            "N": len(t_nodes) - 1,
            "M": bundle.num_planes(),
            "delta": float(delta),
            "tau": float(tau),
            "newton_iter": info["iterations"],
            "newton_residual": info["residual_norm"],
            "t_nodes_iter": t_nodes.copy(),
            "note": "final_resolve",
            **indicators,
            "action": "final_resolve",
        })

    return {
        "t_nodes": t_nodes,
        "X": X,
        "P": P,
        "p_tau": p_tau,
        "tau": float(tau),
        "bundle": bundle,
        "delta": float(delta),
        "log": log,
        "problem": problem,
        "info": info,
        "settings": {
            "tol_time": config.tol_time,
            "tol_PA": config.tol_PA,
            "tol_delta": config.tol_delta,
            "max_iters": config.max_iters,
            "delta0": config.delta0,
            "s_time": config.s_time,
            "K_time": config.K_time,
            "newton_tol": config.newton_tol,
            "newton_max_iter": config.newton_max_iter,
            "variant": config.name,
        },
    }


def _estimate_switch_time(mesh: np.ndarray, controls: np.ndarray) -> float:
    for i in range(len(mesh) - 1):
        if controls[i] >= 0.0 and controls[i + 1] < 0.0:
            return float(0.5 * (mesh[i] + mesh[i + 1]))
    return float(mesh[np.argmin(np.abs(mesh - 1.0))])


def _state_constraint_violation(config: Example2Config, X: np.ndarray) -> float:
    if config.state_bounds is None:
        return 0.0
    lower, upper = config.state_bounds
    lower_violation = np.maximum(lower - X, 0.0)
    upper_violation = np.maximum(X - upper, 0.0)
    return float(max(np.max(lower_violation), np.max(upper_violation)))


def _mesh_objective(problem: OCPProblem, mesh: np.ndarray, X: np.ndarray, controls: np.ndarray) -> float:
    objective = float(problem.g(X[-1]))
    for i in range(len(mesh) - 1):
        dt = mesh[i + 1] - mesh[i]
        objective += float(problem.l(X[i], controls[i], mesh[i])) * dt
    return float(objective)


def _build_summary(result: dict) -> dict:
    config: Example2Config = result["config"]
    problem: OCPProblem = result["problem"]
    mesh = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)
    P = np.asarray(result["P"], dtype=float)
    p_tau = np.asarray(result["p_tau"], dtype=float) if config.use_tau else None
    controls = np.asarray(result["controls"], dtype=float)
    last = result["log"][-1]

    summary = {
        "variant": config.name,
        "description": config.description,
        "use_tau": bool(config.use_tau),
        "mesh_points": int(len(mesh)),
        "mesh_intervals": int(len(mesh) - 1),
        "planes": int(result["bundle"].num_planes()),
        "delta": float(result["delta"]),
        "objective_mesh_approx": _mesh_objective(problem, mesh, X, controls),
        "estimated_final_time": float(result["estimated_final_time"]),
        "estimated_switch_time_normalized": float(result["estimated_switch_time"]),
        "estimated_switch_time_physical": float(result["estimated_final_time"] * result["estimated_switch_time"])
        if config.use_tau
        else float(result["estimated_switch_time"]),
        "terminal_state": [float(v) for v in X[-1]],
        "terminal_penalty_gradient": [float(v) for v in _tau_terminal_gradient(X[-1], config.target, config.penalty_weight)],
        "state_constraint_violation_sup": _state_constraint_violation(config, X),
        "last_iteration": int(last["iteration"]),
        "outer_iterations_logged": int(len(result["log"])),
        "eta_time": float(last["eta_time"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "newton_iter": int(last["newton_iter"]),
        "newton_residual": float(last["newton_residual"]),
        "final_action": str(last["action"]),
        "all_indicators_within_tolerance": bool(
            last["eta_time"] <= last["tol_time_star"]
            and last["eta_PA"] <= config.tol_PA
            and last["eta_delta"] <= config.tol_delta
        ),
        "settings": {
            "tol_time": float(config.tol_time),
            "tol_PA": float(config.tol_PA),
            "tol_delta": float(config.tol_delta),
            "max_iters": int(config.max_iters),
            "delta0": float(config.delta0),
            "s_time": float(config.s_time),
            "K_time": float(config.K_time),
            "newton_tol": float(config.newton_tol),
            "newton_max_iter": int(config.newton_max_iter),
            "penalty_weight": float(config.penalty_weight),
            "initial_bundle_size": int(config.initial_bundle_size),
            "state_constraint_label": config.state_constraint_label,
            "running_cost_label": config.running_cost_label,
        },
    }
    if config.use_tau:
        summary["tau"] = float(result["tau"])
        summary["p_tau_terminal"] = float(p_tau[-1])
        summary["p_tau_initial"] = float(p_tau[0])
    return summary


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    return value


def summarize_example2_results(result: dict, print_log: bool = False) -> dict:
    summary = _build_summary(result)
    print(f"=== Example 2 ({summary['variant']}) ===")
    print(f"mesh intervals:           {summary['mesh_intervals']}")
    print(f"planes:                   {summary['planes']}")
    print(f"delta:                    {summary['delta']:.12e}")
    print(f"objective (mesh approx):  {summary['objective_mesh_approx']:.12e}")
    print(f"estimated final time:     {summary['estimated_final_time']:.12e}")
    print(f"switch time (normalized): {summary['estimated_switch_time_normalized']:.12e}")
    print(f"switch time (physical):   {summary['estimated_switch_time_physical']:.12e}")
    print(f"terminal state:           {summary['terminal_state']}")
    print(f"eta_time:                 {summary['eta_time']:.12e}")
    print(f"eta_PA:                   {summary['eta_PA']:.12e}")
    print(f"eta_delta:                {summary['eta_delta']:.12e}")
    print(f"all indicators OK:        {summary['all_indicators_within_tolerance']}")
    if print_log:
        for entry in result["log"]:
            print(entry)
    return summary


def _save_plot(fig, stem: str, fig_dir: Path, ext: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _keep_plot(fig, stem=None):
    return None


def plot_example2_results(
    result: dict,
    out_prefix: str = "example2",
    save_plots: bool = False,
    plot_ext: str = "pdf",
    fig_dir: Optional[Path] = None,
):
    config: Example2Config = result["config"]
    mesh = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)
    P = np.asarray(result["P"], dtype=float)
    controls = np.asarray(result["controls"], dtype=float)
    log = result.get("log", [])
    last = log[-1] if log else {}

    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir = Path(fig_dir)

    plot_action = partial(_save_plot, fig_dir=fig_dir, ext=plot_ext) if save_plots else _keep_plot
    render_plots = (lambda: None) if save_plots else plt.show

    physical_mesh = result["estimated_final_time"] * mesh if config.use_tau else mesh
    switch_normalized = result["estimated_switch_time"]
    switch_physical = result["estimated_final_time"] * switch_normalized if config.use_tau else switch_normalized

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(physical_mesh, X[:, 0], label=r"$x_1$")
    axes[0].plot(physical_mesh, X[:, 1], label=r"$x_2$")
    axes[0].axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("physical time")
    axes[0].set_ylabel("state")
    axes[0].set_title("State trajectory")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(physical_mesh, P[:, 0], label=r"$p_1$")
    axes[1].plot(physical_mesh, P[:, 1], label=r"$p_2$")
    axes[1].axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("physical time")
    axes[1].set_ylabel("costate")
    axes[1].set_title("Costate trajectory")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    plot_action(fig, f"{out_prefix}_state_costate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].step(physical_mesh, controls[:, 0], where="post", label=r"$u$")
    axes[0].axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("physical time")
    axes[0].set_ylabel("control")
    axes[0].set_title("Control profile")
    axes[0].grid(True)
    axes[0].legend()

    if len(mesh) > 1:
        dt = np.diff(physical_mesh)
        axes[1].step(physical_mesh[:-1], dt, where="post", label=r"$\Delta t_n$")
        axes[1].axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
        axes[1].set_xlabel("physical time")
        axes[1].set_ylabel(r"$\Delta t$")
        axes[1].set_yscale("log")
        axes[1].set_title("Adaptive step sizes")
        axes[1].grid(True, which="both")
        axes[1].legend()
    fig.tight_layout()
    plot_action(fig, f"{out_prefix}_control_stepsize")

    if last and ("rho" in last) and ("r_bar" in last):
        rho = np.asarray(last["rho"], dtype=float)
        r_bar = np.asarray(last["r_bar"], dtype=float)
        mesh_left = physical_mesh[:-1][: len(rho)]

        fig = plt.figure(figsize=(7.0, 4.5))
        plt.step(mesh_left, rho, where="post", label=r"$\rho_n$")
        plt.axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
        plt.xlabel("physical time")
        plt.ylabel(r"$\rho_n$")
        plt.title("Error density")
        plt.grid(True)
        plt.legend()
        fig.tight_layout()
        plot_action(fig, f"{out_prefix}_rho_density")

        fig = plt.figure(figsize=(7.0, 4.5))
        plt.step(mesh_left, r_bar, where="post", label=r"$\bar r_n$")
        plt.axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
        plt.xlabel("physical time")
        plt.ylabel(r"$\bar r_n$")
        plt.yscale("log")
        plt.title("Time indicator")
        plt.grid(True, which="both")
        plt.legend()
        fig.tight_layout()
        plot_action(fig, f"{out_prefix}_r_indicator")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].step(mesh_left, rho, where="post", label=r"$\rho_n$")
        axes[0].axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
        axes[0].set_xlabel("physical time")
        axes[0].set_ylabel(r"$\rho_n$")
        axes[0].set_title("Error density")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].step(mesh_left, r_bar, where="post", label=r"$\bar r_n$")
        axes[1].axvline(switch_physical, color="k", linestyle=":", linewidth=1.0)
        axes[1].set_xlabel("physical time")
        axes[1].set_ylabel(r"$\bar r_n$")
        axes[1].set_yscale("log")
        axes[1].set_title("Time indicator")
        axes[1].grid(True, which="both")
        axes[1].legend()
        fig.tight_layout()
        plot_action(fig, f"{out_prefix}_indicators")

    render_plots()


def export_example2_artifacts(
    result: dict,
    out_dir: Path,
    figure_dir: Optional[Path] = None,
    out_prefix: Optional[str] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if figure_dir is None:
        figure_dir = out_dir / "figures"
    figure_dir = Path(figure_dir)

    summary = _build_summary(result)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    detailed = {
        **summary,
        "mesh": [float(v) for v in result["t_nodes"]],
        "state": [[float(v) for v in row] for row in np.asarray(result["X"])],
        "costate": [[float(v) for v in row] for row in np.asarray(result["P"])],
        "controls": [[float(v) for v in row] for row in np.asarray(result["controls"])],
        "log": [_jsonable(entry) for entry in result["log"]],
    }
    if result["config"].use_tau:
        detailed["p_tau"] = [float(v) for v in np.asarray(result["p_tau"])]
    (out_dir / "summary_detailed.json").write_text(json.dumps(detailed, indent=2))

    with (out_dir / "outer_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "iteration",
                "N",
                "M",
                "delta",
                "tau",
                "newton_iter",
                "newton_residual",
                "eta_time",
                "tol_time_star",
                "eta_PA",
                "eta_delta",
                "action",
                "note",
            ]
        )
        for entry in result["log"]:
            writer.writerow(
                [
                    entry.get("iteration"),
                    entry.get("N"),
                    entry.get("M"),
                    entry.get("delta"),
                    entry.get("tau", ""),
                    entry.get("newton_iter"),
                    entry.get("newton_residual"),
                    entry.get("eta_time"),
                    entry.get("tol_time_star"),
                    entry.get("eta_PA"),
                    entry.get("eta_delta"),
                    entry.get("action", ""),
                    entry.get("note", ""),
                ]
            )

    plot_example2_results(
        result,
        out_prefix=out_prefix or result["config"].name,
        save_plots=True,
        plot_ext="pdf",
        fig_dir=figure_dir,
    )
    return summary


def run_example(variant: str = "archive_fixed_box"):
    config = build_example2_config(variant)
    if config.use_tau:
        result = solve_optimal_control_tau_augmented(config)
        problem = result["problem"]
    else:
        problem = build_example2_problem(config)
        t_nodes = np.linspace(0.0, config.T, config.initial_nodes)
        result = solve_optimal_control(
            problem,
            t_nodes,
            tol_time=config.tol_time,
            tol_PA=config.tol_PA,
            tol_delta=config.tol_delta,
            max_iters=config.max_iters,
            delta0=config.delta0,
            s_time=config.s_time,
            K_time=config.K_time,
            newton_tol=config.newton_tol,
            newton_max_iter=config.newton_max_iter,
        )

    X = result["X"]
    P = result["P"]
    mesh = result["t_nodes"]
    bundle = result["bundle"]

    controls = []
    for i in range(len(mesh)):
        _, u_star = compute_H(problem, P[i], X[i], mesh[i], bundle.controls, restricted=True)
        if u_star is None:
            raise RuntimeError(f"No viable control found at node {i} for variant {variant}.")
        controls.append(u_star)
    controls = np.asarray(controls)

    final_time = result.get("tau", mesh[-1]) if config.use_tau else mesh[-1]
    if (not config.use_tau) and config.state_bounds is not None and np.isfinite(config.state_bounds[1][0]):
        for i in range(len(mesh) - 1):
            if X[i, 0] <= 0 <= X[i + 1, 0]:
                alpha = (0.0 - X[i, 0]) / (X[i + 1, 0] - X[i, 0] + 1e-12)
                final_time = mesh[i] + alpha * (mesh[i + 1] - mesh[i])
                break

    switch_time = _estimate_switch_time(mesh, controls[:, 0])

    print(f"Double Integrator Example ({variant})")
    print(f"Mesh points: {len(mesh)}")
    print(f"Planes: {bundle.num_planes()}")
    print(f"Estimated final time: {final_time}")
    print(f"Estimated switch time: {switch_time}")
    print("Indicator history:")
    for entry in result["log"]:
        print(entry)

    result["config"] = config
    result["controls"] = controls
    result["estimated_final_time"] = float(final_time)
    result["estimated_switch_time"] = float(switch_time)
    return result


if __name__ == "__main__":
    run_example()
