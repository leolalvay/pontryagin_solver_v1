"""
Fixed-final-time penalized Goddard benchmark in Bolza form.

Phase 1:
- exact dynamics, terminal penalty, dynamic-pressure path constraint,
- constrained-arc control candidate for the active path constraint.

Phase 2:
- singular-arc control candidate derived from the switching-function conditions,
- oracle candidate selection over bang / constrained / singular controls.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np

from core.adaptivity import solve_optimal_control
from core.hamiltonian import compute_H
from core.problem import OCPProblem


def goddard_dynamic_pressure(r: float, v: float, *, b: float, beta: float) -> float:
    return float(b * (v ** 2) * np.exp(beta * (1.0 - r)))


def goddard_drag(r: float, v: float, *, b: float, beta: float, C_D: float) -> float:
    return float(C_D * goddard_dynamic_pressure(r, v, b=b, beta=beta))


def goddard_terminal_cost(x: np.ndarray, *, rho_m: float, m_f: float) -> float:
    r, _, m = np.asarray(x, dtype=float)
    mass_defect = float(m - m_f)
    return float(-r + 0.5 * rho_m * (mass_defect ** 2))


def goddard_terminal_gradient(x: np.ndarray, *, rho_m: float, m_f: float) -> np.ndarray:
    _, _, m = np.asarray(x, dtype=float)
    return np.array([-1.0, 0.0, rho_m * (m - m_f)], dtype=float)


def goddard_switching_function(x: np.ndarray, p: np.ndarray, *, c: float) -> float:
    _, _, m = np.asarray(x, dtype=float)
    _, p_v, p_m = np.asarray(p, dtype=float)
    return float(p_v / m - p_m / c)


def goddard_pressure_margin(x: np.ndarray, *, q_max: float, b: float, beta: float) -> float:
    r, v, _ = np.asarray(x, dtype=float)
    return float(goddard_dynamic_pressure(r, v, b=b, beta=beta) - q_max)


def goddard_barrier_margin(x: np.ndarray, *, q_max: float, b: float, beta: float) -> float:
    r, v, _ = np.asarray(x, dtype=float)
    return float(q_max - goddard_dynamic_pressure(r, v, b=b, beta=beta))


def goddard_barrier_stage_cost(x: np.ndarray, *, mu: float, q_max: float, b: float, beta: float, floor: float = 1.0e-14) -> float:
    margin = max(goddard_barrier_margin(x, q_max=q_max, b=b, beta=beta), floor)
    return float(-mu * np.log(margin))


def goddard_barrier_grad_x(x: np.ndarray, *, mu: float, q_max: float, b: float, beta: float, floor: float = 1.0e-14) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    q_val = goddard_dynamic_pressure(x_arr[0], x_arr[1], b=b, beta=beta)
    margin = max(float(q_max - q_val), floor)
    grad_q = np.array(
        [
            -beta * q_val,
            2.0 * b * x_arr[1] * np.exp(beta * (1.0 - x_arr[0])),
            0.0,
        ],
        dtype=float,
    )
    return float(mu) * grad_q / margin


def goddard_pressure_time_derivative(
    x: np.ndarray,
    u: np.ndarray,
    *,
    b: float,
    beta: float,
    C_D: float,
) -> float:
    r, v, m = np.asarray(x, dtype=float)
    control = float(np.asarray(u, dtype=float)[0])
    pressure = goddard_dynamic_pressure(r, v, b=b, beta=beta)
    exp_term = np.exp(beta * (1.0 - r))
    drag = C_D * pressure
    g_r = -beta * pressure
    g_v = 2.0 * b * v * exp_term
    r_dot = v
    v_dot = (control - drag) / m - 1.0 / (r ** 2)
    return float(g_r * r_dot + g_v * v_dot)


def goddard_constrained_control(
    x: np.ndarray,
    *,
    b: float,
    beta: float,
    C_D: float,
) -> float:
    r, v, m = np.asarray(x, dtype=float)
    drag = goddard_drag(r, v, b=b, beta=beta, C_D=C_D)
    return float(drag + m / (r ** 2) + 0.5 * beta * m * (v ** 2))


def goddard_singular_control(
    x: np.ndarray,
    *,
    b: float,
    beta: float,
    C_D: float,
    c: float,
    eps: float = 1e-12,
) -> float | None:
    r, v, m = np.asarray(x, dtype=float)
    drag = goddard_drag(r, v, b=b, beta=beta, C_D=C_D)
    denom = drag * (r ** 3) * (2.0 * (c ** 2) + 4.0 * c * v + v ** 2)
    if abs(denom) <= eps or abs(r) <= eps or abs(m) <= eps:
        return None
    numer = (
        2.0 * c * drag * m * r * (c + v)
        - (c ** 2) * m * (v ** 2) * (beta * drag * (r ** 3) + 2.0 * m)
    )
    value = drag + numer / denom
    if not np.isfinite(value):
        return None
    return float(value)


def goddard_candidate_controls(
    x: np.ndarray,
    p: np.ndarray,
    *,
    T_max: float,
    q_max: float,
    b: float,
    beta: float,
    C_D: float,
    c: float,
    activation_tol: float = 1e-7,
):
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    candidates = [
        ("coast", 0.0),
        ("full_thrust", float(T_max)),
    ]

    pressure_gap = goddard_pressure_margin(x, q_max=q_max, b=b, beta=beta)
    if abs(pressure_gap) <= activation_tol:
        candidates.append(
            (
                "constrained_arc",
                goddard_constrained_control(x, b=b, beta=beta, C_D=C_D),
            )
        )

    singular = goddard_singular_control(x, b=b, beta=beta, C_D=C_D, c=c)
    if singular is not None:
        candidates.append(("singular_arc", singular))

    # Deduplicate by control value while preserving order.
    deduped = []
    for name, value in candidates:
        if not np.isfinite(value):
            continue
        if any(abs(value - prev_value) < 1e-10 for _, prev_value in deduped):
            continue
        deduped.append((name, float(value)))
    return deduped


def build_goddard_problem(
    *,
    T: float = 0.15,
    q_max: float = 10.0,
    rho_m: float = 1.0e4,
    T_max: float = 3.5,
    m_f: float = 0.6,
    b: float = 6200.0,
    beta: float = 500.0,
    c: float = 0.5,
    C_D: float = 0.05,
    feasibility_margin_fraction: float = 0.1,
    mu_barrier: float = 0.0,
):
    x0 = np.array([1.0, 0.0, 1.0], dtype=float)
    params = {
        "problem_name": "fixed-final-time penalized Goddard rocket",
        "source": "Seywald NASA-CR-4393 (penalized terminal mass variant)",
        "T": float(T),
        "q_max": float(q_max),
        "rho_m": float(rho_m),
        "T_max": float(T_max),
        "m_f": float(m_f),
        "b": float(b),
        "beta": float(beta),
        "c": float(c),
        "C_D": float(C_D),
        "feasibility_margin_fraction": float(feasibility_margin_fraction),
        "mu_barrier": float(mu_barrier),
        "x0": x0.copy(),
    }

    def dynamics(x, u, t):
        r, v, m = np.asarray(x, dtype=float)
        control = float(np.asarray(u, dtype=float)[0])
        drag = goddard_drag(r, v, b=b, beta=beta, C_D=C_D)
        return np.array(
            [
                v,
                (control - drag) / m - 1.0 / (r ** 2),
                -control / c,
            ],
            dtype=float,
        )

    def stage_cost(x, u, t):
        return 0.0

    def terminal_cost(x):
        return goddard_terminal_cost(x, rho_m=rho_m, m_f=m_f)

    def tangent_ok_fn(x, u, t, tol):
        pressure_gap = goddard_pressure_margin(x, q_max=q_max, b=b, beta=beta)
        if pressure_gap > tol:
            return False
        if abs(pressure_gap) <= tol:
            return goddard_pressure_time_derivative(x, u, b=b, beta=beta, C_D=C_D) <= tol
        return True

    def step_feasible_control_fn(x, u, t, dt, tol):
        x_arr = np.asarray(x, dtype=float)
        u_arr = np.asarray(u, dtype=float)
        if dt is None:
            return tangent_ok_fn(x_arr, u_arr, t, tol)
        x_trial = x_arr + float(dt) * dynamics(x_arr, u_arr, t)
        return state_feasible_fn(x_trial, float(t) + float(dt), tol)

    def barrier_stage_cost_fn(x, u, t, mu):
        return goddard_barrier_stage_cost(x, mu=mu, q_max=q_max, b=b, beta=beta)

    def barrier_grad_x_fn(x, t, mu):
        return goddard_barrier_grad_x(x, mu=mu, q_max=q_max, b=b, beta=beta)

    def barrier_margin_fn(x, t):
        return goddard_barrier_margin(x, q_max=q_max, b=b, beta=beta)

    def feasibility_refinement_fn(x, p, t, dt, tol):
        u_arr, ok = u_star_local_fn(x, p, t, True, tol, dt)
        if u_arr is None or not ok:
            return None
        x_arr = np.asarray(x, dtype=float)
        x_trial = x_arr + float(dt) * dynamics(x_arr, u_arr, t)
        pressure_margin = float(q_max - goddard_dynamic_pressure(x_trial[0], x_trial[1], b=b, beta=beta))
        margin_threshold = float(feasibility_margin_fraction) * float(q_max)
        if pressure_margin <= margin_threshold:
            return {
                "reason": "thin_step_margin",
                "predicted_pressure_margin": pressure_margin,
                "margin_threshold": margin_threshold,
                "control_dt": np.asarray(u_arr, dtype=float).copy(),
            }
        return None

    def state_feasible_fn(x, t, tol):
        x_arr = np.asarray(x, dtype=float)
        r, v, m = x_arr
        if r <= tol or m <= tol:
            return False
        pressure_gap = goddard_pressure_margin(x_arr, q_max=q_max, b=b, beta=beta)
        return bool(pressure_gap <= tol)

    def project_state_fn(x, t, tol):
        x_arr = np.asarray(x, dtype=float).copy()
        x_arr[0] = max(float(x_arr[0]), 1.0e-6)
        x_arr[2] = max(float(x_arr[2]), 1.0e-6)
        exp_term = np.exp(beta * (1.0 - x_arr[0]))
        v_cap = np.sqrt(max(q_max, 0.0) / max(b * exp_term, 1.0e-300))
        inward_cap = np.nextafter(v_cap, 0.0)
        x_arr[1] = float(np.clip(x_arr[1], -inward_cap, inward_cap))
        return x_arr

    def fraction_to_boundary_fn(x, dx, t, safety, tol):
        x_arr = np.asarray(x, dtype=float)
        dx_arr = np.asarray(dx, dtype=float)
        r, v, m = x_arr
        dr, dv, dm = dx_arr
        lam = 1.0
        r_floor = 1.0e-6
        m_floor = 1.0e-6
        if dr < 0.0:
            lam = min(lam, safety * max(r - r_floor, 0.0) / max(-dr, 1.0e-300))
        if dm < 0.0:
            lam = min(lam, safety * max(m - m_floor, 0.0) / max(-dm, 1.0e-300))
        exp_term = np.exp(beta * (1.0 - r))
        v_cap = np.sqrt(max(q_max, 0.0) / max(b * exp_term, 1.0e-300))
        safe_v_cap = safety * v_cap
        if dv > 0.0:
            lam = min(lam, max(safe_v_cap - v, 0.0) / max(dv, 1.0e-300))
        elif dv < 0.0:
            lam = min(lam, max(safe_v_cap + v, 0.0) / max(-dv, 1.0e-300))
        return float(np.clip(lam, 0.0, 1.0))

    def u_star_fn(x, p, t):
        x_arr = np.asarray(x, dtype=float)
        p_arr = np.asarray(p, dtype=float)
        best_val = np.inf
        best_u = 0.0
        for _, candidate in goddard_candidate_controls(
            x_arr,
            p_arr,
            T_max=T_max,
            q_max=q_max,
            b=b,
            beta=beta,
            C_D=C_D,
            c=c,
        ):
            u_arr = np.array([np.clip(candidate, 0.0, T_max)], dtype=float)
            value = float(np.dot(p_arr, dynamics(x_arr, u_arr, t)) + stage_cost(x_arr, u_arr, t))
            if value < best_val:
                best_val = value
                best_u = float(u_arr[0])
        return np.array([best_u], dtype=float)

    def u_star_local_fn(x, p, t, restricted, tol, dt):
        x_arr = np.asarray(x, dtype=float)
        p_arr = np.asarray(p, dtype=float)
        best_val = np.inf
        best_u = None
        for _, candidate in goddard_candidate_controls(
            x_arr,
            p_arr,
            T_max=T_max,
            q_max=q_max,
            b=b,
            beta=beta,
            C_D=C_D,
            c=c,
        ):
            u_arr = np.array([np.clip(candidate, 0.0, T_max)], dtype=float)
            if restricted and not step_feasible_control_fn(x_arr, u_arr, t, dt, tol):
                continue
            value = float(np.dot(p_arr, dynamics(x_arr, u_arr, t)) + stage_cost(x_arr, u_arr, t))
            if value < best_val:
                best_val = value
                best_u = u_arr
        return best_u, best_u is not None

    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(np.array([0.0]), np.array([T_max])),
        state_bounds=None,
        u_star_fn=u_star_fn,
        u_star_local_fn=u_star_local_fn,
        tangent_ok_fn=tangent_ok_fn,
        step_feasible_control_fn=step_feasible_control_fn,
        feasibility_refinement_fn=feasibility_refinement_fn,
        barrier_stage_cost_fn=barrier_stage_cost_fn,
        barrier_grad_x_fn=barrier_grad_x_fn,
        barrier_margin_fn=barrier_margin_fn,
        state_feasible_fn=state_feasible_fn,
        project_state_fn=project_state_fn,
        fraction_to_boundary_fn=fraction_to_boundary_fn,
    )
    prob.mu_barrier = float(mu_barrier)
    return prob, params


def build_goddard_initial_guess(t_nodes: np.ndarray, *, m_f: float) -> tuple[np.ndarray, np.ndarray]:
    t_nodes = np.asarray(t_nodes, dtype=float)
    T = float(t_nodes[-1]) if len(t_nodes) > 0 else 1.0
    tau = t_nodes / max(T, 1e-12)
    X_guess = np.column_stack(
        [
            1.0 + 0.2 * tau,
            0.2 * tau,
            1.0 - (1.0 - m_f) * tau,
        ]
    )
    P_guess = np.column_stack(
        [
            -np.ones_like(tau),
            np.zeros_like(tau),
            np.zeros_like(tau),
        ]
    )
    return X_guess, P_guess


def make_goddard_pressure_feasible_guess(
    X_guess: np.ndarray,
    *,
    q_max: float,
    b: float,
    beta: float,
    safety_factor: float = 0.98,
) -> np.ndarray:
    X_guess = np.asarray(X_guess, dtype=float).copy()
    if X_guess.ndim != 2 or X_guess.shape[1] != 3:
        raise ValueError("X_guess must have shape (N+1, 3) for the Goddard problem.")

    safe_q = float(safety_factor) * float(q_max)
    for i in range(1, X_guess.shape[0]):
        r_i = float(X_guess[i, 0])
        v_i = float(X_guess[i, 1])
        exp_term = np.exp(beta * (1.0 - r_i))
        v_cap = np.sqrt(max(safe_q, 0.0) / max(b * exp_term, 1.0e-300))
        if abs(v_i) > v_cap:
            X_guess[i, 1] = np.sign(v_i) * v_cap
    return X_guess


def summarize_goddard_result(result: dict) -> dict:
    X = np.asarray(result["X"], dtype=float)
    P = np.asarray(result["P"], dtype=float)
    t_nodes = np.asarray(result["t_nodes"], dtype=float)
    problem_data = result.get("problem_data", {})
    final_summary = {
        "problem_name": problem_data.get("problem_name", "fixed-final-time penalized Goddard rocket"),
        "T": float(problem_data.get("T", t_nodes[-1] if len(t_nodes) > 0 else 0.0)),
        "q_max": float(problem_data.get("q_max", np.nan)),
        "rho_m": float(problem_data.get("rho_m", np.nan)),
        "mesh_intervals": int(len(t_nodes) - 1),
        "planes": int(result["bundle"].num_planes()),
        "delta": float(result["delta"]),
        "terminal_state": X[-1].tolist(),
        "terminal_costate": P[-1].tolist(),
        "terminal_mass_defect": float(X[-1, 2] - problem_data.get("m_f", 0.0)),
        "objective": float(result["objective_mesh_approx"]),
        "final_action": str(result["log"][-1]["action"]),
        "all_indicators_within_tolerance": bool(result["log"][-1]["all_indicators_within_tolerance"]),
    }
    return final_summary


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12e}"
    return str(value)


def _short_display(value):
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return str(value)


def _latex_escape(text):
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def _write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _latex_table_from_pairs(caption, label, rows):
    body = "\n".join(f"{_latex_escape(name)} & {_latex_escape(value)} \\\\" for name, value in rows)
    return dedent(
        f"""
        \\begin{{table}}[H]
        \\centering
        \\begin{{tabular}}{{ll}}
        \\toprule
        Quantity & Value \\\\
        \\midrule
        {body}
        \\bottomrule
        \\end{{tabular}}
        \\caption{{{_latex_escape(caption)}}}
        \\label{{{label}}}
        \\end{{table}}
        """
    ).strip()


def _latex_longtable_from_rows(caption, label, fieldnames, rows):
    header = " & ".join(_latex_escape(name) for name in fieldnames) + r" \\"
    body = "\n".join(" & ".join(_latex_escape(row.get(name, "")) for name in fieldnames) + r" \\" for row in rows)
    return dedent(
        f"""
        {{\\scriptsize
        \\begin{{longtable}}{{{'l' * len(fieldnames)}}}
        \\caption{{{_latex_escape(caption)}}}\\label{{{label}}}\\\\
        \\toprule
        {header}
        \\midrule
        \\endfirsthead
        \\toprule
        {header}
        \\midrule
        \\endhead
        {body}
        \\bottomrule
        \\end{{longtable}}
        }}
        """
    ).strip()


def _compute_node_controls(prob, bundle, X, P, t_nodes):
    controls = []
    for i, t_i in enumerate(t_nodes):
        dt_i = None
        if len(t_nodes) > 1:
            dt_i = float(t_nodes[i + 1] - t_nodes[i]) if i < len(t_nodes) - 1 else float(t_nodes[-1] - t_nodes[-2])
        _, u_star = compute_H(prob, P[i], X[i], float(t_i), bundle.controls, restricted=True, use_oracle=True, dt=dt_i)
        if u_star is None:
            controls.append(np.array([np.nan], dtype=float))
        else:
            controls.append(np.asarray(u_star, dtype=float).reshape(-1))
    return np.vstack(controls)


def _input_parameter_rows(result):
    pdata = result["problem_data"]
    settings = result["settings"]
    return [
        {"parameter": "problem_name", "value": pdata["problem_name"]},
        {"parameter": "source", "value": pdata["source"]},
        {"parameter": "T", "value": _format_value(pdata["T"])},
        {"parameter": "q_max", "value": _format_value(pdata["q_max"])},
        {"parameter": "rho_m", "value": _format_value(pdata["rho_m"])},
        {"parameter": "T_max", "value": _format_value(pdata["T_max"])},
        {"parameter": "m_f", "value": _format_value(pdata["m_f"])},
        {"parameter": "b", "value": _format_value(pdata["b"])},
        {"parameter": "beta", "value": _format_value(pdata["beta"])},
        {"parameter": "c", "value": _format_value(pdata["c"])},
        {"parameter": "C_D", "value": _format_value(pdata["C_D"])},
        {"parameter": "n_init", "value": _format_value(len(result["log"][0]["t_nodes_iter"]) - 1)},
        {"parameter": "tol_time", "value": _format_value(settings["tol_time"])},
        {"parameter": "tol_PA", "value": _format_value(settings["tol_PA"])},
        {"parameter": "tol_delta", "value": _format_value(settings["tol_delta"])},
        {"parameter": "max_iters", "value": _format_value(settings["max_iters"])},
        {"parameter": "delta0", "value": _format_value(settings["delta0"])},
        {"parameter": "s_time", "value": _format_value(settings["s_time"])},
        {"parameter": "time_balance_ratio", "value": _format_value(settings["time_balance_ratio"])},
        {"parameter": "pa_add_fraction", "value": _format_value(settings["pa_add_fraction"])},
        {"parameter": "pa_time_separation_factor", "value": _format_value(settings["pa_time_separation_factor"])},
        {"parameter": "pa_gap_floor_ratio", "value": _format_value(settings["pa_gap_floor_ratio"])},
        {"parameter": "initial_guess_label", "value": settings["initial_guess_label"]},
        {"parameter": "newton_tol", "value": _format_value(settings["newton_tol"])},
        {"parameter": "newton_max_iter", "value": _format_value(settings["newton_max_iter"])},
        {"parameter": "fallback_solver", "value": settings["fallback_solver"]},
    ]


def _reference_rows(result):
    pdata = result["problem_data"]
    return [
        {"quantity": "reference_source", "value": pdata["source"]},
        {"quantity": "literature_class", "value": "fixed-final-time Goddard rocket with dynamic-pressure path constraint"},
        {"quantity": "implementation_variant", "value": "quadratically penalized terminal mass defect"},
        {"quantity": "reference_note", "value": "current archive run is an exploratory executable baseline; no exact closed-form reference is available"},
    ]


def _outer_history_rows(log):
    rows = []
    for entry in log:
        rows.append(
            {
                "iteration": str(entry.get("iteration")),
                "note": entry.get("note", ""),
                "action": entry.get("action", ""),
                "N": str(entry.get("N")),
                "M": str(entry.get("M")),
                "delta": _format_value(entry.get("delta")),
                "objective_mesh_approx": _format_value(entry.get("objective_mesh_approx")),
                "eta_time": _format_value(entry.get("eta_time")),
                "eta_time_sum": _format_value(entry.get("eta_time_sum")),
                "tol_time_star": _format_value(entry.get("tol_time_star")),
                "mark_thr": _format_value(entry.get("mark_thr")),
                "eta_PA": _format_value(entry.get("eta_PA")),
                "eta_delta": _format_value(entry.get("eta_delta")),
                "newton_iter": str(entry.get("newton_iter")),
                "newton_residual": _format_value(entry.get("newton_residual")),
                "solver_phase": entry.get("solver_phase", ""),
                "all_indicators_within_tolerance": _format_value(entry.get("all_indicators_within_tolerance", False)),
            }
        )
    return rows


def _slim_log_entry_for_json(entry):
    return {
        "iteration": int(entry.get("iteration", -1)),
        "note": entry.get("note", ""),
        "action": entry.get("action", ""),
        "N": int(entry.get("N", 0)) if entry.get("N", None) is not None else None,
        "M": int(entry.get("M", 0)) if entry.get("M", None) is not None else None,
        "delta": float(entry.get("delta", np.nan)),
        "objective_mesh_approx": float(entry.get("objective_mesh_approx", np.nan)),
        "eta_time": float(entry.get("eta_time", np.nan)),
        "eta_time_sum": float(entry.get("eta_time_sum", np.nan)),
        "tol_time_star": float(entry.get("tol_time_star", np.nan)),
        "mark_thr": float(entry.get("mark_thr", np.nan)),
        "eta_PA": float(entry.get("eta_PA", np.nan)),
        "eta_delta": float(entry.get("eta_delta", np.nan)),
        "newton_iter": int(entry.get("newton_iter", 0)) if entry.get("newton_iter", None) is not None else None,
        "newton_residual": float(entry.get("newton_residual", np.nan)),
        "solver_phase": entry.get("solver_phase", ""),
        "fallback_used": bool(entry.get("fallback_used", False)),
        "all_indicators_within_tolerance": bool(entry.get("all_indicators_within_tolerance", False)),
    }


def _slim_log_for_json(log):
    return [_slim_log_entry_for_json(entry) for entry in log]


def _compact_result_for_warm_start(result):
    compact = {
        "X": np.asarray(result["X"], dtype=float).copy(),
        "P": np.asarray(result["P"], dtype=float).copy(),
        "t_nodes": np.asarray(result["t_nodes"], dtype=float).copy(),
        "delta": float(result["delta"]),
        "problem_data": dict(result.get("problem_data", {})),
        "settings": dict(result.get("settings", {})),
        "objective_mesh_approx": float(result.get("objective_mesh_approx", np.nan)),
    }
    log = result.get("log", [])
    compact["log"] = [_slim_log_entry_for_json(log[-1])] if log else []
    return compact


def _final_summary(result, prob):
    t_nodes = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)
    P = np.asarray(result["P"], dtype=float)
    pdata = result["problem_data"]
    last = result["log"][-1]
    q_values = np.array(
        [goddard_dynamic_pressure(x[0], x[1], b=pdata["b"], beta=pdata["beta"]) for x in X],
        dtype=float,
    )
    node_controls = _compute_node_controls(prob, result["bundle"], X, P, t_nodes)
    return {
        "objective_mesh_approx": float(last["objective_mesh_approx"]),
        "outer_iterations_logged": len(result["log"]),
        "last_outer_iteration": int(last["iteration"]),
        "mesh_points": int(len(t_nodes)),
        "mesh_intervals": int(len(t_nodes) - 1),
        "planes": int(result["bundle"].num_planes()),
        "delta": float(last["delta"]),
        "terminal_r": float(X[-1, 0]),
        "terminal_v": float(X[-1, 1]),
        "terminal_m": float(X[-1, 2]),
        "terminal_pr": float(P[-1, 0]),
        "terminal_pv": float(P[-1, 1]),
        "terminal_pm": float(P[-1, 2]),
        "terminal_mass_defect": float(X[-1, 2] - pdata["m_f"]),
        "max_dynamic_pressure": float(np.max(q_values)),
        "min_pressure_margin": float(np.min(pdata["q_max"] - q_values)),
        "pressure_constraint_nearly_active": bool(np.min(np.abs(pdata["q_max"] - q_values)) <= 1.0e-2),
        "pressure_constraint_violated_at_nodes": bool(np.any(q_values > pdata["q_max"] + 1.0e-10)),
        "max_node_control": float(np.nanmax(node_controls[:, 0])),
        "min_node_control": float(np.nanmin(node_controls[:, 0])),
        "eta_time": float(last["eta_time"]),
        "eta_time_sum": float(last.get("eta_time_sum", 0.0)),
        "tol_time_star": float(last["tol_time_star"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "newton_iter": int(last["newton_iter"]),
        "newton_residual": float(last["newton_residual"]),
        "solver_phase": last.get("solver_phase", "newton"),
        "fallback_used": bool(last.get("fallback_used", False)),
        "final_action": last.get("action", ""),
        "final_note": last.get("note", ""),
        "all_indicators_within_tolerance": bool(last.get("all_indicators_within_tolerance", False)),
        "wall_time_sec": float(result.get("wall_time_sec", float("nan"))),
    }


def _plot_iteration_state_costate(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)
    P = np.asarray(entry["P_iter"], dtype=float)
    support_points = entry.get("bundle_support_points_so_far", [])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, X[:, 0], label="r(t)")
    axes[0].plot(t, X[:, 1], label="v(t)")
    axes[0].plot(t, X[:, 2], label="m(t)")
    if support_points:
        support_t = np.array([float(point["time"]) for point in support_points], dtype=float)
        support_r = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in support_points], dtype=float)
        axes[0].scatter(
            support_t,
            support_r,
            s=26,
            color="tab:red",
            marker="o",
            label="bundle support points",
            zorder=5,
        )
    axes[0].set_ylabel("state")
    axes[0].set_title("State trajectories" + (" with support points" if support_points else ""))
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, P[:, 0], label="p_r(t)")
    axes[1].plot(t, P[:, 1], label="p_v(t)")
    axes[1].plot(t, P[:, 2], label="p_m(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("costate")
    axes[1].set_title("Costate trajectories")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_bundle_support_points(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)
    support_points = entry.get("bundle_support_points_so_far", [])
    current_iter = int(entry["iteration"])
    current_points = [point for point in support_points if int(point.get("iteration", -1)) == current_iter]
    previous_points = [point for point in support_points if int(point.get("iteration", -1)) < current_iter]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, X[:, 0], color="tab:blue", linewidth=1.8, label="r(t)")
    axes[0].plot(t, X[:, 1], color="tab:orange", linewidth=1.4, label="v(t)")
    axes[0].plot(t, X[:, 2], color="tab:green", linewidth=1.4, label="m(t)")

    if previous_points:
        prev_t = np.array([float(point["time"]) for point in previous_points], dtype=float)
        prev_r = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in previous_points], dtype=float)
        axes[0].scatter(
            prev_t,
            prev_r,
            s=26,
            color="tab:red",
            marker="o",
            alpha=0.75,
            label="previous support points",
            zorder=5,
        )
        for time_value in prev_t:
            axes[1].axvline(time_value, color="tab:red", alpha=0.10, linewidth=0.8)

    if current_points:
        curr_t = np.array([float(point["time"]) for point in current_points], dtype=float)
        curr_r = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in current_points], dtype=float)
        axes[0].scatter(
            curr_t,
            curr_r,
            s=72,
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            marker="s",
            label="support points added this iteration",
            zorder=6,
        )
        for time_value in curr_t:
            axes[1].axvline(time_value, color="black", alpha=0.35, linewidth=1.2, linestyle="--")

    axes[0].set_ylabel("state")
    axes[0].set_title("Bundle support locations on the state trajectory")
    axes[0].grid(True)
    axes[0].legend()

    if support_points:
        support_t_all = np.array(sorted(float(point["time"]) for point in support_points), dtype=float)
        cumulative_count = np.searchsorted(support_t_all, t, side="right")
        title = "Time locations of affine-plane support points"
        label = "cumulative support points up to time"
    else:
        cumulative_count = np.zeros_like(t)
        title = "No post-bootstrap support points were added in this run"
        label = "cumulative added support points"
    axes[1].step(t, cumulative_count, where="post", color="tab:purple", label=label)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("count")
    axes[1].set_title(title)
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_control(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    U = np.asarray(entry["U_iter"], dtype=float)[:, 0]
    fig = plt.figure(figsize=(9, 4))
    plt.step(t, U, where="post", label="restricted oracle control")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title("Nodewise control selected by the restricted oracle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_rho(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    rho = np.asarray(entry["rho"], dtype=float)
    rho_bar = np.asarray(entry["rho_bar"], dtype=float)
    fig = plt.figure(figsize=(9, 4))
    plt.step(t[:-1], np.maximum(np.abs(rho), 1e-18), where="post", label=r"$|\rho_n|$")
    plt.step(t[:-1], np.maximum(np.abs(rho_bar), 1e-18), where="post", label=r"$|\bar{\rho}_n|$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|\rho|$")
    plt.title("Estimated time-discretization error density")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_mesh_and_indicator(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    dt = np.diff(t)
    r_bar = np.asarray(entry["r_bar"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].step(t[:-1], dt, where="post", label=r"$\Delta t_n$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\Delta t$")
    axes[0].set_title("Adaptive mesh")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].step(t[:-1], np.maximum(np.abs(r_bar), 1e-18), where="post", label=r"$\bar r_n$")
    axes[1].axhline(float(entry["mark_thr"]), color="tab:orange", linestyle="--", label="refine threshold")
    axes[1].axhline(float(entry["tol_time_star"]), color="tab:green", linestyle=":", label="stop threshold")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\bar r_n$")
    axes[1].set_title("Time indicators and thresholds")
    axes[1].grid(True, which="both")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_pressure(entry, out_path, pdata):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)
    q_values = np.array(
        [goddard_dynamic_pressure(x[0], x[1], b=pdata["b"], beta=pdata["beta"]) for x in X],
        dtype=float,
    )
    fig = plt.figure(figsize=(9, 4))
    plt.plot(t, q_values, label="q(r,v)")
    plt.axhline(float(pdata["q_max"]), color="tab:red", linestyle="--", label=r"$q_{\max}$")
    plt.xlabel("t")
    plt.ylabel("dynamic pressure")
    plt.title("Dynamic-pressure path constraint diagnostic")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_pa_delta_contributions(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    t_int = t[:-1]
    eta_pa_local = np.asarray(entry.get("eta_PA_local", np.zeros_like(t_int)), dtype=float)
    eta_delta_local = np.asarray(entry.get("eta_delta_local", np.zeros_like(t_int)), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].step(
        t_int[: len(eta_pa_local)],
        np.maximum(np.abs(eta_pa_local), 1e-18),
        where="post",
        label=r"local $\eta_{\mathrm{PA}}$",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("PA contribution")
    axes[0].set_title("Per-interval plane-approximation contributions")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].step(
        t_int[: len(eta_delta_local)],
        np.maximum(np.abs(eta_delta_local), 1e-18),
        where="post",
        label=r"local $\eta_{\delta}$",
        color="tab:green",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("smoothing contribution")
    axes[1].set_title("Per-interval smoothing contributions")
    axes[1].grid(True, which="both")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_error_indicator_history(log, out_path):
    iterations = [int(entry["iteration"]) for entry in log]
    eta_time = [float(entry["eta_time"]) for entry in log]
    eta_time_sum = [float(entry.get("eta_time_sum", 0.0)) for entry in log]
    tol_time = [float(entry["tol_time_star"]) for entry in log]
    eta_pa = [float(entry["eta_PA"]) for entry in log]
    eta_delta = [float(entry["eta_delta"]) for entry in log]
    fig = plt.figure(figsize=(9, 5))
    plt.semilogy(iterations, eta_time, marker="o", label=r"$\eta_{\mathrm{time}}=\max_n \bar r_n$")
    plt.semilogy(iterations, eta_time_sum, marker="D", label=r"$\sum_n \bar r_n$")
    plt.semilogy(iterations, tol_time, linestyle="--", label=r"$\mathrm{tol}_{\mathrm{time}}^\star$")
    plt.semilogy(iterations, eta_pa, marker="s", label=r"$\eta_{\mathrm{PA}}$")
    plt.semilogy(iterations, eta_delta, marker="^", label=r"$\eta_{\delta}$")
    plt.xlabel("outer iteration")
    plt.ylabel("indicator value")
    plt.title("Error-estimate history")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_iteration_artifacts(log, iteration_root, pdata):
    for entry in log:
        if "X_iter" not in entry or "P_iter" not in entry or "U_iter" not in entry:
            continue
        iter_idx = int(entry["iteration"])
        iter_dir = iteration_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "iteration_data.json").write_text(json.dumps(_jsonable(entry), indent=2))
        _plot_iteration_state_costate(entry, iter_dir / f"iter_{iter_idx:02d}_state_costate.pdf")
        _plot_iteration_bundle_support_points(entry, iter_dir / f"iter_{iter_idx:02d}_bundle_support_points.pdf")
        _plot_iteration_control(entry, iter_dir / f"iter_{iter_idx:02d}_control.pdf")
        _plot_iteration_rho(entry, iter_dir / f"iter_{iter_idx:02d}_rho_density.pdf")
        _plot_iteration_mesh_and_indicator(entry, iter_dir / f"iter_{iter_idx:02d}_mesh_and_indicator.pdf")
        _plot_iteration_pressure(entry, iter_dir / f"iter_{iter_idx:02d}_pressure.pdf", pdata)
        _plot_iteration_pa_delta_contributions(entry, iter_dir / f"iter_{iter_idx:02d}_pa_delta_contributions.pdf")


def _generate_report_tex(out_dir, summary, input_rows, reference_rows, outer_rows, log):
    intro_status = (
        "The current exported baseline does not yet certify a fully accepted constrained rocket benchmark; it is a reproducible executable phase-3 report used to inspect the adaptive behavior and the dynamic-pressure margin."
    )
    final_pairs = [(k, _format_value(v)) for k, v in summary.items()]
    input_pairs = [(row["parameter"], row["value"]) for row in input_rows]
    reference_pairs = [(row["quantity"], row["value"]) for row in reference_rows]
    outer_fields = ["iter", "action", "N", "M", "delta", "J_h", "eta_time / tol*", "eta_PA", "eta_delta", "Newton it"]
    outer_display_rows = []
    for row in outer_rows:
        outer_display_rows.append(
            {
                "iter": row["iteration"],
                "action": row["action"],
                "N": row["N"],
                "M": row["M"],
                "delta": _short_display(row["delta"]),
                "J_h": _short_display(row["objective_mesh_approx"]),
                "eta_time / tol*": f"{_short_display(row['eta_time'])} / {_short_display(row['tol_time_star'])}",
                "eta_PA": _short_display(row["eta_PA"]),
                "eta_delta": _short_display(row["eta_delta"]),
                "Newton it": row["newton_iter"],
            }
        )
    iteration_sections = []
    for entry in log:
        iter_idx = int(entry["iteration"])
        if "X_iter" not in entry or "P_iter" not in entry or "U_iter" not in entry:
            iteration_sections.append(
                dedent(
                    f"""
                    \\subsection{{Iteration {iter_idx}}}
                    Iteration {iter_idx} ends with action \\texttt{{{_latex_escape(entry.get("action", ""))}}}. This was a refinement-only outer step ({_latex_escape(entry.get("note", ""))}), so no new TPBVP iterate was exported for plotting at this stage.
                    """
                ).strip()
            )
            continue
        iter_rel = Path("iterations") / f"iter_{iter_idx:02d}"
        status_text = "all indicators below tolerance" if entry.get("all_indicators_within_tolerance", False) else "refinement still required"
        iteration_sections.append(
            dedent(
                f"""
                \\subsection{{Iteration {iter_idx}}}
                Iteration {iter_idx} ends with action \\texttt{{{_latex_escape(entry.get("action", ""))}}}, mesh intervals $N={int(entry["N"])}$, planes $M={int(entry["M"])}$, and status {_latex_escape(status_text)}.

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_state_costate.pdf").as_posix()}}}
                \\caption{{State and costate trajectories at iteration {iter_idx}. Red support markers are overlaid on the state plot when post-bootstrap affine planes exist.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_bundle_support_points.pdf").as_posix()}}}
                \\caption{{Dedicated bundle-support diagnostic at iteration {iter_idx}. If no markers appear, the run stayed on the seeded bundle and no new PA support points were added during the adaptive loop.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_control.pdf").as_posix()}}}
                \\caption{{Restricted-oracle control selected at the mesh nodes at iteration {iter_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_pressure.pdf").as_posix()}}}
                \\caption{{Dynamic-pressure path-constraint diagnostic at iteration {iter_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_rho_density.pdf").as_posix()}}}
                \\caption{{Estimated time-discretization error density at iteration {iter_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_mesh_and_indicator.pdf").as_posix()}}}
                \\caption{{Time mesh and time-indicator thresholds at iteration {iter_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_pa_delta_contributions.pdf").as_posix()}}}
                \\caption{{Per-interval $\\eta_{{\\mathrm{{PA}}}}$ and $\\eta_\\delta$ contributions at iteration {iter_idx}. These are the local contributions whose sums produce the reported global plane-approximation and smoothing indicators.}}
                \\end{{figure}}
                """
            ).strip()
        )
    iteration_sections_text = "\n\n".join(iteration_sections)

    tex = dedent(
        f"""
        \\documentclass[11pt]{{article}}
        \\usepackage[a4paper,margin=1in]{{geometry}}
        \\usepackage{{amsmath}}
        \\usepackage{{amssymb}}
        \\usepackage{{booktabs}}
        \\usepackage{{float}}
        \\usepackage{{graphicx}}
        \\usepackage{{hyperref}}
        \\usepackage{{longtable}}
        \\title{{Standalone Deep Report for the Fixed-Time Penalized Goddard Benchmark}}
        \\author{{Archive Python adaptive Pontryagin solver}}
        \\date{{\\today}}
        \\begin{{document}}
        \\maketitle

        \\section{{Fixed-Time Penalized Goddard Benchmark}}
        This report documents the current executable fixed-final-time Goddard benchmark implemented in penalized Bolza form. {_latex_escape(intro_status)}

        \\subsection{{Input parameters}}
        {_latex_table_from_pairs("Input parameters for the Goddard phase-3 baseline.", "tab:goddard-input", input_pairs)}

        \\subsection{{Reference values and interpretation}}
        {_latex_table_from_pairs("Reference and interpretation entries for the current Goddard baseline.", "tab:goddard-reference", reference_pairs)}

        \\subsection{{Outer-loop summary}}
        {_latex_longtable_from_rows("Adaptive outer-loop iterations of the proposed solver on the Goddard phase-3 baseline.", "tab:goddard-outer", outer_fields, outer_display_rows)}

        \\subsection{{Final reported numbers}}
        {_latex_table_from_pairs("Final numbers reported by the proposed solver on the Goddard phase-3 baseline.", "tab:goddard-final", final_pairs)}

        \\subsection{{Error-estimate history}}
        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.9\\textwidth]{{figures/error_indicator_history.pdf}}
        \\caption{{Error-estimate history across the adaptive outer loop.}}
        \\end{{figure}}

        \\section{{Per-Iteration Dossier}}
        {iteration_sections_text}
        \\end{{document}}
        """
    ).strip() + "\n"
    report_path = Path(out_dir) / "report.tex"
    report_path.write_text(tex)
    return report_path


def export_goddard_deep_report_artifacts(result, prob, out_dir):
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    iteration_dir = out_dir / "iterations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    iteration_dir.mkdir(parents=True, exist_ok=True)

    input_rows = _input_parameter_rows(result)
    reference_rows = _reference_rows(result)
    outer_rows = _outer_history_rows(result["log"])
    summary = _final_summary(result, prob)

    (tables_dir / "final_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (tables_dir / "final_summary.csv").write_text(
        "quantity,value\n" + "\n".join(f"{k},{_format_value(v)}" for k, v in summary.items()) + "\n"
    )
    _write_csv(input_rows, tables_dir / "input_parameters.csv", ["parameter", "value"])
    _write_csv(reference_rows, tables_dir / "reference_values.csv", ["quantity", "value"])
    _write_csv(
        outer_rows,
        tables_dir / "outer_loop_history.csv",
        [
            "iteration",
            "note",
            "action",
            "N",
            "M",
            "delta",
            "objective_mesh_approx",
            "eta_time",
            "eta_time_sum",
            "tol_time_star",
            "mark_thr",
            "eta_PA",
            "eta_delta",
            "newton_iter",
            "newton_residual",
            "solver_phase",
            "all_indicators_within_tolerance",
        ],
    )
    (tables_dir / "outer_loop_history.json").write_text(json.dumps(_jsonable(_slim_log_for_json(result["log"])), indent=2))
    _write_iteration_artifacts(result["log"], iteration_dir, result["problem_data"])
    _plot_error_indicator_history(result["log"], figures_dir / "error_indicator_history.pdf")
    report_path = _generate_report_tex(out_dir, summary, input_rows, reference_rows, outer_rows, result["log"])
    return {
        "summary": summary,
        "report_tex": str(report_path),
        "out_dir": str(out_dir),
    }


def _plot_barrier_stage_history(barrier_run, out_path):
    stages = [stage for stage in barrier_run["stages"] if stage.get("status") == "ok"]
    if not stages:
        return
    mus = np.array([float(stage["mu_barrier"]) for stage in stages], dtype=float)
    mesh = np.array([float(stage["mesh_intervals"]) for stage in stages], dtype=float)
    margins = np.array([float(stage["min_barrier_margin"]) for stage in stages], dtype=float)
    delta_vals = np.array([float(stage["delta"]) for stage in stages], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].semilogx(mus, mesh, marker="o")
    axes[0].set_ylabel("mesh intervals")
    axes[0].grid(True, which="both")
    axes[0].set_title("Barrier continuation diagnostics")
    axes[1].semilogx(mus, margins, marker="s")
    axes[1].set_ylabel("min barrier margin")
    axes[1].grid(True, which="both")
    axes[2].semilogx(mus, delta_vals, marker="^")
    axes[2].set_ylabel(r"$\delta$")
    axes[2].set_xlabel(r"$\mu$")
    axes[2].grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_pressure_profiles(results_map, out_path):
    fig = plt.figure(figsize=(8, 4.5))
    for label, payload in results_map.items():
        if payload is None:
            continue
        t_nodes = np.asarray(payload["t_nodes"], dtype=float)
        X = np.asarray(payload["X"], dtype=float)
        pdata = payload["problem_data"]
        q_values = np.array(
            [goddard_dynamic_pressure(x[0], x[1], b=pdata["b"], beta=pdata["beta"]) for x in X],
            dtype=float,
        )
        plt.plot(t_nodes, q_values, label=label)
        plt.axhline(float(pdata["q_max"]), color="tab:red", linestyle="--", linewidth=1.0)
    plt.xlabel("t")
    plt.ylabel("dynamic pressure")
    plt.title("Pressure profiles: baseline vs handoff")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_mu_path_following(stages, out_path):
    ok_stages = [stage for stage in stages if stage.get("status") == "ok"]
    if not ok_stages:
        return
    mus = np.array([float(stage["mu_barrier"]) for stage in ok_stages], dtype=float)
    objective = np.array([float(stage["objective_mesh_approx"]) for stage in ok_stages], dtype=float)
    max_q = np.array([float(stage["max_dynamic_pressure"]) for stage in ok_stages], dtype=float)
    min_margin = np.array([float(stage["min_pressure_margin"]) for stage in ok_stages], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].semilogx(mus, objective, marker="o")
    axes[0].set_ylabel(r"$J_h$")
    axes[0].set_title(r"Log-barrier path following in $\mu$")
    axes[0].grid(True, which="both")
    axes[1].semilogx(mus, max_q, marker="s")
    axes[1].set_ylabel(r"$\max q(r_h,v_h)$")
    axes[1].grid(True, which="both")
    axes[2].semilogx(mus, min_margin, marker="^")
    axes[2].set_ylabel("min margin")
    axes[2].set_xlabel(r"$\mu$")
    axes[2].grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _result_final_plot_entry(result, prob):
    last = result["log"][-1]
    entry = dict(last)
    entry["t_nodes_iter"] = np.asarray(result["t_nodes"], dtype=float).copy()
    entry["X_iter"] = np.asarray(result["X"], dtype=float).copy()
    entry["P_iter"] = np.asarray(result["P"], dtype=float).copy()
    entry["U_iter"] = _compute_node_controls(
        prob,
        result["bundle"],
        np.asarray(result["X"], dtype=float),
        np.asarray(result["P"], dtype=float),
        np.asarray(result["t_nodes"], dtype=float),
    )
    return entry


def _write_barrier_stage_artifacts(stage_payloads, stage_root):
    rows = []
    for payload in stage_payloads:
        stage = payload["stage"]
        result = payload["result"]
        prob = payload["problem"]
        stage_idx = int(stage["stage_index"])
        mu = float(stage["mu_barrier"])
        stage_dir = stage_root / f"mu_stage_{stage_idx:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        entry = _result_final_plot_entry(result, prob)
        (stage_dir / "stage_summary.json").write_text(json.dumps(_jsonable(stage), indent=2))
        _plot_iteration_state_costate(entry, stage_dir / f"mu_stage_{stage_idx:02d}_state_costate.pdf")
        _plot_iteration_bundle_support_points(entry, stage_dir / f"mu_stage_{stage_idx:02d}_bundle_support_points.pdf")
        _plot_iteration_control(entry, stage_dir / f"mu_stage_{stage_idx:02d}_control.pdf")
        _plot_iteration_rho(entry, stage_dir / f"mu_stage_{stage_idx:02d}_rho_density.pdf")
        _plot_iteration_mesh_and_indicator(entry, stage_dir / f"mu_stage_{stage_idx:02d}_mesh_and_indicator.pdf")
        _plot_iteration_pressure(entry, stage_dir / f"mu_stage_{stage_idx:02d}_pressure.pdf", result["problem_data"])
        _plot_iteration_pa_delta_contributions(entry, stage_dir / f"mu_stage_{stage_idx:02d}_pa_delta_contributions.pdf")
        _plot_error_indicator_history(result["log"], stage_dir / f"mu_stage_{stage_idx:02d}_error_history.pdf")
        rows.append(
            {
                "stage_index": stage_idx,
                "mu": mu,
                "status": stage.get("status", ""),
                "N": int(stage.get("mesh_intervals", 0)),
                "delta": _short_display(stage.get("delta", np.nan)),
                "action": stage.get("final_action", ""),
                "objective": _short_display(stage.get("objective_mesh_approx", np.nan)),
                "max_q": _short_display(stage.get("max_dynamic_pressure", np.nan)),
                "min_margin": _short_display(stage.get("min_pressure_margin", np.nan)),
            }
        )
    return rows


def export_goddard_barrier_report_artifacts(barrier_run, handoff_bundle, baseline_result, baseline_prob, out_dir):
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    handoff_result = handoff_bundle["handoff_result"]
    handoff_prob = handoff_bundle["handoff_problem"]
    handoff_summary = _final_summary(handoff_result, handoff_prob)
    baseline_summary = _final_summary(baseline_result, baseline_prob)

    comparison = {
        "handoff_vs_baseline_objective_delta": float(handoff_summary["objective_mesh_approx"] - baseline_summary["objective_mesh_approx"]),
        "handoff_vs_baseline_mesh_delta": int(handoff_summary["mesh_intervals"] - baseline_summary["mesh_intervals"]),
        "handoff_vs_baseline_pressure_margin_delta": float(handoff_summary["min_pressure_margin"] - baseline_summary["min_pressure_margin"]),
        "handoff_final_action": handoff_summary["final_action"],
        "baseline_final_action": baseline_summary["final_action"],
        "handoff_all_indicators_within_tolerance": bool(handoff_summary["all_indicators_within_tolerance"]),
        "baseline_all_indicators_within_tolerance": bool(baseline_summary["all_indicators_within_tolerance"]),
    }

    barrier_summary = {
        "mu_schedule": barrier_run.get("mu_schedule", []),
        "q_max": barrier_run.get("q_max", np.nan),
        "stages": barrier_run.get("stages", []),
    }
    (out_dir / "barrier_continuation_summary.json").write_text(json.dumps(_jsonable(barrier_summary), indent=2))
    (tables_dir / "handoff_final_summary.json").write_text(json.dumps(_jsonable(handoff_summary), indent=2))
    (tables_dir / "baseline_final_summary.json").write_text(json.dumps(_jsonable(baseline_summary), indent=2))
    (tables_dir / "comparison.json").write_text(json.dumps(_jsonable(comparison), indent=2))

    _plot_barrier_stage_history(barrier_run, figures_dir / "barrier_stage_history.pdf")
    _plot_pressure_profiles(
        {
            "baseline true constrained": baseline_result,
            "true from barrier": handoff_result,
        },
        figures_dir / "pressure_comparison.pdf",
    )

    stage_rows = []
    for stage in barrier_run["stages"]:
        stage_rows.append(
            {
                "mu": stage.get("mu_barrier", ""),
                "status": stage.get("status", ""),
                "N": stage.get("mesh_intervals", ""),
                "delta": _short_display(stage.get("delta", "")),
                "action": stage.get("final_action", ""),
                "min margin": _short_display(stage.get("min_barrier_margin", "")),
                "max q": _short_display(stage.get("max_dynamic_pressure", "")),
            }
        )
    tex = dedent(
        f"""
        \\documentclass[11pt]{{article}}
        \\usepackage[a4paper,margin=1in]{{geometry}}
        \\usepackage{{amsmath}}
        \\usepackage{{booktabs}}
        \\usepackage{{float}}
        \\usepackage{{graphicx}}
        \\usepackage{{longtable}}
        \\title{{Barrier Continuation Note for the Fixed-Time Goddard Benchmark}}
        \\author{{Archive Python adaptive Pontryagin solver}}
        \\date{{\\today}}
        \\begin{{document}}
        \\maketitle

        \\section{{Purpose}}
        This note documents the interior-point log-barrier continuation used only as a warm-start mechanism for the state-constrained Goddard benchmark. The final benchmark remains the subsequent true constrained solve with $\\mu=0$.

        \\section{{Barrier Continuation Stages}}
        {_latex_longtable_from_rows('Barrier stages in decreasing $\\mu$.', 'tab:barrier-stages', ['mu','status','N','delta','action','min margin','max q'], stage_rows)}

        \\section{{Barrier Diagnostics}}
        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.9\\textwidth]{{figures/barrier_stage_history.pdf}}
        \\caption{{Barrier-stage diagnostics.}}
        \\end{{figure}}

        \\section{{True Constrained Handoff vs Direct Baseline}}
        {_latex_table_from_pairs('True constrained handoff summary.', 'tab:handoff-final', [(k, _format_value(v)) for k, v in handoff_summary.items()])}

        {_latex_table_from_pairs('Direct true constrained baseline summary.', 'tab:baseline-final', [(k, _format_value(v)) for k, v in baseline_summary.items()])}

        {_latex_table_from_pairs('Comparison between the handoff run and the direct baseline.', 'tab:barrier-comparison', [(k, _format_value(v)) for k, v in comparison.items()])}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.9\\textwidth]{{figures/pressure_comparison.pdf}}
        \\caption{{Dynamic-pressure profiles for the direct baseline and the true constrained handoff from the barrier path.}}
        \\end{{figure}}
        \\end{{document}}
        """
    ).strip() + "\n"
    report_path = out_dir / "report.tex"
    report_path.write_text(tex)
    return {
        "report_tex": str(report_path),
        "out_dir": str(out_dir),
        "handoff_summary": handoff_summary,
        "baseline_summary": baseline_summary,
        "comparison": comparison,
    }


def export_goddard_barrier_handoff_rich_report_artifacts(barrier_run, handoff_bundle, out_dir):
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    barrier_stage_dir = out_dir / "barrier_stages"
    handoff_iter_dir = out_dir / "iterations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    barrier_stage_dir.mkdir(parents=True, exist_ok=True)
    handoff_iter_dir.mkdir(parents=True, exist_ok=True)

    handoff_result = handoff_bundle["handoff_result"]
    handoff_prob = handoff_bundle["handoff_problem"]
    handoff_summary = _final_summary(handoff_result, handoff_prob)
    input_rows = _input_parameter_rows(handoff_result)
    reference_rows = _reference_rows(handoff_result)
    outer_rows = _outer_history_rows(handoff_result["log"])

    (tables_dir / "final_summary.json").write_text(json.dumps(_jsonable(handoff_summary), indent=2))
    (tables_dir / "outer_loop_history.json").write_text(json.dumps(_jsonable(_slim_log_for_json(handoff_result["log"])), indent=2))
    _write_csv(input_rows, tables_dir / "input_parameters.csv", ["parameter", "value"])
    _write_csv(reference_rows, tables_dir / "reference_values.csv", ["quantity", "value"])
    _write_csv(
        outer_rows,
        tables_dir / "outer_loop_history.csv",
        [
            "iteration",
            "note",
            "action",
            "N",
            "M",
            "delta",
            "objective_mesh_approx",
            "eta_time",
            "eta_time_sum",
            "tol_time_star",
            "mark_thr",
            "eta_PA",
            "eta_delta",
            "newton_iter",
            "newton_residual",
            "solver_phase",
            "all_indicators_within_tolerance",
        ],
    )
    _write_iteration_artifacts(handoff_result["log"], handoff_iter_dir, handoff_result["problem_data"])
    _plot_error_indicator_history(handoff_result["log"], figures_dir / "handoff_error_indicator_history.pdf")

    stage_payloads = barrier_run.get("stage_payloads", [])
    stage_rows = _write_barrier_stage_artifacts(stage_payloads, barrier_stage_dir) if stage_payloads else []
    _plot_barrier_stage_history(barrier_run, figures_dir / "barrier_stage_history.pdf")
    _plot_mu_path_following(barrier_run.get("stages", []), figures_dir / "mu_path_following.pdf")

    final_pairs = [(k, _format_value(v)) for k, v in handoff_summary.items()]
    input_pairs = [(row["parameter"], row["value"]) for row in input_rows]
    reference_pairs = [(row["quantity"], row["value"]) for row in reference_rows]
    outer_fields = ["iter", "action", "N", "M", "delta", "J_h", "eta_time / tol*", "eta_PA", "eta_delta", "Newton it"]
    outer_display_rows = []
    for row in outer_rows:
        outer_display_rows.append(
            {
                "iter": row["iteration"],
                "action": row["action"],
                "N": row["N"],
                "M": row["M"],
                "delta": _short_display(row["delta"]),
                "J_h": _short_display(row["objective_mesh_approx"]),
                "eta_time / tol*": f"{_short_display(row['eta_time'])} / {_short_display(row['tol_time_star'])}",
                "eta_PA": _short_display(row["eta_PA"]),
                "eta_delta": _short_display(row["eta_delta"]),
                "Newton it": row["newton_iter"],
            }
        )

    barrier_stage_sections = []
    for row in stage_rows:
        stage_idx = int(row["stage_index"])
        rel = Path("barrier_stages") / f"mu_stage_{stage_idx:02d}"
        barrier_stage_sections.append(
            dedent(
                f"""
                \\subsection{{Barrier stage {stage_idx} ($\\mu={row["mu"]}$)}}
                Final status \\texttt{{{_latex_escape(row["status"])}}}, action \\texttt{{{_latex_escape(row["action"])}}}, mesh intervals $N={_latex_escape(row["N"])}$, and objective {_latex_escape(row["objective"])}.

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_state_costate.pdf").as_posix()}}}
                \\caption{{State and costate trajectories at barrier stage {stage_idx}. Red support markers are overlaid when post-bootstrap affine planes exist at this stage.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_bundle_support_points.pdf").as_posix()}}}
                \\caption{{Dedicated bundle-support diagnostic at barrier stage {stage_idx}. If no support markers appear, the stage remained on the seeded bundle.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_control.pdf").as_posix()}}}
                \\caption{{Restricted-oracle control at barrier stage {stage_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_pressure.pdf").as_posix()}}}
                \\caption{{Dynamic-pressure diagnostic at barrier stage {stage_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_rho_density.pdf").as_posix()}}}
                \\caption{{Estimated time-discretization density at barrier stage {stage_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_mesh_and_indicator.pdf").as_posix()}}}
                \\caption{{Time mesh and thresholds at barrier stage {stage_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_pa_delta_contributions.pdf").as_posix()}}}
                \\caption{{Per-interval $\\eta_{{\\mathrm{{PA}}}}$ and $\\eta_\\delta$ contributions at barrier stage {stage_idx}.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(rel / f"mu_stage_{stage_idx:02d}_error_history.pdf").as_posix()}}}
                \\caption{{Outer-loop error history within barrier stage {stage_idx}.}}
                \\end{{figure}}
                """
            ).strip()
        )

    handoff_report_path = _generate_report_tex(out_dir, handoff_summary, input_rows, reference_rows, outer_rows, handoff_result["log"])
    handoff_tex = Path(handoff_report_path).read_text()
    handoff_body = handoff_tex.split("\\section{Per-Iteration Dossier}", 1)[1].rsplit("\\end{document}", 1)[0].strip()

    barrier_stage_sections_text = "\n\n".join(barrier_stage_sections)

    tex = dedent(
        f"""
        \\documentclass[11pt]{{article}}
        \\usepackage[a4paper,margin=1in]{{geometry}}
        \\usepackage{{amsmath}}
        \\usepackage{{amssymb}}
        \\usepackage{{booktabs}}
        \\usepackage{{float}}
        \\usepackage{{graphicx}}
        \\usepackage{{hyperref}}
        \\usepackage{{longtable}}
        \\title{{Rich Deep Report for the Barrier-Assisted Goddard Handoff at $q_{{\\max}}=19$}}
        \\author{{Archive Python adaptive Pontryagin solver}}
        \\date{{\\today}}
        \\begin{{document}}
        \\maketitle

        \\section{{Barrier-Assisted Continuation Overview}}
        This report documents the full log-barrier path following in $\\mu$ together with the per-iteration dossier of the final true constrained handoff at $q_{{\\max}}=19$.

        \\subsection{{Input parameters}}
        {_latex_table_from_pairs("Input parameters for the barrier-assisted Goddard handoff.", "tab:goddard19-input", input_pairs)}

        \\subsection{{Reference values and interpretation}}
        {_latex_table_from_pairs("Reference and interpretation entries for the barrier-assisted Goddard handoff.", "tab:goddard19-reference", reference_pairs)}

        \\subsection{{Final reported numbers}}
        {_latex_table_from_pairs("Final numbers for the accepted barrier-assisted handoff at $q_{{\\max}}=19$.", "tab:goddard19-final", final_pairs)}

        {_latex_longtable_from_rows("Barrier stages along the $\\mu$ path following.", "tab:goddard19-barrier-stages", ["stage_index","mu","status","N","delta","action","objective","max_q","min_margin"], stage_rows)}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.9\\textwidth]{{figures/barrier_stage_history.pdf}}
        \\caption{{Barrier-stage diagnostics.}}
        \\end{{figure}}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.9\\textwidth]{{figures/mu_path_following.pdf}}
        \\caption{{Path following in $\\mu$: objective, maximum dynamic pressure, and minimum margin.}}
        \\end{{figure}}

        \\section{{Barrier Stage Dossier}}
        {barrier_stage_sections_text}

        \\section{{True Constrained Handoff Outer-Loop Summary}}
        {_latex_longtable_from_rows("Adaptive outer-loop iterations of the true constrained handoff at $q_{{\\max}}=19$.", "tab:goddard19-outer", outer_fields, outer_display_rows)}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.9\\textwidth]{{figures/handoff_error_indicator_history.pdf}}
        \\caption{{Error-estimate history across the true constrained handoff iterations.}}
        \\end{{figure}}

        \\section{{Per-Iteration Dossier}}
        {handoff_body}
        \\end{{document}}
        """
    ).strip() + "\n"
    report_path = out_dir / "report.tex"
    report_path.write_text(tex)
    return {
        "report_tex": str(report_path),
        "out_dir": str(out_dir),
        "final_summary": handoff_summary,
    }


def run_goddard_solver(
    *,
    n_init: int = 8,
    T: float = 0.15,
    q_max: float = 100.0,
    rho_m: float = 1.0e4,
    tol_time: float = 1.0e-3,
    tol_PA: float = 1.0e-3,
    tol_delta: float = 1.0e-3,
    delta0: float = 5.0e-2,
    max_iters: int = 2,
    store_iterates: bool = False,
    verbose: bool = True,
    mu_barrier: float = 0.0,
    initial_mesh: np.ndarray | None = None,
    initial_X_guess: np.ndarray | None = None,
    initial_P_guess: np.ndarray | None = None,
    initial_guess_label: str = "goddard_bootstrap",
):
    prob, params = build_goddard_problem(T=T, q_max=q_max, rho_m=rho_m, mu_barrier=mu_barrier)
    t_nodes = np.asarray(initial_mesh, dtype=float).copy() if initial_mesh is not None else np.linspace(0.0, params["T"], int(n_init))
    if initial_X_guess is None or initial_P_guess is None:
        X_guess, P_guess = build_goddard_initial_guess(t_nodes, m_f=params["m_f"])
    else:
        X_guess = np.asarray(initial_X_guess, dtype=float).copy()
        P_guess = np.asarray(initial_P_guess, dtype=float).copy()

    t0_wall = time.perf_counter()
    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        max_iters=max_iters,
        delta0=delta0,
        use_oracle_bootstrap=True,
        use_oracle_PA=True,
        use_explicit_hamiltonian_gradients=False,
        store_iterates=store_iterates,
        verbose=verbose,
        initial_X_guess=X_guess,
        initial_P_guess=P_guess,
        initial_guess_label=initial_guess_label,
    )
    result["problem_data"] = params
    result["wall_time_sec"] = time.perf_counter() - t0_wall
    result["objective_mesh_approx"] = float(result["log"][-1]["objective_mesh_approx"])
    return result, prob


def run_goddard_qmax_continuation(
    *,
    q_schedule,
    T: float = 0.15,
    rho_m: float = 1.0e4,
    n_init: int = 8,
    tol_time: float = 1.0e-4,
    tol_PA: float = 1.0e-4,
    tol_delta: float = 1.0e-4,
    delta0: float = 5.0e-2,
    stage_max_iters: int = 2,
    verbose: bool = False,
    pressure_feasible_init: bool = True,
    pressure_feasible_safety_factor: float = 0.98,
    store_iterates: bool = False,
    initial_mesh: np.ndarray | None = None,
    initial_X_guess: np.ndarray | None = None,
    initial_P_guess: np.ndarray | None = None,
    initial_delta0: float | None = None,
    initial_guess_label_prefix: str = "qmax",
):
    q_schedule = [float(q) for q in q_schedule]
    mesh = (
        np.asarray(initial_mesh, dtype=float).copy()
        if initial_mesh is not None
        else np.linspace(0.0, float(T), int(n_init))
    )
    X_guess = None if initial_X_guess is None else np.asarray(initial_X_guess, dtype=float).copy()
    P_guess = None if initial_P_guess is None else np.asarray(initial_P_guess, dtype=float).copy()
    current_delta0 = float(initial_delta0) if initial_delta0 is not None else float(delta0)
    stages = []
    last_successful_result = None
    last_successful_problem = None

    for stage_idx, q_max in enumerate(q_schedule):
        prob, params = build_goddard_problem(T=T, q_max=q_max, rho_m=rho_m)
        if X_guess is None:
            X_guess, P_guess = build_goddard_initial_guess(mesh, m_f=params["m_f"])
        elif pressure_feasible_init:
            X_guess = make_goddard_pressure_feasible_guess(
                X_guess,
                q_max=q_max,
                b=params["b"],
                beta=params["beta"],
                safety_factor=pressure_feasible_safety_factor,
            )
        try:
            result = solve_optimal_control(
                prob,
                mesh,
                tol_time=tol_time,
                tol_PA=tol_PA,
                tol_delta=tol_delta,
                max_iters=stage_max_iters,
                delta0=current_delta0,
                use_oracle_bootstrap=True,
                use_oracle_PA=True,
                store_iterates=store_iterates,
                verbose=verbose,
                initial_X_guess=X_guess,
                initial_P_guess=P_guess,
                initial_guess_label=f"{initial_guess_label_prefix}_{q_max}",
            )
            X_guess = np.asarray(result["X"], dtype=float)
            P_guess = np.asarray(result["P"], dtype=float)
            mesh = np.asarray(result["t_nodes"], dtype=float)
            current_delta0 = float(result["delta"])
            q_values = np.array(
                [goddard_dynamic_pressure(x[0], x[1], b=params["b"], beta=params["beta"]) for x in X_guess],
                dtype=float,
            )
            stage_info = {
                "stage_index": int(stage_idx),
                "q_max": float(q_max),
                "status": "ok",
                "mesh_intervals": int(len(mesh) - 1),
                "delta": float(result["delta"]),
                "final_action": str(result["log"][-1]["action"]),
                "eta_time": float(result["log"][-1]["eta_time"]),
                "eta_delta": float(result["log"][-1]["eta_delta"]),
                "max_dynamic_pressure": float(np.max(q_values)),
                "min_pressure_margin": float(np.min(q_max - q_values)),
                "terminal_state": X_guess[-1].tolist(),
                "objective_mesh_approx": float(result["log"][-1]["objective_mesh_approx"]),
            }
            stages.append(stage_info)
            result["problem_data"] = params
            result["wall_time_sec"] = float(result.get("wall_time_sec", np.nan))
            last_successful_result = result
            last_successful_problem = prob
        except Exception as exc:
            stages.append(
                {
                    "stage_index": int(stage_idx),
                    "q_max": float(q_max),
                    "status": "error",
                    "error": str(exc),
                }
            )
            break

    return {
        "q_schedule": q_schedule,
        "stages": stages,
        "last_successful_result": last_successful_result,
        "last_successful_problem": last_successful_problem,
    }


def run_goddard_barrier_then_qmax_continuation(
    *,
    mu_schedule,
    q_schedule,
    T: float = 0.15,
    rho_m: float = 1.0e4,
    n_init: int = 8,
    tol_time: float = 1.0e-4,
    tol_PA: float = 1.0e-4,
    tol_delta: float = 1.0e-4,
    delta0: float = 5.0e-2,
    barrier_stage_max_iters: int = 3,
    handoff_max_iters: int = 3,
    continuation_stage_max_iters: int = 3,
    verbose: bool = False,
    pressure_feasible_init: bool = True,
    pressure_feasible_safety_factor: float = 0.98,
    store_iterates: bool = False,
):
    q_schedule = [float(q) for q in q_schedule]
    if not q_schedule:
        raise ValueError("q_schedule must contain at least one value.")

    start_q_max = float(q_schedule[0])
    barrier_run = run_goddard_barrier_continuation(
        mu_schedule=mu_schedule,
        T=T,
        q_max=start_q_max,
        rho_m=rho_m,
        n_init=n_init,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        delta0=delta0,
        stage_max_iters=barrier_stage_max_iters,
        verbose=verbose,
        pressure_feasible_init=pressure_feasible_init,
        pressure_feasible_safety_factor=pressure_feasible_safety_factor,
        store_iterates=store_iterates,
    )
    handoff_bundle = run_goddard_true_from_barrier_handoff(
        barrier_run,
        T=T,
        q_max=start_q_max,
        rho_m=rho_m,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        delta0=None,
        max_iters=handoff_max_iters,
        verbose=verbose,
        store_iterates=store_iterates,
    )

    handoff_result = handoff_bundle["handoff_result"]
    handoff_prob = handoff_bundle["handoff_problem"]
    handoff_params = handoff_result["problem_data"]
    handoff_X = np.asarray(handoff_result["X"], dtype=float)
    handoff_q_values = np.array(
        [goddard_dynamic_pressure(x[0], x[1], b=handoff_params["b"], beta=handoff_params["beta"]) for x in handoff_X],
        dtype=float,
    )
    combined_stages = [
        {
            "stage_index": 0,
            "q_max": start_q_max,
            "status": "ok",
            "source": "true_from_barrier_handoff",
            "mesh_intervals": int(len(handoff_result["t_nodes"]) - 1),
            "delta": float(handoff_result["delta"]),
            "final_action": str(handoff_result["log"][-1]["action"]),
            "eta_time": float(handoff_result["log"][-1]["eta_time"]),
            "eta_delta": float(handoff_result["log"][-1]["eta_delta"]),
            "max_dynamic_pressure": float(np.max(handoff_q_values)),
            "min_pressure_margin": float(np.min(start_q_max - handoff_q_values)),
            "terminal_state": handoff_X[-1].tolist(),
            "objective_mesh_approx": float(handoff_result["log"][-1]["objective_mesh_approx"]),
        }
    ]

    continuation = {
        "q_schedule": [],
        "stages": [],
        "last_successful_result": handoff_result,
        "last_successful_problem": handoff_prob,
    }
    if len(q_schedule) > 1:
        continuation = run_goddard_qmax_continuation(
            q_schedule=q_schedule[1:],
            T=T,
            rho_m=rho_m,
            n_init=n_init,
            tol_time=tol_time,
            tol_PA=tol_PA,
            tol_delta=tol_delta,
            delta0=delta0,
            stage_max_iters=continuation_stage_max_iters,
            verbose=verbose,
            pressure_feasible_init=pressure_feasible_init,
            pressure_feasible_safety_factor=pressure_feasible_safety_factor,
            store_iterates=store_iterates,
            initial_mesh=np.asarray(handoff_result["t_nodes"], dtype=float),
            initial_X_guess=np.asarray(handoff_result["X"], dtype=float),
            initial_P_guess=np.asarray(handoff_result["P"], dtype=float),
            initial_delta0=float(handoff_result["delta"]),
            initial_guess_label_prefix="true_after_barrier_qmax",
        )
        for offset, stage in enumerate(continuation["stages"], start=1):
            stage_copy = dict(stage)
            stage_copy["stage_index"] = offset
            stage_copy.setdefault("source", "true_qmax_continuation")
            combined_stages.append(stage_copy)

    last_successful_result = continuation["last_successful_result"] or handoff_result
    last_successful_problem = continuation["last_successful_problem"] or handoff_prob
    return {
        "mu_schedule": [float(mu) for mu in mu_schedule],
        "q_schedule": q_schedule,
        "barrier_run": barrier_run,
        "handoff_bundle": handoff_bundle,
        "continuation": continuation,
        "stages": combined_stages,
        "last_successful_result": last_successful_result,
        "last_successful_problem": last_successful_problem,
    }


def run_goddard_barrier_continuation(
    *,
    mu_schedule,
    T: float = 0.15,
    q_max: float = 20.0,
    rho_m: float = 1.0e4,
    n_init: int = 8,
    tol_time: float = 1.0e-4,
    tol_PA: float = 1.0e-4,
    tol_delta: float = 1.0e-4,
    delta0: float = 5.0e-2,
    stage_max_iters: int = 3,
    verbose: bool = False,
    pressure_feasible_init: bool = True,
    pressure_feasible_safety_factor: float = 0.98,
    store_iterates: bool = False,
    retain_stage_results: bool = False,
):
    mu_schedule = [float(mu) for mu in mu_schedule]
    mesh = np.linspace(0.0, float(T), int(n_init))
    X_guess = None
    P_guess = None
    stages = []
    stage_payloads = []
    last_successful_result = None
    last_successful_problem = None

    for stage_idx, mu_barrier in enumerate(mu_schedule):
        prob, params = build_goddard_problem(T=T, q_max=q_max, rho_m=rho_m, mu_barrier=mu_barrier)
        if X_guess is None:
            X_guess, P_guess = build_goddard_initial_guess(mesh, m_f=params["m_f"])
        elif pressure_feasible_init:
            X_guess = make_goddard_pressure_feasible_guess(
                X_guess,
                q_max=q_max,
                b=params["b"],
                beta=params["beta"],
                safety_factor=pressure_feasible_safety_factor,
            )
        try:
            result, stage_prob = run_goddard_solver(
                n_init=n_init,
                T=T,
                q_max=q_max,
                rho_m=rho_m,
                tol_time=tol_time,
                tol_PA=tol_PA,
                tol_delta=tol_delta,
                delta0=delta0,
                max_iters=stage_max_iters,
                store_iterates=store_iterates,
                verbose=verbose,
                mu_barrier=mu_barrier,
                initial_mesh=mesh,
                initial_X_guess=X_guess,
                initial_P_guess=P_guess,
                initial_guess_label=f"barrier_mu_{mu_barrier:g}",
            )
            X_guess = np.asarray(result["X"], dtype=float)
            P_guess = np.asarray(result["P"], dtype=float)
            mesh = np.asarray(result["t_nodes"], dtype=float)
            q_values = np.array(
                [goddard_dynamic_pressure(x[0], x[1], b=params["b"], beta=params["beta"]) for x in X_guess],
                dtype=float,
            )
            margins = np.array([stage_prob.barrier_margin(x, t) for x, t in zip(X_guess, mesh)], dtype=float)
            stage_info = {
                "stage_index": int(stage_idx),
                "mu_barrier": float(mu_barrier),
                "status": "ok",
                "mesh_intervals": int(len(mesh) - 1),
                "delta": float(result["delta"]),
                "final_action": str(result["log"][-1]["action"]),
                "eta_time": float(result["log"][-1]["eta_time"]),
                "eta_delta": float(result["log"][-1]["eta_delta"]),
                "max_dynamic_pressure": float(np.max(q_values)),
                "min_pressure_margin": float(np.min(q_max - q_values)),
                "min_barrier_margin": float(np.min(margins)),
                "terminal_state": X_guess[-1].tolist(),
                "objective_mesh_approx": float(result["log"][-1]["objective_mesh_approx"]),
            }
            stages.append(stage_info)
            if retain_stage_results:
                stage_payloads.append(
                    {
                        "stage": dict(stage_info),
                        "result": result,
                        "problem": stage_prob,
                    }
                )
            last_successful_result = _compact_result_for_warm_start(result)
            last_successful_problem = stage_prob
        except Exception as exc:
            stages.append(
                {
                    "stage_index": int(stage_idx),
                    "mu_barrier": float(mu_barrier),
                    "status": "error",
                    "error": str(exc),
                }
            )
            break

    return {
        "mu_schedule": mu_schedule,
        "q_max": float(q_max),
        "stages": stages,
        "stage_payloads": stage_payloads,
        "last_successful_result": last_successful_result,
        "last_successful_problem": last_successful_problem,
    }


def run_goddard_true_from_barrier_handoff(
    barrier_run: dict,
    *,
    T: float = 0.15,
    q_max: float = 20.0,
    rho_m: float = 1.0e4,
    tol_time: float = 1.0e-4,
    tol_PA: float = 1.0e-4,
    tol_delta: float = 1.0e-4,
    delta0: float | None = None,
    max_iters: int = 3,
    verbose: bool = False,
    store_iterates: bool = True,
):
    last_result = barrier_run.get("last_successful_result")
    if last_result is None:
        raise ValueError("Barrier continuation did not produce a successful stage to hand off from.")
    mesh = np.asarray(last_result["t_nodes"], dtype=float)
    X_guess = np.asarray(last_result["X"], dtype=float)
    P_guess = np.asarray(last_result["P"], dtype=float)
    if delta0 is None:
        delta0 = float(last_result["delta"])
    result, prob = run_goddard_solver(
        T=T,
        q_max=q_max,
        rho_m=rho_m,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        delta0=delta0,
        max_iters=max_iters,
        verbose=verbose,
        store_iterates=store_iterates,
        mu_barrier=0.0,
        initial_mesh=mesh,
        initial_X_guess=X_guess,
        initial_P_guess=P_guess,
        initial_guess_label="true_from_barrier",
    )
    return {
        "barrier_run": barrier_run,
        "handoff_result": result,
        "handoff_problem": prob,
    }


def run_example(out_path: str | None = None, **solver_kwargs):
    result, _ = run_goddard_solver(verbose=True, **solver_kwargs)
    summary = summarize_goddard_result(result)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    summary = run_example(
        out_path="/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/archive_runs/goddard_fixedtime_smoke_summary.json",
        q_max=100.0,
    )
    print(json.dumps(summary, indent=2))
