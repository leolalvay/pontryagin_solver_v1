"""
Legacy 2014 Example 3.2 / archive Example 6: 1D nonsmooth Hamiltonian case.
"""
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
from core.smoothing import eval_H_smooth


def build_nonsmooth_problem():
    x0 = np.array([0.5])
    T = 1.0

    def dynamics(x, u, t):
        return np.array([u[0]], dtype=float)

    def stage_cost(x, u, t):
        return float(x[0] ** 10)

    def terminal_cost(x):
        return 0.0

    def hamiltonian_true(x, p, t):
        return float(x[0] ** 10 - abs(p[0]))

    def hamiltonian_smooth_fn(x, p, t, delta):
        x0 = float(x[0])
        p0 = float(p[0])
        delta = max(float(delta), 1e-14)
        radial = float(np.sqrt(p0 * p0 + delta * delta))
        H_delta = x0 ** 10 - radial
        grad_p = np.array([-p0 / radial], dtype=float)
        grad_x = np.array([10.0 * (x0 ** 9)], dtype=float)
        return H_delta, grad_p, grad_x

    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(np.array([-1.0]), np.array([1.0])),
        state_bounds=None,
        hamiltonian_true=hamiltonian_true,
        u_star_fn=None,
        hamiltonian_grad_fn=None,
        hamiltonian_smooth_fn=hamiltonian_smooth_fn,
    )
    problem_data = {
        "legacy_example": "3.2",
        "problem_name": "simple nonsmooth Hamiltonian",
        "x0": x0,
        "T": T,
        "u_min": -1.0,
        "u_max": 1.0,
        "exact_switch_time": 0.5,
        "exact_objective_J": float((0.5 ** 11) / 11.0),
    }
    return prob, problem_data


def exact_solution(t_nodes):
    t_nodes = np.asarray(t_nodes, dtype=float)
    x_exact = np.maximum(0.5 - t_nodes, 0.0)
    p_exact = np.where(t_nodes <= 0.5, (0.5 - t_nodes) ** 10, 0.0)
    u_exact = np.where(t_nodes[:-1] < 0.5, -1.0, 0.0)
    return x_exact, p_exact, u_exact


def build_initial_guess_arrays(t_nodes, mode="default"):
    t_nodes = np.asarray(t_nodes, dtype=float)
    if mode == "default":
        return None, None

    x_exact, p_exact, _ = exact_solution(t_nodes)
    X_init = x_exact.reshape(-1, 1)

    if mode == "exact_state":
        return X_init, None
    if mode == "exact_state_costate":
        return X_init, p_exact.reshape(-1, 1)

    raise ValueError(
        "Unknown initial guess mode. Expected one of: "
        "'default', 'exact_state', 'exact_state_costate'."
    )


def run_nonsmooth_solver(
    n_init=20,
    tol_time=1.0e-6,
    tol_PA=1.0e-6,
    tol_delta=1.0e-6,
    max_iters=25,
    delta0=0.02,
    use_oracle_bootstrap=False,
    use_oracle_PA=False,
    use_explicit_hamiltonian_gradients=False,
    store_iterates=False,
    fallback_solver="least_squares",
    verbose=True,
    initial_guess_mode="default",
):
    prob, problem_data = build_nonsmooth_problem()
    t_nodes = np.linspace(0.0, problem_data["T"], n_init)
    initial_X_guess, initial_P_guess = build_initial_guess_arrays(t_nodes, mode=initial_guess_mode)

    t0 = time.perf_counter()
    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        max_iters=max_iters,
        delta0=delta0,
        use_oracle_bootstrap=use_oracle_bootstrap,
        use_oracle_PA=use_oracle_PA,
        use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
        store_iterates=store_iterates,
        fallback_solver=fallback_solver,
        verbose=verbose,
        initial_X_guess=initial_X_guess,
        initial_P_guess=initial_P_guess,
        initial_guess_label=initial_guess_mode,
    )
    wall_time = time.perf_counter() - t0
    result["problem_data"] = problem_data
    result["wall_time_sec"] = wall_time
    result["legacy_reference_label"] = "2014 reference manuscript Example 3.2"
    return result, prob


def _compute_node_controls(prob, bundle, X, P, t_nodes):
    controls = []
    for i, t_i in enumerate(t_nodes):
        _, u_star = compute_H(prob, P[i], X[i], t_i, bundle.controls, restricted=True, use_oracle=False)
        if u_star is None:
            raise RuntimeError(f"No admissible control found at nonsmooth node {i}.")
        controls.append(np.asarray(u_star, dtype=float).reshape(-1))
    return np.vstack(controls)


def _compute_interval_controls(prob, bundle, X, P, t_nodes):
    controls = []
    for i, t_i in enumerate(t_nodes[:-1]):
        _, u_star = compute_H(prob, P[i + 1], X[i], t_i, bundle.controls, restricted=True, use_oracle=False)
        if u_star is None:
            raise RuntimeError(f"No admissible interval control found at nonsmooth interval {i}.")
        controls.append(np.asarray(u_star, dtype=float).reshape(-1))
    if not controls:
        return np.zeros((0, 1))
    return np.vstack(controls)


def _compute_interval_effective_controls(prob, bundle, X, P, t_nodes, delta):
    controls = []
    for i, t_i in enumerate(t_nodes[:-1]):
        _, grad_p, _ = eval_H_smooth(prob, bundle, P[i + 1], X[i], t_i, delta)
        controls.append(np.asarray(grad_p, dtype=float).reshape(-1))
    if not controls:
        return np.zeros((0, 1))
    return np.vstack(controls)


def _objective_mesh_approx(prob, t_nodes, X, interval_controls):
    objective = float(prob.g(X[-1]))
    for i in range(len(t_nodes) - 1):
        dt = float(t_nodes[i + 1] - t_nodes[i])
        objective += float(prob.l(X[i], interval_controls[i], t_nodes[i])) * dt
    return objective


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
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.2e}"


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
    out = []
    for ch in str(text):
        out.append(repl.get(ch, ch))
    return "".join(out)


def _write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _latex_table_from_pairs(caption, label, rows):
    body = "\n".join(
        f"{_latex_escape(name)} & {_latex_escape(value)} \\\\" for name, value in rows
    )
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
    body_lines = []
    for row in rows:
        body_lines.append(
            " & ".join(_latex_escape(row.get(name, "")) for name in fieldnames) + r" \\"
        )
    body = "\n".join(body_lines)
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


def _legacy_reference_rows():
    summary_path = (
        Path("/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen")
        / "archive_runs"
        / "legacy_2014"
        / "ex32_nonsmooth"
        / "summary.json"
    )
    rows = [
        {"quantity": "reference_source", "value": "2014 manuscript Example 3.2 exact solution and executable legacy baseline"},
        {"quantity": "exact_objective_J", "value": _format_value((0.5 ** 11) / 11.0)},
        {"quantity": "exact_terminal_state", "value": _format_value(0.0)},
        {"quantity": "exact_switch_time", "value": _format_value(0.5)},
    ]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        rows.extend(
            [
                {"quantity": "legacy_J_mesh", "value": _format_value(summary["J_mesh"])},
                {"quantity": "legacy_relative_objective_error", "value": _format_value(summary["relative_objective_error"])},
                {"quantity": "legacy_terminal_state", "value": _format_value(summary["terminal_state"])},
                {"quantity": "legacy_N_final", "value": _format_value(summary["N_final"])},
                {"quantity": "legacy_M_final", "value": _format_value(summary["M_final"])},
                {"quantity": "legacy_delta_final", "value": _format_value(summary["delta_final"])},
                {"quantity": "legacy_eta_time", "value": _format_value(summary["eta_time"])},
                {"quantity": "legacy_eta_PA", "value": _format_value(summary["eta_PA"])},
                {"quantity": "legacy_eta_delta", "value": _format_value(summary["eta_delta"])},
            ]
        )
    return rows


def _input_parameter_rows(result):
    pdata = result["problem_data"]
    settings = result["settings"]
    return [
        {"parameter": "legacy_example", "value": pdata["legacy_example"]},
        {"parameter": "problem_name", "value": pdata["problem_name"]},
        {"parameter": "x0", "value": _format_value(pdata["x0"][0])},
        {"parameter": "T", "value": _format_value(pdata["T"])},
        {"parameter": "u_min", "value": _format_value(pdata["u_min"])},
        {"parameter": "u_max", "value": _format_value(pdata["u_max"])},
        {"parameter": "exact_switch_time", "value": _format_value(pdata["exact_switch_time"])},
        {"parameter": "n_init", "value": _format_value(len(result["log"][0]["t_nodes_iter"]) - 1)},
        {"parameter": "tol_time", "value": _format_value(settings["tol_time"])},
        {"parameter": "tol_PA", "value": _format_value(settings["tol_PA"])},
        {"parameter": "tol_delta", "value": _format_value(settings["tol_delta"])},
        {"parameter": "max_iters", "value": _format_value(settings["max_iters"])},
        {"parameter": "delta0", "value": _format_value(settings["delta0"])},
        {"parameter": "s_time", "value": _format_value(settings["s_time"])},
        {"parameter": "K_time", "value": _format_value(settings["K_time"])},
        {"parameter": "time_balance_ratio", "value": _format_value(settings["time_balance_ratio"])},
        {"parameter": "pa_add_fraction", "value": _format_value(settings["pa_add_fraction"])},
        {"parameter": "pa_time_separation_factor", "value": _format_value(settings["pa_time_separation_factor"])},
        {"parameter": "pa_gap_floor_ratio", "value": _format_value(settings["pa_gap_floor_ratio"])},
        {"parameter": "initial_guess_label", "value": settings["initial_guess_label"]},
        {"parameter": "newton_tol", "value": _format_value(settings["newton_tol"])},
        {"parameter": "newton_max_iter", "value": _format_value(settings["newton_max_iter"])},
        {"parameter": "fallback_solver", "value": settings["fallback_solver"]},
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


def _final_summary(result, prob):
    last = result["log"][-1]
    t_nodes = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)
    P = np.asarray(result["P"], dtype=float)
    U_nodes = _compute_node_controls(prob, result["bundle"], X, P, t_nodes)
    U_bundle_intervals = _compute_interval_controls(prob, result["bundle"], X, P, t_nodes)
    U_effective_intervals = _compute_interval_effective_controls(prob, result["bundle"], X, P, t_nodes, result["delta"])
    objective_mesh = _objective_mesh_approx(prob, t_nodes, X, U_effective_intervals)
    x_exact, p_exact, u_exact = exact_solution(t_nodes)
    control_error = float(np.max(np.abs(U_effective_intervals[:, 0] - u_exact))) if len(u_exact) else 0.0
    bundle_control_error = float(np.max(np.abs(U_bundle_intervals[:, 0] - u_exact))) if len(u_exact) else 0.0
    return {
        "terminal_state": float(X[-1, 0]),
        "objective_mesh_approx": float(objective_mesh),
        "exact_objective_J": float((0.5 ** 11) / 11.0),
        "objective_gap_vs_exact": float(objective_mesh - (0.5 ** 11) / 11.0),
        "relative_objective_error": float(abs(objective_mesh - (0.5 ** 11) / 11.0) / max(abs((0.5 ** 11) / 11.0), 1e-16)),
        "max_state_error": float(np.max(np.abs(X[:, 0] - x_exact))),
        "max_costate_error": float(np.max(np.abs(P[:, 0] - p_exact))),
        "max_interval_effective_control_error": control_error,
        "max_interval_bundle_control_error": bundle_control_error,
        "outer_iterations_logged": len(result["log"]),
        "last_outer_iteration": int(last["iteration"]),
        "mesh_points": int(len(t_nodes)),
        "mesh_intervals": int(len(t_nodes) - 1),
        "planes": int(result["bundle"].num_planes()),
        "eta_time": float(last["eta_time"]),
        "eta_time_sum": float(last.get("eta_time_sum", 0.0)),
        "tol_time_star": float(last["tol_time_star"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "delta": float(last["delta"]),
        "newton_iter": int(last["newton_iter"]),
        "newton_residual": float(last["newton_residual"]),
        "solver_phase": last.get("solver_phase", "newton"),
        "fallback_used": bool(last.get("fallback_used", False)),
        "final_action": last.get("action", ""),
        "final_note": last.get("note", ""),
        "all_indicators_within_tolerance": bool(last.get("all_indicators_within_tolerance", False)),
        "wall_time_sec": float(result.get("wall_time_sec", float("nan"))),
        "final_bundle_node_control": float(U_nodes[-1, 0]),
    }


def _plot_iteration_state_costate(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)[:, 0]
    P = np.asarray(entry["P_iter"], dtype=float)[:, 0]
    x_exact, p_exact, _ = exact_solution(t)
    support_points = entry.get("bundle_support_points_so_far", [])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, X, label="x(t)")
    axes[0].plot(t, x_exact, "--", label=r"$x^*(t)$")
    if support_points:
        support_t = np.array([float(point["time"]) for point in support_points], dtype=float)
        support_x = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in support_points], dtype=float)
        axes[0].scatter(support_t, support_x, s=20, color="tab:red", marker="o", label="bundle support points", zorder=5)
    axes[0].set_ylabel("state")
    axes[0].set_title("State trajectory with exact comparison and support points")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, P, label="p(t)")
    axes[1].plot(t, p_exact, "--", label=r"$p^*(t)$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("costate")
    axes[1].set_title("Costate trajectory with exact comparison")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_bundle_support_points(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)[:, 0]
    support_points = entry.get("bundle_support_points_so_far", [])
    current_iter = int(entry["iteration"])
    current_points = [point for point in support_points if int(point.get("iteration", -1)) == current_iter]
    previous_points = [point for point in support_points if int(point.get("iteration", -1)) < current_iter]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, X, color="tab:blue", linewidth=1.8, label="state trajectory")
    if previous_points:
        prev_t = np.array([float(point["time"]) for point in previous_points], dtype=float)
        prev_x = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in previous_points], dtype=float)
        axes[0].scatter(prev_t, prev_x, s=26, color="tab:red", marker="o", alpha=0.75, label="previous support points", zorder=5)
        for time_value in prev_t:
            axes[1].axvline(time_value, color="tab:red", alpha=0.10, linewidth=0.8)
    if current_points:
        curr_t = np.array([float(point["time"]) for point in current_points], dtype=float)
        curr_x = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in current_points], dtype=float)
        axes[0].scatter(
            curr_t,
            curr_x,
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

    cumulative_count = np.zeros_like(t)
    if support_points:
        support_t_all = np.array(sorted(float(point["time"]) for point in support_points), dtype=float)
        cumulative_count = np.searchsorted(support_t_all, t, side="right")
    axes[1].step(t, cumulative_count, where="post", color="tab:purple", label="cumulative support points up to time")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("count")
    axes[1].set_title("Time locations of affine-plane support points")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_control(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    P = np.asarray(entry["P_iter"], dtype=float)[:, 0]
    delta = float(entry["delta"])
    U_bundle = np.asarray(entry["U_iter"], dtype=float)[:, 0]
    U_effective = -P[1:] / np.sqrt(P[1:] * P[1:] + delta * delta)
    _, _, u_exact = exact_solution(t)
    fig = plt.figure(figsize=(9, 4))
    plt.step(t[:-1], U_effective, where="post", label="effective smoothed control")
    plt.step(t[:-1], U_bundle[:-1], where="post", linestyle=":", label="post-processed bundle minimizer")
    plt.step(t[:-1], u_exact, where="post", linestyle="--", label=r"$u^*(t)$")
    plt.xlabel("t")
    plt.ylabel("control")
    plt.title("Effective control used in the state update")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_rho(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    rho = np.asarray(entry["rho"], dtype=float)
    rho_bar = np.asarray(entry["rho_bar"], dtype=float)
    t_int = t[:-1][: len(rho)]
    fig = plt.figure(figsize=(9, 4))
    plt.step(t_int, np.abs(rho), where="post", label=r"$|\rho_n|$")
    plt.step(t_int, np.abs(rho_bar), where="post", label=r"$|\bar{\rho}_n|$")
    plt.xlabel("t")
    plt.ylabel(r"$|\rho|$")
    plt.title("Estimated time-discretization error density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_mesh_and_indicator(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    dt = np.diff(t)
    r_bar = np.asarray(entry["r_bar"], dtype=float)
    t_int = t[:-1][: len(r_bar)]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].step(t[:-1], dt, where="post", label=r"$\Delta t_n$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\Delta t$")
    axes[0].set_title("Time mesh discretization")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].step(t_int, r_bar, where="post", label=r"$\bar r_n$")
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


def _plot_iteration_pa_delta_contributions(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    t_int = t[:-1]
    eta_pa_local = np.asarray(entry.get("eta_PA_local", np.zeros_like(t_int)), dtype=float)
    eta_delta_local = np.asarray(entry.get("eta_delta_local", np.zeros_like(t_int)), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].step(t_int[: len(eta_pa_local)], np.maximum(np.abs(eta_pa_local), 1e-18), where="post", label=r"local $\eta_{\mathrm{PA}}$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("PA contribution")
    axes[0].set_title("Per-interval plane-approximation contributions")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].step(t_int[: len(eta_delta_local)], np.maximum(np.abs(eta_delta_local), 1e-18), where="post", label=r"local $\eta_{\delta}$", color="tab:green")
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


def _write_iteration_artifacts(log, iteration_root):
    for entry in log:
        if "X_iter" not in entry or "P_iter" not in entry or "U_iter" not in entry:
            raise ValueError("Deep report export requires store_iterates=True in the nonsmooth solver run.")
        iter_idx = int(entry["iteration"])
        iter_dir = iteration_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "iteration_data.json").write_text(json.dumps(_jsonable(entry), indent=2))
        _plot_iteration_state_costate(entry, iter_dir / f"iter_{iter_idx:02d}_state_costate.pdf")
        _plot_iteration_bundle_support_points(entry, iter_dir / f"iter_{iter_idx:02d}_bundle_support_points.pdf")
        _plot_iteration_control(entry, iter_dir / f"iter_{iter_idx:02d}_control.pdf")
        _plot_iteration_rho(entry, iter_dir / f"iter_{iter_idx:02d}_rho_density.pdf")
        _plot_iteration_mesh_and_indicator(entry, iter_dir / f"iter_{iter_idx:02d}_mesh_and_indicator.pdf")
        _plot_iteration_pa_delta_contributions(entry, iter_dir / f"iter_{iter_idx:02d}_pa_delta_contributions.pdf")


def _generate_ex32_report_tex(out_dir, summary, input_rows, reference_rows, outer_rows, log):
    intro_status = (
        f"The exported artifact ends with action {summary['final_action']}; all three indicators are below tolerance at that reported final iterate."
        if summary["all_indicators_within_tolerance"]
        else f"The exported artifact ends with action {summary['final_action']}, and at least one indicator is still above tolerance."
    )
    final_pairs = [(key, _format_value(value)) for key, value in summary.items()]
    input_pairs = [(row["parameter"], row["value"]) for row in input_rows]
    reference_pairs = [(row["quantity"], row["value"]) for row in reference_rows]
    outer_display_fields = [
        "iter",
        "action",
        "N",
        "M",
        "delta",
        "J_h",
        "eta_time / tol*",
        "eta_PA",
        "eta_delta",
        "Newton it",
        "all below tol",
    ]
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
                "all below tol": row["all_indicators_within_tolerance"],
            }
        )

    iteration_sections = []
    for entry in log:
        iter_idx = int(entry["iteration"])
        iter_rel = Path("iterations") / f"iter_{iter_idx:02d}"
        status_text = "all indicators below tolerance" if entry.get("all_indicators_within_tolerance", False) else "refinement still required"
        note_text = entry.get("note", "")
        note_line = f" Note: {_latex_escape(note_text)}." if note_text else ""
        pa_plan = entry.get("pa_addition_plan", {})
        pa_tables = ""
        if entry.get("action") == "add_plane" and pa_plan.get("target_count", 0) > 0:
            summary_rows = [
                ("target candidate count", _format_value(pa_plan.get("target_count", 0))),
                ("selected candidate count", _format_value(len(pa_plan.get("selected_node_indices", [])))),
                ("rejected by time-separation", _format_value(pa_plan.get("rejected_by_time_separation", 0))),
                ("rejected as duplicate control", _format_value(pa_plan.get("rejected_as_duplicate_control", 0))),
                ("rejected below score floor", _format_value(pa_plan.get("rejected_below_score_floor", 0))),
                ("max PA gap", _format_value(pa_plan.get("max_gap", 0.0))),
                ("max PA score", _format_value(pa_plan.get("max_score", 0.0))),
                ("PA score floor", _format_value(pa_plan.get("score_floor", 0.0))),
            ]
            selected_rows = []
            for node_idx, time_value, gap_value, dt_value, score_value, added_flag, bundle_size_after in zip(
                pa_plan.get("selected_node_indices", []),
                pa_plan.get("selected_times", []),
                pa_plan.get("selected_pa_gaps", []),
                pa_plan.get("selected_time_steps", []),
                pa_plan.get("selected_pa_scores", []),
                pa_plan.get("selected_added_to_bundle", []),
                pa_plan.get("selected_bundle_size_after", []),
            ):
                selected_rows.append(
                    {
                        "node": _format_value(node_idx),
                        "time": _format_value(time_value),
                        "pa_gap": _format_value(gap_value),
                        "delta_t": _format_value(dt_value),
                        "pa_score": _format_value(score_value),
                        "added": _format_value(added_flag),
                        "bundle size after": _format_value(bundle_size_after),
                    }
                )
            pa_tables = dedent(
                f"""
                {_latex_table_from_pairs(f"PA enrichment summary at iteration {iter_idx}.", f"tab:ex32-pa-summary-{iter_idx}", summary_rows)}

                {_latex_longtable_from_rows(
                    f"Selected PA enrichment candidates at iteration {iter_idx}.",
                    f"tab:ex32-pa-candidates-{iter_idx}",
                    ["node", "time", "pa_gap", "delta_t", "pa_score", "added", "bundle size after"],
                    selected_rows,
                )}
                """
            ).strip()
        iteration_sections.append(
            dedent(
                f"""
                \\subsection{{Iteration {iter_idx}}}
                Iteration {iter_idx} ends with action \\texttt{{{_latex_escape(entry.get("action", ""))}}}, mesh intervals $N={int(entry["N"])}$, planes $M={int(entry["M"])}$, and status {_latex_escape(status_text)}.{note_line}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_state_costate.pdf").as_posix()}}}
                \\caption{{State and costate trajectories at iteration {iter_idx}, with the exact solution overlaid. Red support markers are also overlaid on the state plot, but the dedicated support-point figure below should be used as the primary record of where affine planes were sampled.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_bundle_support_points.pdf").as_posix()}}}
                \\caption{{Dedicated bundle-support diagnostic at iteration {iter_idx}. The upper panel shows the state trajectory with all cumulative affine-plane support points; square black markers identify support points added at the current iteration. The lower panel shows their time locations explicitly so the plane construction can be audited.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_control.pdf").as_posix()}}}
                \\caption{{Control diagnostic at iteration {iter_idx}. The solid curve is the effective smoothed control actually used by the discrete state update, the dotted curve is the post-processed bundle minimizer, and the dashed curve is the exact bang-off control.}}
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
                \\caption{{Per-interval $\\eta_{{\\mathrm{{PA}}}}$ and $\\eta_\\delta$ contributions at iteration {iter_idx}. These are the local intervalwise contributions whose sums produce the reported global plane-approximation and smoothing indicators.}}
                \\end{{figure}}

                {pa_tables}
                """
            ).strip()
        )

    report = dedent(
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
        \\title{{Standalone Deep Report for Legacy 2014 Example 3.2}}
        \\author{{Archive Python adaptive Pontryagin solver}}
        \\date{{\\today}}

        \\begin{{document}}
        \\maketitle

        \\section{{Legacy 2014 Example 3.2: Simple Nonsmooth Hamiltonian}}
        This standalone report documents the current executable nonsmooth example derived from the 2014 reference manuscript. The purpose is diagnostic and comparative: every adaptive outer iterate is exposed with the same plot family and the same summary fields used in the Example 1 and Example 3.1 deep reports, so the solver trajectory can be inspected carefully against both the exact solution and the archived executable baseline.

        The current final status is: {_latex_escape(intro_status)}

        Each iteration subsection includes a dedicated affine-plane support diagnostic and a separate intervalwise $\\eta_{{\\mathrm{{PA}}}}$ / $\\eta_\\delta$ contribution figure. Those two figures should be read together whenever we assess whether the bundle enrichment and smoothing adaptivity are behaving sensibly.

        \\subsection{{Input parameters}}
        {_latex_table_from_pairs("Input parameters used for the legacy Example 3.2 run.", "tab:ex32-input-parameters", input_pairs)}

        \\subsection{{Reference values}}
        {_latex_table_from_pairs("Reference values currently available for legacy Example 3.2.", "tab:ex32-reference-values", reference_pairs)}

        \\subsection{{Outer-loop summary}}
        {_latex_longtable_from_rows("Adaptive outer-loop iterations of the proposed solver on legacy Example 3.2.", "tab:ex32-outer-loop", outer_display_fields, outer_display_rows)}

        \\subsection{{Final reported numbers}}
        {_latex_table_from_pairs("Final numbers reported by the proposed solver on the nonsmooth example.", "tab:ex32-final-summary", final_pairs)}

        \\subsection{{Error-estimate history}}
        The figure below shows the progress of the time-discretization, plane-approximation, and smoothing indicators against the outer iteration count. In particular, it distinguishes the stopping quantity $\\eta_{{\\mathrm{{time}}}}=\\max_n \\bar r_n$ from the total estimated time-discretization error $\\sum_n \\bar r_n$.

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.9\\textwidth]{{figures/error_indicator_history.pdf}}
        \\caption{{Error-estimate history across the adaptive outer loop, including both the stopping quantity $\\eta_{{\\mathrm{{time}}}}=\\max_n \\bar r_n$ and the total estimated time-discretization error $\\sum_n \\bar r_n$.}}
        \\end{{figure}}

        \\section{{Per-Iteration Dossier}}
        {"\n\n".join(iteration_sections)}

        \\end{{document}}
        """
    ).strip() + "\n"
    report_path = Path(out_dir) / "report.tex"
    report_path.write_text(report)
    return report_path


def export_ex32_deep_report_artifacts(result, prob, out_dir):
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    iteration_dir = out_dir / "iterations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    iteration_dir.mkdir(parents=True, exist_ok=True)

    input_rows = _input_parameter_rows(result)
    reference_rows = _legacy_reference_rows()
    outer_rows = _outer_history_rows(result["log"])
    summary = _final_summary(result, prob)

    (tables_dir / "final_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (tables_dir / "final_summary.csv").write_text(
        "quantity,value\n"
        + "\n".join(f"{key},{_format_value(value)}" for key, value in summary.items())
        + "\n"
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
    (tables_dir / "outer_loop_history.json").write_text(json.dumps(_jsonable(result["log"]), indent=2))

    _write_iteration_artifacts(result["log"], iteration_dir)
    _plot_error_indicator_history(result["log"], figures_dir / "error_indicator_history.pdf")
    report_path = _generate_ex32_report_tex(out_dir, summary, input_rows, reference_rows, outer_rows, result["log"])
    return {
        "summary": summary,
        "report_tex": str(report_path),
        "out_dir": str(out_dir),
    }


def summarize_nonsmooth_results(result, print_last_log_only=False):
    if print_last_log_only:
        last = result["log"][-1]
        print("\nLegacy 2014 Example 3.2 (Simple nonsmooth Hamiltonian)")
        print("last outer iter =", last["iteration"])
        print("len(t_nodes) =", len(result["t_nodes"]))
        print("final delta =", result["delta"])
        print("last log entry =", last)
        return

    print("\nLegacy 2014 Example 3.2 (Simple nonsmooth Hamiltonian)")
    print("len(log) =", len(result["log"]))
    print("last outer iter =", result["log"][-1]["iteration"])
    print("len(t_nodes) =", len(result["t_nodes"]))
    print("X.shape =", result["X"].shape)
    print("P.shape =", result["P"].shape)
    print("final delta =", result["delta"])
    print("last log entry =", result["log"][-1])


def run_example():
    result, _ = run_nonsmooth_solver(store_iterates=False, verbose=True)
    summarize_nonsmooth_results(result)
    return result


if __name__ == "__main__":
    run_example()
