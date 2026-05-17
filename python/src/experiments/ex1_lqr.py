"""
Example 1: linear quadratic regulator with deep reporting support.
"""
import csv
import json
from functools import partial
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np

from core.adaptivity import solve_optimal_control
from core.hamiltonian import compute_H
from core.problem import OCPProblem


def build_ex1_lqr_problem(u_min=-11.0, u_max=5.0):
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = 1.0e-2 * np.eye(1)
    Qf = Q.copy()
    x0 = np.array([1.0, 0.0])
    T = 1.0

    def dynamics(x, u, t):
        return A @ x + B @ u

    def stage_cost(x, u, t):
        return float(x @ Q @ x + u.T @ R @ u)

    def terminal_cost(x):
        return float(x @ Qf @ x)

    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(np.array([u_min]), np.array([u_max])),
        state_bounds=None,
    )
    problem_data = {
        "A": A,
        "B": B,
        "Q": Q,
        "R": R,
        "Qf": Qf,
        "x0": x0,
        "T": T,
        "u_min": float(u_min),
        "u_max": float(u_max),
    }
    return prob, problem_data


def run_ex1_lqr_solver(
    n_init=20,
    tol_time=2e-2,
    tol_PA=1e-2,
    tol_delta=1e-2,
    max_iters=15,
    delta0=0.15,
    u_min=-11.0,
    u_max=5.0,
    store_iterates=False,
    fallback_solver="least_squares",
):
    """
    Build and solve Example 1 LQR with the adaptive Pontryagin solver.
    """
    prob, problem_data = build_ex1_lqr_problem(u_min=u_min, u_max=u_max)
    t_nodes = np.linspace(0.0, problem_data["T"], n_init + 1)

    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        max_iters=max_iters,
        delta0=delta0,
        store_iterates=store_iterates,
        fallback_solver=fallback_solver,
    )
    result["problem_data"] = problem_data
    return result, prob


def _compute_controls(prob, bundle, X, P, t_nodes):
    controls = []
    for i in range(len(t_nodes)):
        _, u_star = compute_H(prob, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
        if u_star is None:
            raise RuntimeError(f"No admissible control found at Example 1 node {i}.")
        controls.append(np.asarray(u_star, dtype=float).reshape(-1))
    return np.vstack(controls)


def _objective_mesh_approx(prob, t_nodes, X, controls):
    obj = float(prob.g(X[-1]))
    for i in range(len(t_nodes) - 1):
        dt = float(t_nodes[i + 1] - t_nodes[i])
        obj += float(prob.l(X[i], controls[i], t_nodes[i])) * dt
    return obj


def summarize_ex1_results(result, prob, print_last_log_only=True):
    """
    Print compact diagnostics for Example 1 solver output.
    """
    log = result["log"]
    last = log[-1]

    t_nodes = np.asarray(result["t_nodes"])
    X = np.asarray(result["X"])
    P = np.asarray(result["P"])
    bundle = result["bundle"]
    controls = _compute_controls(prob, bundle, X, P, t_nodes)
    obj = _objective_mesh_approx(prob, t_nodes, X, controls)

    print("=== Example 1 (LQR) ===")
    print(f"outer iterations logged: {len(log)}")
    print(f"last outer iteration:    {last.get('iteration')}")
    print(f"mesh points:             {len(t_nodes)}")
    print(f"state shape:             {X.shape}")
    print(f"costate shape:           {P.shape}")
    print(f"planes:                  {bundle.num_planes()}")
    print(f"objective (mesh approx): {obj:.12e}")

    for key in [
        "eta_time",
        "eta_PA",
        "eta_delta",
        "delta",
        "newton_iter",
        "newton_residual",
        "tol_time_star",
        "mark_thr",
        "note",
        "action",
        "solver_phase",
    ]:
        if key in last:
            print(f"{key:24s}: {last[key]}")

    if not print_last_log_only:
        print("\nIndicator history (compact):")
        for entry in log:
            msg = (
                f"it={entry.get('iteration')} "
                f"N={entry.get('N')} M={entry.get('M')} "
                f"eta_time={entry.get('eta_time', float('nan')):.2e} "
                f"eta_PA={entry.get('eta_PA', float('nan')):.2e} "
                f"eta_delta={entry.get('eta_delta', float('nan')):.2e} "
                f"newton_it={entry.get('newton_iter')} "
                f"res={entry.get('newton_residual', float('nan')):.2e} "
                f"action={entry.get('action')}"
            )
            if "note" in entry and entry["note"]:
                msg += f" note={entry['note']}"
            print(msg)


def save_plot(fig, stem, fig_dir, ext="pdf"):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def keep_plot(fig, stem=None):
    pass


def plot_ex1_results(
    result,
    prob,
    out_prefix="example1_solver",
    save_plots=False,
    plot_ext="pdf",
    fig_dir=None,
):
    t = np.asarray(result["t_nodes"])
    X = np.asarray(result["X"])
    P = np.asarray(result["P"])
    bundle = result["bundle"]
    log = result.get("log", [])

    last_with_indicators = {}
    for entry in reversed(log):
        if ("rho" in entry) and ("rho_bar" in entry) and ("r_bar" in entry):
            last_with_indicators = entry
            break

    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent / "figures"

    plot_action = partial(save_plot, fig_dir=fig_dir, ext=plot_ext) if save_plots else keep_plot
    render_plots = (lambda: None) if save_plots else plt.show
    U = _compute_controls(prob, bundle, X, P, t)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, X[:, 0], label="x1(t)")
    plt.plot(t, X[:, 1], label="x2(t)")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("State trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_state_x")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, P[:, 0], label="p1(t)")
    plt.plot(t, P[:, 1], label="p2(t)")
    plt.xlabel("t")
    plt.ylabel("costate")
    plt.title("Costate trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_costate_p")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, U[:, 0], label="u(t)")
    plt.xlabel("t")
    plt.ylabel("control")
    plt.title("Control trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_control_u")

    if len(t) > 1:
        dt = np.diff(t)
        fig = plt.figure(figsize=(8, 5))
        plt.step(t[:-1], dt, where="post", label=r"$\Delta t_n$")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel(r"$\Delta t$")
        plt.title("Time mesh step sizes")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plot_action(fig, f"{out_prefix}_t_vs_dt")

    if "rho" in last_with_indicators and "rho_bar" in last_with_indicators:
        rho = np.asarray(last_with_indicators["rho"], dtype=float)
        rho_bar = np.asarray(last_with_indicators["rho_bar"], dtype=float)
        t_int = t[:-1][: len(rho)]
        fig = plt.figure(figsize=(8, 5))
        plt.step(t_int, rho, where="post", label=r"$\rho_n$")
        plt.step(t_int, rho_bar, where="post", label=r"$\bar{\rho}_n$")
        plt.xlabel("t")
        plt.ylabel(r"$\rho$")
        plt.title(r"Error density: $\rho_n,\ \bar{\rho}_n$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_action(fig, f"{out_prefix}_rho_density")

    if "r_bar" in last_with_indicators:
        r_bar = np.asarray(last_with_indicators["r_bar"], dtype=float)
        t_int = t[:-1][: len(r_bar)]
        fig = plt.figure(figsize=(8, 5))
        plt.step(t_int, r_bar, where="post", label=r"$\bar r_n = |\bar\rho_n|\Delta t_n^2$")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel(r"$\bar r$")
        plt.title(r"Time indicator: $\bar r_n$")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plot_action(fig, f"{out_prefix}_r_indicator")

    render_plots()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _matrix_str(matrix):
    return np.array2string(
        np.asarray(matrix),
        precision=6,
        separator=", ",
        max_line_width=10**6,
    ).replace("\n", " ")


def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12e}"
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
    out = []
    for ch in str(text):
        out.append(repl.get(ch, ch))
    return "".join(out)


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


def _latex_table_from_rows(caption, label, fieldnames, rows):
    header = " & ".join(_latex_escape(name) for name in fieldnames) + r" \\"
    body_lines = []
    for row in rows:
        body_lines.append(
            " & ".join(_latex_escape(row.get(name, "")) for name in fieldnames) + r" \\"
        )
    body = "\n".join(body_lines) if body_lines else r"\multicolumn{"+str(len(fieldnames))+r"}{c}{No rows} \\"
    return dedent(
        f"""
        {{\\scriptsize
        \\begin{{table}}[H]
        \\centering
        \\begin{{tabular}}{{{'l' * len(fieldnames)}}}
        \\toprule
        {header}
        \\midrule
        {body}
        \\bottomrule
        \\end{{tabular}}
        \\caption{{{_latex_escape(caption)}}}
        \\label{{{label}}}
        \\end{{table}}
        }}
        """
    ).strip()


def _short_display(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.2e}"


def _reference_benchmark():
    from tests.ex1_lqr_test import run_lqr_riccati_benchmark

    return run_lqr_riccati_benchmark()


def _reference_table_rows(reference):
    data = reference["data"]
    ric = reference["riccati_diagnostics"]
    cost = reference["cost_info"]
    pmp = reference["pmp_diagnostics"]
    return [
        {"quantity": "reference_objective_J", "value": _format_value(cost["J"])},
        {"quantity": "running_cost_integral", "value": _format_value(cost["running_cost_integral"])},
        {"quantity": "terminal_cost", "value": _format_value(cost["terminal_cost"])},
        {"quantity": "terminal_condition_error", "value": _format_value(ric["terminal_condition_error"])},
        {"quantity": "max_symmetry_error", "value": _format_value(ric["max_symmetry_error"])},
        {"quantity": "max_stationarity_error", "value": _format_value(pmp["max_stationarity_error"])},
        {"quantity": "reference_x0", "value": _matrix_str(data["x0"])},
        {"quantity": "reference_T", "value": _format_value(data["T"])},
    ]


def _input_parameter_rows(result):
    pdata = result["problem_data"]
    settings = result["settings"]
    return [
        {"parameter": "A", "value": _matrix_str(pdata["A"])},
        {"parameter": "B", "value": _matrix_str(pdata["B"])},
        {"parameter": "Q", "value": _matrix_str(pdata["Q"])},
        {"parameter": "R", "value": _matrix_str(pdata["R"])},
        {"parameter": "Qf", "value": _matrix_str(pdata["Qf"])},
        {"parameter": "x0", "value": _matrix_str(pdata["x0"])},
        {"parameter": "T", "value": _format_value(pdata["T"])},
        {"parameter": "u_min", "value": _format_value(pdata["u_min"])},
        {"parameter": "u_max", "value": _format_value(pdata["u_max"])},
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


def _final_summary(result, prob, reference):
    last = result["log"][-1]
    t_nodes = np.asarray(result["t_nodes"])
    X = np.asarray(result["X"])
    P = np.asarray(result["P"])
    U = _compute_controls(prob, result["bundle"], X, P, t_nodes)
    reference_J = float(reference["cost_info"]["J"])
    objective_mesh = _objective_mesh_approx(prob, t_nodes, X, U)
    return {
        "outer_iterations_logged": len(result["log"]),
        "last_outer_iteration": int(last["iteration"]),
        "mesh_points": int(len(t_nodes)),
        "mesh_intervals": int(len(t_nodes) - 1),
        "planes": int(result["bundle"].num_planes()),
        "objective_mesh_approx": float(objective_mesh),
        "reference_objective_J": reference_J,
        "objective_gap_vs_reference": float(objective_mesh - reference_J),
        "eta_time": float(last["eta_time"]),
        "eta_time_sum": float(last.get("eta_time_sum", float("nan"))),
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
    }


def _plot_iteration_state_costate(entry, out_path):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)
    P = np.asarray(entry["P_iter"], dtype=float)
    support_points = entry.get("bundle_support_points_so_far", [])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, X[:, 0], label="x1(t)")
    axes[0].plot(t, X[:, 1], label="x2(t)")
    if support_points:
        support_t = np.array([float(point["time"]) for point in support_points], dtype=float)
        support_x1 = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in support_points], dtype=float)
        axes[0].scatter(
            support_t,
            support_x1,
            s=20,
            color="tab:red",
            marker="o",
            label="bundle support points",
            zorder=5,
        )
    axes[0].set_ylabel("state")
    axes[0].set_title("State trajectories with bundle support points")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, P[:, 0], label="p1(t)")
    axes[1].plot(t, P[:, 1], label="p2(t)")
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

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, X[:, 0], color="tab:blue", linewidth=1.8, label="x1(t)")
    axes[0].plot(t, X[:, 1], color="tab:orange", linewidth=1.5, label="x2(t)")
    if previous_points:
        prev_t = np.array([float(point["time"]) for point in previous_points], dtype=float)
        prev_x1 = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in previous_points], dtype=float)
        axes[0].scatter(
            prev_t,
            prev_x1,
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
        curr_x1 = np.array([float(np.asarray(point["state"], dtype=float)[0]) for point in current_points], dtype=float)
        axes[0].scatter(
            curr_t,
            curr_x1,
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
    U = np.asarray(entry["U_iter"], dtype=float)
    fig = plt.figure(figsize=(9, 4))
    plt.plot(t, U[:, 0], label="u(t)")
    plt.xlabel("t")
    plt.ylabel("control")
    plt.title("Control trajectory")
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
    plt.step(t_int, rho, where="post", label=r"$\rho_n$")
    plt.step(t_int, rho_bar, where="post", label=r"$\bar{\rho}_n$")
    plt.xlabel("t")
    plt.ylabel(r"$\rho$")
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
    eta_time_sum = [float(entry.get("eta_time_sum", float("nan"))) for entry in log]
    tol_time = [float(entry["tol_time_star"]) for entry in log]
    eta_pa = [float(entry["eta_PA"]) for entry in log]
    eta_delta = [float(entry["eta_delta"]) for entry in log]

    fig = plt.figure(figsize=(9, 5))
    plt.semilogy(iterations, eta_time, marker="o", label=r"$\eta_{\mathrm{time}}=\max_n \bar r_n$")
    plt.semilogy(iterations, eta_time_sum, marker="d", label=r"$\sum_n \bar r_n$")
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
            raise ValueError("Deep report export requires store_iterates=True in the solver run.")

        iter_idx = int(entry["iteration"])
        iter_dir = iteration_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (_jsonable(entry))
        (iter_dir / "iteration_data.json").write_text(json.dumps(_jsonable(entry), indent=2))
        _plot_iteration_state_costate(entry, iter_dir / f"iter_{iter_idx:02d}_state_costate.pdf")
        _plot_iteration_bundle_support_points(entry, iter_dir / f"iter_{iter_idx:02d}_bundle_support_points.pdf")
        _plot_iteration_control(entry, iter_dir / f"iter_{iter_idx:02d}_control.pdf")
        _plot_iteration_rho(entry, iter_dir / f"iter_{iter_idx:02d}_rho_density.pdf")
        _plot_iteration_mesh_and_indicator(entry, iter_dir / f"iter_{iter_idx:02d}_mesh_and_indicator.pdf")
        _plot_iteration_pa_delta_contributions(entry, iter_dir / f"iter_{iter_idx:02d}_pa_delta_contributions.pdf")


def _generate_example1_report_tex(out_dir, summary, input_rows, reference_rows, outer_rows, log):
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
        status_text = (
            "all indicators below tolerance"
            if entry.get("all_indicators_within_tolerance", False)
            else "refinement still required"
        )
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
                {_latex_table_from_pairs(f"PA enrichment summary at iteration {iter_idx}.", f"tab:ex1-pa-summary-{iter_idx}", summary_rows)}

                {_latex_table_from_rows(
                    f"Selected PA enrichment candidates at iteration {iter_idx}.",
                    f"tab:ex1-pa-candidates-{iter_idx}",
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
                \\caption{{State and costate trajectories at iteration {iter_idx}. Red support markers are also overlaid on the state plot, but the dedicated support-point figure below should be used as the primary record of where affine planes were sampled.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_bundle_support_points.pdf").as_posix()}}}
                \\caption{{Dedicated bundle-support diagnostic at iteration {iter_idx}. The upper panel shows the state trajectory with all cumulative affine-plane support points; square black markers identify support points added at the current iteration. The lower panel shows their time locations explicitly so the plane construction can be audited.}}
                \\end{{figure}}

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_control.pdf").as_posix()}}}
                \\caption{{Control trajectory at iteration {iter_idx}.}}
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
        \\title{{Standalone Deep Report for Example 1}}
        \\author{{Archive Python adaptive Pontryagin solver}}
        \\date{{\\today}}

        \\begin{{document}}
        \\maketitle

        \\section{{Example 1: Linear Quadratic Regulator}}
        This standalone report documents the current executable Example 1 pipeline in a reproducible way. The purpose is diagnostic as well as presentational: every adaptive outer iterate is exposed with the same plot family and the same summary fields so the algorithm can be inspected carefully and improved systematically.

        The current final status is: {_latex_escape(intro_status)}

        Each iteration subsection includes a dedicated affine-plane support diagnostic and a separate intervalwise $\\eta_{{\\mathrm{{PA}}}}$ / $\\eta_\\delta$ contribution figure. Those two figures should be read together whenever we assess whether the bundle enrichment and smoothing adaptivity are behaving sensibly.

        \\subsection{{Input parameters}}
        {_latex_table_from_pairs("Input parameters used for the Example 1 run.", "tab:ex1-input-parameters", input_pairs)}

        \\subsection{{Reference values}}
        {_latex_table_from_pairs("Riccati-based reference values for Example 1.", "tab:ex1-reference-values", reference_pairs)}

        \\subsection{{Outer-loop summary}}
        {_latex_longtable_from_rows("Adaptive outer-loop iterations of the proposed solver on Example 1.", "tab:ex1-outer-loop", outer_display_fields, outer_display_rows)}

        \\subsection{{Final reported numbers}}
        {_latex_table_from_pairs("Final numbers reported by the proposed solver on Example 1.", "tab:ex1-final-summary", final_pairs)}

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


def export_example1_deep_report_artifacts(result, prob, out_dir, figure_ext="pdf"):
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    iteration_dir = out_dir / "iterations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    iteration_dir.mkdir(parents=True, exist_ok=True)

    reference = _reference_benchmark()
    input_rows = _input_parameter_rows(result)
    reference_rows = _reference_table_rows(reference)
    outer_rows = _outer_history_rows(result["log"])
    summary = _final_summary(result, prob, reference)

    summary_path = tables_dir / "final_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2))
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
    _plot_error_indicator_history(result["log"], figures_dir / f"error_indicator_history.{figure_ext}")
    report_path = _generate_example1_report_tex(
        out_dir,
        summary,
        input_rows,
        reference_rows,
        outer_rows,
        result["log"],
    )

    return {
        "summary": summary,
        "report_tex": str(report_path),
        "out_dir": str(out_dir),
    }


def run_example():
    result, prob = run_ex1_lqr_solver()
    summarize_ex1_results(result, prob, print_last_log_only=True)
    plot_ex1_results(
        result,
        prob,
        out_prefix="example1_solver",
        save_plots=False,
        plot_ext="pdf",
    )
    return result


if __name__ == "__main__":
    run_example()
