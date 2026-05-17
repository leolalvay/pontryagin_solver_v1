"""
Legacy 2014 Example 3.3: singular tracking problem with explicit time dependence.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hyp2f1

from core.adaptivity import solve_optimal_control
from core.problem import OCPProblem


def singular_reference_x(t, epsilon=1.0e-10, beta=0.75, t0=5.0 / 3.0):
    t_arr = np.asarray(t, dtype=float)
    z = -((t_arr - t0) ** 2) / (epsilon ** 2)
    prefactor = (t_arr - t0) / (epsilon ** beta)
    values = np.exp(prefactor * hyp2f1(0.5, beta / 2.0, 1.5, z))
    return values


def build_singular_problem(epsilon=1.0e-10, beta=0.75, t0=5.0 / 3.0, T=4.0):
    x0_scalar = float(singular_reference_x(np.array([0.0]), epsilon=epsilon, beta=beta, t0=t0)[0])
    x_ref_T = float(singular_reference_x(np.array([T]), epsilon=epsilon, beta=beta, t0=t0)[0])
    x0 = np.array([x0_scalar, 0.0], dtype=float)

    def denom(s):
        return ((s - t0) ** 2 + epsilon ** 2) ** (beta / 2.0)

    def dynamics(x, u, t):
        d = denom(float(x[1]))
        return np.array([float(u[0]) / d, 1.0], dtype=float)

    def stage_cost(x, u, t):
        return float((u[0] - x[0]) ** 2)

    def terminal_cost(x):
        return float((x[0] - x_ref_T) ** 2)

    def u_star_fn(x, p, t):
        d = denom(float(x[1]))
        return np.array([float(x[0] - p[0] * d / 2.0)], dtype=float)

    def hamiltonian_true(x, p, t):
        d = denom(float(x[1]))
        return float((p[0] * x[0]) / d - (p[0] ** 2) / 4.0 + p[1])

    def hamiltonian_grad_fn(x, p, t):
        s = float(x[1])
        d = denom(s)
        grad_p = np.array([float(x[0] / d - p[0] / 2.0), 1.0], dtype=float)

        d_term = ((s - t0) ** 2 + epsilon ** 2)
        d_inv = d_term ** (-beta / 2.0)
        ds_term = -beta * (s - t0) * (d_term ** (-beta / 2.0 - 1.0))
        grad_x0 = float(p[0] * d_inv)
        grad_s = float(p[0] * x[0] * ds_term)
        grad_x = np.array([grad_x0, grad_s], dtype=float)
        return grad_p, grad_x

    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=None,
        state_bounds=None,
        hamiltonian_true=hamiltonian_true,
        u_star_fn=u_star_fn,
        hamiltonian_grad_fn=hamiltonian_grad_fn,
    )
    problem_data = {
        "legacy_example": "3.3",
        "problem_name": "singular tracking problem",
        "epsilon": float(epsilon),
        "beta": float(beta),
        "t0": float(t0),
        "T": float(T),
        "x0": x0.copy(),
        "x_ref_T": float(x_ref_T),
    }
    return prob, problem_data


def exact_solution(t_nodes, epsilon=1.0e-10, beta=0.75, t0=5.0 / 3.0):
    t_nodes = np.asarray(t_nodes, dtype=float)
    x_ref = singular_reference_x(t_nodes, epsilon=epsilon, beta=beta, t0=t0)
    X_exact = np.column_stack([x_ref, t_nodes])
    P_exact = np.zeros((len(t_nodes), 2), dtype=float)
    U_exact = x_ref[:-1].reshape(-1, 1)
    return X_exact, P_exact, U_exact


def run_singular_solver(
    n_init=30,
    tol_time=1.0e-4,
    tol_PA=1.0e-4,
    tol_delta=1.0e-4,
    max_iters=18,
    delta0=1.0e-2,
    store_iterates=False,
    verbose=True,
):
    prob, problem_data = build_singular_problem()
    t_nodes = np.linspace(0.0, problem_data["T"], n_init)
    X_exact, P_exact, _ = exact_solution(
        t_nodes,
        epsilon=problem_data["epsilon"],
        beta=problem_data["beta"],
        t0=problem_data["t0"],
    )

    t0_wall = time.perf_counter()
    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        max_iters=max_iters,
        delta0=delta0,
        use_oracle_PA=True,
        use_explicit_hamiltonian_gradients=True,
        store_iterates=store_iterates,
        verbose=verbose,
        initial_X_guess=X_exact,
        initial_P_guess=P_exact,
        initial_guess_label="exact_state_costate",
    )
    result["problem_data"] = problem_data
    result["wall_time_sec"] = time.perf_counter() - t0_wall
    result["legacy_reference_label"] = "2014 reference manuscript Example 3.3"
    return result, prob


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
    return "".join(repl.get(ch, ch) for ch in str(text))


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


def _input_parameter_rows(result):
    pdata = result["problem_data"]
    settings = result["settings"]
    return [
        {"parameter": "legacy_example", "value": pdata["legacy_example"]},
        {"parameter": "problem_name", "value": pdata["problem_name"]},
        {"parameter": "epsilon", "value": _format_value(pdata["epsilon"])},
        {"parameter": "beta", "value": _format_value(pdata["beta"])},
        {"parameter": "t0", "value": _format_value(pdata["t0"])},
        {"parameter": "T", "value": _format_value(pdata["T"])},
        {"parameter": "x0_x", "value": _format_value(pdata["x0"][0])},
        {"parameter": "x0_s", "value": _format_value(pdata["x0"][1])},
        {"parameter": "x_ref_T", "value": _format_value(pdata["x_ref_T"])},
        {"parameter": "n_init", "value": _format_value(len(result["log"][0]["t_nodes_iter"]) - 1)},
        {"parameter": "tol_time", "value": _format_value(settings["tol_time"])},
        {"parameter": "tol_PA", "value": _format_value(settings["tol_PA"])},
        {"parameter": "tol_delta", "value": _format_value(settings["tol_delta"])},
        {"parameter": "max_iters", "value": _format_value(settings["max_iters"])},
        {"parameter": "delta0", "value": _format_value(settings["delta0"])},
        {"parameter": "newton_tol", "value": _format_value(settings["newton_tol"])},
        {"parameter": "newton_max_iter", "value": _format_value(settings["newton_max_iter"])},
        {"parameter": "use_oracle_PA", "value": _format_value(settings["use_oracle_PA"])},
        {"parameter": "use_explicit_hamiltonian_gradients", "value": _format_value(settings["use_explicit_hamiltonian_gradients"])},
        {"parameter": "initial_guess_label", "value": settings["initial_guess_label"]},
    ]


def _reference_rows(result):
    pdata = result["problem_data"]
    return [
        {"quantity": "reference_source", "value": "2014 manuscript Example 3.3 explicit reference trajectory"},
        {"quantity": "true_objective", "value": _format_value(0.0)},
        {"quantity": "x_ref_T", "value": _format_value(pdata["x_ref_T"])},
        {"quantity": "p_ref_terminal_norm", "value": _format_value(0.0)},
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
                "newton_iter": str(entry.get("newton_iter")),
                "newton_residual": _format_value(entry.get("newton_residual")),
                "solver_phase": entry.get("solver_phase", ""),
                "all_indicators_within_tolerance": _format_value(entry.get("all_indicators_within_tolerance", False)),
            }
        )
    return rows


def _objective_mesh_approx(prob, t_nodes, X, interval_controls):
    objective = float(prob.g(X[-1]))
    for i in range(len(t_nodes) - 1):
        dt = float(t_nodes[i + 1] - t_nodes[i])
        objective += float(prob.l(X[i], interval_controls[i], t_nodes[i])) * dt
    return objective


def _compute_interval_controls_from_state(X):
    return np.asarray(X[:-1, 0], dtype=float).reshape(-1, 1)


def _final_summary(result, prob):
    t_nodes = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)
    P = np.asarray(result["P"], dtype=float)
    U = _compute_interval_controls_from_state(X)
    X_exact, P_exact, U_exact = exact_solution(
        t_nodes,
        epsilon=result["problem_data"]["epsilon"],
        beta=result["problem_data"]["beta"],
        t0=result["problem_data"]["t0"],
    )
    last = result["log"][-1]
    objective_mesh = _objective_mesh_approx(prob, t_nodes, X, U)
    return {
        "objective_mesh_approx": float(objective_mesh),
        "true_objective": 0.0,
        "terminal_state_error": float(X[-1, 0] - X_exact[-1, 0]),
        "max_state_error": float(np.max(np.abs(X[:, 0] - X_exact[:, 0]))),
        "max_aux_state_error": float(np.max(np.abs(X[:, 1] - X_exact[:, 1]))),
        "max_costate1_error": float(np.max(np.abs(P[:, 0] - P_exact[:, 0]))),
        "max_costate2_error": float(np.max(np.abs(P[:, 1] - P_exact[:, 1]))),
        "max_control_error": float(np.max(np.abs(U[:, 0] - U_exact[:, 0]))),
        "outer_iterations_logged": len(result["log"]),
        "mesh_points": int(len(t_nodes)),
        "mesh_intervals": int(len(t_nodes) - 1),
        "eta_time": float(last["eta_time"]),
        "eta_time_sum": float(last.get("eta_time_sum", 0.0)),
        "tol_time_star": float(last["tol_time_star"]),
        "newton_iter": int(last["newton_iter"]),
        "newton_residual": float(last["newton_residual"]),
        "solver_phase": last.get("solver_phase", "newton"),
        "final_action": last.get("action", ""),
        "final_note": last.get("note", ""),
        "all_indicators_within_tolerance": bool(last.get("all_indicators_within_tolerance", False)),
        "wall_time_sec": float(result.get("wall_time_sec", float("nan"))),
    }


def _plot_iteration_state_costate(entry, out_path, pdata):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)
    P = np.asarray(entry["P_iter"], dtype=float)
    X_exact, P_exact, _ = exact_solution(t, epsilon=pdata["epsilon"], beta=pdata["beta"], t0=pdata["t0"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, X[:, 0], label="state $X_h$")
    axes[0].plot(t, X_exact[:, 0], "--", label=r"$X_{ref}$")
    axes[0].plot(t, X[:, 1], label="auxiliary state $s_h$")
    axes[0].plot(t, X_exact[:, 1], ":", label=r"$s(t)=t$")
    axes[0].set_ylabel("state")
    axes[0].set_title("State trajectories")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, P[:, 0], label=r"$\lambda_1$")
    axes[1].plot(t, P[:, 1], label=r"$\lambda_2$")
    axes[1].plot(t, P_exact[:, 0], "--", label=r"exact $\lambda_1$")
    axes[1].plot(t, P_exact[:, 1], ":", label=r"exact $\lambda_2$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("costate")
    axes[1].set_title("Costate trajectories")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_control(entry, out_path, pdata):
    t = np.asarray(entry["t_nodes_iter"], dtype=float)
    X = np.asarray(entry["X_iter"], dtype=float)
    _, _, U_exact = exact_solution(t, epsilon=pdata["epsilon"], beta=pdata["beta"], t0=pdata["t0"])
    U = np.asarray(X[:-1, 0], dtype=float)
    fig = plt.figure(figsize=(9, 4))
    plt.step(t[:-1], U, where="post", label="control $\\alpha_h$")
    plt.step(t[:-1], U_exact[:, 0], where="post", linestyle="--", label=r"exact $\alpha^*$")
    plt.xlabel("t")
    plt.ylabel(r"$\alpha$")
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
    fig = plt.figure(figsize=(9, 4))
    plt.step(t[:-1], np.abs(rho), where="post", label=r"$|\rho_n|$")
    plt.step(t[:-1], np.abs(rho_bar), where="post", label=r"$|\bar \rho_n|$")
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

    axes[1].step(t[:-1], r_bar, where="post", label=r"$\bar r_n$")
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


def _plot_error_indicator_history(log, out_path):
    iterations = [int(entry["iteration"]) for entry in log]
    eta_time = [float(entry["eta_time"]) for entry in log]
    eta_time_sum = [float(entry.get("eta_time_sum", 0.0)) for entry in log]
    tol_time = [float(entry["tol_time_star"]) for entry in log]
    fig = plt.figure(figsize=(9, 5))
    plt.semilogy(iterations, eta_time, marker="o", label=r"$\eta_{\mathrm{time}}=\max_n \bar r_n$")
    plt.semilogy(iterations, eta_time_sum, marker="D", label=r"$\sum_n \bar r_n$")
    plt.semilogy(iterations, tol_time, linestyle="--", label=r"$\mathrm{tol}_{\mathrm{time}}^\star$")
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
        if "X_iter" not in entry:
            raise ValueError("Deep report export requires store_iterates=True.")
        iter_idx = int(entry["iteration"])
        iter_dir = iteration_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "iteration_data.json").write_text(json.dumps(_jsonable(entry), indent=2))
        _plot_iteration_state_costate(entry, iter_dir / f"iter_{iter_idx:02d}_state_costate.pdf", pdata)
        _plot_iteration_control(entry, iter_dir / f"iter_{iter_idx:02d}_control.pdf", pdata)
        _plot_iteration_rho(entry, iter_dir / f"iter_{iter_idx:02d}_rho_density.pdf")
        _plot_iteration_mesh_and_indicator(entry, iter_dir / f"iter_{iter_idx:02d}_mesh_and_indicator.pdf")


def _generate_report_tex(out_dir, summary, input_rows, reference_rows, outer_rows, log):
    final_pairs = [(key, _format_value(value)) for key, value in summary.items()]
    input_pairs = [(row["parameter"], row["value"]) for row in input_rows]
    reference_pairs = [(row["quantity"], row["value"]) for row in reference_rows]
    outer_fields = ["iter", "action", "N", "delta", "J_h", "eta_time / tol*", "Newton it", "all below tol"]
    outer_display_rows = []
    for row in outer_rows:
        outer_display_rows.append(
            {
                "iter": row["iteration"],
                "action": row["action"],
                "N": row["N"],
                "delta": _short_display(row["delta"]),
                "J_h": _short_display(row["objective_mesh_approx"]),
                "eta_time / tol*": f"{_short_display(row['eta_time'])} / {_short_display(row['tol_time_star'])}",
                "Newton it": row["newton_iter"],
                "all below tol": row["all_indicators_within_tolerance"],
            }
        )

    iteration_sections = []
    for entry in log:
        iter_idx = int(entry["iteration"])
        iter_rel = Path("iterations") / f"iter_{iter_idx:02d}"
        iteration_sections.append(
            dedent(
                f"""
                \\subsection{{Iteration {iter_idx}}}
                Iteration {iter_idx} ends with action \\texttt{{{_latex_escape(entry.get("action", ""))}}}, mesh intervals $N={int(entry["N"])}$, and time-indicator status {_latex_escape("below tolerance" if entry.get("all_indicators_within_tolerance", False) else "refinement still required")}.

                \\begin{{figure}}[H]
                \\centering
                \\includegraphics[width=0.9\\textwidth]{{{(iter_rel / f"iter_{iter_idx:02d}_state_costate.pdf").as_posix()}}}
                \\caption{{State and costate trajectories at iteration {iter_idx}, with the exact reference state and exact zero dual overlaid.}}
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
                \\caption{{Time mesh and time indicators at iteration {iter_idx}.}}
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
        \\title{{Standalone Deep Report for Legacy 2014 Example 3.3}}
        \\author{{Archive Python adaptive Pontryagin solver}}
        \\date{{\\today}}
        \\begin{{document}}
        \\maketitle

        \\section{{Legacy 2014 Example 3.3: Singular Tracking Problem}}
        This report documents the archive Python reproduction of the singular tracking problem from the 2014 legacy manuscript. The problem is smooth after regularization but explicitly time dependent through the auxiliary state $s(t)=t$ and the parameter set $(\\varepsilon,\\beta,t_0)$.

        \\subsection{{Input parameters}}
        {_latex_table_from_pairs("Input parameters for the singular legacy run.", "tab:ex33-input", input_pairs)}

        \\subsection{{Reference values}}
        {_latex_table_from_pairs("Reference values for the singular tracking problem.", "tab:ex33-reference", reference_pairs)}

        \\subsection{{Outer-loop summary}}
        {_latex_longtable_from_rows("Adaptive outer-loop iterations of the proposed solver on legacy Example 3.3.", "tab:ex33-outer", outer_fields, outer_display_rows)}

        \\subsection{{Final reported numbers}}
        {_latex_table_from_pairs("Final numbers reported by the proposed solver on the singular tracking problem.", "tab:ex33-final", final_pairs)}

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


def export_ex33_deep_report_artifacts(result, prob, out_dir):
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
            "newton_iter",
            "newton_residual",
            "solver_phase",
            "all_indicators_within_tolerance",
        ],
    )
    (tables_dir / "outer_loop_history.json").write_text(json.dumps(_jsonable(result["log"]), indent=2))

    _write_iteration_artifacts(result["log"], iteration_dir, result["problem_data"])
    _plot_error_indicator_history(result["log"], figures_dir / "error_indicator_history.pdf")
    report_path = _generate_report_tex(out_dir, summary, input_rows, reference_rows, outer_rows, result["log"])
    return {
        "summary": summary,
        "report_tex": str(report_path),
        "out_dir": str(out_dir),
    }


def summarize_singular_results(result):
    print("\nLegacy 2014 Example 3.3 (Singular tracking problem)")
    print("len(log) =", len(result["log"]))
    print("last outer iter =", result["log"][-1]["iteration"])
    print("len(t_nodes) =", len(result["t_nodes"]))
    print("X.shape =", result["X"].shape)
    print("P.shape =", result["P"].shape)
    print("final delta =", result["delta"])
    print("last log entry =", result["log"][-1])


def run_example():
    result, _ = run_singular_solver(store_iterates=False, verbose=True)
    summarize_singular_results(result)
    return result


if __name__ == "__main__":
    run_example()
