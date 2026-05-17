from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.smoothing import eval_H_smooth
from experiments.ex5_hypersensitive import run_example as run_ex31
from experiments.ex6_nonsmoothham import run_example as run_ex32
from experiments.ex7_singular import exact_solution as exact_solution_ex33
from experiments.ex7_singular import run_example as run_ex33


ROOT = Path("/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen")
RUN_ROOT = ROOT / "archive_runs" / "legacy_2014"
FIG_ROOT = ROOT / "figures" / "legacy_2014"


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


def _write_log_csv(log, path: Path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "iteration",
                "N",
                "M",
                "delta",
                "eta_time",
                "tol_time_star",
                "eta_PA",
                "eta_delta",
                "newton_iter",
                "newton_residual",
                "action",
                "note",
            ]
        )
        for row in log:
            writer.writerow(
                [
                    row.get("iteration"),
                    row.get("N"),
                    row.get("M"),
                    row.get("delta"),
                    row.get("eta_time"),
                    row.get("tol_time_star"),
                    row.get("eta_PA"),
                    row.get("eta_delta"),
                    row.get("newton_iter"),
                    row.get("newton_residual"),
                    row.get("action", ""),
                    row.get("note", ""),
                ]
            )


def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def export_example31():
    plt.show = lambda *args, **kwargs: None
    result = run_ex31()

    out_dir = RUN_ROOT / "ex31_hypersensitive"
    fig_dir = FIG_ROOT / "ex31_hypersensitive"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)[:, 0]
    P = np.asarray(result["P"], dtype=float)[:, 0]
    dt = np.diff(t)
    alpha = -0.5 * P
    last = result["log"][-1]
    rho = np.abs(np.asarray(last["rho_bar"], dtype=float))
    rbar = np.asarray(last["r_bar"], dtype=float)

    summary = {
        "legacy_example": "3.1",
        "problem": "hyper-sensitive optimal control",
        "terminal_state": float(X[-1]),
        "N_final": int(last["N"]),
        "M_final": int(last["M"]),
        "delta_final": float(result["delta"]),
        "eta_time": float(last["eta_time"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "newton_iter_final": int(last["newton_iter"]),
        "newton_residual_final": float(last["newton_residual"]),
    }

    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (out_dir / "log.json").write_text(json.dumps(_jsonable(result["log"]), indent=2))
    _write_log_csv(result["log"], out_dir / "outer_trace.csv")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.plot(t, X, label="state $X$")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Legacy Example 3.1: state")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "state.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.plot(t, alpha, label=r"control $\alpha=-p/2$")
    plt.xlabel("t")
    plt.ylabel(r"$\alpha$")
    plt.title("Legacy Example 3.1: control")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "control.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.plot(t, P, label=r"costate $\lambda$")
    plt.xlabel("t")
    plt.ylabel(r"$\lambda$")
    plt.title("Legacy Example 3.1: costate")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "costate.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], dt, where="post", label=r"$\Delta t_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta t$")
    plt.title("Legacy Example 3.1: adaptive mesh")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "mesh.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], rho, where="post", label=r"$|\bar{\rho}_n|$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|\bar{\rho}_n|$")
    plt.title("Legacy Example 3.1: density")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "rho.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], rbar, where="post", label=r"$\bar r_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar r_n$")
    plt.title("Legacy Example 3.1: time indicator")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "rbar.pdf")


def export_example32():
    plt.show = lambda *args, **kwargs: None
    result = run_ex32()

    out_dir = RUN_ROOT / "ex32_nonsmooth"
    fig_dir = FIG_ROOT / "ex32_nonsmooth"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    prob = result["problem"]
    t = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)[:, 0]
    P = np.asarray(result["P"], dtype=float)[:, 0]
    bundle = result["bundle"]
    dt = np.diff(t)
    last = result["log"][-1]
    rho = np.abs(np.asarray(last["rho_bar"], dtype=float))
    rbar = np.asarray(last["r_bar"], dtype=float)

    x_exact = np.maximum(0.5 - t, 0.0)
    p_exact = np.where(t <= 0.5, (0.5 - t) ** 10, 0.0)
    a_exact = np.where(t[:-1] < 0.5, -1.0, 0.0)

    a_bar = np.zeros_like(t[:-1])
    u_delta = np.zeros_like(t[:-1])
    for i in range(len(t) - 1):
        _, idx = bundle.evaluate(prob, np.array([P[i + 1]]), np.array([X[i]]), float(t[i]))
        a_bar[i] = float(bundle.controls[int(idx)][0])
        _, grad_p, _ = eval_H_smooth(
            prob,
            bundle,
            np.array([P[i + 1]]),
            np.array([X[i]]),
            float(t[i]),
            float(result["delta"]),
        )
        u_delta[i] = float(grad_p[0])

    J_mesh = float(np.sum(dt * (X[:-1] ** 10)))
    J_star = float((0.5 ** 11) / 11.0)
    summary = {
        "legacy_example": "3.2",
        "problem": "simple non-smooth optimal control",
        "J_mesh": J_mesh,
        "J_star": J_star,
        "relative_objective_error": abs(J_mesh - J_star) / J_star,
        "terminal_state": float(X[-1]),
        "N_final": int(last["N"]),
        "M_final": int(last["M"]),
        "delta_final": float(result["delta"]),
        "eta_time": float(last["eta_time"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "newton_iter_final": int(last["newton_iter"]),
        "newton_residual_final": float(last["newton_residual"]),
    }

    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (out_dir / "log.json").write_text(json.dumps(_jsonable(result["log"]), indent=2))
    _write_log_csv(result["log"], out_dir / "outer_trace.csv")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.plot(t, X, label="state $X_h$")
    plt.plot(t, x_exact, "--", label="exact $X^*$")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Legacy Example 3.2: state")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "state.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.plot(t, P, label="costate $P_h$")
    plt.plot(t, p_exact, "--", label="exact $P^*$")
    plt.xlabel("t")
    plt.ylabel("P")
    plt.title("Legacy Example 3.2: costate")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "costate.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], a_bar, where="post", label=r"active-plane control $\bar a$")
    plt.step(t[:-1], u_delta, where="post", label=r"$u_\delta=\partial_p H_\delta$")
    plt.step(t[:-1], a_exact, where="post", linestyle="--", label=r"exact $a^*$")
    plt.xlabel("t")
    plt.ylabel("a")
    plt.title("Legacy Example 3.2: controls")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "control.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], dt, where="post", label=r"$\Delta t_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta t$")
    plt.title("Legacy Example 3.2: adaptive mesh")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "mesh.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], rho, where="post", label=r"$|\bar{\rho}_n|$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|\bar{\rho}_n|$")
    plt.title("Legacy Example 3.2: density")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "rho.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], rbar, where="post", label=r"$\bar r_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar r_n$")
    plt.title("Legacy Example 3.2: time indicator")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "rbar.pdf")


def export_example33():
    plt.show = lambda *args, **kwargs: None
    result = run_ex33()

    out_dir = RUN_ROOT / "ex33_singular"
    fig_dir = FIG_ROOT / "ex33_singular"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(result["t_nodes"], dtype=float)
    X = np.asarray(result["X"], dtype=float)
    P = np.asarray(result["P"], dtype=float)
    dt = np.diff(t)
    last = result["log"][-1]
    rho = np.abs(np.asarray(last["rho_bar"], dtype=float))
    rbar = np.asarray(last["r_bar"], dtype=float)

    X_exact, P_exact, U_exact = exact_solution_ex33(
        t,
        epsilon=result["problem_data"]["epsilon"],
        beta=result["problem_data"]["beta"],
        t0=result["problem_data"]["t0"],
    )
    U = X[:-1, 0]

    summary = {
        "legacy_example": "3.3",
        "problem": "singular tracking problem",
        "objective_mesh_approx": float(last["objective_mesh_approx"]),
        "true_objective": 0.0,
        "terminal_state_error": float(X[-1, 0] - X_exact[-1, 0]),
        "N_final": int(last["N"]),
        "M_final": int(last["M"]),
        "delta_final": float(result["delta"]),
        "eta_time": float(last["eta_time"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "newton_iter_final": int(last["newton_iter"]),
        "newton_residual_final": float(last["newton_residual"]),
    }

    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (out_dir / "log.json").write_text(json.dumps(_jsonable(result["log"]), indent=2))
    _write_log_csv(result["log"], out_dir / "outer_trace.csv")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.plot(t, X[:, 0], label="state $X_h$")
    plt.plot(t, X_exact[:, 0], "--", label="reference $X_{ref}$")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Legacy Example 3.3: state")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "state.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.plot(t, P[:, 0], label=r"costate $\lambda_1$")
    plt.plot(t, P_exact[:, 0], "--", label=r"exact $\lambda_1$")
    plt.xlabel("t")
    plt.ylabel(r"$\lambda_1$")
    plt.title("Legacy Example 3.3: primary costate")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "costate.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], U, where="post", label=r"control $\alpha_h$")
    plt.step(t[:-1], U_exact[:, 0], where="post", linestyle="--", label=r"reference $\alpha^*$")
    plt.xlabel("t")
    plt.ylabel(r"$\alpha$")
    plt.title("Legacy Example 3.3: control")
    plt.grid(True)
    plt.legend()
    _savefig(fig, fig_dir / "control.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], dt, where="post", label=r"$\Delta t_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta t$")
    plt.title("Legacy Example 3.3: adaptive mesh")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "mesh.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], rho, where="post", label=r"$|\bar{\rho}_n|$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|\bar{\rho}_n|$")
    plt.title("Legacy Example 3.3: density")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "rho.pdf")

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.step(t[:-1], rbar, where="post", label=r"$\bar r_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar r_n$")
    plt.title("Legacy Example 3.3: time indicator")
    plt.grid(True, which="both")
    plt.legend()
    _savefig(fig, fig_dir / "rbar.pdf")


if __name__ == "__main__":
    export_example31()
    export_example32()
    export_example33()
