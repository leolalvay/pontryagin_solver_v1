from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from experiments.ex8_goddard_fixedtime import (
    export_goddard_barrier_report_artifacts,
    run_goddard_barrier_continuation,
    run_goddard_solver,
    run_goddard_true_from_barrier_handoff,
)


def main():
    parser = argparse.ArgumentParser(description="Generate the barrier-continuation note for the fixed-time Goddard benchmark.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/reports/goddard_barrier_continuation_report"),
    )
    parser.add_argument("--T", type=float, default=0.15)
    parser.add_argument("--q-max", type=float, default=17.0)
    parser.add_argument("--rho-m", type=float, default=1.0e4)
    parser.add_argument("--tol-time", type=float, default=1.0e-4)
    parser.add_argument("--tol-pa", type=float, default=1.0e-4)
    parser.add_argument("--tol-delta", type=float, default=1.0e-4)
    parser.add_argument("--delta0", type=float, default=5.0e-2)
    parser.add_argument("--stage-max-iters", type=int, default=3)
    parser.add_argument("--max-iters-handoff", type=int, default=3)
    parser.add_argument("--mu-schedule", type=str, default="1e-1,3e-2,1e-2,3e-3,1e-3")
    args = parser.parse_args()

    mu_schedule = [float(item) for item in args.mu_schedule.split(",") if item.strip()]

    barrier_run = run_goddard_barrier_continuation(
        mu_schedule=mu_schedule,
        T=args.T,
        q_max=args.q_max,
        rho_m=args.rho_m,
        tol_time=args.tol_time,
        tol_PA=args.tol_pa,
        tol_delta=args.tol_delta,
        delta0=args.delta0,
        stage_max_iters=args.stage_max_iters,
        verbose=False,
        pressure_feasible_init=True,
        pressure_feasible_safety_factor=0.90,
        store_iterates=True,
    )

    handoff_bundle = run_goddard_true_from_barrier_handoff(
        barrier_run,
        T=args.T,
        q_max=args.q_max,
        rho_m=args.rho_m,
        tol_time=args.tol_time,
        tol_PA=args.tol_pa,
        tol_delta=args.tol_delta,
        delta0=None,
        max_iters=args.max_iters_handoff,
        verbose=False,
        store_iterates=True,
    )

    baseline_result, baseline_prob = run_goddard_solver(
        T=args.T,
        q_max=args.q_max,
        rho_m=args.rho_m,
        tol_time=args.tol_time,
        tol_PA=args.tol_pa,
        tol_delta=args.tol_delta,
        delta0=args.delta0,
        max_iters=args.max_iters_handoff,
        store_iterates=True,
        verbose=False,
        mu_barrier=0.0,
        initial_guess_label="direct_true_baseline",
    )

    artifacts = export_goddard_barrier_report_artifacts(
        barrier_run,
        handoff_bundle,
        baseline_result,
        baseline_prob,
        args.out_dir,
    )
    report_path = Path(artifacts["report_tex"])
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", report_path.name],
            cwd=report_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    main()
