from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from experiments.ex8_goddard_fixedtime import (
    export_goddard_deep_report_artifacts,
    run_goddard_solver,
)


def main():
    parser = argparse.ArgumentParser(description="Generate the deep report for the fixed-time penalized Goddard benchmark.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/reports/goddard_fixedtime_deep_report"),
    )
    parser.add_argument("--T", type=float, default=0.15)
    parser.add_argument("--q-max", type=float, default=12.75)
    parser.add_argument("--rho-m", type=float, default=1.0e4)
    parser.add_argument("--tol-time", type=float, default=1.0e-4)
    parser.add_argument("--tol-pa", type=float, default=1.0e-4)
    parser.add_argument("--tol-delta", type=float, default=1.0e-4)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--delta0", type=float, default=5.0e-2)
    args = parser.parse_args()

    result, prob = run_goddard_solver(
        T=args.T,
        q_max=args.q_max,
        rho_m=args.rho_m,
        n_init=8,
        tol_time=args.tol_time,
        tol_PA=args.tol_pa,
        tol_delta=args.tol_delta,
        max_iters=args.max_iters,
        delta0=args.delta0,
        store_iterates=True,
        verbose=False,
    )
    artifacts = export_goddard_deep_report_artifacts(result, prob, args.out_dir)
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
