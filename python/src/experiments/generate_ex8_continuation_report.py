from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.ex8_goddard_fixedtime import (
    export_goddard_deep_report_artifacts,
    run_goddard_qmax_continuation,
)


def _parse_schedule(text: str):
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("q_schedule must contain at least one numeric value.")
    return values


def main():
    parser = argparse.ArgumentParser(description="Generate a Goddard deep report from a q_max continuation run.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/reports/goddard_fixedtime_qmax17_deep_report"),
    )
    parser.add_argument("--q-schedule", type=str, default="100,40,20,18,17")
    parser.add_argument("--T", type=float, default=0.15)
    parser.add_argument("--rho-m", type=float, default=1.0e4)
    parser.add_argument("--tol-time", type=float, default=1.0e-4)
    parser.add_argument("--tol-pa", type=float, default=1.0e-4)
    parser.add_argument("--tol-delta", type=float, default=1.0e-4)
    parser.add_argument("--delta0", type=float, default=5.0e-2)
    parser.add_argument("--stage-max-iters", type=int, default=3)
    parser.add_argument("--pressure-feasible-safety-factor", type=float, default=0.90)
    args = parser.parse_args()

    q_schedule = _parse_schedule(args.q_schedule)
    continuation = run_goddard_qmax_continuation(
        q_schedule=q_schedule,
        T=args.T,
        rho_m=args.rho_m,
        tol_time=args.tol_time,
        tol_PA=args.tol_pa,
        tol_delta=args.tol_delta,
        delta0=args.delta0,
        stage_max_iters=args.stage_max_iters,
        pressure_feasible_init=True,
        pressure_feasible_safety_factor=args.pressure_feasible_safety_factor,
        store_iterates=True,
        verbose=False,
    )
    if continuation["last_successful_result"] is None or continuation["last_successful_problem"] is None:
        raise RuntimeError("Continuation produced no successful stage; no report can be generated.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "continuation_summary.json").write_text(
        json.dumps({"q_schedule": continuation["q_schedule"], "stages": continuation["stages"]}, indent=2)
    )

    artifacts = export_goddard_deep_report_artifacts(
        continuation["last_successful_result"],
        continuation["last_successful_problem"],
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
