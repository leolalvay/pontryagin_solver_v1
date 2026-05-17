"""
Generate a standalone deep report for legacy 2014 Example 3.3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ex7_singular import export_ex33_deep_report_artifacts, run_singular_solver, summarize_singular_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/reports/legacy2014_ex33_deep_report"),
    )
    args = parser.parse_args()

    result, prob = run_singular_solver(store_iterates=True, verbose=True)
    summarize_singular_results(result)
    artifacts = export_ex33_deep_report_artifacts(result, prob, args.out_dir)
    print(f"wrote report sources to {artifacts['out_dir']}")
    print(f"report tex: {artifacts['report_tex']}")


if __name__ == "__main__":
    main()
