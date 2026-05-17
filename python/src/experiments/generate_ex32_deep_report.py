import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.ex6_nonsmoothham import (
    export_ex32_deep_report_artifacts,
    run_nonsmooth_solver,
    summarize_nonsmooth_results,
)


def main():
    parser = argparse.ArgumentParser(description="Generate the standalone deep report for legacy Example 3.2.")
    parser.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parents[3]
            / "reports"
            / "legacy2014_ex32_deep_report"
        ),
        help="Output directory for the report artifacts.",
    )
    args = parser.parse_args()

    result, prob = run_nonsmooth_solver(store_iterates=True)
    summarize_nonsmooth_results(result, print_last_log_only=True)
    artifacts = export_ex32_deep_report_artifacts(result, prob, out_dir=args.out_dir)
    print(f"Legacy Example 3.2 deep report written to: {artifacts['out_dir']}")
    print(f"LaTeX report: {artifacts['report_tex']}")


if __name__ == "__main__":
    main()
