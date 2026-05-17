import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.ex1_lqr import export_example1_deep_report_artifacts, run_ex1_lqr_solver, summarize_ex1_results


def main():
    parser = argparse.ArgumentParser(description="Generate the standalone deep report for Example 1.")
    parser.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parents[3]
            / "reports"
            / "example1_deep_report"
        ),
        help="Output directory for the report artifacts.",
    )
    args = parser.parse_args()

    result, prob = run_ex1_lqr_solver(store_iterates=True)
    summarize_ex1_results(result, prob, print_last_log_only=True)
    artifacts = export_example1_deep_report_artifacts(result, prob, out_dir=args.out_dir)
    print(f"Example 1 deep report written to: {artifacts['out_dir']}")
    print(f"LaTeX report: {artifacts['report_tex']}")


if __name__ == "__main__":
    main()
