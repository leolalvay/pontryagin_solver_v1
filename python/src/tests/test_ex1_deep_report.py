import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

from experiments.ex1_lqr import export_example1_deep_report_artifacts, run_ex1_lqr_solver


class Example1DeepReportTests(unittest.TestCase):
    def test_example1_report_export_writes_tables_figures_and_tex(self):
        result, prob = run_ex1_lqr_solver(
            n_init=10,
            tol_time=5.0e-3,
            tol_PA=1.0e-2,
            tol_delta=1.0e-2,
            max_iters=3,
            delta0=0.15,
            store_iterates=True,
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "example1_deep_report"
            artifacts = export_example1_deep_report_artifacts(
                result,
                prob,
                out_dir=out_dir,
                figure_ext="pdf",
            )

            report_path = Path(artifacts["report_tex"])
            summary_path = out_dir / "tables" / "final_summary.json"
            outer_history_path = out_dir / "tables" / "outer_loop_history.csv"
            input_path = out_dir / "tables" / "input_parameters.csv"
            reference_path = out_dir / "tables" / "reference_values.csv"
            history_figure = out_dir / "figures" / "error_indicator_history.pdf"
            iter0_dir = out_dir / "iterations" / "iter_00"

            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(outer_history_path.exists())
            self.assertTrue(input_path.exists())
            self.assertTrue(reference_path.exists())
            self.assertTrue(history_figure.exists())
            self.assertTrue((iter0_dir / "iter_00_state_costate.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_bundle_support_points.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_control.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_rho_density.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_mesh_and_indicator.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_pa_delta_contributions.pdf").exists())

            summary = json.loads(summary_path.read_text())
            self.assertIn("objective_mesh_approx", summary)
            self.assertIn("all_indicators_within_tolerance", summary)
            self.assertIn("final_action", summary)

            with outer_history_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("iteration", rows[0])
            self.assertIn("action", rows[0])
            self.assertIn("all_indicators_within_tolerance", rows[0])

            with input_path.open(newline="") as handle:
                input_rows = list(csv.DictReader(handle))
            parameter_names = {row["parameter"] for row in input_rows}
            self.assertIn("pa_add_fraction", parameter_names)
            self.assertIn("pa_time_separation_factor", parameter_names)
            self.assertIn("pa_gap_floor_ratio", parameter_names)

            tex = report_path.read_text()
            self.assertIn(r"\section{Example 1", tex)
            self.assertIn("Input parameters", tex)
            self.assertIn("Reference values", tex)
            self.assertIn("Outer-loop summary", tex)
            self.assertIn("Error-estimate history", tex)
            self.assertIn(r"\subsection{Iteration 0}", tex)


if __name__ == "__main__":
    unittest.main()
